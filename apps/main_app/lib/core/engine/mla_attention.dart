// Multi-Head Latent Attention (MLA).
//
// Reference architecture: DeepSeek-V2 / DeepSeek-V3 low-rank KV joint
// compression. Vanilla multi-head attention materialises one key and one value
// vector per head per token, so the KV cache grows as
// `2 * numHeads * headDim * sizeof(float)` bytes per token. MLA instead
// compresses the whole KV state into a single low-rank latent vector and keeps
// only the *decoupled* rotary key alongside it:
//
//     c_kv   = W_dkv · x                     (kvLoraRank dims — the KV cache)
//     k_nope = W_uk  · c_kv                  (per-head, regenerated on the fly)
//     v      = W_uv  · c_kv                  (per-head, regenerated on the fly)
//     k_pe   = RoPE(W_kr · x)                (ropeHeadDim dims — the KV cache)
//     q      = W_uq  · (W_dq · x)            (low-rank query projection)
//
// Because `k_nope` and `v` are deterministic linear maps of `c_kv`, they never
// need to be stored. The cache therefore holds `kvLoraRank + ropeHeadDim`
// floats per token instead of `2 * numHeads * headDim`, which is the memory win
// that makes long-context on-device inference practical.
//
// This implementation is a pure-Dart, dependency-free reference: it performs
// real float32 linear algebra (no stubs) and is intended to run in an isolate
// on device, with platform-specific kernels substituted behind the same API.

import 'dart:math' as math;
import 'dart:typed_data';

/// Thrown when tensors are shaped incompatibly with the declared config.
class MlaShapeException implements Exception {
  MlaShapeException(this.message);

  final String message;

  @override
  String toString() => 'MlaShapeException: $message';
}

/// Thrown when the KV cache is asked for a position beyond its window.
class MlaCacheException implements Exception {
  MlaCacheException(this.message);

  final String message;

  @override
  String toString() => 'MlaCacheException: $message';
}

/// Immutable geometry of an MLA layer.
class MlaConfig {
  const MlaConfig({
    required this.hiddenSize,
    required this.numHeads,
    required this.headDim,
    required this.kvLoraRank,
    required this.ropeHeadDim,
    this.qLoraRank = 0,
    this.ropeTheta = 10000.0,
    this.maxSeqLen = 2048,
    this.epsilon = 1e-6,
  })  : assert(hiddenSize > 0, 'hiddenSize must be positive'),
        assert(numHeads > 0, 'numHeads must be positive'),
        assert(headDim > 0, 'headDim must be positive'),
        assert(kvLoraRank > 0, 'kvLoraRank must be positive'),
        assert(ropeHeadDim > 0, 'ropeHeadDim must be positive'),
        assert(ropeHeadDim <= headDim,
            'ropeHeadDim cannot exceed headDim (the non-RoPE part would be empty)'),
        assert(maxSeqLen > 0, 'maxSeqLen must be positive');

  /// Model width feeding this layer.
  final int hiddenSize;

  /// Number of attention heads.
  final int numHeads;

  /// Total per-head width (RoPE part + non-RoPE part).
  final int headDim;

  /// Rank of the compressed KV latent — the dominant cache term.
  final int kvLoraRank;

  /// Width of the decoupled rotary key, shared by every head.
  final int ropeHeadDim;

  /// Rank of the query low-rank projection; `0` disables query compression.
  final int qLoraRank;

  /// RoPE base frequency.
  final double ropeTheta;

  /// Sliding window capacity of the KV cache, in tokens.
  final int maxSeqLen;

  /// Numerical floor for softmax denominators.
  final double epsilon;

  /// Per-head width of the non-RoPE key/value part.
  int get nopeHeadDim => headDim - ropeHeadDim;

  /// Floats stored in the KV cache per token.
  int get cacheFloatsPerToken => kvLoraRank + ropeHeadDim;

  int get _qProjectRank => qLoraRank > 0 ? qLoraRank : hiddenSize;

  MlaConfig copyWith({int? maxSeqLen}) => MlaConfig(
        hiddenSize: hiddenSize,
        numHeads: numHeads,
        headDim: headDim,
        kvLoraRank: kvLoraRank,
        ropeHeadDim: ropeHeadDim,
        qLoraRank: qLoraRank,
        ropeTheta: ropeTheta,
        maxSeqLen: maxSeqLen ?? this.maxSeqLen,
        epsilon: epsilon,
      );
}

/// Memory accounting for a cache layout, in bytes (float32 elements).
class MlaCacheStats {
  const MlaCacheStats({
    required this.tokens,
    required this.config,
  });

  final int tokens;
  final MlaConfig config;

  /// Bytes this MLA layer stores per token: latent + decoupled rotary key.
  int get mlaBytesPerToken => config.cacheFloatsPerToken * 4;

  /// Bytes a vanilla MHA layer would store per token for the same geometry.
  int get vanillaBytesPerToken => 2 * config.numHeads * config.headDim * 4;

  /// Bytes currently resident in the MLA cache.
  int get mlaBytes => tokens * mlaBytesPerToken;

  /// Bytes an equivalent vanilla MHA cache would occupy.
  int get vanillaBytes => tokens * vanillaBytesPerToken;

  /// `vanilla / mla` — how many times smaller the latent cache is.
  double get compressionRatio =>
      mlaBytesPerToken == 0 ? 0 : vanillaBytesPerToken / mlaBytesPerToken;

  /// Bytes avoided versus vanilla MHA.
  int get bytesSaved => vanillaBytes - mlaBytes;

  Map<String, Object?> toJson() => <String, Object?>{
        'tokens': tokens,
        'mla_bytes_per_token': mlaBytesPerToken,
        'vanilla_bytes_per_token': vanillaBytesPerToken,
        'mla_bytes': mlaBytes,
        'vanilla_bytes': vanillaBytes,
        'bytes_saved': bytesSaved,
        'compression_ratio': double.parse(compressionRatio.toStringAsFixed(3)),
        'kv_lora_rank': config.kvLoraRank,
        'rope_head_dim': config.ropeHeadDim,
      };
}

/// Sliding-window KV cache holding only `c_kv` and `k_pe` per token.
class MlaKvCache {
  MlaKvCache(this.config)
      : _latent = Float32List(config.maxSeqLen * config.kvLoraRank),
        _rope = Float32List(config.maxSeqLen * config.ropeHeadDim);

  final MlaConfig config;
  final Float32List _latent;
  final Float32List _rope;

  int _length = 0;
  int _writeIndex = 0;

  /// Number of tokens currently resident.
  int get length => _length;

  /// Whether the window has wrapped and is discarding the oldest tokens.
  bool get isSaturated => _length >= config.maxSeqLen;

  Float32List get latentView => _latent;
  Float32List get ropeView => _rope;

  /// Logical-to-physical slot mapping, oldest token first.
  int slotFor(int logicalIndex) {
    if (logicalIndex < 0 || logicalIndex >= _length) {
      throw MlaCacheException(
        'logical index $logicalIndex outside cache length $_length',
      );
    }
    final int start = _length == config.maxSeqLen ? _writeIndex : 0;
    return (start + logicalIndex) % config.maxSeqLen;
  }

  /// Appends one token's compressed latent and rotary key.
  void append(Float32List latent, Float32List rope) {
    if (latent.length != config.kvLoraRank) {
      throw MlaShapeException(
        'latent must have ${config.kvLoraRank} elements, got ${latent.length}',
      );
    }
    if (rope.length != config.ropeHeadDim) {
      throw MlaShapeException(
        'rope key must have ${config.ropeHeadDim} elements, got ${rope.length}',
      );
    }
    final int slot = _writeIndex;
    _latent.setRange(
      slot * config.kvLoraRank,
      slot * config.kvLoraRank + config.kvLoraRank,
      latent,
    );
    _rope.setRange(
      slot * config.ropeHeadDim,
      slot * config.ropeHeadDim + config.ropeHeadDim,
      rope,
    );
    _writeIndex = (_writeIndex + 1) % config.maxSeqLen;
    if (_length < config.maxSeqLen) {
      _length++;
    }
  }

  /// Reads the latent vector of a logical token.
  Float32List latentAt(int logicalIndex) {
    final int slot = slotFor(logicalIndex);
    final int base = slot * config.kvLoraRank;
    return Float32List.sublistView(
      _latent,
      base,
      base + config.kvLoraRank,
    );
  }

  /// Reads the decoupled rotary key of a logical token.
  Float32List ropeAt(int logicalIndex) {
    final int slot = slotFor(logicalIndex);
    final int base = slot * config.ropeHeadDim;
    return Float32List.sublistView(_rope, base, base + config.ropeHeadDim);
  }

  /// Clears the window (called between independent sequences).
  void reset() {
    _length = 0;
    _writeIndex = 0;
  }

  MlaCacheStats get stats => MlaCacheStats(tokens: _length, config: config);
}

/// A dense weight matrix stored row-major as `[outFeatures][inFeatures]`.
class MlaLinear {
  MlaLinear(this.outFeatures, this.inFeatures, this.weights, [this.bias])
      : assert(weights.length == outFeatures * inFeatures, 'weight size');

  final int outFeatures;
  final int inFeatures;
  final Float32List weights;
  final Float32List? bias;

  /// Deterministic Xavier/Glorot-uniform initialisation from a seeded LCG so
  /// that results are reproducible in tests and across cold starts.
  factory MlaLinear.initialized(
    int outFeatures,
    int inFeatures,
    int seed,
  ) {
    final Float32List w = Float32List(outFeatures * inFeatures);
    final math.Random rng = math.Random(seed);
    final double limit = math.sqrt(6.0 / (inFeatures + outFeatures));
    for (int i = 0; i < w.length; i++) {
      w[i] = (rng.nextDouble() * 2.0 - 1.0) * limit;
    }
    return MlaLinear(outFeatures, inFeatures, w);
  }

  /// `y = W · x (+ bias)`.
  Float32List project(Float32List x) {
    if (x.length != inFeatures) {
      throw MlaShapeException(
        'expected input of $inFeatures, got ${x.length}',
      );
    }
    final Float32List out = Float32List(outFeatures);
    for (int o = 0; o < outFeatures; o++) {
      final int base = o * inFeatures;
      double acc = 0.0;
      for (int i = 0; i < inFeatures; i++) {
        acc += weights[base + i] * x[i];
      }
      final Float32List? b = bias;
      out[o] = b == null ? acc : acc + b[o];
    }
    return out;
  }
}

/// Rotary position embedding applied to the decoupled key (and queries).
class RotaryEmbedding {
  RotaryEmbedding({required this.dim, required this.theta, required this.maxSeqLen})
      : assert(dim.isEven, 'RoPE dimension must be even'),
        _cos = Float32List(maxSeqLen * (dim ~/ 2)),
        _sin = Float32List(maxSeqLen * (dim ~/ 2)) {
    _precompute();
  }

  final int dim;
  final double theta;
  final int maxSeqLen;
  final Float32List _cos;
  final Float32List _sin;

  void _precompute() {
    final int half = dim ~/ 2;
    for (int pos = 0; pos < maxSeqLen; pos++) {
      for (int i = 0; i < half; i++) {
        final double freq = 1.0 / math.pow(theta, (2 * i) / dim);
        final double angle = pos * freq;
        _cos[pos * half + i] = math.cos(angle);
        _sin[pos * half + i] = math.sin(angle);
      }
    }
  }

  /// Rotates adjacent element pairs of [vector] in place for [position].
  void applyInPlace(Float32List vector, int position) {
    if (vector.length != dim) {
      throw MlaShapeException('RoPE expected dim $dim, got ${vector.length}');
    }
    if (position < 0 || position >= maxSeqLen) {
      throw MlaCacheException(
        'position $position outside RoPE table of $maxSeqLen',
      );
    }
    final int half = dim ~/ 2;
    final int base = position * half;
    for (int i = 0; i < half; i++) {
      final double c = _cos[base + i];
      final double s = _sin[base + i];
      final double x0 = vector[2 * i];
      final double x1 = vector[2 * i + 1];
      vector[2 * i] = x0 * c - x1 * s;
      vector[2 * i + 1] = x0 * s + x1 * c;
    }
  }

  /// Returns a rotated copy of [vector].
  Float32List apply(Float32List vector, int position) {
    final Float32List out = Float32List.fromList(vector);
    applyInPlace(out, position);
    return out;
  }
}

/// Output of a single MLA forward pass.
class MlaForwardResult {
  const MlaForwardResult({
    required this.output,
    required this.position,
    required this.attendedTokens,
  });

  /// Layer output, `hiddenSize` wide.
  final Float32List output;

  /// Position of the token that was just processed.
  final int position;

  /// Number of cached tokens the query attended over (inclusive).
  final int attendedTokens;
}

/// Multi-Head Latent Attention layer with an O(1)-per-token compressed cache.
class MultiHeadLatentAttention {
  MultiHeadLatentAttention(this.config, {int seed = 0x5EED})
      : _wDq = MlaLinear.initialized(config._qProjectRank, config.hiddenSize, seed + 1),
        _wUq = MlaLinear.initialized(
          config.numHeads * config.headDim,
          config._qProjectRank,
          seed + 2,
        ),
        _wDkv = MlaLinear.initialized(config.kvLoraRank, config.hiddenSize, seed + 3),
        _wUk = MlaLinear.initialized(
          config.numHeads * config.nopeHeadDim,
          config.kvLoraRank,
          seed + 4,
        ),
        _wUv = MlaLinear.initialized(
          config.numHeads * config.headDim,
          config.kvLoraRank,
          seed + 5,
        ),
        _wKr = MlaLinear.initialized(config.ropeHeadDim, config.hiddenSize, seed + 6),
        _wO = MlaLinear.initialized(config.hiddenSize, config.numHeads * config.headDim, seed + 7),
        cache = MlaKvCache(config),
        _rope = RotaryEmbedding(
          dim: config.ropeHeadDim,
          theta: config.ropeTheta,
          maxSeqLen: config.maxSeqLen,
        );

  final MlaConfig config;

  final MlaLinear _wDq;
  final MlaLinear _wUq;
  final MlaLinear _wDkv;
  final MlaLinear _wUk;
  final MlaLinear _wUv;
  final MlaLinear _wKr;
  final MlaLinear _wO;

  /// Sliding-window compressed KV cache. Exposed for memory introspection.
  final MlaKvCache cache;
  final RotaryEmbedding _rope;

  /// Runs one decode step for a single token.
  ///
  /// [x] must be `hiddenSize` wide; [position] is the absolute token index and
  /// must match the cache length for a left-to-right decode.
  MlaForwardResult forward(Float32List x, int position) {
    if (x.length != config.hiddenSize) {
      throw MlaShapeException(
        'input must have ${config.hiddenSize} elements, got ${x.length}',
      );
    }

    // 1. Low-rank query projection: q = W_uq · (W_dq · x)
    final Float32List qCompressed = _wDq.project(x);
    final Float32List q = _wUq.project(qCompressed);

    // 2. Compressed KV latent: the only per-token K/V state we persist.
    final Float32List cKv = _wDkv.project(x);

    // 3. Decoupled rotary key, shared across heads.
    final Float32List kPe = _rope.apply(_wKr.project(x), position);

    cache.append(cKv, kPe);
    final int attended = cache.length;

    // 4. Rebuild non-RoPE keys and values for every cached token on the fly.
    //    These are never stored, which is the source of the memory saving.
    final int nope = config.nopeHeadDim;
    final int headDim = config.headDim;
    final int ropeDim = config.ropeHeadDim;

    final Float32List kFlat = Float32List(attended * config.numHeads * headDim);
    final Float32List vFlat = Float32List(attended * config.numHeads * headDim);
    for (int t = 0; t < attended; t++) {
      final Float32List latentT = cache.latentAt(t);
      final Float32List kNoPeT = _wUk.project(latentT);
      final Float32List vT = _wUv.project(latentT);
      final Float32List kPeT = cache.ropeAt(t);
      for (int h = 0; h < config.numHeads; h++) {
        final int kBase = (t * config.numHeads + h) * headDim;
        final int nopeBase = h * nope;
        for (int d = 0; d < nope; d++) {
          kFlat[kBase + d] = kNoPeT[nopeBase + d];
        }
        for (int d = 0; d < ropeDim; d++) {
          kFlat[kBase + nope + d] = kPeT[d];
        }
        final int vBase = h * headDim;
        for (int d = 0; d < headDim; d++) {
          vFlat[kBase + d] = vT[vBase + d];
        }
      }
    }

    // 5. Scaled dot-product attention per head over the window.
    final Float32List context = Float32List(config.numHeads * headDim);
    final double scale = 1.0 / math.sqrt(headDim.toDouble());
    final Float32List scores = Float32List(attended);
    for (int h = 0; h < config.numHeads; h++) {
      final int qBase = h * headDim;
      for (int t = 0; t < attended; t++) {
        final int kBase = (t * config.numHeads + h) * headDim;
        double dot = 0.0;
        for (int d = 0; d < headDim; d++) {
          dot += q[qBase + d] * kFlat[kBase + d];
        }
        scores[t] = dot * scale;
      }
      _softmaxInPlace(scores);
      final int ctxBase = h * headDim;
      for (int t = 0; t < attended; t++) {
        final double p = scores[t];
        if (p == 0.0) {
          continue;
        }
        final int vBase = (t * config.numHeads + h) * headDim;
        for (int d = 0; d < headDim; d++) {
          context[ctxBase + d] += p * vFlat[vBase + d];
        }
      }
    }

    // 6. Output projection back to model width.
    return MlaForwardResult(
      output: _wO.project(context),
      position: position,
      attendedTokens: attended,
    );
  }

  /// Runs a full prompt through the layer, returning the last hidden state.
  Float32List forwardSequence(List<Float32List> tokens, {int startPosition = 0}) {
    if (tokens.isEmpty) {
      throw MlaShapeException('forwardSequence requires at least one token');
    }
    Float32List out = Float32List(config.hiddenSize);
    for (int i = 0; i < tokens.length; i++) {
      out = forward(tokens[i], startPosition + i).output;
    }
    return out;
  }

  void _softmaxInPlace(Float32List logits) {
    if (logits.isEmpty) {
      return;
    }
    double max = logits[0];
    for (int i = 1; i < logits.length; i++) {
      if (logits[i] > max) {
        max = logits[i];
      }
    }
    double sum = 0.0;
    for (int i = 0; i < logits.length; i++) {
      final double e = math.exp(logits[i] - max);
      logits[i] = e;
      sum += e;
    }
    final double denom = sum < config.epsilon ? config.epsilon : sum;
    for (int i = 0; i < logits.length; i++) {
      logits[i] = logits[i] / denom;
    }
  }

  /// Wipes the KV cache. Call between unrelated requests.
  void resetCache() => cache.reset();

  /// Current cache memory report.
  MlaCacheStats get cacheStats => cache.stats;

  /// Human-readable summary used by the diagnostics screen.
  Map<String, Object?> describe() => <String, Object?>{
        'architecture': 'multi_head_latent_attention',
        'hidden_size': config.hiddenSize,
        'num_heads': config.numHeads,
        'head_dim': config.headDim,
        'nope_head_dim': config.nopeHeadDim,
        'rope_head_dim': config.ropeHeadDim,
        'kv_lora_rank': config.kvLoraRank,
        'q_lora_rank': config._qProjectRank,
        'max_seq_len': config.maxSeqLen,
        ...cacheStats.toJson(),
      };
}