// The transformer building blocks: linear projections with attachable LoRA,
// multi-head latent attention (MLA), and a fine-grained mixture-of-experts
// feed-forward network.
//
// Every block implements `forward` and `backward` as a matched pair that share
// an explicit activation cache. The cache is *passed* rather than stored on the
// block, which is what makes the same weight object usable at several sequence
// positions in one pass without the backward pass picking up the wrong input.
// That mistake is the single easiest way to write a transformer that trains to
// a plausible-looking but wrong loss, so the type system is used to prevent it.

import 'dart:math' as math;
import 'dart:typed_data';

import 'encoding.dart';
import 'linalg.dart';

/// Everything the backward pass of a [Linear] needs from its forward pass.
class LinearCache {
  /// Creates a cache from the forward pass's input, output and LoRA
  /// bottleneck activation.
  LinearCache(this.input, this.output, this.loraHidden);

  /// The input vector.
  final Float64List input;

  /// The output vector, `W x + b (+ scaling · B A x)`.
  final Float64List output;

  /// The LoRA bottleneck activation `A x`, or null when no adapter is attached.
  final Float64List? loraHidden;
}

/// A dense projection `y = W x + b` with an optional LoRA adapter.
///
/// When an adapter is attached the base weight is *frozen*: `gradWeight` is no
/// longer written and only the low-rank factors receive gradients. That is the
/// whole point of LoRA — a handful of trainable scalars on top of a fixed
/// backbone — and it is enforced here rather than left to convention.
class Linear {
  /// Builds a projection of [inFeatures] to [outFeatures].
  ///
  /// [name] prefixes every parameter name; names are used for checkpoint keys
  /// and for the diagnostics table, so they must be unique per model.
  Linear({
    required this.name,
    required this.inFeatures,
    required this.outFeatures,
    required math.Random random,
    bool bias = true,
    Matrix? weight,
    Float64List? biasVector,
  })  : weight = weight ?? Matrix.xavier(outFeatures, inFeatures, random),
        bias = bias ? (biasVector ?? Float64List(outFeatures)) : Float64List(0),
        gradBias = Float64List(bias ? outFeatures : 0) {
    if (this.weight.rows != outFeatures || this.weight.cols != inFeatures) {
      throw ArgumentError(
        'supplied weight for "$name" is ${this.weight.rows}x'
        '${this.weight.cols}, expected ${outFeatures}x$inFeatures',
      );
    }
    if (biasVector != null && biasVector.length != outFeatures) {
      throw ArgumentError(
        'supplied bias for "$name" has ${biasVector.length} entries, expected '
        '$outFeatures',
      );
    }
  }

  /// Parameter-name prefix.
  final String name;

  /// Input width.
  final int inFeatures;

  /// Output width.
  final int outFeatures;

  /// The base weight, `[outFeatures, inFeatures]`.
  final Matrix weight;

  /// The base bias; empty when the projection is bias-free.
  final Float64List bias;

  /// Gradient of [bias]; empty when bias-free.
  final Float64List gradBias;

  Matrix? _loraA;
  Matrix? _loraB;
  Matrix? _gradLoraA;
  Matrix? _gradLoraB;
  double _loraScaling = 1.0;

  /// True when a LoRA adapter is currently contributing to the output.
  bool get hasAdapter => _loraA != null && _loraB != null;

  /// The adapter rank, or zero when no adapter is attached.
  int get adapterRank => _loraA?.rows ?? 0;

  /// The factor applied to the low-rank branch, `alpha / rank` by default.
  double get adapterScaling => _loraScaling;

  /// The LoRA `A` factor, `[rank, inFeatures]`, or null.
  Matrix? get adapterA => _loraA;

  /// The LoRA `B` factor, `[outFeatures, rank]`, or null.
  Matrix? get adapterB => _loraB;

  /// Attaches a fresh low-rank adapter.
  ///
  /// `A` is initialised from a normal distribution and `B` to zero, so the
  /// adapted model starts out numerically identical to the base model — the
  /// standard LoRA initialisation, and the only one that makes a training curve
  /// comparable across runs.
  void attachAdapter(int rank, math.Random random, {double alpha = 8.0}) {
    if (rank <= 0) {
      throw ArgumentError.value(rank, 'rank', 'must be positive');
    }
    if (rank > math.min(inFeatures, outFeatures)) {
      throw ArgumentError(
        'rank $rank exceeds the smallest dimension of "$name" '
        '(${math.min(inFeatures, outFeatures)})',
      );
    }
    _loraA = Matrix.normal(rank, inFeatures, random, std: 0.02);
    _loraB = Matrix(outFeatures, rank);
    _gradLoraA = Matrix(rank, inFeatures);
    _gradLoraB = Matrix(outFeatures, rank);
    _loraScaling = alpha / rank;
  }

  /// Removes the adapter and every gradient buffer that belonged to it.
  void detachAdapter() {
    _loraA = null;
    _loraB = null;
    _gradLoraA = null;
    _gradLoraB = null;
  }

  /// The rank-`r` update the adapter currently represents, `scaling · B A`.
  ///
  /// Exposed so a test can assert that folding the adapter into the base weight
  /// is exactly equivalent to evaluating the adapted projection, which is the
  /// property that makes an exported adapter meaningful.
  Matrix adapterDelta() {
    final Matrix delta = Matrix(outFeatures, inFeatures);
    if (!hasAdapter) {
      return delta;
    }
    final Matrix a = _loraA!;
    final Matrix b = _loraB!;
    final int rank = a.rows;
    for (int r = 0; r < outFeatures; r++) {
      for (int c = 0; c < inFeatures; c++) {
        double sum = 0;
        for (int k = 0; k < rank; k++) {
          sum += b.at(r, k) * a.at(k, c);
        }
        delta.set(r, c, sum * _loraScaling);
      }
    }
    return delta;
  }

  /// Folds the adapter into the base weight and detaches it.
  ///
  /// After this call `forward` produces exactly what it produced while the
  /// adapter was attached, with no change in cost.
  void mergeAdapter() {
    if (!hasAdapter) {
      return;
    }
    final Matrix delta = adapterDelta();
    for (int i = 0; i < weight.values.length; i++) {
      weight.values[i] += delta.values[i];
    }
    detachAdapter();
  }

  /// `y = W x + b`, plus the low-rank branch when an adapter is attached.
  LinearCache forward(Float64List x) {
    if (x.length != inFeatures) {
      throw ArgumentError(
        '"$name" expected $inFeatures inputs, got ${x.length}',
      );
    }
    final Float64List out = weight.matVec(x);
    for (int i = 0; i < bias.length; i++) {
      out[i] += bias[i];
    }
    Float64List? hidden;
    if (hasAdapter) {
      hidden = _loraA!.matVec(x);
      final Float64List up = _loraB!.matVec(hidden);
      for (int i = 0; i < out.length; i++) {
        out[i] += _loraScaling * up[i];
      }
    }
    return LinearCache(x, out, hidden);
  }

  /// The vector-Jacobian product of [forward], accumulating parameter grads.
  Float64List backward(Float64List gradOut, LinearCache cache) {
    if (gradOut.length != outFeatures) {
      throw ArgumentError(
        '"$name" expected $outFeatures output gradients, got ${gradOut.length}',
      );
    }
    final Float64List gradInput = weight.matVecTranspose(gradOut);
    if (hasAdapter) {
      final Float64List gradUp = Float64List(outFeatures);
      for (int i = 0; i < outFeatures; i++) {
        gradUp[i] = gradOut[i] * _loraScaling;
      }
      _gradLoraB!.addOuter(gradUp, cache.loraHidden!);
      final Float64List gradHidden = _loraB!.matVecTranspose(gradUp);
      _gradLoraA!.addOuter(gradHidden, cache.input);
      final Float64List fromAdapter = _loraA!.matVecTranspose(gradHidden);
      for (int i = 0; i < gradInput.length; i++) {
        gradInput[i] += fromAdapter[i];
      }
      // The base weight stays frozen while an adapter is attached.
      return gradInput;
    }
    weight.addOuter(gradOut, cache.input);
    for (int i = 0; i < gradBias.length; i++) {
      gradBias[i] += gradOut[i];
    }
    return gradInput;
  }

  /// Every trainable tensor, in a deterministic order.
  List<ParamRef> parameters() {
    // While an adapter is attached the base matrix is frozen, so the base
    // tensors must not appear here. This list is what the optimizer iterates:
    // returning the base weights as well would apply LoRA-shaped gradients to
    // the frozen matrix and quietly turn "train the adapter" into "train
    // everything", which is a different experiment with the same name.
    if (hasAdapter) {
      return adapterParameters();
    }
    return <ParamRef>[
      ParamRef('$name.weight', weight.values, weight.grads),
      if (bias.isNotEmpty) ParamRef('$name.bias', bias, gradBias),
    ];
  }

  /// The adapter tensors only; empty when no adapter is attached.
  List<ParamRef> adapterParameters() {
    if (!hasAdapter) {
      return const <ParamRef>[];
    }
    return <ParamRef>[
      ParamRef('$name.lora_a', _loraA!.values, _gradLoraA!.grads),
      ParamRef('$name.lora_b', _loraB!.values, _gradLoraB!.grads),
    ];
  }

  /// Number of *trainable* scalars, which excludes the frozen base weight while
  /// an adapter is attached.
  int get trainableParameterCount {
    if (hasAdapter) {
      return _loraA!.length + _loraB!.length;
    }
    return weight.length + bias.length;
  }

  /// Zeroes every gradient buffer, including the adapter's.
  void zeroGrad() {
    weight.zeroGrad();
    gradBias.fillRange(0, gradBias.length, 0);
    _gradLoraA?.zeroGrad();
    _gradLoraB?.zeroGrad();
  }

  /// Serialises the weights, and the adapter when one is attached.
  Map<String, Object?> toJson() => <String, Object?>{
        'in': inFeatures,
        'out': outFeatures,
        'w': encodeFloats(weight.values),
        if (bias.isNotEmpty) 'b': encodeFloats(bias),
        if (hasAdapter) 'lora_a': encodeFloats(_loraA!.values),
        if (hasAdapter) 'lora_b': encodeFloats(_loraB!.values),
        if (hasAdapter) 'lora_scaling': _loraScaling,
      };

  /// Restores weights written by [toJson].
  ///
  /// Shape mismatches are rejected rather than partially applied: a checkpoint
  /// from a differently-sized model must fail here, not produce a silently
  /// corrupted network.
  void loadJson(Map<String, Object?> json) {
    final Object? inRaw = json['in'];
    final Object? outRaw = json['out'];
    if (inRaw != inFeatures || outRaw != outFeatures) {
      throw FormatException(
        '"$name" checkpoint is ${inRaw}x$outRaw, model expects '
        '${inFeatures}x$outFeatures',
      );
    }
    final Object? weightRaw = json['w'];
    if (weightRaw is! String) {
      throw FormatException('"$name" checkpoint has no weight payload');
    }
    final Float64List restored = decodeFloats(weightRaw);
    if (restored.length != weight.length) {
      throw FormatException(
        '"$name" checkpoint weight has ${restored.length} values, expected '
        '${weight.length}',
      );
    }
    weight.values.setAll(0, restored);

    final Object? biasRaw = json['b'];
    if (biasRaw is String) {
      final Float64List restoredBias = decodeFloats(biasRaw);
      if (restoredBias.length != bias.length) {
        throw FormatException('"$name" checkpoint bias has the wrong length');
      }
      bias.setAll(0, restoredBias);
    }

    final Object? loraA = json['lora_a'];
    final Object? loraB = json['lora_b'];
    if (loraA is String && loraB is String) {
      final Float64List a = decodeFloats(loraA);
      final Float64List b = decodeFloats(loraB);
      final int rank = a.length ~/ inFeatures;
      if (a.length % inFeatures != 0 || b.length != outFeatures * rank) {
        throw FormatException('"$name" checkpoint adapter has the wrong shape');
      }
      // `attachAdapter` takes alpha and derives `scaling = alpha / rank`, so
      // the stored scaling is recovered by multiplying it back out.
      final Object? scalingRaw = json['lora_scaling'];
      final double scaling =
          scalingRaw is num ? scalingRaw.toDouble() : 8.0 / rank;
      attachAdapter(rank, math.Random(0), alpha: scaling * rank);
      _loraA!.values.setAll(0, a);
      _loraB!.values.setAll(0, b);
    }
  }
}

/// The activation cache of one MLA block over a whole sequence.
class MlaAttentionCache {
  /// Creates an empty cache sized for [length] positions.
  MlaAttentionCache(this.length);

  /// Number of positions.
  final int length;

  /// Per-position input vectors.
  late final List<Float64List> inputs = List<Float64List>.filled(
    length,
    Float64List(0),
  );

  /// Per-position projection caches.
  late final List<LinearCache> queryCaches = List<LinearCache>.filled(
    length,
    LinearCache(Float64List(0), Float64List(0), null),
  );

  /// Per-position latent down-projection caches.
  late final List<LinearCache> latentCaches = List<LinearCache>.filled(
    length,
    LinearCache(Float64List(0), Float64List(0), null),
  );

  /// Per-position key up-projection caches.
  late final List<LinearCache> keyCaches = List<LinearCache>.filled(
    length,
    LinearCache(Float64List(0), Float64List(0), null),
  );

  /// Per-position value up-projection caches.
  late final List<LinearCache> valueCaches = List<LinearCache>.filled(
    length,
    LinearCache(Float64List(0), Float64List(0), null),
  );

  /// Per-position output projection caches.
  late final List<LinearCache> outputCaches = List<LinearCache>.filled(
    length,
    LinearCache(Float64List(0), Float64List(0), null),
  );

  /// Post-rotation queries, `[nHeads * headDim]` per position.
  late final List<Float64List> queries = List<Float64List>.filled(
    length,
    Float64List(0),
  );

  /// Post-rotation keys, `[nHeads * headDim]` per position.
  late final List<Float64List> keys = List<Float64List>.filled(
    length,
    Float64List(0),
  );

  /// Values, `[nHeads * headDim]` per position.
  late final List<Float64List> values = List<Float64List>.filled(
    length,
    Float64List(0),
  );

  /// Attention probabilities per position, laid out as
  /// `[head * (position + 1) + keyIndex]` so the causal triangle needs no
  /// padding.
  late final List<Float64List> probabilities = List<Float64List>.filled(
    length,
    Float64List(0),
  );

  /// Concatenated per-head attention outputs, the input to the output
  /// projection.
  late final List<Float64List> headOutputs = List<Float64List>.filled(
    length,
    Float64List(0),
  );

  /// The per-position block outputs, before the residual addition.
  late final List<Float64List> outputs = List<Float64List>.filled(
    length,
    Float64List(0),
  );
}

/// An incremental key/value cache for one MLA block.
///
/// Holds the post-rotation keys and values for every position decoded so far,
/// plus the shared latent each pair was reconstructed from. The latent is what
/// makes the compression real: a plain multi-head cache would carry
/// `2 · nHeads · headDim` values per position, while the latent this
/// architecture actually has to persist is `kvRank` — and the keys and values
/// stored alongside it are a cache, not the model state that has to survive a
/// restart.
class MlaKvCache {
  /// Post-rotation keys, one `nHeads * headDim` vector per position.
  final List<Float64List> keys = <Float64List>[];

  /// Values, one `nHeads * headDim` vector per position.
  final List<Float64List> values = <Float64List>[];

  /// The shared latent `W_dkv x` each position's keys and values came from.
  final List<Float64List> latents = <Float64List>[];

  /// Number of positions cached.
  int get length => keys.length;

  /// Number of scalar values the *latents* account for — the figure the
  /// compression ratio is measured against.
  int get storedLatentValues {
    if (latents.isEmpty) {
      return 0;
    }
    return latents.length * latents.first.length;
  }

  /// Empties the cache.
  void clear() {
    keys.clear();
    values.clear();
    latents.clear();
  }
}

/// Multi-head latent attention.
///
/// Keys and values are not projected directly from the residual stream. Instead
/// a single low-rank latent `c = W_dkv x` is stored per position and keys and
/// values are reconstructed from it (`k = W_uk c`, `v = W_uv c`). The cache a
/// decoder must carry therefore shrinks from `2 · nHeads · headDim` to `kvRank`
/// values per position per layer — the compression this architecture exists for.
class MlaAttention {
  /// Builds the block.
  ///
  /// [headDim] must be even because rotary embeddings rotate channel pairs, and
  /// [contextLength] bounds the rotary table as well as the attention window.
  MlaAttention({
    required this.name,
    required this.dModel,
    required this.nHeads,
    required this.headDim,
    required this.kvRank,
    required this.contextLength,
    required math.Random random,
  })  : query = Linear(
          name: '$name.q',
          inFeatures: dModel,
          outFeatures: nHeads * headDim,
          random: random,
          bias: false,
        ),
        latentDown = Linear(
          name: '$name.kv_down',
          inFeatures: dModel,
          outFeatures: kvRank,
          random: random,
          bias: false,
        ),
        keyUp = Linear(
          name: '$name.k_up',
          inFeatures: kvRank,
          outFeatures: nHeads * headDim,
          random: random,
          bias: false,
        ),
        valueUp = Linear(
          name: '$name.v_up',
          inFeatures: kvRank,
          outFeatures: nHeads * headDim,
          random: random,
          bias: false,
        ),
        output = Linear(
          name: '$name.out',
          inFeatures: nHeads * headDim,
          outFeatures: dModel,
          random: random,
          bias: false,
        ),
        rope = RotaryEmbedding(dim: headDim, maxPositions: contextLength) {
    if (headDim.isOdd) {
      throw ArgumentError.value(headDim, 'headDim', 'must be even for rotary');
    }
    if (kvRank <= 0) {
      throw ArgumentError.value(kvRank, 'kvRank', 'must be positive');
    }
  }

  /// Parameter-name prefix.
  final String name;

  /// Residual-stream width.
  final int dModel;

  /// Number of attention heads.
  final int nHeads;

  /// Width of one head.
  final int headDim;

  /// Rank of the shared key/value latent.
  final int kvRank;

  /// Maximum number of attended positions.
  final int contextLength;

  /// Query projection, `dModel -> nHeads * headDim`.
  final Linear query;

  /// Latent down-projection, `dModel -> kvRank`.
  final Linear latentDown;

  /// Key up-projection, `kvRank -> nHeads * headDim`.
  final Linear keyUp;

  /// Value up-projection, `kvRank -> nHeads * headDim`.
  final Linear valueUp;

  /// Output projection, `nHeads * headDim -> dModel`.
  final Linear output;

  /// Rotary position embedding tables.
  final RotaryEmbedding rope;

  /// Number of values a KV cache stores per position with this block.
  int get cacheValuesPerPosition => kvRank;

  /// Number of values a plain multi-head attention cache would store.
  int get uncompressedCacheValuesPerPosition => 2 * nHeads * headDim;

  /// The compression factor of the cache, always `>= 1` for a sane rank.
  double get cacheCompressionRatio =>
      uncompressedCacheValuesPerPosition / cacheValuesPerPosition;

  /// Every trainable tensor in this block.
  List<ParamRef> parameters() => <ParamRef>[
        ...query.parameters(),
        ...latentDown.parameters(),
        ...keyUp.parameters(),
        ...valueUp.parameters(),
        ...output.parameters(),
      ];

  /// Zeroes every gradient in this block.
  void zeroGrad() {
    query.zeroGrad();
    latentDown.zeroGrad();
    keyUp.zeroGrad();
    valueUp.zeroGrad();
    output.zeroGrad();
  }

  /// Runs the block over [inputs], returning a cache whose `outputs` field
  /// holds one output vector per position.
  ///
  /// Causal masking is implicit: position `i` only attends to positions `<= i`.
  /// The cache is returned rather than retained so the same block can be reused
  /// for a second sequence without the backward pass of the first picking up
  /// the wrong activations.
  MlaAttentionCache forward(List<Float64List> inputs) {
    if (inputs.isEmpty) {
      throw ArgumentError.value(inputs, 'inputs', 'must not be empty');
    }
    if (inputs.length > contextLength) {
      throw ArgumentError(
        '"$name" received ${inputs.length} positions but its rotary table and '
        'context window cover $contextLength',
      );
    }
    final MlaAttentionCache cache = MlaAttentionCache(inputs.length);
    final double scale = 1.0 / math.sqrt(headDim);

    for (int i = 0; i < inputs.length; i++) {
      cache.inputs[i] = inputs[i];
      final LinearCache qCache = query.forward(inputs[i]);
      final LinearCache latentCache = latentDown.forward(inputs[i]);
      final LinearCache kCache = keyUp.forward(latentCache.output);
      final LinearCache vCache = valueUp.forward(latentCache.output);

      final Float64List q = Float64List.fromList(qCache.output);
      final Float64List k = Float64List.fromList(kCache.output);
      for (int h = 0; h < nHeads; h++) {
        rope.apply(
          Float64List.sublistView(q, h * headDim, (h + 1) * headDim),
          i,
        );
        rope.apply(
          Float64List.sublistView(k, h * headDim, (h + 1) * headDim),
          i,
        );
      }

      cache
        ..queryCaches[i] = qCache
        ..latentCaches[i] = latentCache
        ..keyCaches[i] = kCache
        ..valueCaches[i] = vCache
        ..queries[i] = q
        ..keys[i] = k
        ..values[i] = Float64List.fromList(vCache.output);
    }

    for (int i = 0; i < inputs.length; i++) {
      final Float64List headOutput = Float64List(nHeads * headDim);
      cache.headOutputs[i] = headOutput;
      for (int h = 0; h < nHeads; h++) {
        final int qBase = h * headDim;
        final Float64List probs = Float64List(i + 1);
        double maxScore = double.negativeInfinity;
        for (int j = 0; j <= i; j++) {
          double dot = 0;
          final int kBase = h * headDim;
          for (int d = 0; d < headDim; d++) {
            dot += cache.queries[i][qBase + d] * cache.keys[j][kBase + d];
          }
          final double score = dot * scale;
          probs[j] = score;
          if (score > maxScore) {
            maxScore = score;
          }
        }
        double total = 0;
        for (int j = 0; j <= i; j++) {
          final double value = math.exp(probs[j] - maxScore);
          probs[j] = value;
          total += value;
        }
        if (total <= 0 || !total.isFinite) {
          final double uniform = 1.0 / (i + 1);
          probs.fillRange(0, i + 1, uniform);
        } else {
          final double inverse = 1.0 / total;
          for (int j = 0; j <= i; j++) {
            probs[j] *= inverse;
          }
        }
        cache.probabilities[i] = _appendRow(cache.probabilities[i], probs);
        for (int j = 0; j <= i; j++) {
          final double weight = probs[j];
          if (weight == 0) {
            continue;
          }
          final int vBase = h * headDim;
          for (int d = 0; d < headDim; d++) {
            headOutput[qBase + d] += weight * cache.values[j][vBase + d];
          }
        }
      }
      final LinearCache outCache = output.forward(headOutput);
      cache.outputCaches[i] = outCache;
      cache.outputs[i] = outCache.output;
    }
    return cache;
  }

  /// Decodes a single position against [cache], appending to it.
  ///
  /// This is the incremental path used for generation: cost per token is linear
  /// in the number of positions already cached rather than quadratic, because
  /// the earlier positions' keys and values are reused instead of recomputed.
  /// `test/model_test.dart` asserts that this path and the full-sequence
  /// [forward] path agree to within floating-point noise, which is the only
  /// thing that makes an incremental decoder trustworthy.
  Float64List forwardStep(Float64List x, MlaKvCache cache) {
    if (x.length != dModel) {
      throw ArgumentError(
        '"$name" expected $dModel inputs, got ${x.length}',
      );
    }
    final int position = cache.length;
    if (position >= contextLength) {
      throw ArgumentError(
        '"$name" cannot decode position $position: the context window is '
        '$contextLength',
      );
    }
    final LinearCache qCache = query.forward(x);
    final LinearCache latentCache = latentDown.forward(x);
    final LinearCache kCache = keyUp.forward(latentCache.output);
    final LinearCache vCache = valueUp.forward(latentCache.output);

    final Float64List q = Float64List.fromList(qCache.output);
    final Float64List k = Float64List.fromList(kCache.output);
    final Float64List v = Float64List.fromList(vCache.output);
    for (int h = 0; h < nHeads; h++) {
      rope.apply(
        Float64List.sublistView(q, h * headDim, (h + 1) * headDim),
        position,
      );
      rope.apply(
        Float64List.sublistView(k, h * headDim, (h + 1) * headDim),
        position,
      );
    }
    cache.keys.add(k);
    cache.values.add(v);
    cache.latents.add(Float64List.fromList(latentCache.output));

    final double scale = 1.0 / math.sqrt(headDim);
    final Float64List headOutput = Float64List(nHeads * headDim);
    final int cached = cache.length;
    for (int h = 0; h < nHeads; h++) {
      final int qBase = h * headDim;
      final Float64List probs = Float64List(cached);
      double maxScore = double.negativeInfinity;
      for (int j = 0; j < cached; j++) {
        double dot = 0;
        final int kBase = h * headDim;
        for (int d = 0; d < headDim; d++) {
          dot += q[qBase + d] * cache.keys[j][kBase + d];
        }
        final double score = dot * scale;
        probs[j] = score;
        if (score > maxScore) {
          maxScore = score;
        }
      }
      double total = 0;
      for (int j = 0; j < cached; j++) {
        final double value = math.exp(probs[j] - maxScore);
        probs[j] = value;
        total += value;
      }
      if (total <= 0 || !total.isFinite) {
        probs.fillRange(0, cached, 1.0 / cached);
      } else {
        final double inverse = 1.0 / total;
        for (int j = 0; j < cached; j++) {
          probs[j] *= inverse;
        }
      }
      for (int j = 0; j < cached; j++) {
        final double weight = probs[j];
        if (weight == 0) {
          continue;
        }
        final int vBase = h * headDim;
        for (int d = 0; d < headDim; d++) {
          headOutput[qBase + d] += weight * cache.values[j][vBase + d];
        }
      }
    }
    return output.forward(headOutput).output;
  }

  /// The vector-Jacobian product of [forward] over the whole sequence.
  ///
  /// [cache] must be the object returned by the matching [forward] call.
  List<Float64List> backward(
    List<Float64List> gradOutputs,
    MlaAttentionCache cache,
  ) {
    if (gradOutputs.length != cache.length) {
      throw ArgumentError(
        '"$name" received ${gradOutputs.length} output gradients but its last '
        'forward pass covered ${cache.length} positions',
      );
    }
    final double scale = 1.0 / math.sqrt(headDim);
    final List<Float64List> gradInputs = List<Float64List>.generate(
      cache.length,
      (int _) => Float64List(dModel),
    );

    // Gradients with respect to post-rotation queries and keys, accumulated
    // across every position that attended to them.
    final List<Float64List> gradQueries = List<Float64List>.generate(
      cache.length,
      (int _) => Float64List(nHeads * headDim),
    );
    final List<Float64List> gradKeys = List<Float64List>.generate(
      cache.length,
      (int _) => Float64List(nHeads * headDim),
    );
    final List<Float64List> gradValues = List<Float64List>.generate(
      cache.length,
      (int _) => Float64List(nHeads * headDim),
    );
    final List<Float64List> gradHeadOutputs = List<Float64List>.generate(
      cache.length,
      (int _) => Float64List(nHeads * headDim),
    );

    // The output projection contributes to every position's head-output
    // gradient independently.
    for (int i = 0; i < cache.length; i++) {
      final Float64List gradHead =
          output.backward(gradOutputs[i], cache.outputCaches[i]);
      gradHeadOutputs[i] = gradHead;
    }

    for (int i = 0; i < cache.length; i++) {
      final Float64List gradHead = gradHeadOutputs[i];
      for (int h = 0; h < nHeads; h++) {
        final int base = h * headDim;
        final Float64List probs = Float64List(i + 1);
        for (int j = 0; j <= i; j++) {
          probs[j] = cache.probabilities[i][h * (i + 1) + j];
        }
        // Gradient of the probabilities.
        final Float64List gradProbs = Float64List(i + 1);
        for (int j = 0; j <= i; j++) {
          double dot = 0;
          final int vBase = h * headDim;
          for (int d = 0; d < headDim; d++) {
            dot += gradHead[base + d] * cache.values[j][vBase + d];
          }
          gradProbs[j] = dot;
          // Gradient of the values.
          final double weight = probs[j];
          if (weight != 0) {
            for (int d = 0; d < headDim; d++) {
              gradValues[j][vBase + d] += weight * gradHead[base + d];
            }
          }
        }
        final Float64List gradScores = softmaxBackward(gradProbs, probs);
        for (int j = 0; j <= i; j++) {
          final double weighted = gradScores[j] * scale;
          if (weighted == 0) {
            continue;
          }
          final int kBase = h * headDim;
          for (int d = 0; d < headDim; d++) {
            gradQueries[i][base + d] += weighted * cache.keys[j][kBase + d];
            gradKeys[j][kBase + d] += weighted * cache.queries[i][base + d];
          }
        }
      }
    }

    // Undo the rotary rotation, then push through the projections.
    for (int i = 0; i < cache.length; i++) {
      final Float64List gradQ = gradQueries[i];
      final Float64List gradK = gradKeys[i];
      for (int h = 0; h < nHeads; h++) {
        rope.applyInverse(
          Float64List.sublistView(gradQ, h * headDim, (h + 1) * headDim),
          i,
        );
        rope.applyInverse(
          Float64List.sublistView(gradK, h * headDim, (h + 1) * headDim),
          i,
        );
      }
      final Float64List gradLatent = Float64List(kvRank);
      _accumulate(keyUp.backward(gradK, cache.keyCaches[i]), gradLatent);
      _accumulate(
        valueUp.backward(gradValues[i], cache.valueCaches[i]),
        gradLatent,
      );
      _accumulate(
        query.backward(gradQ, cache.queryCaches[i]),
        gradInputs[i],
      );
      _accumulate(
        latentDown.backward(gradLatent, cache.latentCaches[i]),
        gradInputs[i],
      );
    }
    return gradInputs;
  }

  /// Adds [source] into [target]; shapes must match.
  static void _accumulate(Float64List source, Float64List target) {
    for (int i = 0; i < source.length; i++) {
      target[i] += source[i];
    }
  }

  /// Appends [row] to a jagged per-head probability buffer.
  static Float64List _appendRow(Float64List existing, Float64List row) {
    final Float64List out = Float64List(existing.length + row.length)
      ..setRange(0, existing.length, existing)
      ..setRange(existing.length, existing.length + row.length, row);
    return out;
  }
}

/// The activation cache of one mixture-of-experts block at one position.
///
/// Fields are mutable rather than `final` because [MoeFfn.forward] fills the
/// object through a cascade after construction; `late` defers the
/// initialisation check to first use, so a cache that is read before it is
/// filled fails immediately instead of silently holding zeros.
class MoeCache {
  /// The block input.
  late Float64List input;

  /// Router probabilities over all experts.
  late Float64List routerProbs;

  /// The routing distribution's logits.
  late Float64List routerLogits;

  /// The router projection cache.
  late LinearCache routerCache;

  /// Indices of the selected experts, best first.
  late List<int> selected;

  /// Normalised weight of each selected expert.
  late List<double> weights;

  /// Sum of the selected experts' probabilities, the normalisation constant.
  late double selectedMass;

  /// First-layer caches per selected expert.
  late Map<int, LinearCache> hiddenCaches;

  /// GELU activations per selected expert.
  late Map<int, Float64List> activations;

  /// Second-layer caches per selected expert.
  late Map<int, LinearCache> outputCaches;

  /// The combined block output.
  late Float64List output;
}

/// A fine-grained mixture-of-experts feed-forward network.
///
/// "Fine-grained" means many small experts with a top-k router, rather than a
/// few large ones: total parameters grow while the FLOPs per token stay fixed
/// at `topK` experts. Only the selected experts receive gradients, which is what
/// makes conditional computation cheap to train as well as to run.
class MoeFfn {
  /// Builds the block with [numExperts] experts of width [hidden].
  MoeFfn({
    required this.name,
    required this.dModel,
    required this.numExperts,
    required this.topK,
    required this.hidden,
    required math.Random random,
  })  : router = Linear(
          name: '$name.router',
          inFeatures: dModel,
          outFeatures: numExperts,
          random: random,
          bias: false,
        ),
        up = List<Linear>.generate(
          numExperts,
          (int e) => Linear(
            name: '$name.e$e.up',
            inFeatures: dModel,
            outFeatures: hidden,
            random: random,
          ),
        ),
        down = List<Linear>.generate(
          numExperts,
          (int e) => Linear(
            name: '$name.e$e.down',
            inFeatures: hidden,
            outFeatures: dModel,
            random: random,
          ),
        ) {
    if (topK <= 0 || topK > numExperts) {
      throw ArgumentError(
        'topK must be between 1 and numExperts ($numExperts), got $topK',
      );
    }
  }

  /// Parameter-name prefix.
  final String name;

  /// Residual-stream width.
  final int dModel;

  /// Number of experts.
  final int numExperts;

  /// Number of experts evaluated per token.
  final int topK;

  /// Hidden width of each expert.
  final int hidden;

  /// The router projection, `dModel -> numExperts`.
  final Linear router;

  /// First expert layer, `dModel -> hidden`, one per expert.
  final List<Linear> up;

  /// Second expert layer, `hidden -> dModel`, one per expert.
  final List<Linear> down;

  /// How many times each expert has been selected since the last reset.
  ///
  /// Reported in the UI: a router that collapses onto one expert is the classic
  /// MoE failure mode, and this is how it becomes visible without a training
  /// run log.
  late final List<int> usageCounts = List<int>.filled(numExperts, 0);

  /// Resets [usageCounts].
  void resetUsage() => usageCounts.fillRange(0, usageCounts.length, 0);

  /// Every trainable tensor in this block.
  List<ParamRef> parameters() => <ParamRef>[
        ...router.parameters(),
        for (final Linear layer in up) ...layer.parameters(),
        for (final Linear layer in down) ...layer.parameters(),
      ];

  /// Zeroes every gradient in this block.
  void zeroGrad() {
    router.zeroGrad();
    for (final Linear layer in up) {
      layer.zeroGrad();
    }
    for (final Linear layer in down) {
      layer.zeroGrad();
    }
  }

  /// Routes one token through [topK] experts and combines their outputs.
  ///
  /// Discards the activation cache; use [forwardCached] when the result will be
  /// differentiated.
  Float64List forward(Float64List x) => forwardCached(x).output;

  /// Routes one token and returns the cache needed to differentiate it.
  ///
  /// The cache is returned rather than retained so a sequence can route every
  /// position through the same block and still back-propagate each one against
  /// its own activations.
  MoeCache forwardCached(Float64List x) {
    if (x.length != dModel) {
      throw ArgumentError('"$name" expected $dModel inputs, got ${x.length}');
    }
    final LinearCache routerCache = router.forward(x);
    final Float64List probs = softmax(routerCache.output);

    final List<int> order = List<int>.generate(numExperts, (int i) => i)
      ..sort((int a, int b) => probs[b].compareTo(probs[a]));
    final List<int> selected = order.sublist(0, topK);

    double mass = 0;
    for (final int e in selected) {
      mass += probs[e];
    }
    final List<double> weights = <double>[
      for (final int e in selected) probs[e] / mass,
    ];

    final Float64List out = Float64List(dModel);
    final Map<int, LinearCache> hiddenCaches = <int, LinearCache>{};
    final Map<int, Float64List> activations = <int, Float64List>{};
    final Map<int, LinearCache> outputCaches = <int, LinearCache>{};

    for (int i = 0; i < selected.length; i++) {
      final int e = selected[i];
      final LinearCache hiddenCache = up[e].forward(x);
      final Float64List activated = Float64List(hidden)
        ..setAll(0, hiddenCache.output);
      for (int j = 0; j < activated.length; j++) {
        activated[j] = gelu(activated[j]);
      }
      final LinearCache outCache = down[e].forward(activated);
      hiddenCaches[e] = hiddenCache;
      activations[e] = activated;
      outputCaches[e] = outCache;
      final double weight = weights[i];
      for (int d = 0; d < dModel; d++) {
        out[d] += weight * outCache.output[d];
      }
    }
    for (final int e in selected) {
      usageCounts[e]++;
    }

    return MoeCache()
      ..input = x
      ..routerProbs = probs
      ..routerLogits = routerCache.output
      ..routerCache = routerCache
      ..selected = selected
      ..weights = weights
      ..selectedMass = mass
      ..hiddenCaches = hiddenCaches
      ..activations = activations
      ..outputCaches = outputCaches
      ..output = out;
  }

  /// The vector-Jacobian product of [forward].
  ///
  /// Gradients reach the selected experts and the router. The router's gradient
  /// accounts for the normalisation `w_i = p_i / Σ_{j∈T} p_j`, which is why the
  /// expression looks more involved than the usual top-k weight gradient: the
  /// weights are *not* independent of one another.
  Float64List backward(Float64List gradOut, MoeCache cache) {
    if (gradOut.length != dModel) {
      throw ArgumentError(
        '"$name" expected $dModel output gradients, got ${gradOut.length}',
      );
    }
    final Float64List gradInput = Float64List(dModel);
    final List<double> gradWeights =
        List<double>.filled(cache.selected.length, 0);

    for (int i = 0; i < cache.selected.length; i++) {
      final int e = cache.selected[i];
      final double weight = cache.weights[i];
      final Float64List expertOutput = cache.outputCaches[e]!.output;
      double dot = 0;
      for (int d = 0; d < dModel; d++) {
        dot += gradOut[d] * expertOutput[d];
      }
      gradWeights[i] = dot;

      final Float64List scaled = Float64List(dModel);
      for (int d = 0; d < dModel; d++) {
        scaled[d] = gradOut[d] * weight;
      }
      final Float64List gradActivated =
          down[e].backward(scaled, cache.outputCaches[e]!);
      final Float64List gradHidden = Float64List(hidden);
      final Float64List hiddenOutput = cache.hiddenCaches[e]!.output;
      for (int j = 0; j < hidden; j++) {
        gradHidden[j] = gradActivated[j] * geluGrad(hiddenOutput[j]);
      }
      final Float64List fromExpert =
          up[e].backward(gradHidden, cache.hiddenCaches[e]!);
      for (int d = 0; d < dModel; d++) {
        gradInput[d] += fromExpert[d];
      }
    }

    // d w_i / d p_m = δ(i=m)/S - p_i·[m ∈ T]/S², so the router's probability
    // gradient is gw_m/S - (Σ_i gw_i p_i)/S² on the selected set and zero
    // elsewhere.
    final double mass = cache.selectedMass;
    double weightedSum = 0;
    for (int i = 0; i < cache.selected.length; i++) {
      weightedSum += gradWeights[i] * cache.routerProbs[cache.selected[i]];
    }
    final double correction = weightedSum / (mass * mass);
    final Float64List gradProbs = Float64List(numExperts);
    for (int i = 0; i < cache.selected.length; i++) {
      gradProbs[cache.selected[i]] = gradWeights[i] / mass - correction;
    }
    final Float64List gradLogits =
        softmaxBackward(gradProbs, cache.routerProbs);
    final Float64List fromRouter = router.backward(gradLogits, cache.routerCache);
    for (int d = 0; d < dModel; d++) {
      gradInput[d] += fromRouter[d];
    }
    return gradInput;
  }
}