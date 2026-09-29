// A decoder-only transformer that trains on-device.
//
// Architecture, per layer, pre-norm with residual connections:
//
//     h ← h + MLA(LayerNorm(h))
//     h ← h + MoE(LayerNorm(h))
//
// followed by a final LayerNorm and a linear head to the vocabulary. MLA and the
// MoE block live in `blocks.dart`; this file owns the stack, the embedding, the
// loss, the backward pass through the residual stream, incremental decoding, and
// checkpoint serialisation.
//
// Two decoding paths exist on purpose:
//
//  * [logitsFor] uses an incremental key/value cache, decoding one token at a
//    time. This is what generation uses.
//  * [logitsFullForward] recomputes the whole window on every call. It is the
//    reference implementation the test suite checks the incremental path
//    against, because an incremental decoder that disagrees with the full
//    forward pass is a bug that is otherwise invisible until the model starts
//    producing subtly worse text.

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'blocks.dart';
import 'encoding.dart';
import 'interfaces.dart';
import 'linalg.dart';
import 'trainable_model.dart';

/// The shape and size of a [TinyLm].
///
/// Every field has a default that is simultaneously valid and small, so tests
/// can construct a working model with `const TinyLmConfig()`-style brevity while
/// production picks one of the named presets.
class TinyLmConfig {
  /// Creates a configuration. Call [validate] before building a model.
  const TinyLmConfig({
    this.vocabSize = 1024,
    this.dModel = 64,
    this.nLayers = 2,
    this.nHeads = 4,
    this.headDim = 16,
    this.kvRank = 16,
    this.contextLength = 128,
    this.numExperts = 4,
    this.topK = 2,
    this.expertHidden = 128,
    this.seed = 0x48415242,
  });

  /// Number of distinct tokens; must match the tokenizer's vocabulary.
  final int vocabSize;

  /// Residual-stream width.
  final int dModel;

  /// Number of transformer layers.
  final int nLayers;

  /// Number of attention heads per layer.
  final int nHeads;

  /// Width of one attention head; must be even for rotary embeddings.
  final int headDim;

  /// Rank of the shared key/value latent.
  final int kvRank;

  /// Maximum number of positions attended over.
  final int contextLength;

  /// Number of experts per mixture-of-experts block.
  final int numExperts;

  /// Number of experts evaluated per token.
  final int topK;

  /// Hidden width of one expert.
  final int expertHidden;

  /// Seed for the initialisation, so a run is reproducible.
  final int seed;

  /// A configuration small enough to train in a browser tab in seconds, used by
  /// the web dashboard's built-in demo run.
  static const TinyLmConfig browserPreset = TinyLmConfig(
    vocabSize: 768,
    dModel: 48,
    nLayers: 2,
    nHeads: 4,
    headDim: 12,
    kvRank: 12,
    // 512 rather than the 96 an earlier revision used: the dashboard chat has to
    // hold a system prompt, a tool catalogue and a few turns, and at 96 tokens
    // the window filled up before any answer could be generated. Attention here
    // is O(T) per token, so 512 is still comfortable in a browser tab.
    contextLength: 512,
    numExperts: 4,
    topK: 2,
    expertHidden: 96,
  );

  /// The default on-device configuration: big enough that the architecture is
  /// exercised (four experts, a real latent bottleneck), small enough that a
  /// training round finishes on battery.
  static const TinyLmConfig onDevicePreset = TinyLmConfig();

  /// A configuration for the test suite: every code path present, everything
  /// tiny, so a full gradient check runs in milliseconds.
  static const TinyLmConfig testPreset = TinyLmConfig(
    // 260 is the floor the validator enforces: ids 0-255 are raw bytes and
    // 256-259 are the control tokens, so a smaller vocabulary would collide
    // with them. A preset that cannot pass its own validation is a trap.
    vocabSize: 260,
    dModel: 16,
    nLayers: 2,
    nHeads: 2,
    headDim: 8,
    kvRank: 6,
    contextLength: 24,
    numExperts: 3,
    topK: 2,
    expertHidden: 24,
  );

  /// Throws [ArgumentError] when the configuration cannot build a model.
  ///
  /// Checked eagerly rather than left to fail somewhere deep inside a matrix
  /// multiply, so a malformed preset is reported at the call site that chose it.
  void validate() {
    if (vocabSize < 260 || vocabSize > 4096) {
      throw ArgumentError.value(
        vocabSize,
        'vocabSize',
        'must be between 260 (the byte vocabulary plus controls) and 4096',
      );
    }
    if (dModel <= 0 || nLayers <= 0 || nHeads <= 0 || headDim <= 0) {
      throw ArgumentError('all dimensions must be positive');
    }
    if (headDim.isOdd) {
      throw ArgumentError.value(headDim, 'headDim', 'must be even for rotary');
    }
    if (kvRank <= 0) {
      throw ArgumentError.value(kvRank, 'kvRank', 'must be positive');
    }
    if (kvRank > nHeads * headDim) {
      throw ArgumentError(
        'kvRank $kvRank exceeds the key/value width ${nHeads * headDim}; the '
        'latent would inflate the cache instead of compressing it',
      );
    }
    if (contextLength <= 0) {
      throw ArgumentError.value(contextLength, 'contextLength', 'must be positive');
    }
    if (numExperts <= 0 || topK <= 0 || topK > numExperts) {
      throw ArgumentError(
        'topK must be between 1 and numExperts ($numExperts), got $topK',
      );
    }
    if (expertHidden <= 0) {
      throw ArgumentError.value(expertHidden, 'expertHidden', 'must be positive');
    }
  }

  /// A copy with selected fields replaced.
  TinyLmConfig copyWith({
    int? vocabSize,
    int? dModel,
    int? nLayers,
    int? nHeads,
    int? headDim,
    int? kvRank,
    int? contextLength,
    int? numExperts,
    int? topK,
    int? expertHidden,
    int? seed,
  }) {
    return TinyLmConfig(
      vocabSize: vocabSize ?? this.vocabSize,
      dModel: dModel ?? this.dModel,
      nLayers: nLayers ?? this.nLayers,
      nHeads: nHeads ?? this.nHeads,
      headDim: headDim ?? this.headDim,
      kvRank: kvRank ?? this.kvRank,
      contextLength: contextLength ?? this.contextLength,
      numExperts: numExperts ?? this.numExperts,
      topK: topK ?? this.topK,
      expertHidden: expertHidden ?? this.expertHidden,
      seed: seed ?? this.seed,
    );
  }

  /// Parameter count computed from the shape, without allocating anything.
  ///
  /// The UI shows this before a model exists, so it must be exact rather than
  /// approximate; `test/model_test.dart` asserts it equals the constructed
  /// model's `totalParameterCount`.
  int get estimatedParameters {
    final int attention = dModel * (nHeads * headDim) + // query
        dModel * kvRank + // latent down
        kvRank * (nHeads * headDim) + // key up
        kvRank * (nHeads * headDim) + // value up
        (nHeads * headDim) * dModel; // output
    final int perExpert =
        dModel * expertHidden + expertHidden + expertHidden * dModel + dModel;
    final int moe = dModel * numExperts + numExperts * perExpert;
    final int norms = 4 * dModel; // two layer norms per layer, scale + shift
    final int perLayer = attention + moe + norms;
    final int embedding = vocabSize * dModel;
    final int head = dModel * vocabSize + vocabSize;
    final int finalNorm = 2 * dModel;
    return embedding + nLayers * perLayer + finalNorm + head;
  }

  /// A compact human description, e.g. `2L·d64·4×16h·kv16·4E/top2·ctx128`.
  String get summary => '${nLayers}L·d$dModel·$nHeads×${headDim}h·kv$kvRank'
      '·${numExperts}E/top$topK·ctx$contextLength';

  /// JSON form, embedded in every checkpoint.
  Map<String, Object?> toJson() => <String, Object?>{
        'vocab_size': vocabSize,
        'd_model': dModel,
        'n_layers': nLayers,
        'n_heads': nHeads,
        'head_dim': headDim,
        'kv_rank': kvRank,
        'context_length': contextLength,
        'num_experts': numExperts,
        'top_k': topK,
        'expert_hidden': expertHidden,
        'seed': seed,
      };

  /// Parses [toJson], rejecting a payload with missing or mistyped fields.
  static TinyLmConfig fromJson(Map<String, Object?> json) {
    int read(String key) {
      final Object? value = json[key];
      if (value is! int) {
        throw FormatException('config field "$key" must be an integer');
      }
      return value;
    }

    return TinyLmConfig(
      vocabSize: read('vocab_size'),
      dModel: read('d_model'),
      nLayers: read('n_layers'),
      nHeads: read('n_heads'),
      headDim: read('head_dim'),
      kvRank: read('kv_rank'),
      contextLength: read('context_length'),
      numExperts: read('num_experts'),
      topK: read('top_k'),
      expertHidden: read('expert_hidden'),
      seed: read('seed'),
    );
  }

  @override
  String toString() => 'TinyLmConfig($summary, vocab $vocabSize)';
}

/// One transformer layer: attention, MoE, and their two layer norms.
class TinyLmLayer {
  /// Builds the layer.
  TinyLmLayer({
    required this.index,
    required TinyLmConfig config,
    required math.Random random,
  })  : attention = MlaAttention(
          name: 'layer$index.attn',
          dModel: config.dModel,
          nHeads: config.nHeads,
          headDim: config.headDim,
          kvRank: config.kvRank,
          contextLength: config.contextLength,
          random: random,
        ),
        moe = MoeFfn(
          name: 'layer$index.moe',
          dModel: config.dModel,
          numExperts: config.numExperts,
          topK: config.topK,
          hidden: config.expertHidden,
          random: random,
        ),
        norm1Gamma = Float64List(config.dModel),
        norm1Beta = Float64List(config.dModel),
        norm2Gamma = Float64List(config.dModel),
        norm2Beta = Float64List(config.dModel),
        gradNorm1Gamma = Float64List(config.dModel),
        gradNorm1Beta = Float64List(config.dModel),
        gradNorm2Gamma = Float64List(config.dModel),
        gradNorm2Beta = Float64List(config.dModel) {
    // Layer-norm gains start at one so a freshly built model is an identity
    // normalisation rather than a zeroing one; a zeroed gain would make the
    // first forward pass output exactly the bias term and the first gradients
    // identically zero.
    norm1Gamma.fillRange(0, norm1Gamma.length, 1);
    norm2Gamma.fillRange(0, norm2Gamma.length, 1);
  }

  /// Zero-based position of this layer, used in parameter names.
  final int index;

  /// The attention block.
  final MlaAttention attention;

  /// The mixture-of-experts block.
  final MoeFfn moe;

  /// First layer-norm scale.
  final Float64List norm1Gamma;

  /// First layer-norm shift.
  final Float64List norm1Beta;

  /// Second layer-norm scale.
  final Float64List norm2Gamma;

  /// Second layer-norm shift.
  final Float64List norm2Beta;

  /// Gradient of [norm1Gamma].
  final Float64List gradNorm1Gamma;

  /// Gradient of [norm1Beta].
  final Float64List gradNorm1Beta;

  /// Gradient of [norm2Gamma].
  final Float64List gradNorm2Gamma;

  /// Gradient of [norm2Beta].
  final Float64List gradNorm2Beta;

  /// Every trainable tensor in this layer.
  List<ParamRef> parameters() => <ParamRef>[
        ...attention.parameters(),
        ...moe.parameters(),
        ParamRef('layer$index.norm1.gamma', norm1Gamma, gradNorm1Gamma),
        ParamRef('layer$index.norm1.beta', norm1Beta, gradNorm1Beta),
        ParamRef('layer$index.norm2.gamma', norm2Gamma, gradNorm2Gamma),
        ParamRef('layer$index.norm2.beta', norm2Beta, gradNorm2Beta),
      ];

  /// Zeroes every gradient in this layer.
  void zeroGrad() {
    attention.zeroGrad();
    moe.zeroGrad();
    gradNorm1Gamma.fillRange(0, gradNorm1Gamma.length, 0);
    gradNorm1Beta.fillRange(0, gradNorm1Beta.length, 0);
    gradNorm2Gamma.fillRange(0, gradNorm2Gamma.length, 0);
    gradNorm2Beta.fillRange(0, gradNorm2Beta.length, 0);
  }

  /// Serialises the layer.
  Map<String, Object?> toJson() => <String, Object?>{
        'norm1_gamma': encodeFloats(norm1Gamma),
        'norm1_beta': encodeFloats(norm1Beta),
        'norm2_gamma': encodeFloats(norm2Gamma),
        'norm2_beta': encodeFloats(norm2Beta),
        'attention': <String, Object?>{
          'q': attention.query.toJson(),
          'kv_down': attention.latentDown.toJson(),
          'k_up': attention.keyUp.toJson(),
          'v_up': attention.valueUp.toJson(),
          'out': attention.output.toJson(),
        },
        'moe': <String, Object?>{
          'router': moe.router.toJson(),
          'up': <Object?>[for (final Linear l in moe.up) l.toJson()],
          'down': <Object?>[for (final Linear l in moe.down) l.toJson()],
        },
      };

  /// Restores the layer from [toJson].
  void loadJson(Map<String, Object?> json) {
    final Object? attentionRaw = json['attention'];
    if (attentionRaw is! Map<String, Object?>) {
      throw FormatException('layer$index has no attention payload');
    }
    _linear(attention.query, attentionRaw['q'], 'layer$index.attn.q');
    _linear(attention.latentDown, attentionRaw['kv_down'], 'layer$index.kv_down');
    _linear(attention.keyUp, attentionRaw['k_up'], 'layer$index.k_up');
    _linear(attention.valueUp, attentionRaw['v_up'], 'layer$index.v_up');
    _linear(attention.output, attentionRaw['out'], 'layer$index.attn.out');

    final Object? moeRaw = json['moe'];
    if (moeRaw is! Map<String, Object?>) {
      throw FormatException('layer$index has no moe payload');
    }
    _linear(moe.router, moeRaw['router'], 'layer$index.moe.router');
    _linearList(moe.up, moeRaw['up'], 'layer$index.moe.up');
    _linearList(moe.down, moeRaw['down'], 'layer$index.moe.down');

    _vector(json['norm1_gamma'], norm1Gamma, 'layer$index.norm1.gamma');
    _vector(json['norm1_beta'], norm1Beta, 'layer$index.norm1.beta');
    _vector(json['norm2_gamma'], norm2Gamma, 'layer$index.norm2.gamma');
    _vector(json['norm2_beta'], norm2Beta, 'layer$index.norm2.beta');
  }

  static void _linear(Linear target, Object? raw, String field) {
    if (raw is! Map<String, Object?>) {
      throw FormatException('"$field" is missing or not a JSON object');
    }
    target.loadJson(raw);
  }

  static void _linearList(List<Linear> targets, Object? raw, String field) {
    if (raw is! List<Object?> || raw.length != targets.length) {
      throw FormatException('"$field" has the wrong expert count');
    }
    for (int i = 0; i < targets.length; i++) {
      _linear(targets[i], raw[i], '$field[$i]');
    }
  }

  static void _vector(Object? raw, Float64List target, String field) {
    if (raw is! String) {
      throw FormatException('"$field" is missing');
    }
    final Float64List restored = decodeFloats(raw);
    if (restored.length != target.length) {
      throw FormatException('"$field" has the wrong length');
    }
    target.setAll(0, restored);
  }
}

/// Everything a backward pass over one window needs.
class _LayerActivations {
  late final List<Float64List> input;
  late final List<LayerNormCache> norm1Caches;
  late final List<Float64List> norm1Out;
  late final MlaAttentionCache attentionCache;
  late final List<Float64List> mid;
  late final List<LayerNormCache> norm2Caches;
  late final List<Float64List> norm2Out;
  late final List<MoeCache> moeCaches;
  late final List<Float64List> output;
}

/// The full activation record of one forward pass.
class _WindowActivations {
  late final List<int> inputs;
  late final List<Float64List> embeddings;
  late final List<_LayerActivations> layers;
  late final List<LayerNormCache> finalNormCaches;
  late final List<Float64List> finalNormOut;
  late final List<LinearCache> headCaches;
  late final List<Float64List> logits;
}

/// A trainable decoder-only transformer.
class TinyLm implements TrainableLanguageModel {
  /// Builds a model from [config].
  ///
  /// [random] is injectable so a test can pin the initialisation; by default the
  /// seed in [config] is used.
  TinyLm(this.config, {math.Random? random})
      : tokenEmbedding = Matrix.normal(
          config.vocabSize,
          config.dModel,
          random ?? math.Random(config.seed),
          std: 0.02,
        ),
        finalGamma = Float64List(config.dModel),
        finalBeta = Float64List(config.dModel),
        gradFinalGamma = Float64List(config.dModel),
        gradFinalBeta = Float64List(config.dModel) {
    config.validate();
    final math.Random rng = random ?? math.Random(config.seed ^ 0x9E3779B9);
    layers = List<TinyLmLayer>.generate(
      config.nLayers,
      (int i) => TinyLmLayer(index: i, config: config, random: rng),
    );
    lmHead = Linear(
      name: 'head',
      inFeatures: config.dModel,
      outFeatures: config.vocabSize,
      random: rng,
    );
    finalGamma.fillRange(0, finalGamma.length, 1);
  }

  /// The shape of this model.
  final TinyLmConfig config;

  /// Token embedding table, `[vocabSize, dModel]`.
  final Matrix tokenEmbedding;

  /// The transformer layers.
  late final List<TinyLmLayer> layers;

  /// Final layer-norm scale.
  final Float64List finalGamma;

  /// Final layer-norm shift.
  final Float64List finalBeta;

  /// Gradient of [finalGamma].
  final Float64List gradFinalGamma;

  /// Gradient of [finalBeta].
  final Float64List gradFinalBeta;

  /// Output projection to the vocabulary.
  late final Linear lmHead;

  List<MlaKvCache>? _decodeCaches;
  List<int>? _warmPrefix;
  List<double>? _lastLogits;

  @override
  int get vocabSize => config.vocabSize;

  @override
  int get contextLength => config.contextLength;

  @override
  int get parameterCount => trainableParameterCount;

  @override
  int get totalParameterCount => config.estimatedParameters + adapterParameterCount;

  @override
  int get trainableParameterCount {
    int total = 0;
    for (final ParamRef parameter in parameters) {
      total += parameter.length;
    }
    return total;
  }

  @override
  int get frozenParameterCount => totalParameterCount - trainableParameterCount;

  @override
  bool get hasAdapter => layers.isNotEmpty && layers.first.attention.query.hasAdapter;

  @override
  int get adapterRank =>
      hasAdapter ? layers.first.attention.query.adapterRank : 0;

  @override
  int get adapterParameterCount {
    int total = 0;
    for (final ParamRef parameter in adapterParameters) {
      total += parameter.length;
    }
    return total;
  }

  /// Every linear projection in the model, in a deterministic order.
  ///
  /// Embeddings and layer norms are plain vectors, not projections, so they are
  /// not adapter targets — matching standard LoRA practice, where only the
  /// attention and feed-forward matrices receive low-rank updates.
  List<Linear> get _linears => <Linear>[
        for (final TinyLmLayer layer in layers) ...<Linear>[
          layer.attention.query,
          layer.attention.latentDown,
          layer.attention.keyUp,
          layer.attention.valueUp,
          layer.attention.output,
          layer.moe.router,
          ...layer.moe.up,
          ...layer.moe.down,
        ],
        lmHead,
      ];

  @override
  void attachAdapter(int rank, {double alpha = 8.0, int? seed}) {
    final math.Random rng = math.Random(seed ?? (config.seed ^ 0x10A4));
    for (final Linear linear in _linears) {
      linear.attachAdapter(
        math.min(rank, math.min(linear.inFeatures, linear.outFeatures)),
        rng,
        alpha: alpha,
      );
    }
    _resetDecode();
  }

  @override
  void detachAdapter() {
    for (final Linear linear in _linears) {
      linear.detachAdapter();
    }
    _resetDecode();
  }

  @override
  void mergeAdapter() {
    for (final Linear linear in _linears) {
      linear.mergeAdapter();
    }
    _resetDecode();
  }

  @override
  List<ParamRef> get adapterParameters => <ParamRef>[
        for (final Linear linear in _linears) ...linear.adapterParameters(),
      ];

  /// Every parameter the next optimizer step may change.
  ///
  /// With an adapter attached that is exactly the adapter factors: the
  /// embeddings and the final norm are part of the frozen base too, so leaving
  /// them in would contradict the freeze the parameter counts report.
  @override
  List<ParamRef> get parameters {
    if (hasAdapter) {
      return adapterParameters;
    }
    return <ParamRef>[
      ParamRef('embedding', tokenEmbedding.values, tokenEmbedding.grads),
      for (final TinyLmLayer layer in layers) ...layer.parameters(),
      ParamRef('final_norm.gamma', finalGamma, gradFinalGamma),
      ParamRef('final_norm.beta', finalBeta, gradFinalBeta),
      ...lmHead.parameters(),
    ];
  }

  @override
  void zeroGrad() {
    tokenEmbedding.zeroGrad();
    for (final TinyLmLayer layer in layers) {
      layer.zeroGrad();
    }
    gradFinalGamma.fillRange(0, gradFinalGamma.length, 0);
    gradFinalBeta.fillRange(0, gradFinalBeta.length, 0);
    lmHead.zeroGrad();
  }

  @override
  double gradNorm() {
    double sum = 0;
    for (final ParamRef parameter in parameters) {
      for (int i = 0; i < parameter.values.length; i++) {
        sum += parameter.grads[i] * parameter.grads[i];
      }
    }
    return math.sqrt(sum);
  }

  @override
  void scaleGradients(double factor) {
    for (final ParamRef parameter in parameters) {
      for (int i = 0; i < parameter.values.length; i++) {
        parameter.grads[i] *= factor;
      }
    }
  }

  // -------------------------------------------------------------------------
  // Forward
  // -------------------------------------------------------------------------

  /// Runs the full stack over [inputs], recording every activation.
  ///
  /// [inputs] are the *input* positions; the caller is responsible for shifting
  /// the targets by one. Causal masking is handled inside the attention block.
  _WindowActivations _forward(List<int> inputs) {
    if (inputs.isEmpty) {
      throw ArgumentError.value(inputs, 'inputs', 'must not be empty');
    }
    if (inputs.length > config.contextLength) {
      throw ArgumentError(
        'window of ${inputs.length} positions exceeds the context length '
        '${config.contextLength}',
      );
    }
    final int d = config.dModel;
    final _WindowActivations acts = _WindowActivations()..inputs = inputs;

    List<Float64List> h = <Float64List>[];
    for (final int token in inputs) {
      if (token < 0 || token >= config.vocabSize) {
        throw ArgumentError(
          'token $token is outside the vocabulary of ${config.vocabSize}',
        );
      }
      // A copy, not a view: the residual stream adds into these vectors, and a
      // view would write the residual back into the embedding table.
      h.add(
        Float64List.fromList(
          Float64List.sublistView(
            tokenEmbedding.values,
            token * d,
            (token + 1) * d,
          ),
        ),
      );
    }
    acts.embeddings = h;

    final List<int> positions = List<int>.generate(inputs.length, (int i) => i);
    final List<_LayerActivations> layerActs = <_LayerActivations>[];
    for (final TinyLmLayer layer in layers) {
      final _LayerActivations la = _LayerActivations()..input = h;

      final List<LayerNormCache> n1Caches = <LayerNormCache>[
        for (final int _ in positions) LayerNormCache(d),
      ];
      final List<Float64List> n1 = <Float64List>[
        for (final int i in positions)
          layerNormForward(h[i], layer.norm1Gamma, layer.norm1Beta, n1Caches[i]),
      ];
      la
        ..norm1Caches = n1Caches
        ..norm1Out = n1;

      final MlaAttentionCache attentionCache = layer.attention.forward(n1);
      la.attentionCache = attentionCache;

      final List<Float64List> mid = <Float64List>[
        for (final int i in positions) _add(h[i], attentionCache.outputs[i]),
      ];
      la.mid = mid;

      final List<LayerNormCache> n2Caches = <LayerNormCache>[
        for (final int _ in positions) LayerNormCache(d),
      ];
      final List<Float64List> n2 = <Float64List>[
        for (final int i in positions)
          layerNormForward(mid[i], layer.norm2Gamma, layer.norm2Beta, n2Caches[i]),
      ];
      la
        ..norm2Caches = n2Caches
        ..norm2Out = n2;

      final List<MoeCache> moeCaches = <MoeCache>[
        for (final int i in positions) layer.moe.forwardCached(n2[i]),
      ];
      la.moeCaches = moeCaches;

      final List<Float64List> output = <Float64List>[
        for (final int i in positions) _add(mid[i], moeCaches[i].output),
      ];
      la.output = output;
      layerActs.add(la);
      h = output;
    }
    acts.layers = layerActs;

    final List<LayerNormCache> finalCaches = <LayerNormCache>[
      for (final int _ in positions) LayerNormCache(d),
    ];
    final List<Float64List> finalOut = <Float64List>[
      for (final int i in positions)
        layerNormForward(h[i], finalGamma, finalBeta, finalCaches[i]),
    ];
    acts
      ..finalNormCaches = finalCaches
      ..finalNormOut = finalOut;

    final List<LinearCache> headCaches = <LinearCache>[
      for (final int i in positions) lmHead.forward(finalOut[i]),
    ];
    acts
      ..headCaches = headCaches
      ..logits = <Float64List>[for (final LinearCache c in headCaches) c.output];
    return acts;
  }

  /// The reference, non-incremental next-token logits for [prefix].
  ///
  /// Recomputes the whole window. Exposed so the test suite can prove that the
  /// incremental path in [logitsFor] agrees with it exactly (to floating-point
  /// noise), which is the property that makes a cached decoder safe.
  List<double> logitsFullForward(List<int> prefix) {
    final List<int> window = _truncate(prefix);
    final _WindowActivations acts = _forward(window);
    return List<double>.of(acts.logits.last);
  }

  /// Mean cross-entropy of [tokens] without recording gradients.
  @override
  double evaluateWindow(List<int> tokens) {
    _requireWindow(tokens);
    final _WindowActivations acts = _forward(tokens.sublist(0, tokens.length - 1));
    double total = 0;
    for (int i = 0; i + 1 < tokens.length; i++) {
      total += _crossEntropy(acts.logits[i], tokens[i + 1]);
    }
    return total / (tokens.length - 1);
  }

  @override
  TrainingOutcome accumulateWindow(List<int> tokens, {double lossScale = 1.0}) {
    _requireWindow(tokens);
    final int scored = tokens.length - 1;
    final _WindowActivations acts = _forward(tokens.sublist(0, scored));

    double totalLoss = 0;
    int correct = 0;
    final List<Float64List> gradLogits = <Float64List>[
      for (int i = 0; i < scored; i++) Float64List(config.vocabSize),
    ];
    for (int i = 0; i < scored; i++) {
      final Float64List logits = acts.logits[i];
      final int target = tokens[i + 1];
      final Float64List probs = softmax(logits);
      totalLoss += -math.log(math.max(probs[target], 1e-12));

      int best = 0;
      for (int v = 1; v < probs.length; v++) {
        if (probs[v] > probs[best]) {
          best = v;
        }
      }
      if (best == target) {
        correct++;
      }

      // d(CE)/d(logits) is `softmax - onehot`; dividing by the number of scored
      // positions makes the accumulated gradient the mean over positions, and
      // `lossScale` carries the outer `1/batchSize`.
      final Float64List grad = gradLogits[i];
      final double factor = lossScale / scored;
      for (int v = 0; v < grad.length; v++) {
        grad[v] = (probs[v] - (v == target ? 1.0 : 0.0)) * factor;
      }
    }

    _backward(acts, gradLogits);
    return TrainingOutcome(
      loss: totalLoss / scored,
      correct: correct,
      predicted: scored,
      tokens: tokens.length,
      gradNorm: gradNorm(),
    );
  }

  /// Back-propagates `dL/dlogits` through the whole stack.
  void _backward(
    _WindowActivations acts,
    List<Float64List> gradLogits,
  ) {
    final int positions = acts.inputs.length;

    // Final layer norm and head.
    List<Float64List> gradH = <Float64List>[
      for (int i = 0; i < positions; i++)
        layerNormBackward(
          lmHead.backward(gradLogits[i], acts.headCaches[i]),
          acts.finalNormCaches[i],
          finalGamma,
          gradFinalGamma,
          gradFinalBeta,
        ),
    ];

    for (int li = layers.length - 1; li >= 0; li--) {
      final TinyLmLayer layer = layers[li];
      final _LayerActivations la = acts.layers[li];

      // `h_out = mid + moe(norm2(mid))`, so the gradient splits into the
      // identity path and the MoE path, and the MoE path continues through the
      // second layer norm.
      final List<Float64List> gradMid = <Float64List>[];
      for (int i = 0; i < positions; i++) {
        final Float64List fromMoe =
            layer.moe.backward(gradH[i], la.moeCaches[i]);
        final Float64List fromNorm = layerNormBackward(
          fromMoe,
          la.norm2Caches[i],
          layer.norm2Gamma,
          layer.gradNorm2Gamma,
          layer.gradNorm2Beta,
        );
        gradMid.add(_add(gradH[i], fromNorm));
      }

      // `mid = h_in + attention(norm1(h_in))`.
      final List<Float64List> gradNorm1 =
          layer.attention.backward(gradMid, la.attentionCache);
      final List<Float64List> gradIn = <Float64List>[];
      for (int i = 0; i < positions; i++) {
        final Float64List fromNorm = layerNormBackward(
          gradNorm1[i],
          la.norm1Caches[i],
          layer.norm1Gamma,
          layer.gradNorm1Gamma,
          layer.gradNorm1Beta,
        );
        gradIn.add(_add(gradMid[i], fromNorm));
      }
      gradH = gradIn;
    }

    // Embedding gradients: every occurrence of a token accumulates, because the
    // same row was read at each of its positions.
    final int d = config.dModel;
    for (int i = 0; i < positions; i++) {
      final int token = acts.inputs[i];
      final Float64List row = Float64List.sublistView(
        tokenEmbedding.grads,
        token * d,
        (token + 1) * d,
      );
      final Float64List grad = gradH[i];
      for (int j = 0; j < d; j++) {
        row[j] += grad[j];
      }
    }
  }

  /// Cross-entropy of [logits] against [target].
  double _crossEntropy(Float64List logits, int target) {
    double max = logits[0];
    for (int i = 1; i < logits.length; i++) {
      if (logits[i] > max) {
        max = logits[i];
      }
    }
    double total = 0;
    for (final double value in logits) {
      total += math.exp(value - max);
    }
    return -(logits[target] - max) + math.log(total);
  }

  // -------------------------------------------------------------------------
  // Incremental decoding
  // -------------------------------------------------------------------------

  @override
  List<double> logitsFor(List<int> prefix) {
    if (prefix.isEmpty) {
      throw ArgumentError.value(prefix, 'prefix', 'must not be empty');
    }
    final List<int> window = _truncate(prefix);
    final List<int>? warm = _warmPrefix;
    if (warm != null && _sameTokens(warm, window) && _lastLogits != null) {
      return List<double>.of(_lastLogits!);
    }

    if (warm != null &&
        window.length == warm.length + 1 &&
        _startsWith(window, warm)) {
      final List<double> logits = _decodeStep(window.last);
      _warmPrefix = window;
      _lastLogits = logits;
      return List<double>.of(logits);
    }

    // Cold start: replay the window one token at a time through the incremental
    // path, which leaves the caches warm for the tokens that follow.
    _resetDecode();
    List<double> logits = const <double>[];
    for (final int token in window) {
      logits = _decodeStep(token);
    }
    _warmPrefix = window;
    _lastLogits = logits;
    return List<double>.of(logits);
  }

  @override
  void resetCache() => _resetDecode();

  /// Pushes one token through the incremental caches and returns its logits.
  List<double> _decodeStep(int token) {
    if (token < 0 || token >= config.vocabSize) {
      throw ArgumentError(
        'token $token is outside the vocabulary of ${config.vocabSize}',
      );
    }
    final int d = config.dModel;
    final List<MlaKvCache> caches = _decodeCaches ??= <MlaKvCache>[
      for (final TinyLmLayer _ in layers) MlaKvCache(),
    ];
    Float64List h = Float64List.fromList(
      Float64List.sublistView(
        tokenEmbedding.values,
        token * d,
        (token + 1) * d,
      ),
    );
    for (int li = 0; li < layers.length; li++) {
      final TinyLmLayer layer = layers[li];
      final Float64List n1 = layerNormForward(
        h,
        layer.norm1Gamma,
        layer.norm1Beta,
        LayerNormCache(d),
      );
      final Float64List attention =
          layer.attention.forwardStep(n1, caches[li]);
      h = _add(h, attention);
      final Float64List n2 = layerNormForward(
        h,
        layer.norm2Gamma,
        layer.norm2Beta,
        LayerNormCache(d),
      );
      h = _add(h, layer.moe.forward(n2));
    }
    final Float64List finalOut = layerNormForward(
      h,
      finalGamma,
      finalBeta,
      LayerNormCache(d),
    );
    return List<double>.of(lmHead.forward(finalOut).output);
  }

  @override
  List<int> generate(
    List<int> prompt, {
    int maxNewTokens = 64,
    Sampler? sampler,
    int? stopToken,
    void Function(int token)? onToken,
  }) {
    if (prompt.isEmpty) {
      throw ArgumentError.value(prompt, 'prompt', 'must not be empty');
    }
    final Sampler effective = sampler ?? Sampler();
    final List<int> prefix = List<int>.of(_truncate(prompt));
    final List<int> produced = <int>[];
    for (int step = 0; step < maxNewTokens; step++) {
      if (prefix.length >= config.contextLength) {
        break;
      }
      final List<double> logits = logitsFor(prefix);
      final int next = effective.pick(
        Float64List.fromList(logits),
        generated: produced,
      );
      if (next == stopToken) {
        break;
      }
      produced.add(next);
      prefix.add(next);
      onToken?.call(next);
    }
    return produced;
  }

  void _resetDecode() {
    _decodeCaches?.clear();
    _decodeCaches = null;
    _warmPrefix = null;
    _lastLogits = null;
  }

  /// Truncates [prefix] to the most recent [contextLength] tokens.
  ///
  /// Keeping the *tail* rather than the head is the only choice that lets a
  /// conversation continue once it outgrows the window: the recent turns are
  /// what the next token depends on.
  List<int> _truncate(List<int> prefix) {
    if (prefix.length <= config.contextLength) {
      return List<int>.of(prefix);
    }
    return prefix.sublist(prefix.length - config.contextLength);
  }

  void _requireWindow(List<int> tokens) {
    if (tokens.length < 2) {
      throw ArgumentError(
        'a training window needs at least two tokens to score one prediction',
      );
    }
    if (tokens.length > config.contextLength) {
      throw ArgumentError(
        'window of ${tokens.length} tokens exceeds the context length '
        '${config.contextLength}',
      );
    }
  }

  static bool _sameTokens(List<int> a, List<int> b) {
    if (a.length != b.length) {
      return false;
    }
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) {
        return false;
      }
    }
    return true;
  }

  static bool _startsWith(List<int> value, List<int> prefix) {
    if (value.length < prefix.length) {
      return false;
    }
    for (int i = 0; i < prefix.length; i++) {
      if (value[i] != prefix[i]) {
        return false;
      }
    }
    return true;
  }

  static Float64List _add(Float64List a, Float64List b) {
    final Float64List out = Float64List(a.length);
    for (int i = 0; i < a.length; i++) {
      out[i] = a[i] + b[i];
    }
    return out;
  }

  // -------------------------------------------------------------------------
  // Checkpoints
  // -------------------------------------------------------------------------

  @override
  Map<String, Object?> toJson() => <String, Object?>{
        'format': 'harbor.tiny_lm.v1',
        'config': config.toJson(),
        'embedding': encodeFloats(tokenEmbedding.values),
        'final_norm': <String, Object?>{
          'gamma': encodeFloats(finalGamma),
          'beta': encodeFloats(finalBeta),
        },
        'layers': <Object?>[for (final TinyLmLayer l in layers) l.toJson()],
        'head': lmHead.toJson(),
      };

  /// Serialises the model to a JSON string.
  String encode() => jsonEncode(toJson());

  /// Parses a checkpoint written by [encode].
  static TinyLm decode(String payload) {
    final Object? decoded = jsonDecode(payload);
    if (decoded is! Map<String, Object?>) {
      throw const FormatException('checkpoint must be a JSON object');
    }
    final Object? configRaw = decoded['config'];
    if (configRaw is! Map<String, Object?>) {
      throw const FormatException('checkpoint has no config');
    }
    final TinyLm model = TinyLm(TinyLmConfig.fromJson(configRaw));
    model.loadJson(decoded);
    return model;
  }

  @override
  void loadJson(Map<String, Object?> json) {
    final Object? configRaw = json['config'];
    if (configRaw is! Map<String, Object?>) {
      throw const FormatException('checkpoint has no config');
    }
    final TinyLmConfig stored = TinyLmConfig.fromJson(configRaw);
    if (stored.vocabSize != config.vocabSize ||
        stored.dModel != config.dModel ||
        stored.nLayers != config.nLayers ||
        stored.nHeads != config.nHeads ||
        stored.headDim != config.headDim ||
        stored.kvRank != config.kvRank ||
        stored.numExperts != config.numExperts ||
        stored.topK != config.topK ||
        stored.expertHidden != config.expertHidden) {
      throw FormatException(
        'checkpoint shape ${stored.summary} does not match this model '
        '(${config.summary})',
      );
    }

    final Object? embeddingRaw = json['embedding'];
    if (embeddingRaw is! String) {
      throw const FormatException('checkpoint has no embedding payload');
    }
    final Float64List embedding = decodeFloats(embeddingRaw);
    if (embedding.length != tokenEmbedding.length) {
      throw FormatException(
        'checkpoint embedding has ${embedding.length} values, expected '
        '${tokenEmbedding.length}',
      );
    }
    tokenEmbedding.values.setAll(0, embedding);

    final Object? finalNormRaw = json['final_norm'];
    if (finalNormRaw is! Map<String, Object?>) {
      throw const FormatException('checkpoint has no final_norm payload');
    }
    TinyLmLayer._vector(finalNormRaw['gamma'], finalGamma, 'final_norm.gamma');
    TinyLmLayer._vector(finalNormRaw['beta'], finalBeta, 'final_norm.beta');

    final Object? layersRaw = json['layers'];
    if (layersRaw is! List<Object?> || layersRaw.length != layers.length) {
      throw const FormatException('checkpoint has the wrong layer count');
    }
    for (int i = 0; i < layers.length; i++) {
      final Object? layerRaw = layersRaw[i];
      if (layerRaw is! Map<String, Object?>) {
        throw FormatException('checkpoint layer $i is not an object');
      }
      layers[i].loadJson(layerRaw);
    }

    final Object? headRaw = json['head'];
    if (headRaw is! Map<String, Object?>) {
      throw const FormatException('checkpoint has no head payload');
    }
    lmHead.loadJson(headRaw);
    _resetDecode();
  }

  /// A short description for logs and the UI.
  String describe() {
    final String adapter = hasAdapter
        ? ' · LoRA r$adapterRank (${adapterParameterCount ~/ 1000}k)'
        : ' · full fine-tune';
    return 'TinyLM ${config.summary} · '
        '${(totalParameterCount / 1000).toStringAsFixed(0)}k params$adapter';
  }

  @override
  String toString() => 'TinyLm(${config.summary}, $totalParameterCount params)';
}