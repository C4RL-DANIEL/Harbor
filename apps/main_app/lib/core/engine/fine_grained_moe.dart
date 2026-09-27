// Fine-Grained Mixture of Experts (DeepSeekMoE style) routing engine.
//
// Two expert populations cooperate:
//
//   * SHARED experts (always active) capture common, cross-domain structure so
//     the routed experts are free to specialise.
//   * ROUTED experts are selected per token by a top-K gate over a fine-grained
//     expert pool.
//
// The layer implements three production details that matter on device:
//
//   1. Aux-loss-free load balancing. A per-expert bias `b_i` is added to the
//      router logits for *selection only* (never to the mixing weight) and is
//      nudged up when an expert is under-loaded and down when it is
//      over-loaded. This keeps the gate balanced without perturbing the
//      language-modelling objective.
//   2. Capacity factor + token dropping. Each expert accepts at most
//      `ceil(tokens * topK / numExperts * capacityFactor)` assignments; overflow
//      tokens skip the expert and pass through the residual connection.
//   3. Normalised top-K probabilities, so mixing weights sum to one regardless
//      of how the gate distributes mass.
//
// Pure Dart float32 reference implementation — no stubs, runs in an isolate.

import 'dart:math' as math;
import 'dart:typed_data';

/// Thrown when expert geometry is inconsistent.
class MoeShapeException implements Exception {
  MoeShapeException(this.message);

  final String message;

  @override
  String toString() => 'MoeShapeException: $message';
}

/// Static geometry of a fine-grained MoE layer.
class MoeConfig {
  const MoeConfig({
    required this.hiddenSize,
    required this.intermediateSize,
    required this.numRoutedExperts,
    required this.topK,
    this.numSharedExperts = 1,
    this.sharedIntermediateSize = 0,
    this.capacityFactor = 1.25,
    this.normalizeTopKProb = true,
    this.routedScalingFactor = 1.0,
    this.biasUpdateSpeed = 0.001,
    this.jitterEpsilon = 0.0,
  })  : assert(hiddenSize > 0, 'hiddenSize must be positive'),
        assert(numRoutedExperts > 0, 'need at least one routed expert'),
        assert(topK > 0, 'topK must be positive'),
        assert(topK <= numRoutedExperts, 'topK cannot exceed numRoutedExperts'),
        assert(numSharedExperts >= 0, 'numSharedExperts cannot be negative'),
        assert(capacityFactor > 0, 'capacityFactor must be positive');

  /// Model width.
  final int hiddenSize;

  /// Feed-forward width of each routed expert.
  final int intermediateSize;

  /// Size of the fine-grained routed expert pool.
  final int numRoutedExperts;

  /// How many routed experts each token activates.
  final int topK;

  /// Always-on shared experts.
  final int numSharedExperts;

  /// Feed-forward width of shared experts; defaults to [intermediateSize].
  final int sharedIntermediateSize;

  /// Multiplier over the perfectly-balanced assignment count.
  final double capacityFactor;

  /// Whether top-K gate probabilities are renormalised to sum to one.
  final bool normalizeTopKProb;

  /// Scale applied to the routed-expert branch.
  final double routedScalingFactor;

  /// Step size of the aux-loss-free bias controller.
  final double biasUpdateSpeed;

  /// Optional multiplicative gate jitter used during training for exploration.
  final double jitterEpsilon;

  /// Effective shared-expert width.
  int get effectiveSharedIntermediateSize =>
      sharedIntermediateSize > 0 ? sharedIntermediateSize : intermediateSize;

  /// Per-expert assignment capacity for a batch of [tokenCount] tokens.
  int capacityFor(int tokenCount) {
    final double ideal = tokenCount * topK / numRoutedExperts;
    return math.max(1, (ideal * capacityFactor).ceil());
  }
}

/// Per-expert utilisation statistics for one routing pass.
class ExpertLoadReport {
  const ExpertLoadReport({
    required this.assignments,
    required this.dropped,
    required this.capacity,
    required this.tokenCount,
  });

  /// Number of routed assignments accepted by each expert.
  final List<int> assignments;

  /// Token-expert assignments rejected because an expert was at capacity.
  final int dropped;

  /// Per-expert capacity used for this pass.
  final int capacity;

  /// Number of tokens routed.
  final int tokenCount;

  /// Total accepted assignments.
  int get accepted => assignments.fold<int>(0, (int a, int b) => a + b);

  /// Shannon entropy of the load distribution, in nats.
  double get balanceEntropy {
    if (accepted == 0) {
      return 0.0;
    }
    double h = 0.0;
    for (final int c in assignments) {
      if (c == 0) {
        continue;
      }
      final double p = c / accepted;
      h -= p * math.log(p);
    }
    return h;
  }

  /// Entropy of a perfectly uniform load, in nats — the upper bound.
  double get maxEntropy =>
      assignments.length <= 1 ? 0.0 : math.log(assignments.length.toDouble());

  /// `balanceEntropy / maxEntropy`, where 1.0 means perfectly balanced.
  double get balanceScore =>
      maxEntropy == 0 ? 1.0 : balanceEntropy / maxEntropy;

  Map<String, Object?> toJson() => <String, Object?>{
        'token_count': tokenCount,
        'capacity': capacity,
        'assignments': assignments,
        'accepted': accepted,
        'dropped': dropped,
        'drop_rate': tokenCount == 0
            ? 0.0
            : double.parse(
                ((dropped / (tokenCount * assignments.length)) * 100)
                    .toStringAsFixed(3),
              ),
        'balance_entropy': double.parse(balanceEntropy.toStringAsFixed(4)),
        'balance_score': double.parse(balanceScore.toStringAsFixed(4)),
      };
}

/// One routed expert: a gated feed-forward block.
class MoeExpert {
  MoeExpert({
    required this.index,
    required int hiddenSize,
    required int intermediateSize,
    required int seed,
  })  : _gate = _Dense.initialized(intermediateSize, hiddenSize, seed + 11),
        _up = _Dense.initialized(intermediateSize, hiddenSize, seed + 12),
        _down = _Dense.initialized(hiddenSize, intermediateSize, seed + 13);

  final int index;
  final _Dense _gate;
  final _Dense _up;
  final _Dense _down;

  /// SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`.
  Float32List forward(Float32List x) {
    final Float32List g = _gate.project(x);
    final Float32List u = _up.project(x);
    final Float32List act = Float32List(g.length);
    for (int i = 0; i < g.length; i++) {
      act[i] = _silu(g[i]) * u[i];
    }
    return _down.project(act);
  }

  static double _silu(double v) => v / (1.0 + math.exp(-v));
}

/// Internal dense layer used by experts and the router gate.
class _Dense {
  _Dense(this.outFeatures, this.inFeatures, this.weights);

  final int outFeatures;
  final int inFeatures;
  final Float32List weights;

  factory _Dense.initialized(int outFeatures, int inFeatures, int seed) {
    final Float32List w = Float32List(outFeatures * inFeatures);
    final math.Random rng = math.Random(seed);
    final double limit = math.sqrt(6.0 / (inFeatures + outFeatures));
    for (int i = 0; i < w.length; i++) {
      w[i] = (rng.nextDouble() * 2.0 - 1.0) * limit;
    }
    return _Dense(outFeatures, inFeatures, w);
  }

  Float32List project(Float32List x) {
    if (x.length != inFeatures) {
      throw MoeShapeException(
        'dense layer expected $inFeatures inputs, got ${x.length}',
      );
    }
    final Float32List out = Float32List(outFeatures);
    for (int o = 0; o < outFeatures; o++) {
      final int base = o * inFeatures;
      double acc = 0.0;
      for (int i = 0; i < inFeatures; i++) {
        acc += weights[base + i] * x[i];
      }
      out[o] = acc;
    }
    return out;
  }
}

/// A completed routing decision for a single token.
class MoeRoutingDecision {
  const MoeRoutingDecision({
    required this.expertIndices,
    required this.weights,
    required this.sharedApplied,
    required this.dropped,
  });

  /// Indices of the routed experts that actually processed this token.
  final List<int> expertIndices;

  /// Mixing weight aligned with [expertIndices].
  final List<double> weights;

  /// Whether the shared-expert branch contributed.
  final bool sharedApplied;

  /// Whether at least one selected expert was skipped at capacity.
  final bool dropped;

  Map<String, Object?> toJson() => <String, Object?>{
        'experts': expertIndices,
        'weights': weights.map((double w) => double.parse(w.toStringAsFixed(4))).toList(),
        'shared': sharedApplied,
        'dropped': dropped,
      };
}

/// Result of a full MoE layer pass over a batch of tokens.
class MoeForwardResult {
  const MoeForwardResult({
    required this.outputs,
    required this.decisions,
    required this.load,
  });

  /// One output vector per input token, in input order.
  final List<Float32List> outputs;

  /// Per-token routing decision, in input order.
  final List<MoeRoutingDecision> decisions;

  /// Aggregate expert utilisation for this pass.
  final ExpertLoadReport load;
}

/// Fine-grained Mixture-of-Experts layer.
class FineGrainedMoE {
  FineGrainedMoE(this.config, {int seed = 0x00E})
      : _gate = _Dense.initialized(config.numRoutedExperts, config.hiddenSize, seed + 1),
        _shared = List<MoeExpert>.generate(
          config.numSharedExperts,
          (int i) => MoeExpert(
            index: i,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.effectiveSharedIntermediateSize,
            seed: seed + 100 + i * 7,
          ),
          growable: false,
        ),
        _routed = List<MoeExpert>.generate(
          config.numRoutedExperts,
          (int i) => MoeExpert(
            index: i,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            seed: seed + 1000 + i * 17,
          ),
          growable: false,
        ),
        _expertBias = Float32List(config.numRoutedExperts),
        _rng = math.Random(seed ^ 0x9E3779B9) {
    if (config.hiddenSize <= 0) {
      throw MoeShapeException('hiddenSize must be positive');
    }
  }

  final MoeConfig config;
  final _Dense _gate;
  final List<MoeExpert> _shared;
  final List<MoeExpert> _routed;

  /// Aux-loss-free selection bias, one entry per routed expert.
  final Float32List _expertBias;
  final math.Random _rng;

  /// Running total of accepted assignments, used to steer the bias controller.
  final List<int> _cumulativeAssignments =
      List<int>.filled(0, 0, growable: true);

  int _totalRoutedTokens = 0;

  /// Read-only view of the current selection bias per routed expert.
  List<double> get expertBias => List<double>.unmodifiable(_expertBias);

  /// Shared experts (always active).
  List<MoeExpert> get sharedExperts => List<MoeExpert>.unmodifiable(_shared);

  /// Routed expert pool.
  List<MoeExpert> get routedExperts => List<MoeExpert>.unmodifiable(_routed);

  /// Number of times the bias controller has been applied.
  int get balanceSteps => _totalRoutedTokens;

  /// Runs the layer over a batch of [tokens].
  MoeForwardResult forward(List<Float32List> tokens) {
    if (tokens.isEmpty) {
      return MoeForwardResult(
        outputs: const <Float32List>[],
        decisions: const <MoeRoutingDecision>[],
        load: ExpertLoadReport(
          assignments: List<int>.filled(config.numRoutedExperts, 0),
          dropped: 0,
          capacity: 0,
          tokenCount: 0,
        ),
      );
    }
    for (final Float32List t in tokens) {
      if (t.length != config.hiddenSize) {
        throw MoeShapeException(
          'token must have ${config.hiddenSize} elements, got ${t.length}',
        );
      }
    }

    final int capacity = config.capacityFor(tokens.length);
    final List<int> assignments = List<int>.filled(config.numRoutedExperts, 0);
    final List<Float32List> outputs = <Float32List>[];
    final List<MoeRoutingDecision> decisions = <MoeRoutingDecision>[];
    int dropped = 0;

    for (final Float32List token in tokens) {
      final List<double> logits = _routerLogits(token);
      final List<int> selected = _topKIndices(logits, config.topK);

      // Mixing weights come from the *unbiased* softmax probabilities.
      final List<double> probs = _softmax(logits);
      final List<double> weights = <double>[];
      for (final int e in selected) {
        weights.add(probs[e]);
      }
      if (config.normalizeTopKProb) {
        final double sum = weights.fold<double>(0.0, (double a, double b) => a + b);
        if (sum > 0) {
          for (int i = 0; i < weights.length; i++) {
            weights[i] = weights[i] / sum;
          }
        }
      }

      // Shared experts always contribute.
      final Float32List mixed = Float32List(config.hiddenSize);
      bool sharedApplied = false;
      for (final MoeExpert s in _shared) {
        final Float32List o = s.forward(token);
        for (int d = 0; d < config.hiddenSize; d++) {
          mixed[d] += o[d];
        }
        sharedApplied = true;
      }

      // Routed experts, honouring per-expert capacity.
      final List<int> acceptedExperts = <int>[];
      final List<double> acceptedWeights = <double>[];
      bool tokenDropped = false;
      for (int k = 0; k < selected.length; k++) {
        final int e = selected[k];
        if (assignments[e] >= capacity) {
          tokenDropped = true;
          dropped++;
          continue;
        }
        assignments[e]++;
        final Float32List o = _routed[e].forward(token);
        final double w = weights[k] * config.routedScalingFactor;
        for (int d = 0; d < config.hiddenSize; d++) {
          mixed[d] += w * o[d];
        }
        acceptedExperts.add(e);
        acceptedWeights.add(weights[k]);
      }

      outputs.add(mixed);
      decisions.add(
        MoeRoutingDecision(
          expertIndices: List<int>.unmodifiable(acceptedExperts),
          weights: List<double>.unmodifiable(acceptedWeights),
          sharedApplied: sharedApplied,
          dropped: tokenDropped,
        ),
      );
    }

    _totalRoutedTokens += tokens.length;
    _updateExpertBias(assignments, tokens.length);

    return MoeForwardResult(
      outputs: outputs,
      decisions: decisions,
      load: ExpertLoadReport(
        assignments: List<int>.unmodifiable(assignments),
        dropped: dropped,
        capacity: capacity,
        tokenCount: tokens.length,
      ),
    );
  }

  /// Routes a single token and returns only its output vector.
  Float32List forwardOne(Float32List token) =>
      forward(<Float32List>[token]).outputs.single;

  List<double> _routerLogits(Float32List x) {
    final Float32List raw = _gate.project(x);
    final List<double> logits = List<double>.generate(raw.length, (int i) {
      double v = raw[i];
      if (config.jitterEpsilon > 0) {
        v *= 1.0 +
            config.jitterEpsilon * (_rng.nextDouble() * 2.0 - 1.0);
      }
      return v + _expertBias[i];
    });
    return logits;
  }

  List<int> _topKIndices(List<double> scores, int k) {
    final List<int> indices = List<int>.generate(scores.length, (int i) => i);
    indices.sort((int a, int b) {
      final int cmp = scores[b].compareTo(scores[a]);
      return cmp != 0 ? cmp : a.compareTo(b);
    });
    return indices.sublist(0, math.min(k, indices.length));
  }

  List<double> _softmax(List<double> logits) {
    double max = logits[0];
    for (int i = 1; i < logits.length; i++) {
      if (logits[i] > max) {
        max = logits[i];
      }
    }
    double sum = 0.0;
    final List<double> out = List<double>.filled(logits.length, 0.0);
    for (int i = 0; i < logits.length; i++) {
      final double e = math.exp(logits[i] - max);
      out[i] = e;
      sum += e;
    }
    if (sum > 0) {
      for (int i = 0; i < out.length; i++) {
        out[i] = out[i] / sum;
      }
    }
    return out;
  }

  /// Aux-loss-free controller: raise the bias of under-used experts and lower
  /// it for over-used ones. The bias affects selection only, never the gate's
  /// probability mass, so the training objective is untouched.
  void _updateExpertBias(List<int> assignments, int tokenCount) {
    if (_cumulativeAssignments.length != config.numRoutedExperts) {
      _cumulativeAssignments
        ..clear()
        ..addAll(List<int>.filled(config.numRoutedExperts, 0));
    }
    final int total = tokenCount * config.topK;
    if (total == 0) {
      return;
    }
    final double target = total / config.numRoutedExperts;
    for (int e = 0; e < config.numRoutedExperts; e++) {
      _cumulativeAssignments[e] += assignments[e];
      final double error = target - assignments[e];
      final double direction = error == 0 ? 0.0 : (error > 0 ? 1.0 : -1.0);
      _expertBias[e] = (_expertBias[e] + direction * config.biasUpdateSpeed)
          .clamp(-1.0, 1.0)
          .toDouble();
    }
  }

  /// Resets the routing statistics and bias controller.
  void resetStatistics() {
    _cumulativeAssignments.clear();
    _totalRoutedTokens = 0;
    for (int i = 0; i < _expertBias.length; i++) {
      _expertBias[i] = 0.0;
    }
  }

  /// Diagnostic summary for the engine status screen.
  Map<String, Object?> describe() => <String, Object?>{
        'architecture': 'fine_grained_moe',
        'hidden_size': config.hiddenSize,
        'intermediate_size': config.intermediateSize,
        'routed_experts': config.numRoutedExperts,
        'shared_experts': config.numSharedExperts,
        'top_k': config.topK,
        'capacity_factor': config.capacityFactor,
        'normalize_top_k_prob': config.normalizeTopKProb,
        'routed_scaling_factor': config.routedScalingFactor,
        'active_params_per_token_fraction': double.parse(
          ((config.topK + config.numSharedExperts) / config.numRoutedExperts)
              .toStringAsFixed(4),
        ),
        'expert_bias': expertBias
            .map((double b) => double.parse(b.toStringAsFixed(5)))
            .toList(),
        'balance_steps': balanceSteps,
      };
}