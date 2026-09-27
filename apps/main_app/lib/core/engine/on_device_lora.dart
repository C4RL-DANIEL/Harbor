// On-device LoRA adapter training for continuous background self-learning.
//
// The frozen base weights never change. A small low-rank correction is learned
// instead:
//
//     y = W0 · x + (alpha / rank) · B · (A · x)
//
// with A ∈ R^{rank×inFeatures} and B ∈ R^{outFeatures×rank}. Because only A and
// B are trained, the optimiser state and the gradient buffers stay tiny, which
// is what makes on-device learning viable. Adapters can also be merged into the
// base weights for zero-overhead inference.
//
// The trainer implements genuine analytical backpropagation through the LoRA
// branch plus a decoupled-weight-decay AdamW optimiser with global-norm gradient
// clipping. `IdleTrainingScheduler` gates training behind the device being idle
// *and* charging, a thermal ceiling, a battery floor, and a daily compute
// budget, so background learning can never be the reason a phone dies.
//
// Pure Dart — no Flutter or platform imports — so it runs in a background
// isolate and is fully unit-testable with simulated device state.

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

/// Thrown when adapter tensors are shaped incompatibly.
class LoraShapeException implements Exception {
  LoraShapeException(this.message);

  final String message;

  @override
  String toString() => 'LoraShapeException: $message';
}

/// Thrown by an [LoraAdapterStore] when persistence fails.
class AdapterStoreException implements Exception {
  AdapterStoreException(this.message, [this.cause]);

  final String message;
  final Object? cause;

  @override
  String toString() =>
      'AdapterStoreException: $message${cause == null ? '' : ' ($cause)'}';
}

/// Thermal pressure reported by the platform, ordered by severity.
enum ThermalState {
  nominal,
  fair,
  serious,
  critical;

  /// Parses a platform string, defaulting to [ThermalState.nominal].
  static ThermalState parse(String? raw) {
    if (raw == null) {
      return ThermalState.nominal;
    }
    for (final ThermalState s in ThermalState.values) {
      if (s.name == raw.toLowerCase()) {
        return s;
      }
    }
    return ThermalState.nominal;
  }

  bool operator >(ThermalState other) => index > other.index;
  bool operator >=(ThermalState other) => index >= other.index;
  bool operator <(ThermalState other) => index < other.index;
  bool operator <=(ThermalState other) => index <= other.index;
}

/// Geometry and hyperparameters of a LoRA adapter.
class LoraConfig {
  const LoraConfig({
    required this.rank,
    required this.inFeatures,
    required this.outFeatures,
    required this.alpha,
    this.dropout = 0.0,
    this.targetModule = 'q_proj',
  })  : assert(rank > 0, 'rank must be positive'),
        assert(inFeatures > 0, 'inFeatures must be positive'),
        assert(outFeatures > 0, 'outFeatures must be positive'),
        assert(alpha > 0, 'alpha must be positive'),
        assert(dropout >= 0.0 && dropout < 1.0, 'dropout must be in [0,1)');

  /// Low-rank bottleneck width.
  final int rank;

  /// Width of the input activations.
  final int inFeatures;

  /// Width of the output activations.
  final int outFeatures;

  /// LoRA scaling numerator; effective scale is `alpha / rank`.
  final double alpha;

  /// Training-time dropout applied to the hidden bottleneck.
  final double dropout;

  /// Name of the frozen module this adapter augments (e.g. `q_proj`).
  final String targetModule;

  /// Effective multiplier applied to the low-rank branch.
  double get scaling => alpha / rank;

  /// Trainable parameter count.
  int get trainableParameters => rank * inFeatures + outFeatures * rank;

  /// Parameter count of the equivalent dense update matrix.
  int get denseEquivalentParameters => inFeatures * outFeatures;

  /// Percentage of a dense fine-tune this adapter needs, in `[0, 1]`.
  double get parameterRatio =>
      denseEquivalentParameters == 0
          ? 0
          : trainableParameters / denseEquivalentParameters;

  Map<String, Object?> toJson() => <String, Object?>{
        'rank': rank,
        'in_features': inFeatures,
        'out_features': outFeatures,
        'alpha': alpha,
        'dropout': dropout,
        'target_module': targetModule,
      };

  static LoraConfig fromJson(Map<String, Object?> json) => LoraConfig(
        rank: (json['rank'] as num).toInt(),
        inFeatures: (json['in_features'] as num).toInt(),
        outFeatures: (json['out_features'] as num).toInt(),
        alpha: (json['alpha'] as num).toDouble(),
        dropout: (json['dropout'] as num?)?.toDouble() ?? 0.0,
        targetModule: json['target_module'] as String? ?? 'q_proj',
      );
}

/// A trainable low-rank adapter.
class LoraAdapter {
  LoraAdapter({
    required this.config,
    required this.a,
    required this.b,
    this.version = 1,
    this.samplesSeen = 0,
    this.lastLoss,
  });

  /// Creates an adapter with standard LoRA initialisation: A ~ Gaussian,
  /// B = 0, so the adapter starts as an exact identity (zero correction) and
  /// cannot degrade the frozen base model at step 0.
  factory LoraAdapter.initialized(LoraConfig config, {int seed = 0xA11CE}) {
    final math.Random rng = math.Random(seed);
    final double std = 1.0 / math.sqrt(config.inFeatures.toDouble());
    final Float32List a = Float32List(config.rank * config.inFeatures);
    for (int i = 0; i < a.length; i++) {
      // Box-Muller for a proper normal sample.
      final double u1 = math.max(rng.nextDouble(), 1e-12);
      final double u2 = rng.nextDouble();
      a[i] = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2) * std;
    }
    return LoraAdapter(
      config: config,
      a: a,
      b: Float32List(config.outFeatures * config.rank),
    );
  }

  /// Immutable geometry.
  final LoraConfig config;

  /// Row-major `[rank][inFeatures]` down-projection.
  final Float32List a;

  /// Row-major `[outFeatures][rank]` up-projection.
  final Float32List b;

  /// Monotonically increasing adapter revision, bumped on every optimiser step.
  int version;

  /// Total training examples consumed by this adapter.
  int samplesSeen;

  /// Most recent training loss, or null before the first step.
  double? lastLoss;

  /// Computes `h = A · x`.
  Float32List projectDown(Float32List x) {
    if (x.length != config.inFeatures) {
      throw LoraShapeException(
        'adapter expects ${config.inFeatures} inputs, got ${x.length}',
      );
    }
    final Float32List h = Float32List(config.rank);
    for (int r = 0; r < config.rank; r++) {
      final int base = r * config.inFeatures;
      double acc = 0.0;
      for (int i = 0; i < config.inFeatures; i++) {
        acc += a[base + i] * x[i];
      }
      h[r] = acc;
    }
    return h;
  }

  /// Computes `z = B · h`.
  Float32List projectUp(Float32List h) {
    if (h.length != config.rank) {
      throw LoraShapeException(
        'adapter bottleneck expects ${config.rank} values, got ${h.length}',
      );
    }
    final Float32List z = Float32List(config.outFeatures);
    for (int o = 0; o < config.outFeatures; o++) {
      final int base = o * config.rank;
      double acc = 0.0;
      for (int r = 0; r < config.rank; r++) {
        acc += b[base + r] * h[r];
      }
      z[o] = acc;
    }
    return z;
  }

  /// Low-rank correction `(alpha/rank) · B · A · x`.
  Float32List forward(Float32List x) {
    final Float32List h = projectDown(x);
    final Float32List z = projectUp(h);
    final double s = config.scaling;
    for (int i = 0; i < z.length; i++) {
      z[i] = z[i] * s;
    }
    return z;
  }

  /// Merges the adapter into a frozen weight matrix laid out `[out][in]`.
  ///
  /// `W' = W + (alpha/rank) · B · A`. Returns a new buffer; [baseWeights] is
  /// left untouched.
  Float32List mergeInto(Float32List baseWeights) {
    final int expected = config.outFeatures * config.inFeatures;
    if (baseWeights.length != expected) {
      throw LoraShapeException(
        'base weights must hold $expected values, got ${baseWeights.length}',
      );
    }
    final Float32List merged = Float32List.fromList(baseWeights);
    final double s = config.scaling;
    for (int o = 0; o < config.outFeatures; o++) {
      for (int r = 0; r < config.rank; r++) {
        final double bVal = b[o * config.rank + r] * s;
        if (bVal == 0.0) {
          continue;
        }
        final int aBase = r * config.inFeatures;
        final int wBase = o * config.inFeatures;
        for (int i = 0; i < config.inFeatures; i++) {
          merged[wBase + i] += bVal * a[aBase + i];
        }
      }
    }
    return merged;
  }

  /// Inverse of [mergeInto]: recovers the frozen weights from merged weights.
  Float32List unmergeFrom(Float32List mergedWeights) {
    final int expected = config.outFeatures * config.inFeatures;
    if (mergedWeights.length != expected) {
      throw LoraShapeException(
        'merged weights must hold $expected values, got ${mergedWeights.length}',
      );
    }
    final Float32List base = Float32List.fromList(mergedWeights);
    final double s = config.scaling;
    for (int o = 0; o < config.outFeatures; o++) {
      for (int r = 0; r < config.rank; r++) {
        final double bVal = b[o * config.rank + r] * s;
        if (bVal == 0.0) {
          continue;
        }
        final int aBase = r * config.inFeatures;
        final int wBase = o * config.inFeatures;
        for (int i = 0; i < config.inFeatures; i++) {
          base[wBase + i] -= bVal * a[aBase + i];
        }
      }
    }
    return base;
  }

  /// Deep copy, used to snapshot an adapter before a risky training round.
  LoraAdapter clone() => LoraAdapter(
        config: config,
        a: Float32List.fromList(a),
        b: Float32List.fromList(b),
        version: version,
        samplesSeen: samplesSeen,
        lastLoss: lastLoss,
      );

  /// L2 norm of all trainable parameters.
  double parameterNorm() {
    double acc = 0.0;
    for (int i = 0; i < a.length; i++) {
      acc += a[i] * a[i];
    }
    for (int i = 0; i < b.length; i++) {
      acc += b[i] * b[i];
    }
    return math.sqrt(acc);
  }

  Map<String, Object?> toJson() => <String, Object?>{
        'config': config.toJson(),
        'a': a.toList(),
        'b': b.toList(),
        'version': version,
        'samples_seen': samplesSeen,
        'last_loss': lastLoss,
        'parameter_norm': double.parse(parameterNorm().toStringAsFixed(6)),
      };

  static LoraAdapter fromJson(Map<String, Object?> json) {
    final Map<String, Object?> rawConfig =
        (json['config'] as Map<Object?, Object?>).cast<String, Object?>();
    final LoraConfig config = LoraConfig.fromJson(rawConfig);
    final List<Object?> rawA = json['a'] as List<Object?>;
    final List<Object?> rawB = json['b'] as List<Object?>;
    if (rawA.length != config.rank * config.inFeatures) {
      throw LoraShapeException(
        'serialized A has ${rawA.length} values, expected '
        '${config.rank * config.inFeatures}',
      );
    }
    if (rawB.length != config.outFeatures * config.rank) {
      throw LoraShapeException(
        'serialized B has ${rawB.length} values, expected '
        '${config.outFeatures * config.rank}',
      );
    }
    return LoraAdapter(
      config: config,
      a: Float32List.fromList(
        rawA.map((Object? v) => (v as num).toDouble()).toList(growable: false),
      ),
      b: Float32List.fromList(
        rawB.map((Object? v) => (v as num).toDouble()).toList(growable: false),
      ),
      version: (json['version'] as num?)?.toInt() ?? 1,
      samplesSeen: (json['samples_seen'] as num?)?.toInt() ?? 0,
      lastLoss: (json['last_loss'] as num?)?.toDouble(),
    );
  }
}

/// One supervised training pair for the adapter: an input activation and the
/// residual the adapter should learn to produce.
class LoraTrainingExample {
  const LoraTrainingExample({required this.input, required this.target});

  final Float32List input;
  final Float32List target;
}

/// Outcome of a single optimiser step.
class TrainingStepResult {
  const TrainingStepResult({
    required this.loss,
    required this.gradientNorm,
    required this.examplesUsed,
    required this.step,
    required this.clipped,
  });

  final double loss;
  final double gradientNorm;
  final int examplesUsed;
  final int step;
  final bool clipped;

  Map<String, Object?> toJson() => <String, Object?>{
        'loss': double.parse(loss.toStringAsFixed(6)),
        'gradient_norm': double.parse(gradientNorm.toStringAsFixed(6)),
        'examples_used': examplesUsed,
        'step': step,
        'clipped': clipped,
      };
}

/// AdamW optimiser state, one slot per trainable parameter.
class _AdamState {
  _AdamState(int size)
      : m = Float32List(size),
        v = Float32List(size);

  final Float32List m;
  final Float32List v;
  int t = 0;
}

/// Analytical LoRA trainer with decoupled-weight-decay AdamW.
class OnDeviceLoRATrainer {
  OnDeviceLoRATrainer({
    required this.adapter,
    this.learningRate = 0.002,
    this.weightDecay = 0.01,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.epsilon = 1e-8,
    this.maxGradientNorm = 1.0,
    this.maxReplayBuffer = 512,
    int seed = 0x7A1E,
  })  : assert(learningRate > 0, 'learningRate must be positive'),
        assert(beta1 > 0 && beta1 < 1, 'beta1 must be in (0,1)'),
        assert(beta2 > 0 && beta2 < 1, 'beta2 must be in (0,1)'),
        _rng = math.Random(seed),
        _aState = _AdamState(adapter.a.length),
        _bState = _AdamState(adapter.b.length);

  /// The adapter being trained, mutated in place.
  final LoraAdapter adapter;

  /// Step size.
  final double learningRate;

  /// Decoupled weight decay coefficient.
  final double weightDecay;

  final double beta1;
  final double beta2;
  final double epsilon;

  /// Global gradient-norm clip threshold.
  final double maxGradientNorm;

  /// Maximum number of retained replay examples.
  final int maxReplayBuffer;

  final math.Random _rng;
  final _AdamState _aState;
  final _AdamState _bState;
  final List<LoraTrainingExample> _replay = <LoraTrainingExample>[];

  int _step = 0;

  /// Number of optimiser steps taken.
  int get step => _step;

  /// Number of examples currently retained for replay.
  int get replaySize => _replay.length;

  /// Adds an example to the bounded reservoir-style replay buffer.
  void addExample(LoraTrainingExample example) {
    if (example.input.length != adapter.config.inFeatures) {
      throw LoraShapeException(
        'example input must have ${adapter.config.inFeatures} values, '
        'got ${example.input.length}',
      );
    }
    if (example.target.length != adapter.config.outFeatures) {
      throw LoraShapeException(
        'example target must have ${adapter.config.outFeatures} values, '
        'got ${example.target.length}',
      );
    }
    if (_replay.length < maxReplayBuffer) {
      _replay.add(example);
      return;
    }
    // Reservoir replacement keeps the buffer representative over time.
    final int victim = _rng.nextInt(_replay.length + 1);
    if (victim < _replay.length) {
      _replay[victim] = example;
    }
  }

  /// Draws a random mini-batch from the replay buffer.
  List<LoraTrainingExample> sampleBatch(int batchSize) {
    if (_replay.isEmpty) {
      return const <LoraTrainingExample>[];
    }
    final int n = math.min(batchSize, _replay.length);
    final List<LoraTrainingExample> batch = <LoraTrainingExample>[];
    for (int i = 0; i < n; i++) {
      batch.add(_replay[_rng.nextInt(_replay.length)]);
    }
    return batch;
  }

  /// Runs one AdamW step over a random batch of [batchSize] replay examples.
  ///
  /// Returns `null` when the replay buffer is empty — the caller can then keep
  /// the device idle instead of spinning on no data.
  TrainingStepResult? trainOnReplay({int batchSize = 8}) {
    final List<LoraTrainingExample> batch = sampleBatch(batchSize);
    if (batch.isEmpty) {
      return null;
    }
    return trainStep(batch);
  }

  /// Runs one AdamW step over an explicit [batch].
  TrainingStepResult trainStep(List<LoraTrainingExample> batch) {
    if (batch.isEmpty) {
      throw LoraShapeException('training batch must not be empty');
    }
    final LoraConfig cfg = adapter.config;
    final int rank = cfg.rank;
    final int inF = cfg.inFeatures;
    final int outF = cfg.outFeatures;
    final double scale = cfg.scaling;

    final Float32List gradA = Float32List(adapter.a.length);
    final Float32List gradB = Float32List(adapter.b.length);
    double totalLoss = 0.0;

    for (final LoraTrainingExample ex in batch) {
      final Float32List x = ex.input;
      final Float32List target = ex.target;

      // ---- forward ------------------------------------------------------
      final Float32List h = adapter.projectDown(x);
      if (cfg.dropout > 0) {
        for (int r = 0; r < rank; r++) {
          if (_rng.nextDouble() < cfg.dropout) {
            h[r] = 0.0;
          }
        }
      }
      final Float32List z = adapter.projectUp(h);
      final Float32List y = Float32List(outF);
      double sampleLoss = 0.0;
      for (int o = 0; o < outF; o++) {
        y[o] = scale * z[o];
        final double diff = y[o] - target[o];
        sampleLoss += diff * diff;
      }
      totalLoss += sampleLoss / outF;

      // ---- backward -----------------------------------------------------
      // dL/dz = scale * 2 * (y - t) / outFeatures
      final Float32List gradZ = Float32List(outF);
      for (int o = 0; o < outF; o++) {
        gradZ[o] = scale * 2.0 * (y[o] - target[o]) / outF;
      }
      // dL/dB[o][r] += gradZ[o] * h[r]
      for (int o = 0; o < outF; o++) {
        final double g = gradZ[o];
        if (g == 0.0) {
          continue;
        }
        final int bBase = o * rank;
        for (int r = 0; r < rank; r++) {
          gradB[bBase + r] += g * h[r];
        }
      }
      // dL/dh[r] = sum_o gradZ[o] * B[o][r]
      final Float32List gradH = Float32List(rank);
      for (int o = 0; o < outF; o++) {
        final double g = gradZ[o];
        if (g == 0.0) {
          continue;
        }
        final int bBase = o * rank;
        for (int r = 0; r < rank; r++) {
          gradH[r] += g * adapter.b[bBase + r];
        }
      }
      // dL/dA[r][i] += dL/dh[r] * x[i]
      for (int r = 0; r < rank; r++) {
        final double g = gradH[r];
        if (g == 0.0) {
          continue;
        }
        final int aBase = r * inF;
        for (int i = 0; i < inF; i++) {
          gradA[aBase + i] += g * x[i];
        }
      }
    }

    final double invBatch = 1.0 / batch.length;
    for (int i = 0; i < gradA.length; i++) {
      gradA[i] *= invBatch;
    }
    for (int i = 0; i < gradB.length; i++) {
      gradB[i] *= invBatch;
    }

    // ---- global-norm gradient clipping ---------------------------------
    double sq = 0.0;
    for (int i = 0; i < gradA.length; i++) {
      sq += gradA[i] * gradA[i];
    }
    for (int i = 0; i < gradB.length; i++) {
      sq += gradB[i] * gradB[i];
    }
    final double gradNorm = math.sqrt(sq);
    bool clipped = false;
    if (maxGradientNorm > 0 && gradNorm > maxGradientNorm) {
      final double clipScale = maxGradientNorm / (gradNorm + epsilon);
      for (int i = 0; i < gradA.length; i++) {
        gradA[i] *= clipScale;
      }
      for (int i = 0; i < gradB.length; i++) {
        gradB[i] *= clipScale;
      }
      clipped = true;
    }

    // ---- AdamW update ---------------------------------------------------
    _step++;
    _adamwUpdate(adapter.a, gradA, _aState);
    _adamwUpdate(adapter.b, gradB, _bState);

    final double meanLoss = totalLoss * invBatch;
    adapter
      ..version = adapter.version + 1
      ..samplesSeen = adapter.samplesSeen + batch.length
      ..lastLoss = meanLoss;

    return TrainingStepResult(
      loss: meanLoss,
      gradientNorm: gradNorm,
      examplesUsed: batch.length,
      step: _step,
      clipped: clipped,
    );
  }

  void _adamwUpdate(Float32List params, Float32List grads, _AdamState state) {
    state.t++;
    final double bc1 = 1.0 - math.pow(beta1, state.t).toDouble();
    final double bc2 = 1.0 - math.pow(beta2, state.t).toDouble();
    for (int i = 0; i < params.length; i++) {
      final double g = grads[i];
      state.m[i] = beta1 * state.m[i] + (1 - beta1) * g;
      state.v[i] = beta2 * state.v[i] + (1 - beta2) * g * g;
      final double mHat = state.m[i] / bc1;
      final double vHat = state.v[i] / bc2;
      // Decoupled weight decay: applied to the parameter, not the gradient.
      final double decay = weightDecay * params[i];
      params[i] -= learningRate * (mHat / (math.sqrt(vHat) + epsilon) + decay);
    }
  }

  /// Clears the replay buffer and the optimiser moments.
  void reset() {
    _replay.clear();
    _aState.m.fillRange(0, _aState.m.length, 0.0);
    _aState.v.fillRange(0, _aState.v.length, 0.0);
    _bState.m.fillRange(0, _bState.m.length, 0.0);
    _bState.v.fillRange(0, _bState.v.length, 0.0);
    _aState.t = 0;
    _bState.t = 0;
    _step = 0;
  }

  Map<String, Object?> describe() => <String, Object?>{
        'architecture': 'on_device_lora',
        'target_module': adapter.config.targetModule,
        'rank': adapter.config.rank,
        'alpha': adapter.config.alpha,
        'scaling': double.parse(adapter.config.scaling.toStringAsFixed(6)),
        'trainable_parameters': adapter.config.trainableParameters,
        'dense_equivalent_parameters':
            adapter.config.denseEquivalentParameters,
        'parameter_ratio': double.parse(
          adapter.config.parameterRatio.toStringAsFixed(6),
        ),
        'learning_rate': learningRate,
        'weight_decay': weightDecay,
        'max_gradient_norm': maxGradientNorm,
        'optimizer_steps': _step,
        'replay_size': replaySize,
        'adapter_version': adapter.version,
        'samples_seen': adapter.samplesSeen,
        'last_loss': adapter.lastLoss,
      };
}

/// Persistence boundary for adapters, so the trainer has no hard dependency on
/// the file system and can be exercised in tests without touching disk.
abstract class LoraAdapterStore {
  /// Persists [adapter] under [name].
  Future<void> save(String name, LoraAdapter adapter);

  /// Loads an adapter, or returns null when [name] is unknown.
  Future<LoraAdapter?> load(String name);

  /// Lists stored adapter names.
  Future<List<String>> list();

  /// Deletes an adapter; returns whether anything was removed.
  Future<bool> delete(String name);
}

/// JSON-file backed adapter store, one `<name>.lora.json` file per adapter.
class FileLoraAdapterStore implements LoraAdapterStore {
  FileLoraAdapterStore(this.directory);

  final Directory directory;

  File _fileFor(String name) {
    final String safe =
        name.replaceAll(RegExp(r'[^A-Za-z0-9_.-]'), '_');
    return File('${directory.path}/$safe.lora.json');
  }

  Future<void> _ensureDirectory() async {
    if (!await directory.exists()) {
      await directory.create(recursive: true);
    }
  }

  @override
  Future<void> save(String name, LoraAdapter adapter) async {
    try {
      await _ensureDirectory();
      final File target = _fileFor(name);
      final File tmp = File('${target.path}.tmp');
      await tmp.writeAsString(jsonEncode(adapter.toJson()), flush: true);
      await tmp.rename(target.path);
    } on FileSystemException catch (e) {
      throw AdapterStoreException('failed to save adapter "$name"', e);
    }
  }

  @override
  Future<LoraAdapter?> load(String name) async {
    try {
      final File target = _fileFor(name);
      if (!await target.exists()) {
        return null;
      }
      final Object? decoded = jsonDecode(await target.readAsString());
      if (decoded is! Map<Object?, Object?>) {
        throw AdapterStoreException(
          'adapter "$name" is not a JSON object',
        );
      }
      return LoraAdapter.fromJson(decoded.cast<String, Object?>());
    } on FileSystemException catch (e) {
      throw AdapterStoreException('failed to load adapter "$name"', e);
    } on FormatException catch (e) {
      throw AdapterStoreException('adapter "$name" is corrupt JSON', e);
    }
  }

  @override
  Future<List<String>> list() async {
    try {
      if (!await directory.exists()) {
        return const <String>[];
      }
      return directory
          .listSync()
          .whereType<File>()
          .map((File f) => f.path)
          .where((String p) => p.endsWith('.lora.json'))
          .map((String p) => p
              .split('/')
              .last
              .replaceAll('.lora.json', ''))
          .toList(growable: false)
        ..sort();
    } on FileSystemException catch (e) {
      throw AdapterStoreException('failed to list adapters', e);
    }
  }

  @override
  Future<bool> delete(String name) async {
    try {
      final File target = _fileFor(name);
      if (!await target.exists()) {
        return false;
      }
      await target.delete();
      return true;
    } on FileSystemException catch (e) {
      throw AdapterStoreException('failed to delete adapter "$name"', e);
    }
  }
}

/// In-memory adapter store for tests and ephemeral sessions.
class InMemoryLoraAdapterStore implements LoraAdapterStore {
  final Map<String, LoraAdapter> _adapters = <String, LoraAdapter>{};

  @override
  Future<void> save(String name, LoraAdapter adapter) async {
    _adapters[name] = adapter.clone();
  }

  @override
  Future<LoraAdapter?> load(String name) async => _adapters[name]?.clone();

  @override
  Future<List<String>> list() async => _adapters.keys.toList()..sort();

  @override
  Future<bool> delete(String name) async => _adapters.remove(name) != null;
}

/// Why a training round was allowed or denied.
class TrainingGateDecision {
  const TrainingGateDecision({required this.allowed, required this.reason});

  final bool allowed;
  final String reason;

  Map<String, Object?> toJson() =>
      <String, Object?>{'allowed': allowed, 'reason': reason};
}

/// Guards that must all pass before background training may run.
class TrainingGuards {
  const TrainingGuards({
    this.minBatteryLevel = 0.35,
    this.maxThermalState = ThermalState.fair,
    this.dailyComputeBudget = const Duration(minutes: 20),
    this.maxConsecutiveFailures = 3,
  });

  /// Battery percentage (0..1) required to start training.
  final double minBatteryLevel;

  /// Highest thermal state at which training is still permitted.
  final ThermalState maxThermalState;

  /// Total training time allowed per rolling day.
  final Duration dailyComputeBudget;

  /// Failures tolerated before the scheduler disables itself for the session.
  final int maxConsecutiveFailures;
}

/// Snapshot of the device conditions the scheduler cares about.
class DeviceState {
  const DeviceState({
    required this.isIdle,
    required this.isCharging,
    required this.batteryLevel,
    required this.thermalState,
  });

  final bool isIdle;
  final bool isCharging;
  final double batteryLevel;
  final ThermalState thermalState;

  static const DeviceState unknown = DeviceState(
    isIdle: false,
    isCharging: false,
    batteryLevel: 0.0,
    thermalState: ThermalState.nominal,
  );

  DeviceState copyWith({
    bool? isIdle,
    bool? isCharging,
    double? batteryLevel,
    ThermalState? thermalState,
  }) =>
      DeviceState(
        isIdle: isIdle ?? this.isIdle,
        isCharging: isCharging ?? this.isCharging,
        batteryLevel: batteryLevel ?? this.batteryLevel,
        thermalState: thermalState ?? this.thermalState,
      );

  @override
  String toString() =>
      'DeviceState(idle: $isIdle, charging: $isCharging, '
      'battery: ${(batteryLevel * 100).toStringAsFixed(0)}%, '
      'thermal: ${thermalState.name})';
}

/// A completed background training round.
class TrainingRoundReport {
  const TrainingRoundReport({
    required this.steps,
    required this.examplesConsumed,
    required this.firstLoss,
    required this.lastLoss,
    required this.improvement,
    required this.duration,
    required this.reason,
  });

  final int steps;
  final int examplesConsumed;
  final double firstLoss;
  final double lastLoss;

  /// `firstLoss - lastLoss`; positive means the adapter improved.
  final double improvement;
  final Duration duration;
  final String reason;

  bool get improved => improvement > 0;

  Map<String, Object?> toJson() => <String, Object?>{
        'steps': steps,
        'examples_consumed': examplesConsumed,
        'first_loss': double.parse(firstLoss.toStringAsFixed(6)),
        'last_loss': double.parse(lastLoss.toStringAsFixed(6)),
        'improvement': double.parse(improvement.toStringAsFixed(6)),
        'duration_ms': duration.inMilliseconds,
        'reason': reason,
        'improved': improved,
      };
}

/// Idle + charging gated background training loop.
///
/// The scheduler is deliberately platform-agnostic: the host pushes device
/// state in through [updateDeviceState] (or feeds it from real platform
/// channels), and the scheduler decides whether a round may run.
class IdleTrainingScheduler {
  IdleTrainingScheduler({
    required this.trainer,
    this.guards = const TrainingGuards(),
    this.batchSize = 8,
    this.stepsPerRound = 16,
    this.onRoundCompleted,
    this.onError,
  });

  final OnDeviceLoRATrainer trainer;
  final TrainingGuards guards;
  final int batchSize;
  final int stepsPerRound;

  /// Invoked after every successful round.
  final void Function(TrainingRoundReport report)? onRoundCompleted;

  /// Invoked when a round throws; the scheduler records and keeps going until
  /// the consecutive-failure ceiling is hit.
  final void Function(Object error, StackTrace stack)? onError;

  DeviceState _deviceState = DeviceState.unknown;
  Duration _computeUsedToday = Duration.zero;
  DateTime _budgetWindowStart = DateTime.now().toUtc();
  int _consecutiveFailures = 0;
  bool _running = false;
  bool _disposed = false;
  final List<TrainingRoundReport> _history = <TrainingRoundReport>[];
  final StreamController<TrainingRoundReport> _reports =
      StreamController<TrainingRoundReport>.broadcast();

  /// Reports emitted whenever a round completes.
  Stream<TrainingRoundReport> get reports => _reports.stream;

  /// Most recent device state.
  DeviceState get deviceState => _deviceState;

  /// Compute consumed inside the current daily window.
  Duration get computeUsedToday => _computeUsedToday;

  /// Completed rounds this session.
  List<TrainingRoundReport> get history =>
      List<TrainingRoundReport>.unmodifiable(_history);

  /// Whether a round is currently executing.
  bool get isRunning => _running;

  /// Remaining compute budget in the current window.
  Duration get remainingBudget {
    final Duration left = guards.dailyComputeBudget - _computeUsedToday;
    return left.isNegative ? Duration.zero : left;
  }

  /// Pushes a new device state into the scheduler.
  void updateDeviceState(DeviceState state) {
    _deviceState = state;
  }

  /// Rolls the daily budget window over when 24h have elapsed.
  void _maybeRollBudgetWindow(DateTime now) {
    if (now.difference(_budgetWindowStart) >= const Duration(days: 1)) {
      _budgetWindowStart = now;
      _computeUsedToday = Duration.zero;
    }
  }

  /// Decides whether training may run right now.
  TrainingGateDecision evaluateGate({DateTime? now}) {
    final DateTime clock = (now ?? DateTime.now()).toUtc();
    _maybeRollBudgetWindow(clock);

    if (_disposed) {
      return const TrainingGateDecision(
        allowed: false,
        reason: 'scheduler disposed',
      );
    }
    if (_running) {
      return const TrainingGateDecision(
        allowed: false,
        reason: 'a training round is already in progress',
      );
    }
    if (_consecutiveFailures >= guards.maxConsecutiveFailures) {
      return TrainingGateDecision(
        allowed: false,
        reason: 'disabled after ${guards.maxConsecutiveFailures} '
            'consecutive failures',
      );
    }
    if (!_deviceState.isIdle) {
      return const TrainingGateDecision(
        allowed: false,
        reason: 'device is not idle',
      );
    }
    if (!_deviceState.isCharging) {
      return const TrainingGateDecision(
        allowed: false,
        reason: 'device is not charging',
      );
    }
    if (_deviceState.batteryLevel < guards.minBatteryLevel) {
      return TrainingGateDecision(
        allowed: false,
        reason: 'battery ${(_deviceState.batteryLevel * 100).toStringAsFixed(0)}% '
            'below floor ${(guards.minBatteryLevel * 100).toStringAsFixed(0)}%',
      );
    }
    if (_deviceState.thermalState > guards.maxThermalState) {
      return TrainingGateDecision(
        allowed: false,
        reason: 'thermal state ${_deviceState.thermalState.name} exceeds '
            'ceiling ${guards.maxThermalState.name}',
      );
    }
    if (remainingBudget <= Duration.zero) {
      return const TrainingGateDecision(
        allowed: false,
        reason: 'daily compute budget exhausted',
      );
    }
    if (trainer.replaySize == 0) {
      return const TrainingGateDecision(
        allowed: false,
        reason: 'no replay examples available',
      );
    }
    return const TrainingGateDecision(
      allowed: true,
      reason: 'idle, charging, within thermal and compute budget',
    );
  }

  /// Runs one round if the gate allows it. Returns null when it does not.
  Future<TrainingRoundReport?> maybeRunRound({DateTime? now}) async {
    final TrainingGateDecision gate = evaluateGate(now: now);
    if (!gate.allowed) {
      return null;
    }
    return runRound(reason: gate.reason);
  }

  /// Runs one round unconditionally, recording failures.
  Future<TrainingRoundReport> runRound({String reason = 'manual'}) async {
    if (_running) {
      throw StateError('a training round is already in progress');
    }
    _running = true;
    final DateTime started = DateTime.now().toUtc();
    try {
      final List<TrainingStepResult> steps = <TrainingStepResult>[];
      for (int i = 0; i < stepsPerRound; i++) {
        final TrainingStepResult? result =
            trainer.trainOnReplay(batchSize: batchSize);
        if (result == null) {
          break;
        }
        steps.add(result);
      }
      if (steps.isEmpty) {
        throw StateError('replay buffer became empty during the round');
      }
      final Duration elapsed = DateTime.now().toUtc().difference(started);
      _computeUsedToday += elapsed;
      _consecutiveFailures = 0;

      final TrainingRoundReport report = TrainingRoundReport(
        steps: steps.length,
        examplesConsumed: steps.fold<int>(
          0,
          (int a, TrainingStepResult s) => a + s.examplesUsed,
        ),
        firstLoss: steps.first.loss,
        lastLoss: steps.last.loss,
        improvement: steps.first.loss - steps.last.loss,
        duration: elapsed,
        reason: reason,
      );
      _history.add(report);
      if (!_reports.isClosed) {
        _reports.add(report);
      }
      onRoundCompleted?.call(report);
      return report;
    } catch (error, stack) {
      _consecutiveFailures++;
      onError?.call(error, stack);
      rethrow;
    } finally {
      _running = false;
    }
  }

  /// Convenience helper: runs [rounds] gated rounds back to back.
  Future<List<TrainingRoundReport>> runGatedRounds(int rounds) async {
    final List<TrainingRoundReport> out = <TrainingRoundReport>[];
    for (int i = 0; i < rounds; i++) {
      final TrainingRoundReport? r = await maybeRunRound();
      if (r == null) {
        break;
      }
      out.add(r);
    }
    return out;
  }

  /// Releases stream resources.
  Future<void> dispose() async {
    _disposed = true;
    await _reports.close();
  }

  Map<String, Object?> describe() => <String, Object?>{
        'architecture': 'idle_training_scheduler',
        'device_state': _deviceState.toString(),
        'is_running': _running,
        'consecutive_failures': _consecutiveFailures,
        'compute_used_today_ms': _computeUsedToday.inMilliseconds,
        'remaining_budget_ms': remainingBudget.inMilliseconds,
        'rounds_completed': _history.length,
        'gate': evaluateGate().toJson(),
      };
}