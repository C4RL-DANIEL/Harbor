// Unit tests for the pure-Dart fine-grained Mixture-of-Experts router and the
// on-device LoRA self-learning stack.
//
// Systems under test:
//   * lib/core/engine/fine_grained_moe.dart
//   * lib/core/engine/on_device_lora.dart
//
// Only the public API is exercised. Run with:
//   flutter test test/engine_moe_lora_test.dart

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:main_app/core/engine/fine_grained_moe.dart';
import 'package:main_app/core/engine/on_device_lora.dart';

/// A small MoE geometry that keeps every test fast.
MoeConfig _moeConfig({
  double capacityFactor = 4.0,
  int numSharedExperts = 1,
  int topK = 2,
  int numRoutedExperts = 4,
}) =>
    MoeConfig(
      hiddenSize: 8,
      intermediateSize: 4,
      numRoutedExperts: numRoutedExperts,
      topK: topK,
      numSharedExperts: numSharedExperts,
      sharedIntermediateSize: 4,
      capacityFactor: capacityFactor,
    );

Float32List _token(int hiddenSize, int seedIndex) => Float32List.fromList(
      List<double>.generate(
        hiddenSize,
        (int i) => ((i + 1) * (seedIndex + 1) % 11) / 11.0 - 0.5,
      ),
    );

List<Float32List> _batch(int hiddenSize, int count) =>
    List<Float32List>.generate(count, (int i) => _token(hiddenSize, i));

/// A minimal LoRA geometry shared by the trainer tests.
LoraConfig _loraConfig() => const LoraConfig(
      rank: 3,
      inFeatures: 6,
      outFeatures: 5,
      alpha: 3.0,
      targetModule: 'q_proj',
    );

/// Mean squared error of the adapter's low-rank correction on [examples].
double _evalMse(LoraAdapter adapter, List<LoraTrainingExample> examples) {
  double total = 0.0;
  for (final LoraTrainingExample ex in examples) {
    final Float32List y = adapter.forward(ex.input);
    for (int o = 0; o < y.length; o++) {
      final double diff = y[o] - ex.target[o];
      total += diff * diff;
    }
  }
  return total / examples.length;
}

/// Builds a fixed, deterministic linear regression task: `target = T · x`.
List<LoraTrainingExample> _linearTask(
  LoraConfig config,
  int count, {
  int seed = 99,
}) {
  final math.Random rng = math.Random(seed);
  final List<List<double>> target = List<List<double>>.generate(
    config.outFeatures,
    (int o) => List<double>.generate(
      config.inFeatures,
      (int i) => rng.nextDouble() * 2.0 - 1.0,
    ),
  );
  return List<LoraTrainingExample>.generate(count, (int n) {
    final Float32List x = Float32List.fromList(
      List<double>.generate(config.inFeatures, (int i) => rng.nextDouble() * 2.0 - 1.0),
    );
    final Float32List y = Float32List(config.outFeatures);
    for (int o = 0; o < config.outFeatures; o++) {
      double acc = 0.0;
      for (int i = 0; i < config.inFeatures; i++) {
        acc += target[o][i] * x[i];
      }
      y[o] = acc;
    }
    return LoraTrainingExample(input: x, target: y);
  });
}

/// A trainer with an initially-zero adapter and a populated replay buffer.
({OnDeviceLoRATrainer trainer, LoraAdapter adapter, List<LoraTrainingExample> data})
    _populatedTrainer({int examples = 24, int maxReplayBuffer = 128}) {
  final LoraConfig config = _loraConfig();
  final LoraAdapter adapter = LoraAdapter.initialized(config, seed: 4242);
  final OnDeviceLoRATrainer trainer = OnDeviceLoRATrainer(
    adapter: adapter,
    learningRate: 0.05,
    weightDecay: 0.0,
    maxGradientNorm: 10.0,
    maxReplayBuffer: maxReplayBuffer,
    seed: 7,
  );
  final List<LoraTrainingExample> data = _linearTask(config, examples);
  for (final LoraTrainingExample ex in data) {
    trainer.addExample(ex);
  }
  return (trainer: trainer, adapter: adapter, data: data);
}

const DeviceState _goodDevice = DeviceState(
  isIdle: true,
  isCharging: true,
  batteryLevel: 0.9,
  thermalState: ThermalState.nominal,
);

void main() {
  // -------------------------------------------------------------------------
  // Fine-grained MoE
  // -------------------------------------------------------------------------
  group('MoeConfig.capacityFor', () {
    test('matches the documented ceiling formula exactly', () {
      final MoeConfig config = _moeConfig(capacityFactor: 1.25);
      // ceil(8 * 2 / 4 * 1.25) = ceil(5.0) = 5.
      expect(config.capacityFor(8), 5);
      // ceil(4 * 2 / 4 * 1.25) = ceil(2.5) = 3.
      expect(config.capacityFor(4), 3);
      // ceil(1 * 2 / 4 * 1.25) = ceil(0.625) = 1.
      expect(config.capacityFor(1), 1);
    });

    test('is never below one, even for an empty batch', () {
      final MoeConfig config = _moeConfig(capacityFactor: 0.01);
      expect(config.capacityFor(0), 1);
      expect(config.capacityFor(1), greaterThanOrEqualTo(1));
      expect(
        _moeConfig(capacityFactor: 0.0001, numRoutedExperts: 64, topK: 1)
            .capacityFor(1),
        1,
      );
    });
  });

  group('FineGrainedMoE.forward', () {
    test('returns one hiddenSize output and one decision per token', () {
      final MoeConfig config = _moeConfig();
      final FineGrainedMoE moe = FineGrainedMoE(config, seed: 11);
      final List<Float32List> tokens = _batch(config.hiddenSize, 5);

      final MoeForwardResult result = moe.forward(tokens);
      expect(result.outputs.length, tokens.length);
      expect(result.decisions.length, tokens.length);
      for (final Float32List out in result.outputs) {
        expect(out.length, config.hiddenSize);
      }
      expect(result.load.tokenCount, tokens.length);
      expect(result.load.assignments.length, config.numRoutedExperts);
    });

    test('empty batch yields empty outputs and load', () {
      final MoeConfig config = _moeConfig();
      final FineGrainedMoE moe = FineGrainedMoE(config, seed: 3);
      final MoeForwardResult result = moe.forward(<Float32List>[]);
      expect(result.outputs, isEmpty);
      expect(result.decisions, isEmpty);
      expect(result.load.tokenCount, 0);
      expect(result.load.dropped, 0);
    });

    test('with headroom each token activates exactly topK unique experts', () {
      final MoeConfig config = _moeConfig(capacityFactor: 4.0, topK: 2);
      final FineGrainedMoE moe = FineGrainedMoE(config, seed: 5);
      final MoeForwardResult result = moe.forward(_batch(config.hiddenSize, 4));

      for (final MoeRoutingDecision d in result.decisions) {
        expect(d.expertIndices.length, config.topK);
        expect(d.dropped, isFalse);
        expect(d.expertIndices.toSet().length, d.expertIndices.length);
        for (final int e in d.expertIndices) {
          expect(e, inInclusiveRange(0, config.numRoutedExperts - 1));
        }
      }
      expect(result.load.dropped, 0);
      expect(result.load.accepted, 4 * config.topK);
    });

    test('normalised top-K accepted weights sum to one', () {
      final MoeConfig config = _moeConfig(capacityFactor: 4.0);
      final FineGrainedMoE moe = FineGrainedMoE(config, seed: 6);
      final MoeForwardResult result = moe.forward(_batch(config.hiddenSize, 6));
      for (final MoeRoutingDecision d in result.decisions) {
        final double sum =
            d.weights.fold<double>(0.0, (double a, double b) => a + b);
        expect(sum, closeTo(1.0, 1e-6));
        expect(d.weights.length, d.expertIndices.length);
      }
    });

    test('shared experts always fire; zero shared experts never apply', () {
      final MoeConfig withShared = _moeConfig(numSharedExperts: 2);
      final FineGrainedMoE sharedMoe = FineGrainedMoE(withShared, seed: 8);
      final MoeForwardResult sharedResult =
          sharedMoe.forward(_batch(withShared.hiddenSize, 4));
      expect(sharedResult.decisions.every((MoeRoutingDecision d) => d.sharedApplied), isTrue);

      final MoeConfig withoutShared = _moeConfig(numSharedExperts: 0);
      final FineGrainedMoE bareMoe = FineGrainedMoE(withoutShared, seed: 8);
      final MoeForwardResult bareResult =
          bareMoe.forward(_batch(withoutShared.hiddenSize, 4));
      expect(bareResult.decisions.every((MoeRoutingDecision d) => d.sharedApplied), isFalse);
      expect(bareMoe.sharedExperts, isEmpty);
    });

    test('a tiny capacity factor drops assignments but keeps output shape', () {
      final MoeConfig config = _moeConfig(capacityFactor: 0.01, topK: 2);
      final FineGrainedMoE moe = FineGrainedMoE(config, seed: 13);
      final List<Float32List> tokens = _batch(config.hiddenSize, 64);

      final MoeForwardResult result = moe.forward(tokens);
      expect(result.load.capacity, 1);
      expect(result.load.dropped, greaterThan(0));
      expect(
        result.load.accepted + result.load.dropped,
        tokens.length * config.topK,
      );
      expect(
        result.load.accepted,
        result.load.assignments.fold<int>(0, (int a, int b) => a + b),
      );

      final int acceptedFromDecisions = result.decisions.fold<int>(
        0,
        (int a, MoeRoutingDecision d) => a + d.expertIndices.length,
      );
      expect(acceptedFromDecisions, result.load.accepted);
      for (final Float32List out in result.outputs) {
        expect(out.length, config.hiddenSize);
      }
      expect(result.decisions.any((MoeRoutingDecision d) => d.dropped), isTrue);
    });

    test('top-K selection distributes across more than one expert', () {
      final MoeConfig config = _moeConfig(capacityFactor: 4.0);
      final FineGrainedMoE moe = FineGrainedMoE(config, seed: 17);
      final MoeForwardResult result = moe.forward(_batch(config.hiddenSize, 48));

      final Set<int> selected = <int>{};
      for (final MoeRoutingDecision d in result.decisions) {
        selected.addAll(d.expertIndices);
      }
      expect(selected.length, greaterThan(1));
      expect(result.load.balanceScore, inInclusiveRange(0.0, 1.0));
      expect(result.load.dropped, 0);
    });

    test('rejects topK greater than the routed expert count', () {
      expect(
        () => MoeConfig(
          hiddenSize: 8,
          intermediateSize: 4,
          numRoutedExperts: 2,
          topK: 3,
        ),
        throwsA(isA<AssertionError>()),
      );
    });

    test('throws MoeShapeException on a wrongly sized token', () {
      final MoeConfig config = _moeConfig();
      final FineGrainedMoE moe = FineGrainedMoE(config, seed: 19);
      expect(
        () => moe.forward(<Float32List>[Float32List(config.hiddenSize + 1)]),
        throwsA(isA<MoeShapeException>()),
      );
      expect(
        () => moe.forward(<Float32List>[
          _token(config.hiddenSize, 0),
          Float32List(config.hiddenSize - 1),
        ]),
        throwsA(isA<MoeShapeException>()),
      );
    });

    test('forwardOne matches a single-token batch', () {
      final MoeConfig config = _moeConfig();
      final FineGrainedMoE moe = FineGrainedMoE(config, seed: 23);
      final Float32List single = moe.forwardOne(_token(config.hiddenSize, 1));
      final FineGrainedMoE reference = FineGrainedMoE(config, seed: 23);
      final Float32List batched =
          reference.forward(<Float32List>[_token(config.hiddenSize, 1)]).outputs.single;
      expect(single.length, config.hiddenSize);
      expect(single, orderedEquals(batched));
    });

    test('describe reports configured expert geometry', () {
      final MoeConfig config = _moeConfig(topK: 2, numRoutedExperts: 4);
      final FineGrainedMoE moe = FineGrainedMoE(config, seed: 29);
      final Map<String, Object?> info = moe.describe();
      expect(info['architecture'], 'fine_grained_moe');
      expect(info['routed_experts'], 4);
      expect(info['top_k'], 2);
      expect(info['hidden_size'], 8);
      expect(info['shared_experts'], 1);
    });
  });

  group('ExpertLoadReport', () {
    test('a perfectly uniform load scores 1.0 with max entropy ln(n)', () {
      const ExpertLoadReport report = ExpertLoadReport(
        assignments: <int>[2, 2, 2, 2],
        dropped: 0,
        capacity: 2,
        tokenCount: 4,
      );
      expect(report.accepted, 8);
      expect(report.maxEntropy, closeTo(math.log(4), 1e-12));
      expect(report.balanceEntropy, closeTo(math.log(4), 1e-12));
      expect(report.balanceScore, closeTo(1.0, 1e-12));
    });

    test('an unbalanced load scores below one but stays in range', () {
      const ExpertLoadReport report = ExpertLoadReport(
        assignments: <int>[8, 0, 0, 0],
        dropped: 0,
        capacity: 8,
        tokenCount: 8,
      );
      expect(report.balanceEntropy, closeTo(0.0, 1e-12));
      expect(report.balanceScore, closeTo(0.0, 1e-12));
      expect(report.balanceScore, inInclusiveRange(0.0, 1.0));
    });

    test('toJson exposes the documented contract', () {
      const ExpertLoadReport report = ExpertLoadReport(
        assignments: <int>[1, 1, 1, 1],
        dropped: 2,
        capacity: 1,
        tokenCount: 6,
      );
      final Map<String, Object?> json = report.toJson();
      expect(
        json.keys.toSet(),
        <String>{
          'token_count',
          'capacity',
          'assignments',
          'accepted',
          'dropped',
          'drop_rate',
          'balance_entropy',
          'balance_score',
        },
      );
      expect(json['token_count'], 6);
      expect(json['capacity'], 1);
      expect(json['assignments'], orderedEquals(<int>[1, 1, 1, 1]));
      expect(json['accepted'], 4);
      expect(json['dropped'], 2);
      expect(json['drop_rate'], closeTo(2 / (6 * 4) * 100, 1e-3));
      expect(json['balance_score'], closeTo(1.0, 1e-4));
    });
  });

  // -------------------------------------------------------------------------
  // LoRA adapter geometry and identity initialisation
  // -------------------------------------------------------------------------
  group('LoraAdapter initialisation', () {
    test('B starts at zero, so the adapter is an exact identity', () {
      final LoraConfig config = _loraConfig();
      final LoraAdapter adapter = LoraAdapter.initialized(config, seed: 123);
      expect(adapter.b.every((double v) => v == 0.0), isTrue);

      final Float32List x = Float32List.fromList(<double>[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]);
      final Float32List y = adapter.forward(x);
      expect(y.length, config.outFeatures);
      expect(y.every((double v) => v == 0.0), isTrue);
    });

    test('mergeInto returns a bit-identical copy at initialisation', () {
      final LoraConfig config = _loraConfig();
      final LoraAdapter adapter = LoraAdapter.initialized(config, seed: 321);
      final Float32List w = Float32List.fromList(
        List<double>.generate(config.outFeatures * config.inFeatures, (int i) => i * 0.25),
      );
      final Float32List merged = adapter.mergeInto(w);
      expect(merged.length, w.length);
      expect(merged, orderedEquals(w));
      // The base buffer must be untouched.
      expect(w[0], 0.0);
      expect(w[1], 0.25);
    });
  });

  group('LoraConfig accounting', () {
    test('scaling is alpha over rank', () {
      const LoraConfig config = LoraConfig(
        rank: 4,
        inFeatures: 8,
        outFeatures: 6,
        alpha: 8.0,
      );
      expect(config.scaling, 8.0 / 4.0);
      expect(config.scaling, 2.0);
    });

    test('trainable parameters and parameter ratio match the LoRA math', () {
      const LoraConfig config = LoraConfig(
        rank: 4,
        inFeatures: 32,
        outFeatures: 64,
        alpha: 16.0,
      );
      expect(config.trainableParameters, 4 * 32 + 64 * 4);
      expect(config.trainableParameters, 384);
      expect(config.denseEquivalentParameters, 32 * 64);
      expect(config.parameterRatio, closeTo(384 / 2048, 1e-12));
      expect(config.parameterRatio, greaterThan(0.0));
      expect(config.parameterRatio, lessThan(1.0));
    });
  });

  group('merge / unmerge round-trip', () {
    test('unmergeFrom(mergeInto(W)) recovers W after training', () {
      final LoraConfig config = _loraConfig();
      final LoraAdapter adapter = LoraAdapter.initialized(config, seed: 909);
      final OnDeviceLoRATrainer trainer = OnDeviceLoRATrainer(
        adapter: adapter,
        learningRate: 0.05,
        seed: 3,
      );
      final List<LoraTrainingExample> data = _linearTask(config, 16);
      for (int s = 0; s < 10; s++) {
        trainer.trainStep(data.sublist(0, 4));
      }
      expect(adapter.b.any((double v) => v != 0.0), isTrue);

      final Float32List w = Float32List.fromList(
        List<double>.generate(
          config.outFeatures * config.inFeatures,
          (int i) => math.sin(i * 0.37) * 0.5,
        ),
      );
      final Float32List merged = adapter.mergeInto(w);
      final Float32List recovered = adapter.unmergeFrom(merged);
      expect(recovered.length, w.length);
      for (int i = 0; i < w.length; i++) {
        expect(recovered[i], closeTo(w[i], 1e-4));
      }
      // The applied correction must actually be non-trivial.
      expect(merged, isNot(orderedEquals(w)));
    });
  });

  // -------------------------------------------------------------------------
  // OnDeviceLoRATrainer
  // -------------------------------------------------------------------------
  group('OnDeviceLoRATrainer', () {
    test('actually learns a synthetic linear mapping', () {
      final ({OnDeviceLoRATrainer trainer, LoraAdapter adapter, List<LoraTrainingExample> data})
          setup = _populatedTrainer(examples: 32);
      final LoraAdapter adapter = setup.adapter;
      final List<LoraTrainingExample> data = setup.data;
      final OnDeviceLoRATrainer trainer = setup.trainer;

      final double initialEval = _evalMse(adapter, data);
      expect(initialEval, greaterThan(0.0));

      TrainingStepResult? last;
      const int steps = 200;
      for (int s = 0; s < steps; s++) {
        last = trainer.trainOnReplay(batchSize: 8);
        expect(last, isNotNull);
      }

      final double finalEval = _evalMse(adapter, data);
      expect(finalEval, lessThan(initialEval * 0.5));
      expect(finalEval, lessThan(initialEval));
      expect(last!.step, steps);
      expect(trainer.step, steps);
      expect(adapter.samplesSeen, steps * 8);
      expect(adapter.samplesSeen, greaterThan(0));
      expect(last.loss, isNonNegative);
      expect(adapter.lastLoss, isNotNull);
      expect(adapter.version, greaterThan(1));
    });

    test('clips the first step with a tiny threshold and not with a huge one', () {
      final LoraConfig config = _loraConfig();
      final List<LoraTrainingExample> data = _linearTask(config, 8).sublist(0, 4);

      final LoraAdapter clippedAdapter = LoraAdapter.initialized(config, seed: 1);
      final OnDeviceLoRATrainer tiny = OnDeviceLoRATrainer(
        adapter: clippedAdapter,
        maxGradientNorm: 1e-6,
        seed: 2,
      );
      final TrainingStepResult clippedResult = tiny.trainStep(data);
      expect(clippedResult.gradientNorm, greaterThan(0.0));
      expect(clippedResult.clipped, isTrue);

      final LoraAdapter freeAdapter = LoraAdapter.initialized(config, seed: 1);
      final OnDeviceLoRATrainer huge = OnDeviceLoRATrainer(
        adapter: freeAdapter,
        maxGradientNorm: 1e9,
        seed: 2,
      );
      final TrainingStepResult freeResult = huge.trainStep(data);
      expect(freeResult.gradientNorm, greaterThan(0.0));
      expect(freeResult.clipped, isFalse);
    });

    test('trainOnReplay returns null on an empty buffer; trainStep throws', () {
      final LoraConfig config = _loraConfig();
      final LoraAdapter adapter = LoraAdapter.initialized(config, seed: 5);
      final OnDeviceLoRATrainer trainer = OnDeviceLoRATrainer(adapter: adapter, seed: 5);
      expect(trainer.replaySize, 0);
      expect(trainer.trainOnReplay(batchSize: 8), isNull);
      expect(
        () => trainer.trainStep(<LoraTrainingExample>[]),
        throwsA(isA<LoraShapeException>()),
      );
      expect(trainer.step, 0);
    });

    test('addExample validates input and target widths', () {
      final LoraConfig config = _loraConfig();
      final LoraAdapter adapter = LoraAdapter.initialized(config, seed: 6);
      final OnDeviceLoRATrainer trainer = OnDeviceLoRATrainer(adapter: adapter, seed: 6);

      expect(
        () => trainer.addExample(LoraTrainingExample(
          input: Float32List(config.inFeatures + 1),
          target: Float32List(config.outFeatures),
        )),
        throwsA(isA<LoraShapeException>()),
      );
      expect(
        () => trainer.addExample(LoraTrainingExample(
          input: Float32List(config.inFeatures),
          target: Float32List(config.outFeatures - 1),
        )),
        throwsA(isA<LoraShapeException>()),
      );
      expect(trainer.replaySize, 0);
    });

    test('the replay buffer never exceeds maxReplayBuffer', () {
      final LoraConfig config = _loraConfig();
      final LoraAdapter adapter = LoraAdapter.initialized(config, seed: 7);
      final OnDeviceLoRATrainer trainer = OnDeviceLoRATrainer(
        adapter: adapter,
        maxReplayBuffer: 8,
        seed: 7,
      );
      final List<LoraTrainingExample> data = _linearTask(config, 60);
      for (final LoraTrainingExample ex in data) {
        trainer.addExample(ex);
      }
      expect(trainer.replaySize, 8);
      expect(trainer.replaySize, lessThanOrEqualTo(8));
      expect(trainer.sampleBatch(8).length, 8);
      expect(trainer.sampleBatch(100).length, 8);
    });

    test('describe reports optimizer and adapter state', () {
      final ({OnDeviceLoRATrainer trainer, LoraAdapter adapter, List<LoraTrainingExample> data})
          setup = _populatedTrainer(examples: 8);
      setup.trainer.trainOnReplay(batchSize: 4);
      final Map<String, Object?> info = setup.trainer.describe();
      expect(info['architecture'], 'on_device_lora');
      expect(info['rank'], 3);
      expect(info['trainable_parameters'], 3 * 6 + 5 * 3);
      expect(info['optimizer_steps'], 1);
      expect(info['samples_seen'], 4);
      expect(info['replay_size'], 8);
    });
  });

  group('LoraAdapter JSON round-trip', () {
    test('toJson / fromJson preserves config, A and B exactly', () {
      final LoraConfig config = _loraConfig();
      final LoraAdapter adapter = LoraAdapter.initialized(config, seed: 2024);
      final OnDeviceLoRATrainer trainer = OnDeviceLoRATrainer(adapter: adapter, seed: 8);
      final List<LoraTrainingExample> data = _linearTask(config, 12);
      for (int s = 0; s < 5; s++) {
        trainer.trainStep(data.sublist(0, 4));
      }
      adapter
        ..version = 42
        ..samplesSeen = 123;
      adapter.lastLoss = 0.5;

      final String encoded = jsonEncode(adapter.toJson());
      final Map<String, Object?> decoded =
          (jsonDecode(encoded) as Map<Object?, Object?>).cast<String, Object?>();
      final LoraAdapter restored = LoraAdapter.fromJson(decoded);

      expect(restored.config.rank, config.rank);
      expect(restored.config.inFeatures, config.inFeatures);
      expect(restored.config.outFeatures, config.outFeatures);
      expect(restored.config.alpha, config.alpha);
      expect(restored.config.targetModule, config.targetModule);
      expect(restored.a.length, adapter.a.length);
      expect(restored.b.length, adapter.b.length);
      for (int i = 0; i < adapter.a.length; i++) {
        expect(restored.a[i], adapter.a[i]);
      }
      for (int i = 0; i < adapter.b.length; i++) {
        expect(restored.b[i], adapter.b[i]);
      }
      expect(restored.version, 42);
      expect(restored.samplesSeen, 123);
      expect(restored.lastLoss, 0.5);
    });

    test('fromJson rejects a truncated A tensor', () {
      final LoraConfig config = _loraConfig();
      final LoraAdapter adapter = LoraAdapter.initialized(config, seed: 9);
      final Map<String, Object?> json = adapter.toJson();
      final List<Object?> shortA = (json['a'] as List<Object?>).sublist(1);
      json['a'] = shortA;
      expect(
        () => LoraAdapter.fromJson(json),
        throwsA(isA<LoraShapeException>()),
      );
    });

    test('fromJson rejects a truncated B tensor', () {
      final LoraConfig config = _loraConfig();
      final LoraAdapter adapter = LoraAdapter.initialized(config, seed: 10);
      final Map<String, Object?> json = adapter.toJson();
      json['b'] = (json['b'] as List<Object?>).sublist(1);
      expect(
        () => LoraAdapter.fromJson(json),
        throwsA(isA<LoraShapeException>()),
      );
    });
  });

  group('LoraAdapterStore', () {
    Directory? tempDir;

    tearDown(() async {
      final Directory? dir = tempDir;
      if (dir != null && await dir.exists()) {
        await dir.delete(recursive: true);
      }
      tempDir = null;
    });

    test('InMemoryLoraAdapterStore save / load / list / delete round-trips', () async {
      final LoraConfig config = _loraConfig();
      final LoraAdapter adapter = LoraAdapter.initialized(config, seed: 77);
      final InMemoryLoraAdapterStore store = InMemoryLoraAdapterStore();

      expect(await store.load('missing'), isNull);
      expect(await store.list(), isEmpty);
      expect(await store.delete('missing'), isFalse);

      await store.save('alpha', adapter);
      final LoraAdapter? loaded = await store.load('alpha');
      expect(loaded, isNotNull);
      expect(loaded!.config.rank, config.rank);
      expect(loaded.a, orderedEquals(adapter.a));
      expect(loaded.b, orderedEquals(adapter.b));
      expect(await store.list(), <String>['alpha']);

      await store.save('beta', adapter);
      expect(await store.list(), <String>['alpha', 'beta']);
      expect(await store.delete('alpha'), isTrue);
      expect(await store.load('alpha'), isNull);
      expect(await store.list(), <String>['beta']);
    });

    test('FileLoraAdapterStore persists, lists and deletes adapters', () async {
      tempDir = await Directory.systemTemp.createTemp('harbor_lora_store_');
      final FileLoraAdapterStore store = FileLoraAdapterStore(tempDir!);
      final LoraConfig config = _loraConfig();
      final LoraAdapter adapter = LoraAdapter.initialized(config, seed: 88);

      final String encoded = jsonEncode(adapter.toJson());
      final Map<String, Object?> decoded =
          (jsonDecode(encoded) as Map<Object?, Object?>).cast<String, Object?>();
      final LoraAdapter restored = LoraAdapter.fromJson(decoded);
      restored
        ..version = 5
        ..samplesSeen = 20;

      expect(await store.load('nope'), isNull);
      expect(await store.list(), isEmpty);
      expect(await store.delete('nope'), isFalse);

      await store.save('checkpoint', restored);
      expect(await store.list(), <String>['checkpoint']);
      final LoraAdapter? loaded = await store.load('checkpoint');
      expect(loaded, isNotNull);
      expect(loaded!.version, 5);
      expect(loaded.samplesSeen, 20);
      expect(loaded.a, orderedEquals(restored.a));
      expect(loaded.b, orderedEquals(restored.b));

      await store.save('other', restored);
      expect((await store.list()).toSet(), <String>{'checkpoint', 'other'});
      expect(await store.delete('checkpoint'), isTrue);
      expect(await store.load('checkpoint'), isNull);
      expect(await store.list(), <String>['other']);
    });
  });

  // -------------------------------------------------------------------------
  // IdleTrainingScheduler gating
  // -------------------------------------------------------------------------
  group('IdleTrainingScheduler gating', () {
    IdleTrainingScheduler buildScheduler({
      TrainingGuards guards = const TrainingGuards(),
      bool populateReplay = true,
      int stepsPerRound = 3,
    }) {
      final LoraConfig config = _loraConfig();
      final LoraAdapter adapter = LoraAdapter.initialized(config, seed: 55);
      final OnDeviceLoRATrainer trainer = OnDeviceLoRATrainer(
        adapter: adapter,
        learningRate: 0.01,
        seed: 55,
      );
      if (populateReplay) {
        for (final LoraTrainingExample ex in _linearTask(config, 12)) {
          trainer.addExample(ex);
        }
      }
      return IdleTrainingScheduler(
        trainer: trainer,
        guards: guards,
        batchSize: 4,
        stepsPerRound: stepsPerRound,
      );
    }

    test('denies training when the device is not idle', () {
      final IdleTrainingScheduler scheduler = buildScheduler();
      scheduler.updateDeviceState(_goodDevice.copyWith(isIdle: false));
      final TrainingGateDecision gate = scheduler.evaluateGate();
      expect(gate.allowed, isFalse);
      expect(gate.reason.toLowerCase(), contains('idle'));
      expect(gate.toJson()['allowed'], isFalse);
      expect(gate.toJson()['reason'], gate.reason);
    });

    test('denies training when idle but not charging', () {
      final IdleTrainingScheduler scheduler = buildScheduler();
      scheduler.updateDeviceState(_goodDevice.copyWith(isCharging: false));
      final TrainingGateDecision gate = scheduler.evaluateGate();
      expect(gate.allowed, isFalse);
      expect(gate.reason.toLowerCase(), contains('charging'));
    });

    test('denies training when battery is below the floor', () {
      final IdleTrainingScheduler scheduler = buildScheduler();
      scheduler.updateDeviceState(_goodDevice.copyWith(batteryLevel: 0.1));
      final TrainingGateDecision gate = scheduler.evaluateGate();
      expect(gate.allowed, isFalse);
      expect(gate.reason.toLowerCase(), contains('battery'));
    });

    test('denies training when thermal state exceeds the ceiling', () {
      final IdleTrainingScheduler scheduler = buildScheduler();
      scheduler.updateDeviceState(
        _goodDevice.copyWith(thermalState: ThermalState.critical),
      );
      final TrainingGateDecision gate = scheduler.evaluateGate();
      expect(gate.allowed, isFalse);
      expect(gate.reason.toLowerCase(), contains('thermal'));
    });

    test('denies training when the daily compute budget is exhausted', () {
      final IdleTrainingScheduler scheduler = buildScheduler(
        guards: const TrainingGuards(dailyComputeBudget: Duration.zero),
      );
      scheduler.updateDeviceState(_goodDevice);
      final TrainingGateDecision gate = scheduler.evaluateGate();
      expect(gate.allowed, isFalse);
      expect(gate.reason.toLowerCase(), contains('budget'));
      expect(scheduler.remainingBudget, Duration.zero);
      expect(scheduler.maybeRunRound(), completion(isNull));
    });

    test('denies training when the replay buffer is empty', () {
      final IdleTrainingScheduler scheduler = buildScheduler(populateReplay: false);
      scheduler.updateDeviceState(_goodDevice);
      final TrainingGateDecision gate = scheduler.evaluateGate();
      expect(gate.allowed, isFalse);
      expect(gate.reason.toLowerCase(), contains('replay'));
    });

    test('allows training when every guard passes and replay is populated', () async {
      final IdleTrainingScheduler scheduler = buildScheduler();
      scheduler.updateDeviceState(_goodDevice);
      final TrainingGateDecision gate = scheduler.evaluateGate();
      expect(gate.allowed, isTrue);
      expect(gate.reason.toLowerCase(), contains('idle'));

      final TrainingRoundReport? report = await scheduler.maybeRunRound();
      expect(report, isNotNull);
      expect(report!.steps, greaterThan(0));
      expect(report.examplesConsumed, greaterThan(0));
      expect(report.firstLoss, greaterThanOrEqualTo(0.0));
      expect(report.lastLoss, greaterThanOrEqualTo(0.0));
      expect(report.duration, greaterThanOrEqualTo(Duration.zero));
      expect(scheduler.isRunning, isFalse);
    });

    test('maybeRunRound returns null whenever the gate denies', () async {
      final IdleTrainingScheduler scheduler = buildScheduler();
      scheduler.updateDeviceState(_goodDevice.copyWith(isIdle: false));
      expect(await scheduler.maybeRunRound(), isNull);
      expect(scheduler.history, isEmpty);
    });

    test('consecutive failures disable the scheduler', () async {
      final IdleTrainingScheduler scheduler = buildScheduler(populateReplay: false);
      scheduler.updateDeviceState(_goodDevice);
      int failures = 0;
      for (int i = 0; i < 3; i++) {
        try {
          await scheduler.runRound(reason: 'forced');
          fail('runRound should have thrown on an empty replay buffer');
        } catch (_) {
          failures++;
        }
      }
      expect(failures, 3);
      final TrainingGateDecision gate = scheduler.evaluateGate();
      expect(gate.allowed, isFalse);
      expect(gate.reason.toLowerCase(), contains('consecutive failures'));
    });

    test('a successful round records history, emits the report and uses budget',
        () async {
      final IdleTrainingScheduler scheduler = buildScheduler();
      scheduler.updateDeviceState(_goodDevice);
      final Duration before = scheduler.computeUsedToday;
      final Future<TrainingRoundReport> emitted = scheduler.reports.first;

      final TrainingRoundReport? report = await scheduler.maybeRunRound();
      expect(report, isNotNull);
      final TrainingRoundReport streamed = await emitted;
      expect(streamed.steps, report!.steps);
      expect(streamed.lastLoss, report.lastLoss);
      expect(scheduler.history.length, 1);
      expect(scheduler.history.single.steps, report.steps);
      expect(scheduler.computeUsedToday, greaterThanOrEqualTo(before));
      expect(scheduler.computeUsedToday, greaterThanOrEqualTo(Duration.zero));
      expect(scheduler.remainingBudget, lessThanOrEqualTo(const Duration(minutes: 20)));
    });

    test('dispose closes the stream and denies further rounds', () async {
      final IdleTrainingScheduler scheduler = buildScheduler();
      scheduler.updateDeviceState(_goodDevice);
      bool done = false;
      final StreamSubscription<TrainingRoundReport> subscription =
          scheduler.reports.listen((_) {}, onDone: () => done = true);
      await scheduler.dispose();
      await Future<void>.delayed(Duration.zero);
      expect(done, isTrue);
      final TrainingGateDecision gate = scheduler.evaluateGate();
      expect(gate.allowed, isFalse);
      expect(gate.reason.toLowerCase(), contains('disposed'));
      expect(await scheduler.maybeRunRound(), isNull);
      await subscription.cancel();
    });
  });

  group('ThermalState', () {
    test('parse maps known names and defaults to nominal', () {
      expect(ThermalState.parse('critical'), ThermalState.critical);
      expect(ThermalState.parse('serious'), ThermalState.serious);
      expect(ThermalState.parse('fair'), ThermalState.fair);
      expect(ThermalState.parse('nominal'), ThermalState.nominal);
      expect(ThermalState.parse('CRITICAL'), ThermalState.critical);
      expect(ThermalState.parse('unknown'), ThermalState.nominal);
      expect(ThermalState.parse('None'), ThermalState.nominal);
      expect(ThermalState.parse(null), ThermalState.nominal);
    });

    test('comparison operators order the states by severity', () {
      expect(ThermalState.nominal < ThermalState.fair, isTrue);
      expect(ThermalState.fair < ThermalState.serious, isTrue);
      expect(ThermalState.serious < ThermalState.critical, isTrue);
      expect(ThermalState.critical > ThermalState.serious, isTrue);
      expect(ThermalState.critical >= ThermalState.critical, isTrue);
      expect(ThermalState.nominal <= ThermalState.nominal, isTrue);
      expect(ThermalState.nominal > ThermalState.critical, isFalse);
    });
  });
}