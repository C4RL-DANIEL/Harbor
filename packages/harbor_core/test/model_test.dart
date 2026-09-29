import 'dart:math' as math;
import 'dart:typed_data';

import 'package:harbor_core/src/model/interfaces.dart';
import 'package:harbor_core/src/model/linalg.dart';
import 'package:harbor_core/src/model/tiny_lm.dart';
import 'package:harbor_core/src/tokenizer/byte_tokenizer.dart';
import 'package:test/test.dart';

/// A model shape that is [TinyLmConfig.testPreset] with the vocabulary raised
/// to the smallest value the validator accepts (and to the size a byte-level
/// tokenizer always has). Everything else is the test preset verbatim.
TinyLmConfig _config({int vocabSize = 260}) =>
    TinyLmConfig.testPreset.copyWith(vocabSize: vocabSize);

const String _passage =
    'the harbor beacon sweeps the fog and guides lost boats home.';

final String _corpus = List<String>.filled(60, _passage).join();

const BpeTrainer _trainer = BpeTrainer(targetMerges: 48, maxSampleChars: 4000);

ByteTokenizer _tokenizer() => ByteTokenizer.train(_corpus, trainer: _trainer);

List<int> _sampleWindow(math.Random random, List<int> tokens, int length) {
  final int maxStart = tokens.length - length;
  if (maxStart <= 0) {
    return tokens.sublist(0, length);
  }
  final int start = random.nextInt(maxStart + 1);
  return tokens.sublist(start, start + length);
}

void _adamwStep(
  TinyLm model,
  double learningRate,
  List<Float64List> moments,
  List<Float64List> velocities,
  int step,
) {
  const double beta1 = 0.9;
  const double beta2 = 0.999;
  const double epsilon = 1e-8;
  const double weightDecay = 0.01;
  final double correction1 = 1 - math.pow(beta1, step).toDouble();
  final double correction2 = 1 - math.pow(beta2, step).toDouble();
  final List<ParamRef> parameters = model.parameters;
  for (int i = 0; i < parameters.length; i++) {
    final ParamRef parameter = parameters[i];
    final Float64List moment = moments[i];
    final Float64List velocity = velocities[i];
    for (int j = 0; j < parameter.length; j++) {
      final double gradient = parameter.grads[j];
      moment[j] = beta1 * moment[j] + (1 - beta1) * gradient;
      velocity[j] = beta2 * velocity[j] + (1 - beta2) * gradient * gradient;
      final double update = (moment[j] / correction1) /
          (math.sqrt(velocity[j] / correction2) + epsilon);
      parameter.values[j] -=
          learningRate * (update + weightDecay * parameter.values[j]);
    }
  }
}

void _train(
  TinyLm model,
  List<int> tokens, {
  required int steps,
  required int seed,
  int windowLength = 8,
  double startLearningRate = 3e-3,
}) {
  final math.Random random = math.Random(seed);
  final List<ParamRef> parameters = model.parameters;
  final List<Float64List> moments = <Float64List>[
    for (final ParamRef parameter in parameters) Float64List(parameter.length),
  ];
  final List<Float64List> velocities = <Float64List>[
    for (final ParamRef parameter in parameters) Float64List(parameter.length),
  ];
  double learningRate = startLearningRate;
  for (int step = 0; step < steps; step++) {
    final List<int> window = _sampleWindow(random, tokens, windowLength);
    model.zeroGrad();
    model.accumulateWindow(window);
    _adamwStep(model, learningRate, moments, velocities, step + 1);
    learningRate *= 0.999;
  }
}

void main() {
  test('shape matches the configuration', () {
    final TinyLmConfig config = _config();
    config.validate();
    final TinyLm model = TinyLm(config);
    expect(model.totalParameterCount, config.estimatedParameters);
    expect(model.frozenParameterCount, 0);
    expect(
      model.totalParameterCount,
      model.frozenParameterCount + model.trainableParameterCount,
    );
    expect(model.hasAdapter, isFalse);
    expect(model.adapterRank, 0);

    final List<double> logits = model.logitsFor(<int>[3]);
    expect(logits.length, config.vocabSize);
    for (final double value in logits) {
      expect(value.isFinite, isTrue);
    }

    // The test preset must itself be constructible; its vocabSize is below the
    // floor the validator enforces, so this documents the source inconsistency.
    expect(() => TinyLmConfig.testPreset.validate(), returnsNormally);
  });

  test('invalid configurations are rejected', () {
    expect(
      () => TinyLm(TinyLmConfig.testPreset.copyWith(headDim: 7)),
      throwsArgumentError,
    );
    expect(
      () => TinyLm(TinyLmConfig.testPreset.copyWith(topK: 0)),
      throwsArgumentError,
    );
    expect(
      () => TinyLm(TinyLmConfig.testPreset.copyWith(topK: 5)),
      throwsArgumentError,
    );
    expect(
      () => TinyLm(TinyLmConfig.testPreset.copyWith(kvRank: 999)),
      throwsArgumentError,
    );
    expect(
      () => TinyLm(TinyLmConfig.testPreset.copyWith(vocabSize: 100)),
      throwsArgumentError,
    );
  });

  test('incremental decoding agrees with the full forward pass', () {
    final TinyLm model = TinyLm(_config());
    const List<int> tokens = <int>[
      3,
      14,
      15,
      9,
      2,
      7,
      8,
      1,
      4,
      5,
      6,
      10,
      11,
      12,
    ];
    for (int length = 1; length <= 12; length++) {
      final List<int> prefix = tokens.sublist(0, length);
      final List<double> first = model.logitsFor(prefix);
      final List<double> second = model.logitsFor(prefix);
      final List<double> reference = model.logitsFullForward(prefix);
      expect(reference.length, model.vocabSize);
      for (int index = 0; index < reference.length; index++) {
        expect(
          first[index],
          closeTo(reference[index], 1e-9),
          reason: 'incremental length $length index $index (cold call)',
        );
        expect(
          second[index],
          closeTo(reference[index], 1e-9),
          reason: 'incremental length $length index $index (warm call)',
        );
      }
    }
    model.resetCache();
    final List<double> replay = model.logitsFor(tokens.sublist(0, 1));
    final List<double> reference =
        model.logitsFullForward(tokens.sublist(0, 1));
    for (int index = 0; index < reference.length; index++) {
      expect(
        replay[index],
        closeTo(reference[index], 1e-9),
        reason: 'post-reset single-token prefix index $index',
      );
    }
  });

  test('causality: a later token cannot affect earlier positions', () {
    final TinyLm model = TinyLm(_config());
    const List<int> original = <int>[3, 14, 15, 9, 2, 7, 8, 1, 4];
    final List<int> changed = List<int>.of(original)..last = 11;
    expect(
      changed.sublist(0, original.length - 1),
      original.sublist(0, original.length - 1),
    );
    for (int length = 1; length < original.length; length++) {
      final List<double> a =
          model.logitsFullForward(original.sublist(0, length));
      final List<double> b =
          model.logitsFullForward(changed.sublist(0, length));
      for (int index = 0; index < a.length; index++) {
        expect(
          a[index],
          closeTo(b[index], 1e-12),
          reason: 'position $length index $index changed when only a later '
              'token differed',
        );
      }
    }
  });

  test('numerical gradient check', () {
    final TinyLm model = TinyLm(_config());
    const List<int> window = <int>[5, 12, 3, 27, 8, 41, 6, 19, 2];
    model.zeroGrad();
    model.accumulateWindow(window);

    const List<String> names = <String>[
      'embedding',
      'layer0.attn.q.weight',
      'layer0.attn.out.weight',
      'layer0.moe.router.weight',
      'layer0.moe.e0.up.weight',
      'layer0.norm1.gamma',
      'head.weight',
      'layer1.norm2.beta',
    ];
    final math.Random random = math.Random(20240917);
    int pairs = 0;
    double worstError = 0;
    String worstErrorDetail = 'none';
    double worstViolation = double.negativeInfinity;
    String worstViolationDetail = 'none';
    bool sawSignal = false;

    for (final String name in names) {
      final ParamRef parameter =
          model.parameters.firstWhere((ParamRef ref) => ref.name == name);
      for (int sample = 0; sample < 8; sample++) {
        final int index = random.nextInt(parameter.length);
        final double original = parameter.values[index];
        parameter.values[index] = original + 1e-5;
        final double plus = model.evaluateWindow(window);
        parameter.values[index] = original - 1e-5;
        final double minus = model.evaluateWindow(window);
        parameter.values[index] = original;

        final double numeric = (plus - minus) / (2 * 1e-5);
        final double analytic = parameter.grads[index];
        final double error = (analytic - numeric).abs();
        final double tolerance = 1e-6 + 1e-4 * numeric.abs();
        final double violation = error - tolerance;
        pairs++;
        if (numeric.abs() > 1e-8) {
          sawSignal = true;
        }
        if (error > worstError) {
          worstError = error;
          worstErrorDetail =
              '$name[$index] analytic=$analytic numeric=$numeric';
        }
        if (violation > worstViolation) {
          worstViolation = violation;
          worstViolationDetail = '$name[$index] analytic=$analytic '
              'numeric=$numeric error=$error tolerance=$tolerance';
        }
      }
    }

    printOnFailure(
      'worst absolute gradient error: $worstError at $worstErrorDetail',
    );
    expect(pairs, greaterThanOrEqualTo(40));
    expect(names.length, greaterThanOrEqualTo(6));
    expect(
      sawSignal,
      isTrue,
      reason: 'no sampled perturbation changed evaluateWindow',
    );
    expect(
      worstViolation,
      lessThanOrEqualTo(0),
      reason: 'worst gradient mismatch: $worstViolationDetail',
    );
  });

  test('end-to-end learning reduces held-out loss', () {
    final ByteTokenizer tokenizer = _tokenizer();
    final TinyLm model = TinyLm(_config(vocabSize: tokenizer.vocabSize));
    final List<int> tokens = tokenizer.encode(_corpus);
    final List<int> heldOut = tokens.sublist(0, 20);
    final double before = model.evaluateWindow(heldOut);
    _train(model, tokens, steps: 150, seed: 7);
    final double after = model.evaluateWindow(heldOut);
    printOnFailure(
      'held-out loss before=$before after=$after ratio=${after / before}',
    );
    expect(after, lessThan(before * 0.7));
  });

  test('training changes generation and is reproducible', () {
    final ByteTokenizer tokenizer = _tokenizer();
    final TinyLm model = TinyLm(_config(vocabSize: tokenizer.vocabSize));
    final List<int> tokens = tokenizer.encode(_corpus);
    final List<int> prompt = tokenizer.encode('the harbor', addBos: true);
    final Sampler sampler = Sampler(temperature: 0);

    model.resetCache();
    final List<int> before =
        model.generate(prompt, maxNewTokens: 6, sampler: sampler);
    _train(model, tokens, steps: 150, seed: 7);
    model.resetCache();
    final List<int> after =
        model.generate(prompt, maxNewTokens: 6, sampler: sampler);
    model.resetCache();
    final List<int> again =
        model.generate(prompt, maxNewTokens: 6, sampler: sampler);

    expect(after, isNot(equals(before)));
    expect(again, after);
  });

  test('LoRA is applied at inference and merging is exact', () {
    final TinyLm model = TinyLm(_config());
    final int trainableBefore = model.trainableParameterCount;
    const List<int> prefix = <int>[4, 9];

    model.attachAdapter(4, seed: 7);
    expect(model.hasAdapter, isTrue);
    expect(model.adapterRank, 4);
    expect(model.adapterParameterCount, greaterThan(0));
    expect(model.frozenParameterCount, greaterThan(0));
    expect(
      model.trainableParameterCount,
      lessThan(model.totalParameterCount),
      reason: 'the frozen base weights must not count as trainable',
    );

    model.resetCache();
    final List<double> adapted = model.logitsFor(prefix);

    model.mergeAdapter();
    expect(model.hasAdapter, isFalse);
    model.resetCache();
    final List<double> merged = model.logitsFor(prefix);
    for (int index = 0; index < adapted.length; index++) {
      expect(merged[index], closeTo(adapted[index], 1e-9));
    }
    expect(model.trainableParameterCount, trainableBefore);
  });

  test('adapter training improves a held-out window', () {
    final ByteTokenizer tokenizer = _tokenizer();
    final TinyLm base = TinyLm(_config(vocabSize: tokenizer.vocabSize));
    final List<int> tokens = tokenizer.encode(_corpus);
    final List<int> heldOut = tokens.sublist(0, 20);

    _train(base, tokens, steps: 150, seed: 11);
    final String checkpoint = base.encode();
    final double baseLoss = base.evaluateWindow(heldOut);

    final TinyLm clone = TinyLm.decode(checkpoint);
    clone.attachAdapter(4, seed: 7);
    _train(clone, tokens, steps: 120, seed: 23);
    final double cloneLoss = clone.evaluateWindow(heldOut);

    final TinyLm redecoded = TinyLm.decode(checkpoint);
    final double redecodedLoss = redecoded.evaluateWindow(heldOut);
    printOnFailure(
      'base=$baseLoss adapter=$cloneLoss redecoded=$redecodedLoss',
    );
    expect(cloneLoss, lessThan(baseLoss * 0.9));
    expect(redecodedLoss, closeTo(baseLoss, 1e-9));
  });

  test('checkpoint round-trip and validation', () {
    final TinyLm model = TinyLm(_config());
    final String checkpoint = model.encode();
    final TinyLm restored = TinyLm.decode(checkpoint);
    expect(restored.totalParameterCount, model.config.estimatedParameters);
    expect(restored.config.summary, model.config.summary);

    const List<int> window = <int>[4, 9];
    expect(
      restored.evaluateWindow(window),
      closeTo(model.evaluateWindow(window), 1e-9),
    );

    final TinyLm other = TinyLm(
      TinyLmConfig.testPreset.copyWith(vocabSize: 260, dModel: 32),
    );
    expect(() => other.loadJson(model.toJson()), throwsFormatException);
    expect(() => TinyLm.decode('{}'), throwsFormatException);

    final TinyLm adapted = TinyLm(_config())..attachAdapter(4, seed: 3);
    final TinyLm decodedAdapted = TinyLm.decode(adapted.encode());
    expect(decodedAdapted.hasAdapter, isTrue);
    expect(decodedAdapted.adapterRank, 4);
  });

  test('generate respects its bounds', () {
    final TinyLm model = TinyLm(_config());
    final Sampler sampler = Sampler(temperature: 0);
    const List<int> prompt = <int>[3];

    expect(model.generate(prompt, maxNewTokens: 0, sampler: sampler), isEmpty);

    final List<double> logits = model.logitsFor(prompt);
    final int greedy = sampler.pick(logits);
    expect(
      model.generate(
        prompt,
        maxNewTokens: 5,
        sampler: sampler,
        stopToken: greedy,
      ),
      isEmpty,
    );

    final List<int> produced =
        model.generate(prompt, maxNewTokens: 2, sampler: sampler);
    final List<int> callbackTokens = <int>[];
    final List<int> counted = model.generate(
      prompt,
      maxNewTokens: 2,
      sampler: sampler,
      onToken: callbackTokens.add,
    );
    expect(counted, produced);
    expect(callbackTokens, counted);
    expect(counted.length, lessThanOrEqualTo(2));

    final List<int> fullPrompt = List<int>.filled(model.contextLength, 3);
    expect(
      model.generate(fullPrompt, maxNewTokens: 5, sampler: sampler),
      isEmpty,
    );
  });
}
