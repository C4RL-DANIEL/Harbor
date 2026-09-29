// Tests for the training loop: the AdamW optimiser, the learning-rate schedule
// and validation in TrainingConfig, and the observable lifecycle of a
// TrainingSession.
//
// The package is Flutter-free and dart:io-free, so this suite runs under the
// plain Dart VM with `dart test`.

import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:harbor_core/harbor_core.dart';
import 'package:test/test.dart';

/// A token id sequence whose ids all lie inside the byte vocabulary.
List<int> _tokens(int length) =>
    List<int>.generate(length, (int i) => i % 260);

/// A tiny model that still exercises the real parameter list.
TinyLm _model() => TinyLm(TinyLmConfig.testPreset);

/// Two parameters whose gradients ([3, 4] and [12]) have a global norm of
/// `sqrt(9 + 16 + 144) = 13`, a value small enough to check by hand.
List<ParamRef> _normParameters() => <ParamRef>[
      ParamRef(
        'first',
        Float64List.fromList(<double>[0, 0]),
        Float64List.fromList(<double>[3, 4]),
      ),
      ParamRef(
        'second',
        Float64List.fromList(<double>[0]),
        Float64List.fromList(<double>[12]),
      ),
    ];

/// Asserts that [config] cannot be run.
void _expectInvalid(TrainingConfig config) {
  expect(config.validate, throwsArgumentError);
}

void main() {
  group('AdamW', () {
    test('globalGradientNorm matches the hand-computed norm', () {
      final AdamW optimizer = AdamW(parameters: _normParameters());
      expect(optimizer.globalGradientNorm(), closeTo(13, 1e-12));
    });

    test('clipByGlobalNorm returns the pre-clip norm and scales under the cap',
        () {
      final List<ParamRef> parameters = _normParameters();
      final AdamW optimizer = AdamW(parameters: parameters);
      final double preClip = optimizer.clipByGlobalNorm(5);
      expect(
        preClip,
        closeTo(13, 1e-12),
        reason: 'the return value is the norm measured before scaling',
      );
      expect(optimizer.globalGradientNorm(), closeTo(5, 1e-12));
      expect(optimizer.globalGradientNorm(), lessThanOrEqualTo(5 + 1e-12));
    });

    test('clipByGlobalNorm is a no-op when already below the cap', () {
      final List<ParamRef> parameters = _normParameters();
      final AdamW optimizer = AdamW(parameters: parameters);
      final double returned = optimizer.clipByGlobalNorm(20);
      expect(returned, closeTo(13, 1e-12));
      expect(parameters[0].grads[0], 3);
      expect(parameters[0].grads[1], 4);
      expect(parameters[1].grads[0], 12);
    });

    test('step applies decoupled weight decay to a zero-gradient parameter',
        () {
      final ParamRef weight = ParamRef(
        'weight',
        Float64List.fromList(<double>[1]),
        Float64List.fromList(<double>[0]),
      );
      final AdamW optimizer = AdamW(
        parameters: <ParamRef>[weight],
        weightDecay: 0.1,
      );
      final double preUpdateNorm = optimizer.step(learningRate: 0.01);
      expect(preUpdateNorm, 0, reason: 'the only gradient is zero');
      expect(
        weight.values[0],
        lessThan(1.0),
        reason: 'decoupled weight decay shrinks a parameter even with no '
            'gradient, which is what "decoupled" means',
      );
      expect(weight.values[0], closeTo(1 - 0.01 * 0.1, 1e-12));
    });

    test('step returns the pre-update gradient norm', () {
      final AdamW optimizer = AdamW(
        parameters: _normParameters(),
        weightDecay: 0,
      );
      expect(optimizer.step(learningRate: 0.01), closeTo(13, 1e-12));
    });

    test('stepCount increments once per step', () {
      final AdamW optimizer = AdamW(
        parameters: _normParameters(),
        weightDecay: 0,
      );
      expect(optimizer.stepCount, 0);
      optimizer.step(learningRate: 0.01);
      optimizer.step(learningRate: 0.01);
      optimizer.step(learningRate: 0.01);
      expect(optimizer.stepCount, 3);
    });

    test('zeroGrad clears every gradient', () {
      final List<ParamRef> parameters = _normParameters();
      final AdamW optimizer = AdamW(parameters: parameters);
      optimizer.zeroGrad();
      for (final ParamRef parameter in parameters) {
        for (int i = 0; i < parameter.grads.length; i++) {
          expect(parameter.grads[i], 0);
        }
      }
      expect(optimizer.globalGradientNorm(), 0);
    });

    test('stats reports the step count and the current gradient norm', () {
      final ParamRef parameter = ParamRef(
        'p',
        Float64List.fromList(<double>[0]),
        Float64List.fromList(<double>[3]),
      );
      final AdamW optimizer = AdamW(
        parameters: <ParamRef>[parameter],
        weightDecay: 0,
      );
      optimizer.step(learningRate: 0.01);
      final Map<String, Object?> stats = optimizer.stats();
      expect(stats['step'], 1);
      // There is no stored "last norm" entry: the real key is `gradient_norm`,
      // which is the live global gradient norm.
      expect(stats['gradient_norm'], closeTo(3, 1e-12));
    });
  });

  group('TrainingConfig.validate', () {
    test('rejects non-positive totalSteps', () {
      _expectInvalid(const TrainingConfig(totalSteps: 0));
    });

    test('rejects non-positive batchSize', () {
      _expectInvalid(const TrainingConfig(batchSize: 0));
    });

    test('rejects windowLength below 2', () {
      _expectInvalid(const TrainingConfig(windowLength: 1));
    });

    test('rejects non-positive checkpointEvery', () {
      _expectInvalid(const TrainingConfig(checkpointEvery: 0));
    });

    test('rejects a negative learningRate', () {
      _expectInvalid(const TrainingConfig(learningRate: -0.1));
    });

    test('rejects minLearningRate above learningRate', () {
      _expectInvalid(
        const TrainingConfig(learningRate: 0.001, minLearningRate: 0.01),
      );
    });

    test('rejects a non-positive maxGradientNorm', () {
      _expectInvalid(const TrainingConfig(maxGradientNorm: 0));
    });

    test('rejects warmupSteps > totalSteps', () {
      const TrainingConfig config =
          TrainingConfig(totalSteps: 4, warmupSteps: 5);
      // A warmup longer than the run never reaches the peak learning rate and
      // then decays from a value it never used, so the schedule it describes is
      // not the schedule anyone asked for. This used to be accepted silently.
      expect(
        config.validate,
        throwsArgumentError,
        reason: 'a warmup longer than the run must be rejected',
      );
    });
  });

  group('TrainingConfig.learningRateAt', () {
    const TrainingConfig cosine = TrainingConfig(
      totalSteps: 100,
      warmupSteps: 10,
    );
    const TrainingConfig flat = TrainingConfig(
      totalSteps: 100,
      warmupSteps: 10,
      cosineSchedule: false,
    );

    test('cosine: step 0 is at most learningRate and warmup ends at it', () {
      expect(
        cosine.learningRateAt(0),
        lessThanOrEqualTo(cosine.learningRate),
        reason: 'warmup must never overshoot the peak rate',
      );
      expect(
        cosine.learningRateAt(cosine.warmupSteps - 1),
        closeTo(cosine.learningRate, 1e-12),
      );
    });

    test('cosine: decays monotonically and never drops below the floor', () {
      double previous = cosine.learningRateAt(cosine.warmupSteps);
      for (int step = cosine.warmupSteps; step <= cosine.totalSteps; step++) {
        final double rate = cosine.learningRateAt(step);
        expect(
          rate,
          lessThanOrEqualTo(previous + 1e-12),
          reason: 'the cosine schedule must not increase after warmup',
        );
        expect(
          rate,
          greaterThanOrEqualTo(cosine.minLearningRate - 1e-12),
          reason: 'minLearningRate is the floor of the decay',
        );
        expect(rate, lessThanOrEqualTo(cosine.learningRate + 1e-12));
        previous = rate;
      }
    });

    test('non-cosine: constant at learningRate after warmup', () {
      for (int step = flat.warmupSteps; step <= flat.totalSteps; step++) {
        expect(flat.learningRateAt(step), closeTo(flat.learningRate, 1e-12));
      }
    });
  });

  group('TrainingSession ordering', () {
    test('delivers the terminal completed event before start() returns',
        () async {
      final TrainingSession session = TrainingSession(
        model: _model(),
        tokens: _tokens(256),
        config: const TrainingConfig(
          totalSteps: 4,
          batchSize: 2,
          windowLength: 8,
          warmupSteps: 1,
          checkpointEvery: 2,
        ),
      );
      final List<TrainingPhase> phases = <TrainingPhase>[];
      session.progress.listen((TrainingProgress progress) {
        phases.add(progress.phase);
      });

      // The listener is attached before start(), so nothing can have arrived
      // yet and the subscription must observe the very first `preparing` event.
      expect(
        phases,
        isEmpty,
        reason: 'no event can arrive before start() is called',
      );

      await session.start();

      // The session's progress controller is deliberately synchronous
      // (`sync: true`) precisely so this ordering holds. With an asynchronous
      // controller the terminal `completed` event would still be queued when
      // start() returns, so a caller doing `await start()` and then `cancel()`
      // (or simply letting the scope end) would never see the final event. That
      // is exactly how the bug manifested: the end-of-run checkpoint was never
      // saved even though the run reported success.
      expect(phases, isNotEmpty);
      expect(
        phases.first,
        TrainingPhase.preparing,
        reason: 'a listener attached before start() must observe the first '
            'event; merely finding completed would not prove ordering',
      );
      expect(
        phases.where((TrainingPhase phase) => phase == TrainingPhase.completed),
        hasLength(1),
        reason: 'the terminal event must be delivered exactly once',
      );
      expect(
        phases.last,
        TrainingPhase.completed,
        reason: 'the terminal event must be delivered by the time start() '
            'returns',
      );
      expect(session.latest.phase, TrainingPhase.completed);
      expect(session.isRunning, isFalse);
    });
  });

  group('TrainingSession.cancel', () {
    test('stops at the next step boundary and reports cancelled', () async {
      final TrainingSession session = TrainingSession(
        model: _model(),
        tokens: _tokens(128),
        config: const TrainingConfig(
          totalSteps: 8,
          batchSize: 1,
          windowLength: 4,
          warmupSteps: 1,
          checkpointEvery: 2,
        ),
      );
      final List<TrainingPhase> phases = <TrainingPhase>[];
      bool requested = false;
      session.progress.listen((TrainingProgress progress) {
        phases.add(progress.phase);
        if (!requested && progress.phase == TrainingPhase.training) {
          requested = true;
          // Called from a synchronous listener, so it cannot be awaited here.
          unawaited(session.cancel());
        }
      });

      await session.start();

      expect(
        requested,
        isTrue,
        reason: 'the run must have emitted at least one training event',
      );
      expect(
        phases.last,
        TrainingPhase.cancelled,
        reason: 'cancel() must be observed at the next step boundary',
      );
      expect(session.latest.phase, TrainingPhase.cancelled);
      expect(session.isRunning, isFalse);
      expect(
        phases.where((TrainingPhase phase) => phase == TrainingPhase.cancelled),
        hasLength(1),
      );
    });
  });

  group('TrainingSession pause and resume', () {
    test('pause is observable and resume lets the run complete', () async {
      final TrainingSession session = TrainingSession(
        model: _model(),
        tokens: _tokens(128),
        config: const TrainingConfig(
          totalSteps: 6,
          batchSize: 1,
          windowLength: 4,
          warmupSteps: 1,
          checkpointEvery: 3,
        ),
      );
      final List<TrainingPhase> phases = <TrainingPhase>[];
      bool pauseRequested = false;
      bool pausedSeen = false;
      session.progress.listen((TrainingProgress progress) {
        phases.add(progress.phase);
        if (!pauseRequested && progress.phase == TrainingPhase.training) {
          pauseRequested = true;
          session.pause();
          expect(
            session.isPaused,
            isTrue,
            reason: 'pause() takes effect immediately for the caller',
          );
        }
        if (progress.phase == TrainingPhase.paused) {
          pausedSeen = true;
          session.resume();
        }
      });

      await session.start();

      expect(
        pausedSeen,
        isTrue,
        reason: 'the loop must emit a paused phase and then wait for resume()',
      );
      expect(phases.last, TrainingPhase.completed);
      expect(session.isPaused, isFalse);
      expect(session.isRunning, isFalse);
    });
  });

  group('TrainingSession persistence', () {
    test('onCheckpoint receives JSON containing the model config', () async {
      final ByteTokenizer tokenizer = ByteTokenizer();
      final List<int> tokens = tokenizer.encode(
        'the harbor beacon sweeps the fog and guides lost boats home. ' * 8,
      );
      final List<String> checkpoints = <String>[];
      final TrainingSession session = TrainingSession(
        model: _model(),
        tokens: tokens,
        config: const TrainingConfig(
          totalSteps: 4,
          batchSize: 2,
          windowLength: 8,
          warmupSteps: 1,
          checkpointEvery: 2,
        ),
        onCheckpoint: checkpoints.add,
      );

      await session.start();

      expect(
        checkpoints,
        isNotEmpty,
        reason: 'checkpointEvery: 2 with totalSteps: 4 must persist at least '
            'once',
      );
      final Map<String, Object?> decoded =
          jsonDecode(checkpoints.first) as Map<String, Object?>;
      final Map<String, Object?> modelJson =
          decoded['config']! as Map<String, Object?>;
      expect(modelJson['vocab_size'], 260);
      expect(modelJson.containsKey('d_model'), isTrue);

      // TrainingSession has no `outcome`/`metrics` getter; its public API
      // exposes `latest` and `optimizer`, which carry the final metrics.
      expect(session.latest.phase, TrainingPhase.completed);
      expect(session.optimizer, isNotNull);
      expect(session.optimizer!.stepCount, 4);
      expect(session.optimizer!.stats()['step'], 4);
    });
  });

  group('TrainingSession robustness', () {
    test('a corpus shorter than a window fails cleanly without throwing',
        () async {
      final TrainingSession session = TrainingSession(
        model: _model(),
        tokens: _tokens(5),
        config: const TrainingConfig(
          totalSteps: 4,
          batchSize: 1,
          windowLength: 8,
          warmupSteps: 1,
          checkpointEvery: 2,
        ),
      );
      final List<TrainingPhase> phases = <TrainingPhase>[];
      session.progress.listen((TrainingProgress progress) {
        phases.add(progress.phase);
      });

      // start() is documented never to throw: a failure is an event.
      await session.start();

      expect(phases.last, TrainingPhase.failed);
      expect(session.latest.phase, TrainingPhase.failed);
      expect(session.latest.message, isNotNull);
      expect(session.latest.message, isNotEmpty);
      expect(session.isRunning, isFalse);
    });
  });

  group('TrainingProgress bookkeeping', () {
    test('fraction is monotone in 0..1 and perplexity stays clamped',
        () async {
      final TrainingSession session = TrainingSession(
        model: _model(),
        tokens: _tokens(128),
        config: const TrainingConfig(
          totalSteps: 5,
          batchSize: 1,
          windowLength: 4,
          warmupSteps: 1,
          checkpointEvery: 2,
        ),
      );
      final List<TrainingProgress> events = <TrainingProgress>[];
      session.progress.listen(events.add);

      await session.start();

      expect(events, isNotEmpty);
      expect(events.last.phase, TrainingPhase.completed);
      double previous = -1;
      for (final TrainingProgress event in events) {
        final double fraction = event.fraction;
        expect(fraction, inInclusiveRange(0.0, 1.0));
        expect(
          fraction,
          greaterThanOrEqualTo(previous),
          reason: 'progress must never move backwards',
        );
        previous = fraction;
        expect(event.percent, (fraction * 100).round());
        expect(event.percent, inInclusiveRange(0, 100));
        expect(
          event.perplexity.isFinite,
          isTrue,
          reason: 'perplexity must be a usable finite number',
        );
        expect(
          event.perplexity,
          lessThanOrEqualTo(math.exp(30)),
          reason: 'perplexity is clamped at exp(30)',
        );
      }
    });
  });
}