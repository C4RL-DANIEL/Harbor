// A live, observable training run.
//
// This is the piece that turns "the model can compute gradients" into "the app
// can show a user their model getting better". It exists as a session object
// rather than a simple loop because three consumers need three different views
// of the same run:
//
//  * the on-device UI needs a stream of progress events to draw a loss curve;
//  * the Android notification needs a cheap, throttled text label;
//  * the watchdog needs `pause`/`cancel` to take effect at a step boundary so a
//    round can stop when the device unplugs or the user picks up the phone.
//
// Every step yields to the event loop. A training loop that never awaits would
// block the Flutter isolate, freeze the UI, and make the notification look like
// a hang — which is exactly the failure mode users report as "it froze my
// phone".

import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;

import '../model/trainable_model.dart';
import 'adamw.dart';

/// Where a [TrainingSession] currently is.
enum TrainingPhase {
  /// Constructed but not started.
  idle,

  /// Tokenising and validating the corpus.
  preparing,

  /// Running optimiser steps.
  training,

  /// Pausing to serialise the model.
  checkpointing,

  /// Stopped, waiting to be resumed.
  paused,

  /// Finished the requested number of steps.
  completed,

  /// Stopped by the user or the scheduler.
  cancelled,

  /// Stopped by an error; the session is no longer usable for this run.
  failed,
}

/// One immutable progress snapshot.
class TrainingProgress {
  /// Creates a snapshot.
  const TrainingProgress({
    required this.phase,
    required this.step,
    required this.totalSteps,
    required this.tokensProcessed,
    required this.loss,
    required this.smoothedLoss,
    required this.accuracy,
    required this.learningRate,
    required this.gradientNorm,
    required this.tokensPerSecond,
    required this.elapsed,
    required this.at,
    this.eta,
    this.message,
  });

  /// The phase this snapshot describes.
  final TrainingPhase phase;

  /// Zero-based step index the snapshot was taken at.
  final int step;

  /// Steps requested by the configuration.
  final int totalSteps;

  /// Training tokens consumed so far.
  final int tokensProcessed;

  /// Mean loss of the most recent step.
  final double loss;

  /// Exponential moving average of the loss.
  ///
  /// The raw per-step loss is extremely noisy at small batch sizes; the UI
  /// draws this so a beginner sees a trend rather than a heartbeat.
  final double smoothedLoss;

  /// Fraction of the batch's next-token predictions that were correct.
  final double accuracy;

  /// Learning rate used for the most recent step.
  final double learningRate;

  /// Global gradient norm measured before clipping.
  final double gradientNorm;

  /// Throughput of the most recent step.
  final double tokensPerSecond;

  /// Wall-clock time since [TrainingSession.start].
  final Duration elapsed;

  /// Estimated time remaining, once a rate is known.
  final Duration? eta;

  /// Human explanation for non-training phases.
  final String? message;

  /// When the snapshot was taken.
  final DateTime at;

  /// Perplexity, the exponential of the loss.
  ///
  /// Clamped at `exp(30)` because an untrained model's first loss can be large
  /// enough that `exp` overflows to infinity, and an infinite perplexity in a
  /// UI is worse than a large finite one.
  double get perplexity => math.exp(math.min(smoothedLoss, 30.0));

  /// Completion fraction in `0..1`.
  double get fraction {
    if (totalSteps <= 0) {
      return 0;
    }
    final double value = step / totalSteps;
    return value.clamp(0.0, 1.0);
  }

  /// Percent complete, for a progress bar label.
  int get percent => (fraction * 100).round();

  /// A single line suitable for a notification body.
  ///
  /// Deliberately short: Android truncates a notification's second line at
  /// roughly forty characters, and a truncated number is worse than no number.
  String get label {
    switch (phase) {
      case TrainingPhase.idle:
        return 'Ready to train';
      case TrainingPhase.preparing:
        return 'Preparing training data';
      case TrainingPhase.training:
        return 'Step $step/$totalSteps · loss '
            '${smoothedLoss.toStringAsFixed(2)} · $percent%';
      case TrainingPhase.checkpointing:
        return 'Saving progress at step $step';
      case TrainingPhase.paused:
        return 'Paused at step $step · loss '
            '${smoothedLoss.toStringAsFixed(2)}';
      case TrainingPhase.completed:
        return 'Training complete · loss ${smoothedLoss.toStringAsFixed(2)}';
      case TrainingPhase.cancelled:
        return 'Training stopped at step $step';
      case TrainingPhase.failed:
        return 'Training failed: ${message ?? 'unknown error'}';
    }
  }

  /// JSON form, appended to the training log.
  Map<String, Object?> toJson() => <String, Object?>{
        'phase': phase.name,
        'step': step,
        'total_steps': totalSteps,
        'tokens_processed': tokensProcessed,
        'loss': loss,
        'smoothed_loss': smoothedLoss,
        'accuracy': accuracy,
        'learning_rate': learningRate,
        'gradient_norm': gradientNorm,
        'tokens_per_second': tokensPerSecond,
        'elapsed_ms': elapsed.inMilliseconds,
        'at': at.toUtc().toIso8601String(),
        if (message != null) 'message': message,
      };

  @override
  String toString() => 'TrainingProgress(${phase.name}, $label)';
}

/// Knobs for a training run.
class TrainingConfig {
  /// Creates a configuration; every field has a usable default.
  const TrainingConfig({
    this.totalSteps = 300,
    this.batchSize = 4,
    this.windowLength = 64,
    this.warmupSteps = 20,
    this.checkpointEvery = 100,
    this.seed = 0x48415242,
    this.learningRate = 3e-3,
    this.minLearningRate = 3e-4,
    this.weightDecay = 0.01,
    this.maxGradientNorm = 1.0,
    this.cosineSchedule = true,
  });

  /// Number of optimiser steps to run.
  final int totalSteps;

  /// Windows accumulated per step.
  final int batchSize;

  /// Tokens per training window, including the single prediction target.
  final int windowLength;

  /// Steps of linear learning-rate warmup.
  final int warmupSteps;

  /// Steps between checkpoints.
  final int checkpointEvery;

  /// Seed for the window sampler, making a run reproducible.
  final int seed;

  /// Peak learning rate.
  final double learningRate;

  /// Floor of the cosine decay.
  final double minLearningRate;

  /// Decoupled weight decay.
  final double weightDecay;

  /// Gradient-norm clip threshold.
  final double maxGradientNorm;

  /// Whether to decay the learning rate after warmup.
  final bool cosineSchedule;

  /// Throws [ArgumentError] when the configuration cannot run.
  void validate() {
    if (totalSteps < 1) {
      throw ArgumentError.value(totalSteps, 'totalSteps', 'must be at least 1');
    }
    if (batchSize < 1) {
      throw ArgumentError.value(batchSize, 'batchSize', 'must be at least 1');
    }
    if (windowLength < 2) {
      throw ArgumentError.value(
        windowLength,
        'windowLength',
        'must be at least 2: a window needs an input and a target',
      );
    }
    if (warmupSteps < 0) {
      throw ArgumentError.value(warmupSteps, 'warmupSteps', 'must not be negative');
    }
    if (checkpointEvery < 1) {
      throw ArgumentError.value(
        checkpointEvery,
        'checkpointEvery',
        'must be at least 1',
      );
    }
    if (learningRate <= 0 || minLearningRate < 0) {
      throw ArgumentError('learningRate must be positive and minLearningRate >= 0');
    }
    if (minLearningRate > learningRate) {
      throw ArgumentError(
        'minLearningRate must not exceed learningRate; the schedule would '
        'increase the rate over time',
      );
    }
    if (maxGradientNorm <= 0) {
      throw ArgumentError.value(
        maxGradientNorm,
        'maxGradientNorm',
        'must be positive',
      );
    }
  }

  /// The learning rate for [step], applying warmup then cosine decay.
  double learningRateAt(int step) {
    if (warmupSteps > 0 && step < warmupSteps) {
      return learningRate * (step + 1) / warmupSteps;
    }
    if (!cosineSchedule) {
      return learningRate;
    }
    final int decaySteps = math.max(1, totalSteps - warmupSteps);
    final double progress = ((step - warmupSteps) / decaySteps).clamp(0.0, 1.0);
    return minLearningRate +
        0.5 * (learningRate - minLearningRate) * (1 + math.cos(math.pi * progress));
  }

  /// JSON form.
  Map<String, Object?> toJson() => <String, Object?>{
        'total_steps': totalSteps,
        'batch_size': batchSize,
        'window_length': windowLength,
        'warmup_steps': warmupSteps,
        'checkpoint_every': checkpointEvery,
        'seed': seed,
        'learning_rate': learningRate,
        'min_learning_rate': minLearningRate,
        'weight_decay': weightDecay,
        'max_gradient_norm': maxGradientNorm,
        'cosine_schedule': cosineSchedule,
      };

  /// Parses [toJson]; tolerant of a missing optional field but strict about
  /// types, so a corrupted config is reported rather than silently defaulted.
  static TrainingConfig fromJson(Map<String, Object?> json) {
    int intAt(String key, int fallback) {
      final Object? value = json[key];
      if (value == null) {
        return fallback;
      }
      if (value is! int) {
        throw FormatException('training config field "$key" must be an integer');
      }
      return value;
    }

    double doubleAt(String key, double fallback) {
      final Object? value = json[key];
      if (value == null) {
        return fallback;
      }
      if (value is! num) {
        throw FormatException('training config field "$key" must be a number');
      }
      return value.toDouble();
    }

    bool boolAt(String key, bool fallback) {
      final Object? value = json[key];
      if (value == null) {
        return fallback;
      }
      if (value is! bool) {
        throw FormatException('training config field "$key" must be a boolean');
      }
      return value;
    }

    return TrainingConfig(
      totalSteps: intAt('total_steps', 300),
      batchSize: intAt('batch_size', 4),
      windowLength: intAt('window_length', 64),
      warmupSteps: intAt('warmup_steps', 20),
      checkpointEvery: intAt('checkpoint_every', 100),
      seed: intAt('seed', 0x48415242),
      learningRate: doubleAt('learning_rate', 3e-3),
      minLearningRate: doubleAt('min_learning_rate', 3e-4),
      weightDecay: doubleAt('weight_decay', 0.01),
      maxGradientNorm: doubleAt('max_gradient_norm', 1.0),
      cosineSchedule: boolAt('cosine_schedule', true),
    );
  }

  /// A copy with selected fields replaced.
  TrainingConfig copyWith({
    int? totalSteps,
    int? batchSize,
    int? windowLength,
    int? warmupSteps,
    int? checkpointEvery,
    int? seed,
    double? learningRate,
    double? minLearningRate,
    double? weightDecay,
    double? maxGradientNorm,
    bool? cosineSchedule,
  }) {
    return TrainingConfig(
      totalSteps: totalSteps ?? this.totalSteps,
      batchSize: batchSize ?? this.batchSize,
      windowLength: windowLength ?? this.windowLength,
      warmupSteps: warmupSteps ?? this.warmupSteps,
      checkpointEvery: checkpointEvery ?? this.checkpointEvery,
      seed: seed ?? this.seed,
      learningRate: learningRate ?? this.learningRate,
      minLearningRate: minLearningRate ?? this.minLearningRate,
      weightDecay: weightDecay ?? this.weightDecay,
      maxGradientNorm: maxGradientNorm ?? this.maxGradientNorm,
      cosineSchedule: cosineSchedule ?? this.cosineSchedule,
    );
  }
}

/// A resumable, observable training run over a token stream.
class TrainingSession {
  /// Creates a session.
  ///
  /// [tokens] is the whole training stream; windows are sampled from it. The
  /// stream is held by reference and never mutated.
  ///
  /// [onCheckpoint] receives the serialised model at each checkpoint, so the
  /// caller decides where a checkpoint goes. A session cannot know whether it is
  /// running on a phone, in a browser or in a test.
  TrainingSession({
    required this.model,
    required List<int> tokens,
    this.config = const TrainingConfig(),
    void Function(String checkpointJson)? onCheckpoint,
  })  : tokens = List<int>.unmodifiable(tokens),
        _onCheckpoint = onCheckpoint;

  /// The model being trained.
  final TrainableLanguageModel model;

  /// The training stream.
  final List<int> tokens;

  /// The run's knobs.
  final TrainingConfig config;

  final void Function(String checkpointJson)? _onCheckpoint;
  // `sync: true` is load-bearing, not a micro-optimisation. With the default
  // asynchronous controller the terminal `completed` event is still queued when
  // `start()` returns, so a caller that does `await session.start()` and then
  // stops listening — which is the obvious way to write it — never sees the
  // final event and therefore never performs its end-of-run work. Here that work
  // is saving the checkpoint and appending the training log, so the failure mode
  // is silent data loss: the run reports success and nothing is persisted.
  //
  // Delivering events synchronously makes "the future completed" and "every
  // event was delivered" the same fact.
  final StreamController<TrainingProgress> _events =
      StreamController<TrainingProgress>.broadcast(sync: true);

  AdamW? _optimizer;
  TrainingProgress _latest = TrainingProgress(
    phase: TrainingPhase.idle,
    step: 0,
    totalSteps: 0,
    tokensProcessed: 0,
    loss: 0,
    smoothedLoss: 0,
    accuracy: 0,
    learningRate: 0,
    gradientNorm: 0,
    tokensPerSecond: 0,
    elapsed: Duration.zero,
    at: DateTime.fromMillisecondsSinceEpoch(0, isUtc: true),
  );
  bool _running = false;
  bool _paused = false;
  bool _cancelled = false;
  Completer<void>? _resumeSignal;

  /// Progress events, broadcast so several widgets can listen.
  ///
  /// Broadcast rather than single-subscription because the notification bridge
  /// and the loss chart subscribe independently, and one of them attaching
  /// second must not steal the stream from the other.
  Stream<TrainingProgress> get progress => _events.stream;

  /// The most recent snapshot.
  TrainingProgress get latest => _latest;

  /// Whether a run is in progress (including while paused).
  bool get isRunning => _running;

  /// Whether the run is paused.
  bool get isPaused => _paused;

  /// The optimiser, available after [start] for diagnostics.
  AdamW? get optimizer => _optimizer;

  /// Runs to completion, or until paused-then-cancelled.
  ///
  /// Never throws: a failure is reported as a [TrainingPhase.failed] event, so
  /// a UI subscription is the single place that has to handle errors.
  Future<void> start() async {
    if (_running) {
      throw StateError('this session is already running');
    }
    if (_events.isClosed) {
      throw StateError('this session has already finished');
    }
    _running = true;
    _paused = false;
    _cancelled = false;

    late final Stopwatch stopwatch;
    try {
      config.validate();
      _emit(
        _snapshot(
          TrainingPhase.preparing,
          step: 0,
          message: 'Validating ${tokens.length} training tokens',
        ),
      );
      if (tokens.length <= config.windowLength) {
        _emit(
          _snapshot(
            TrainingPhase.failed,
            step: 0,
            message: 'the corpus produced ${tokens.length} tokens, but a window '
                'needs ${config.windowLength + 1}; gather more text',
          ),
        );
        return;
      }
      if (tokens.length < config.windowLength * 4) {
        _emit(
          _snapshot(
            TrainingPhase.preparing,
            step: 0,
            message: 'the corpus is small (${tokens.length} tokens); the model '
                'will memorise rather than generalise',
          ),
        );
      }

      _optimizer = AdamW(
        parameters: model.parameters,
        weightDecay: config.weightDecay,
      );
      final math.Random rng = math.Random(config.seed);
      final int span = tokens.length - config.windowLength;
      double smoothed = 0;
      stopwatch = Stopwatch()..start();

      for (int step = 0; step < config.totalSteps; step++) {
        if (_cancelled) {
          break;
        }
        if (_paused) {
          _emit(_snapshot(TrainingPhase.paused, step: step));
          await _waitForResume();
          if (_cancelled) {
            break;
          }
          stopwatch.start();
        }

        model.zeroGrad();
        double stepLoss = 0;
        int stepCorrect = 0;
        int stepPredicted = 0;
        for (int w = 0; w < config.batchSize; w++) {
          final int start = rng.nextInt(span);
          final List<int> window = tokens.sublist(
            start,
            start + config.windowLength + 1,
          );
          final TrainingOutcome outcome =
              model.accumulateWindow(window, lossScale: 1 / config.batchSize);
          stepLoss += outcome.loss * outcome.predicted;
          stepCorrect += outcome.correct;
          stepPredicted += outcome.predicted;
        }
        final double meanLoss =
            stepPredicted == 0 ? 0 : stepLoss / stepPredicted;
        final double accuracy =
            stepPredicted == 0 ? 0 : stepCorrect / stepPredicted;
        final double gradientNorm = _optimizer!.clipByGlobalNorm(
          config.maxGradientNorm,
        );
        final double learningRate = config.learningRateAt(step);
        _optimizer!.step(learningRate: learningRate);

        smoothed = step == 0 ? meanLoss : 0.9 * smoothed + 0.1 * meanLoss;
        final int elapsedMs = math.max(1, stopwatch.elapsedMilliseconds);
        final double tokensPerSecond =
            ((step + 1) * config.batchSize * config.windowLength) /
                (elapsedMs / 1000);
        final int remaining = config.totalSteps - (step + 1);
        _emit(
          _snapshot(
            TrainingPhase.training,
            step: step + 1,
            loss: meanLoss,
            smoothedLoss: smoothed,
            accuracy: accuracy,
            learningRate: learningRate,
            gradientNorm: gradientNorm,
            tokensPerSecond: tokensPerSecond,
            elapsed: stopwatch.elapsed,
            tokensProcessed: (step + 1) * config.batchSize * config.windowLength,
            eta: tokensPerSecond <= 0
                ? null
                : Duration(
                    milliseconds:
                        (remaining * elapsedMs / (step + 1)).round(),
                  ),
          ),
        );

        final bool isCheckpointStep =
            (step + 1) % config.checkpointEvery == 0;
        if (isCheckpointStep) {
          _emit(
            _snapshot(
              TrainingPhase.checkpointing,
              step: step + 1,
              loss: meanLoss,
              smoothedLoss: smoothed,
              tokensProcessed:
                  (step + 1) * config.batchSize * config.windowLength,
            ),
          );
          _writeCheckpoint();
        }

        // Yield so the Flutter isolate can pump frames and service the
        // notification. Without this the UI is frozen for the whole run.
        await Future<void>.delayed(Duration.zero);
      }

      stopwatch.stop();
      if (_cancelled) {
        _writeCheckpoint();
        _emit(
          _snapshot(
            TrainingPhase.cancelled,
            step: _latest.step,
            message: 'stopped; progress was checkpointed',
          ),
        );
        return;
      }

      _writeCheckpoint();
      _emit(
        _snapshot(
          TrainingPhase.completed,
          step: config.totalSteps,
          smoothedLoss: smoothed,
          tokensProcessed:
              config.totalSteps * config.batchSize * config.windowLength,
          elapsed: stopwatch.elapsed,
        ),
      );
    } on Object catch (error) {
      _emit(
        _snapshot(
          TrainingPhase.failed,
          step: _latest.step,
          message: '$error',
        ),
      );
    } finally {
      _running = false;
      _paused = false;
    }
  }

  /// Requests a pause at the next step boundary.
  void pause() {
    if (!_running || _paused) {
      return;
    }
    _paused = true;
    _resumeSignal = Completer<void>();
  }

  /// Resumes a paused run.
  void resume() {
    if (!_paused) {
      return;
    }
    _paused = false;
    final Completer<void>? signal = _resumeSignal;
    _resumeSignal = null;
    if (signal != null && !signal.isCompleted) {
      signal.complete();
    }
  }

  /// Requests cancellation; the run ends after the current step.
  Future<void> cancel() async {
    _cancelled = true;
    resume();
    // Wait for the loop to notice, bounded so a caller can never hang.
    for (int i = 0; i < 400 && _running; i++) {
      await Future<void>.delayed(const Duration(milliseconds: 5));
    }
  }

  /// Closes the event stream. The session cannot be reused afterwards.
  Future<void> dispose() async {
    _cancelled = true;
    resume();
    await _events.close();
  }

  Future<void> _waitForResume() async {
    final Completer<void>? signal = _resumeSignal;
    if (signal != null) {
      await signal.future;
      if (_running) {
        _paused = false;
      }
    }
  }

  void _writeCheckpoint() {
    final void Function(String)? callback = _onCheckpoint;
    if (callback == null) {
      return;
    }
    callback(jsonEncode(model.toJson()));
  }

  TrainingProgress _snapshot(
    TrainingPhase phase, {
    required int step,
    double? loss,
    double? smoothedLoss,
    double? accuracy,
    double? learningRate,
    double? gradientNorm,
    double? tokensPerSecond,
    int? tokensProcessed,
    Duration? elapsed,
    Duration? eta,
    String? message,
  }) {
    return TrainingProgress(
      phase: phase,
      step: step,
      totalSteps: config.totalSteps,
      tokensProcessed: tokensProcessed ?? _latest.tokensProcessed,
      loss: loss ?? _latest.loss,
      smoothedLoss: smoothedLoss ?? _latest.smoothedLoss,
      accuracy: accuracy ?? _latest.accuracy,
      learningRate: learningRate ?? _latest.learningRate,
      gradientNorm: gradientNorm ?? _latest.gradientNorm,
      tokensPerSecond: tokensPerSecond ?? _latest.tokensPerSecond,
      elapsed: elapsed ?? _latest.elapsed,
      eta: eta,
      message: message,
      at: DateTime.now().toUtc(),
    );
  }

  void _emit(TrainingProgress progress) {
    _latest = progress;
    if (!_events.isClosed) {
      _events.add(progress);
    }
  }
}
