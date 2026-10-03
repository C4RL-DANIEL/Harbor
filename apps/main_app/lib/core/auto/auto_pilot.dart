// Auto-Pilot: the one-tap path from an empty phone to a working assistant.
//
// Harbor's pipeline has a real order — gather text, learn a tokenizer, build a
// model sized to that tokenizer, train it, then chat — and a first-time user
// should not have to discover that order by reading four tabs. This controller
// runs the whole chain, reports honest per-step progress, and refuses steps the
// device cannot currently afford (low battery, hot, unplugged) using exactly
// the gate the idle scheduler already uses, so "automatic" never means
// "drains your phone".
//
// Every collaborator is injected. The class owns no model, no store and no
// timer beyond its own poll, which keeps it testable with the same stubs the
// other controller tests use.

import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:harbor_core/harbor_core.dart' as hc;

import '../corpus/corpus_controller.dart';
import '../engine/on_device_lora.dart';
import '../model/harbor_model_runtime.dart';
import '../platform/platform_device_state.dart';
import '../training/training_controller.dart';

/// A stage of the automatic setup, in execution order.
enum AutoStage {
  /// Nothing running; the device may be asked to start.
  idle,

  /// Reading device text into the corpus.
  gathering,

  /// Learning the tokenizer and building the model.
  building,

  /// Running the first training round.
  training,

  /// Pipeline complete; chat is usable and self-maintained.
  ready,

  /// A step failed; [AutoPilotController.status] carries why.
  failed,
}

/// A snapshot of the automatic pipeline, for the UI to render.
class AutoPilotStatus {
  /// Creates a snapshot.
  const AutoPilotStatus({
    required this.stage,
    required this.progress,
    required this.message,
    required this.corpusChars,
    required this.memories,
    required this.modelReady,
    required this.autoLearn,
    this.error,
  });

  /// Current stage.
  final AutoStage stage;

  /// Fraction complete in `[0, 1]`, monotone within a run.
  final double progress;

  /// Human sentence describing what is happening or happened.
  final String message;

  /// Characters of training text available.
  final int corpusChars;

  /// Memories stored.
  final int memories;

  /// Whether a tokenizer+model exist.
  final bool modelReady;

  /// Whether background self-maintenance is on.
  final bool autoLearn;

  /// Failure detail when [stage] is [AutoStage.failed].
  final String? error;

  /// Whether a run is in flight.
  bool get running =>
      stage == AutoStage.gathering ||
      stage == AutoStage.building ||
      stage == AutoStage.training;
}

/// Drives corpus → model → training automatically and keeps it fresh.
class AutoPilotController extends ChangeNotifier {
  /// Creates the controller over the app's existing services.
  AutoPilotController({
    required this.corpus,
    required this.runtime,
    required this.training,
    required this.deviceState,
    this.memory,
    this.pollInterval = const Duration(minutes: 15),
    this.minCorpusChars = 2000,
    this.setupSteps = 240,
    this.maintenanceSteps = 40,
    Future<void> Function()? onSettled,
  }) : _onSettled = onSettled;

  /// Text gathering.
  final CorpusController corpus;

  /// Tokenizer + model ownership.
  final HarborModelRuntime runtime;

  /// The training pipeline.
  final TrainingController training;

  /// Device power/thermal source used to gate background work.
  final DeviceStateSource deviceState;

  /// The assistant's memory, read only to report how much it has learned.
  final hc.MemoryStore? memory;

  /// How often the background poll runs while automatic learning is on.
  final Duration pollInterval;

  /// Below this much text, first training is pointless and setup says so.
  final int minCorpusChars;

  /// Steps run by the explicit one-tap setup.
  final int setupSteps;

  /// Steps run by a background maintenance round.
  final int maintenanceSteps;

  /// Called after a background round finishes (e.g. to notify the user).
  final Future<void> Function()? _onSettled;

  AutoPilotStatus _status = const AutoPilotStatus(
    stage: AutoStage.idle,
    progress: 0,
    message: 'Not started.',
    corpusChars: 0,
    memories: 0,
    modelReady: false,
    autoLearn: false,
  );

  Timer? _timer;
  bool _autoLearn = false;
  Future<void>? _inFlight;

  /// The current snapshot.
  AutoPilotStatus get status => _status;

  /// Whether background self-maintenance is enabled.
  bool get autoLearn => _autoLearn;

  /// Whether a pipeline run is in flight.
  bool get busy => _inFlight != null;

  /// Runs the whole setup: gather → build → train.
  ///
  /// Concurrent calls share one run rather than queueing a second: a double tap
  /// must not start two training sessions fighting over the same checkpoint.
  Future<void> runSetup() {
    return _inFlight ??= _runSetup().whenComplete(() => _inFlight = null);
  }

  Future<void> _runSetup() async {
    try {
      _set(AutoStage.gathering, 0.05, 'Reading text from this device…');
      await corpus.initialize();
      final int before = corpus.stats.totalChars;
      await corpus.scanDevice();
      _set(
        AutoStage.gathering,
        0.2,
        'Corpus ready: ${(corpus.stats.totalChars / 1000).toStringAsFixed(1)}k '
            'characters.',
      );

      if (corpus.stats.totalChars < minCorpusChars && before < minCorpusChars) {
        // Honest failure beats a fabricated model: with almost no text the
        // tokenizer learns nothing and the "training" would be noise.
        _set(
          AutoStage.failed,
          0.2,
          'Not enough text to learn from yet. Add a document on the Corpus '
              'tab or plug the phone in on Wi-Fi and let it gather.',
          error: 'corpus below $minCorpusChars characters',
        );
        return;
      }

      _set(AutoStage.building, 0.35, 'Building tokenizer and model…');
      await runtime.bootstrap(corpus.store.trainingText(maxChars: 24000));
      if (!runtime.ready) {
        _set(
          AutoStage.failed,
          0.35,
          'The model could not be built: ${runtime.error ?? runtime.status}',
          error: runtime.error,
        );
        return;
      }
      await runtime.save();
      _set(
        AutoStage.building,
        0.5,
        'Model ready (${(runtime.model!.totalParameterCount / 1000).toStringAsFixed(0)}k parameters).',
      );

      _set(AutoStage.training, 0.55, 'Training the first round…');
      await training.start(
        override: training.config.copyWith(
          totalSteps: setupSteps,
          checkpointEvery: 20,
        ),
      );
      // start() returns while the session runs; settle means "no longer
      // running", which is the only condition the UI can key "ready" on.
      await _awaitTraining();
      await runtime.save();

      _set(
        AutoStage.ready,
        1.0,
        'Done. Harbor is learning on this device and gets better as it reads.',
      );
      setAutoLearn(true);
    } on Object catch (error) {
      _set(AutoStage.failed, _status.progress, 'Setup stopped: $error',
          error: '$error');
    }
  }

  /// Turns background self-maintenance on or off.
  ///
  /// Turning it on starts a poll that only trains when the device is idle and
  /// charging, mirroring the idle scheduler's gate. Turning it off stops the
  /// poll immediately — a switch that kept a timer alive after `false` would be
  /// a battery leak with a settings label.
  void setAutoLearn(bool enabled) {
    _autoLearn = enabled;
    _timer?.cancel();
    _timer = null;
    if (enabled) {
      _timer = Timer.periodic(pollInterval, (_) => unawaited(maintenanceRound()));
    }
    _refresh();
    notifyListeners();
  }

  /// One gated background round; no-op unless the device can afford it.
  Future<void> maintenanceRound() async {
    if (busy || training.isRunning || !_autoLearn) {
      return;
    }
    if (training.readiness != null) {
      return;
    }
    final DeviceState state = await deviceState.read();
    if (!state.isIdle || !state.isCharging || state.batteryLevel < 0.35 ||
        !_thermalAllows(state.thermalState)) {
      _refresh();
      return;
    }
    try {
      _set(
        AutoStage.training,
        _status.progress,
        'Idle training round while charging…',
      );
      await training.start(
        override: training.config.copyWith(
          totalSteps: maintenanceSteps,
          checkpointEvery: 10,
        ),
      );
      await _awaitTraining();
      await runtime.save();
      await _onSettled?.call();
      _set(
        AutoStage.ready,
        _status.progress,
        'Round complete. Harbor keeps learning while the phone rests.',
      );
    } on Object catch (error) {
      // A failed background round is not an error worth interrupting the user
      // for; the next poll retries when the gate allows.
      _set(AutoStage.ready, _status.progress, 'Last background round skipped: $error');
    }
  }

  void _set(
    AutoStage stage,
    double progress,
    String message, {
    String? error,
  }) {
    _status = AutoPilotStatus(
      stage: stage,
      // A status snapshot may briefly lag a coarser stage, but never move
      // backwards within a run — the progress bar must not stutter.
      progress: progress < _status.progress ? _status.progress : progress,
      message: message,
      corpusChars: corpus.stats.totalChars,
      memories: _status.memories,
      modelReady: runtime.ready,
      autoLearn: _autoLearn,
      error: error,
    );
    notifyListeners();
  }

  /// Updates the passive counters shown alongside the stage.
  void _refresh() {
    _status = AutoPilotStatus(
      stage: _status.stage,
      progress: _status.progress,
      message: _status.message,
      corpusChars: corpus.stats.totalChars,
      memories: _status.memories,
      modelReady: runtime.ready,
      autoLearn: _autoLearn,
      error: _status.error,
    );
  }

  /// Injected memory count (the composition root owns the store).
  void reportMemories(int count) {
    if (_status.memories == count) {
      return;
    }
    _status = AutoPilotStatus(
      stage: _status.stage,
      progress: _status.progress,
      message: _status.message,
      corpusChars: _status.corpusChars,
      memories: count,
      modelReady: _status.modelReady,
      autoLearn: _status.autoLearn,
      error: _status.error,
    );
    notifyListeners();
  }

  /// Whether the device is cool enough to train on.
  ///
  /// Compared against the enum's declared order rather than an index literal,
  /// so reordering [ThermalState] cannot silently invert the gate.
  static bool _thermalAllows(ThermalState state) {
    const List<ThermalState> allowed = <ThermalState>[
      ThermalState.nominal,
      ThermalState.fair,
    ];
    return allowed.contains(state);
  }

  Future<void> _awaitTraining({
    Duration timeout = const Duration(minutes: 10),
  }) async {
    final Stopwatch clock = Stopwatch()..start();
    while (training.isRunning && clock.elapsed < timeout) {
      await Future<void>.delayed(const Duration(milliseconds: 200));
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }
}
