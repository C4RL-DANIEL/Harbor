// Training state for the UI, the notification, and the log.
//
// This controller is where "the model is learning" becomes something the user
// can see while it happens. Three outputs come from one progress stream:
//
//  * a loss curve and a live status line in the app;
//  * an Android foreground notification, throttled, so training survives the
//    app being backgrounded (and so the OS does not kill it as a background
//    service);
//  * a durable JSONL log entry per finished round, so the effect of yesterday's
//    corpus is still inspectable today.
//
// The throttle exists because a notification update per step would be hundreds
// of IPC calls per second and would make the phone warm for no reason.

import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:harbor_core/harbor_core.dart';

import '../corpus/corpus_controller.dart';
import '../corpus/corpus_persistence.dart';
import '../model/harbor_model_runtime.dart';
import '../platform/platform_bridge.dart';

/// Runs training rounds and reports on them.
class TrainingController extends ChangeNotifier {
  /// Creates a controller.
  TrainingController({
    required this.runtime,
    required this.corpus,
    required this.bridge,
    required this.log,
    this.config = const TrainingConfig(),
  });

  /// The model being trained.
  final HarborModelRuntime runtime;

  /// The text being learned from.
  final CorpusController corpus;

  /// The platform channel, for the foreground notification.
  final PlatformBridge bridge;

  /// Where completed rounds are recorded.
  final TrainingLog log;

  /// Knobs for the next run.
  TrainingConfig config;

  TrainingSession? _session;
  StreamSubscription<TrainingProgress>? _subscription;
  TrainingProgress? _latest;
  final List<double> _lossHistory = <double>[];
  String? _error;
  bool _notificationsEnabled = true;
  int _lastNotifiedStep = -1;
  DateTime _lastNotifiedAt = DateTime.fromMillisecondsSinceEpoch(0, isUtc: true);

  /// How many loss points to keep for the chart.
  static const int maxLossPoints = 400;

  /// The most recent progress snapshot.
  TrainingProgress? get latest => _latest;

  /// Whether a round is running.
  bool get isRunning => _session?.isRunning ?? false;

  /// Whether the run is paused.
  bool get isPaused => _session?.isPaused ?? false;

  /// Smoothed loss over time, oldest first.
  List<double> get lossHistory => List<double>.unmodifiable(_lossHistory);

  /// The last error, if any.
  String? get error => _error;

  /// Whether the Android notification is driven by training progress.
  bool get notificationsEnabled => _notificationsEnabled;

  /// Enables or disables the training notification.
  void setNotificationsEnabled(bool value) {
    _notificationsEnabled = value;
    if (!value) {
      unawaited(_stopNotification());
    }
    notifyListeners();
  }

  /// The number of tokens the current corpus would train on.
  int get availableTokens {
    final ByteTokenizer? tokenizer = runtime.tokenizer;
    if (tokenizer == null) {
      return 0;
    }
    final String text = corpus.trainingText(maxChars: 200000);
    if (text.trim().isEmpty) {
      return 0;
    }
    return tokenizer.encode(text, addBos: true).length;
  }

  /// Whether a round can start, with the reason when it cannot.
  String? get readiness {
    if (!runtime.ready) {
      return 'The model is not ready: ${runtime.error ?? runtime.status}';
    }
    final int tokens = availableTokens;
    if (tokens == 0) {
      return 'The corpus is empty. Gather text on the Corpus tab first.';
    }
    if (tokens <= config.windowLength + 1) {
      return 'The corpus produced $tokens tokens, which is too few for a '
          'window of ${config.windowLength + 1}. Add more text.';
    }
    return null;
  }

  /// Starts a round, optionally with different knobs.
  Future<void> start({TrainingConfig? override}) async {
    if (isRunning) {
      return;
    }
    final String? blocked = readiness;
    if (blocked != null) {
      _error = blocked;
      notifyListeners();
      return;
    }
    final TinyLm model = runtime.model!;
    final ByteTokenizer tokenizer = runtime.tokenizer!;
    final String text = corpus.trainingText(maxChars: 200000);
    final List<int> tokens = tokenizer.encode(text, addBos: true);
    config = override ?? config;

    _error = null;
    _lossHistory.clear();
    _lastNotifiedStep = -1;
    notifyListeners();

    final TrainingSession session = TrainingSession(
      model: model,
      tokens: tokens,
      config: config,
      onCheckpoint: (String payload) {
        unawaited(runtime.saveCheckpoint(payload));
      },
    );
    _session = session;
    _subscription = session.progress.listen(_onProgress);
    await _startNotification();
    await session.start();
    await _subscription?.cancel();
    _subscription = null;
    await _stopNotification();
    notifyListeners();
  }

  /// Requests a pause at the next step boundary.
  void pause() {
    _session?.pause();
    notifyListeners();
  }

  /// Resumes a paused run.
  void resume() {
    _session?.resume();
    notifyListeners();
  }

  /// Stops the run, keeping whatever it learned.
  Future<void> cancel() async {
    await _session?.cancel();
    notifyListeners();
  }

  /// Persists the model now, outside the checkpoint cadence.
  Future<void> saveNow() async {
    await runtime.save();
    notifyListeners();
  }

  void _onProgress(TrainingProgress progress) {
    _latest = progress;
    if (progress.phase == TrainingPhase.training) {
      _lossHistory.add(progress.smoothedLoss);
      if (_lossHistory.length > maxLossPoints) {
        _lossHistory.removeRange(0, _lossHistory.length - maxLossPoints);
      }
    }
    if (progress.phase == TrainingPhase.failed) {
      _error = progress.message ?? 'Training failed';
    }
    unawaited(_pushNotification(progress));
    if (_isTerminal(progress.phase)) {
      unawaited(_finishRound(progress));
    }
    notifyListeners();
  }

  Future<void> _finishRound(TrainingProgress progress) async {
    await _stopNotification();
    if (progress.phase == TrainingPhase.completed) {
      await runtime.save();
    }
    try {
      await log.append(<String, Object?>{
        'finished_at': DateTime.now().toUtc().toIso8601String(),
        'phase': progress.phase.name,
        'steps': progress.step,
        'loss': progress.smoothedLoss,
        'accuracy': progress.accuracy,
        'tokens_processed': progress.tokensProcessed,
        'parameters': runtime.model?.totalParameterCount ?? 0,
        'config': config.toJson(),
        'corpus_documents': corpus.stats.documentCount,
        'corpus_chars': corpus.stats.totalChars,
      });
    } on Object {
      // The log is diagnostic, not load-bearing: a failure to append must not
      // turn a successful training round into a reported error.
    }
  }

  Future<void> _startNotification() async {
    if (!_notificationsEnabled || !bridge.isSupported) {
      return;
    }
    try {
      await bridge.startTrainingNotification(
        title: 'Harbor is learning',
        text: 'Preparing $availableTokens tokens',
      );
    } on Object {
      // A denied notification permission must not stop training.
    }
  }

  Future<void> _pushNotification(TrainingProgress progress) async {
    if (!_notificationsEnabled || !bridge.isSupported) {
      return;
    }
    final DateTime now = DateTime.now();
    final bool isStep = progress.phase == TrainingPhase.training;
    final bool dueByStep = progress.step - _lastNotifiedStep >= 5;
    final bool dueByTime =
        now.difference(_lastNotifiedAt) > const Duration(milliseconds: 700);
    if (isStep && !dueByStep && !dueByTime) {
      return;
    }
    _lastNotifiedStep = progress.step;
    _lastNotifiedAt = now;
    try {
      await bridge.updateTrainingNotification(
        progress: progress.step,
        max: progress.totalSteps,
        text: '${_notificationTitle(progress)} · ${progress.label}',
      );
    } on Object {
      // Same reasoning as above.
    }
  }

  String _notificationTitle(TrainingProgress progress) {
    switch (progress.phase) {
      case TrainingPhase.paused:
        return 'Harbor is paused';
      case TrainingPhase.checkpointing:
        return 'Harbor is saving progress';
      case TrainingPhase.completed:
        return 'Harbor finished learning';
      case TrainingPhase.cancelled:
        return 'Harbor stopped learning';
      case TrainingPhase.failed:
        return 'Harbor could not learn';
      case TrainingPhase.idle:
      case TrainingPhase.preparing:
      case TrainingPhase.training:
        return 'Harbor is learning';
    }
  }

  Future<void> _stopNotification() async {
    if (!bridge.isSupported) {
      return;
    }
    try {
      await bridge.stopTrainingNotification();
    } on Object {
      // Ignored: the notification may already have been dismissed.
    }
  }

  static bool _isTerminal(TrainingPhase phase) {
    switch (phase) {
      case TrainingPhase.completed:
      case TrainingPhase.cancelled:
      case TrainingPhase.failed:
        return true;
      case TrainingPhase.idle:
      case TrainingPhase.preparing:
      case TrainingPhase.training:
      case TrainingPhase.checkpointing:
      case TrainingPhase.paused:
        return false;
    }
  }

  @override
  void dispose() {
    unawaited(_subscription?.cancel());
    super.dispose();
  }
}