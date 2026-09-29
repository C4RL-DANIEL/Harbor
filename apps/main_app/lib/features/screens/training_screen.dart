// The training screen: what the model is, what it is doing, and how to steer it.
//
// This is the one place where the app's model runtime and its training
// controller are presented together. The screen is deliberately a thin view:
// every number it shows already lives on the controller or the runtime, and
// every control delegates straight to them, so the training loop keeps running
// — and keeps surviving backgrounding — no matter what this widget tree does.

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:harbor_core/harbor_core.dart';

import '../../core/model/harbor_model_runtime.dart';
import '../../core/training/training_controller.dart';

/// The training tab: model summary, run controls, knobs, and live progress.
///
/// It is stateful for exactly one reason: the configuration sliders replace
/// [TrainingController.config], and `setState` is what makes the new value
/// visible on the very next frame. Everything else rebuilds from the
/// controller's `ChangeNotifier` notifications.
class TrainingScreen extends StatefulWidget {
  /// Creates the training screen for [controller] and [runtime].
  const TrainingScreen({
    super.key,
    required this.controller,
    required this.runtime,
  });

  /// Drives training and reports progress.
  final TrainingController controller;

  /// Owns the on-device tokenizer, weights, and checkpoint file.
  final HarborModelRuntime runtime;

  @override
  State<TrainingScreen> createState() => _TrainingScreenState();
}

class _TrainingScreenState extends State<TrainingScreen> {
  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: widget.controller,
      builder: (BuildContext context, Widget? child) {
        final TrainingController controller = widget.controller;
        final TrainingProgress? latest = controller.latest;
        return ListView(
          padding: const EdgeInsets.all(16),
          children: <Widget>[
            _modelCard(context, widget.runtime),
            const SizedBox(height: 16),
            _runControls(context, controller),
            if (controller.error != null) ...[
              const SizedBox(height: 16),
              _errorCard(context, controller.error!),
            ],
            const SizedBox(height: 16),
            _configurationCard(context, controller),
            const SizedBox(height: 16),
            _notificationCard(controller),
            if (latest != null) ...[
              const SizedBox(height: 16),
              _metricsCard(context, latest),
              const SizedBox(height: 16),
              _chartCard(context, controller, latest),
            ] else if (!controller.isRunning) ...[
              const SizedBox(height: 16),
              _emptyCard(context),
            ],
          ],
        );
      },
    );
  }

  /// Describes the model that training will actually touch.
  ///
  /// The shape, the parameter count and the checkpoint age are shown before any
  /// button, because "train" on an empty or stale model is the single most
  /// confusing thing this screen can offer: the user needs to see what exists
  /// first.
  Widget _modelCard(BuildContext context, HarborModelRuntime runtime) {
    final ThemeData theme = Theme.of(context);
    final TinyLmConfig config = runtime.config;
    final DateTime? checkpointAt = runtime.checkpointAt;
    final String checkpointLabel;
    if (!runtime.hasCheckpoint) {
      checkpointLabel = 'No checkpoint yet';
    } else if (checkpointAt == null) {
      checkpointLabel = 'Checkpoint on disk';
    } else {
      checkpointLabel = 'Checkpoint saved ${_relative(checkpointAt)}';
    }
    final String? runtimeError = runtime.error;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Row(
              children: <Widget>[
                Icon(
                  runtime.ready ? Icons.memory : Icons.hourglass_empty,
                  color: theme.colorScheme.primary,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'On-device model',
                    style: theme.textTheme.titleMedium,
                  ),
                ),
                TextButton(
                  onPressed: () {
                    unawaited(_confirmAndReset());
                  },
                  child: const Text('Reset model'),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(runtime.status),
            const SizedBox(height: 4),
            Text(config.summary, style: theme.textTheme.bodySmall),
            const SizedBox(height: 4),
            Text(
              '${_thousands(config.estimatedParameters)} parameters',
              style: theme.textTheme.bodySmall,
            ),
            const SizedBox(height: 4),
            Text(checkpointLabel, style: theme.textTheme.bodySmall),
            if (runtimeError != null) ...[
              const SizedBox(height: 8),
              Text(
                runtimeError,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.error,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  /// Shows either the blocker that prevents a run, or the run controls.
  ///
  /// The readiness message comes from the controller itself so the reason shown
  /// here is the same reason `start` would refuse, rather than a second,
  /// drifting copy of the rules.
  Widget _runControls(BuildContext context, TrainingController controller) {
    final String? readiness = controller.readiness;
    if (readiness != null) {
      return _warningCard(context, readiness);
    }
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            FilledButton.icon(
              onPressed: () {
                unawaited(controller.start());
              },
              icon: const Icon(Icons.play_arrow),
              label: const Text('Start training'),
            ),
            if (controller.isRunning) ...[
              const SizedBox(height: 16),
              LinearProgressIndicator(
                value: controller.latest?.fraction ?? 0,
              ),
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: <Widget>[
                  FilledButton.tonalIcon(
                    onPressed: controller.isPaused
                        ? controller.resume
                        : controller.pause,
                    icon: Icon(
                      controller.isPaused ? Icons.play_arrow : Icons.pause,
                    ),
                    label: Text(controller.isPaused ? 'Resume' : 'Pause'),
                  ),
                  OutlinedButton.icon(
                    onPressed: () {
                      unawaited(controller.cancel());
                    },
                    icon: const Icon(Icons.stop),
                    label: const Text('Stop'),
                  ),
                  TextButton.icon(
                    onPressed: () {
                      unawaited(controller.saveNow());
                    },
                    icon: const Icon(Icons.save_outlined),
                    label: const Text('Save now'),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  /// Explains why training cannot start, in amber rather than red.
  ///
  /// An empty corpus is a normal first-run state, not a failure; a red error
  /// card would train the user to read the screen as broken before they have
  /// done anything wrong.
  Widget _warningCard(BuildContext context, String readiness) {
    final ThemeData theme = Theme.of(context);
    return Card(
      color: Colors.amber.withOpacity(0.18),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            const Icon(Icons.warning_amber),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Text(readiness),
                  const SizedBox(height: 4),
                  Text(
                    'Gather more text on the Corpus tab, then come back to '
                    'start training.',
                    style: theme.textTheme.bodySmall,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Exposes the two knobs worth changing without a settings file.
  ///
  /// Both are locked while a run is in flight: the controller snapshots the
  /// config when it starts, so a mid-run edit would change the displayed number
  /// without changing the run and quietly turn the screen into a lie.
  Widget _configurationCard(
    BuildContext context,
    TrainingController controller,
  ) {
    final ThemeData theme = Theme.of(context);
    final TrainingConfig config = controller.config;
    final bool enabled = !controller.isRunning;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text('Configuration', style: theme.textTheme.titleMedium),
            const SizedBox(height: 8),
            _LabelledSlider(
              label: 'Steps',
              valueLabel: _thousands(config.totalSteps),
              value: config.totalSteps.toDouble(),
              min: 20,
              max: 2000,
              divisions: 198,
              enabled: enabled,
              onChanged: (double value) {
                setState(() {
                  controller.config = controller.config.copyWith(
                    totalSteps: value.round(),
                  );
                });
              },
            ),
            _LabelledSlider(
              label: 'Window',
              valueLabel: _thousands(config.windowLength),
              value: config.windowLength.toDouble(),
              min: 8,
              max: 128,
              divisions: 60,
              enabled: enabled,
              onChanged: (double value) {
                setState(() {
                  controller.config = controller.config.copyWith(
                    windowLength: value.round(),
                  );
                });
              },
            ),
          ],
        ),
      ),
    );
  }

  /// Exposes the foreground-notification switch.
  ///
  /// The switch is a training-survival control, not a cosmetic one: Android
  /// kills plain background work, so the notification is what lets a long run
  /// finish after the user locks the screen.
  Widget _notificationCard(TrainingController controller) {
    return Card(
      child: SwitchListTile(
        title: const Text('Training notification'),
        subtitle: const Text(
          'Keeps the run alive while the app is in the background; Android '
          'shows a live progress notification.',
        ),
        value: controller.notificationsEnabled,
        onChanged: controller.setNotificationsEnabled,
      ),
    );
  }

  /// Shows the latest training snapshot as a grid of labelled numbers.
  ///
  /// Every value is formatted at a fixed precision so the numbers stop
  /// twitching, and the raw loss is shown beside the smoothed one so the user
  /// can see why the curve they are watching is the smoothed series.
  Widget _metricsCard(BuildContext context, TrainingProgress latest) {
    final ThemeData theme = Theme.of(context);
    final Duration? eta = latest.eta;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text('Live metrics', style: theme.textTheme.titleMedium),
            const SizedBox(height: 12),
            _MetricGrid(
              metrics: <_Metric>[
                _Metric(
                  label: 'Loss',
                  value: latest.loss.toStringAsFixed(4),
                ),
                _Metric(
                  label: 'Smoothed',
                  value: latest.smoothedLoss.toStringAsFixed(4),
                ),
                _Metric(
                  label: 'Perplexity',
                  value: latest.perplexity.toStringAsFixed(1),
                ),
                _Metric(
                  label: 'Accuracy',
                  value: '${(latest.accuracy * 100).toStringAsFixed(1)}%',
                ),
                _Metric(
                  label: 'Learning rate',
                  value: latest.learningRate.toStringAsExponential(2),
                ),
                _Metric(
                  label: 'Gradient norm',
                  value: latest.gradientNorm.toStringAsFixed(3),
                ),
                _Metric(
                  label: 'Tokens/s',
                  value: latest.tokensPerSecond.toStringAsFixed(0),
                ),
                _Metric(
                  label: 'Tokens',
                  value: _thousands(latest.tokensProcessed),
                ),
                _Metric(
                  label: 'Elapsed',
                  value: _duration(latest.elapsed),
                ),
                _Metric(
                  label: 'ETA',
                  value: eta == null ? '—' : _duration(eta),
                ),
                _Metric(
                  label: 'Step',
                  value: '${latest.step}/${latest.totalSteps}',
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  /// Draws the smoothed loss history with its current and best values.
  ///
  /// The chart is the only place a trend is legible; the grid above it is a
  /// snapshot. The best value is computed from the same list the painter uses,
  /// so the headline and the curve can never disagree.
  Widget _chartCard(
    BuildContext context,
    TrainingController controller,
    TrainingProgress latest,
  ) {
    final ThemeData theme = Theme.of(context);
    final List<double> losses = controller.lossHistory;
    final double? best = losses.isEmpty
        ? null
        : losses.reduce((double a, double b) => a < b ? a : b);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            Text('Loss curve', style: theme.textTheme.titleMedium),
            const SizedBox(height: 4),
            Text(
              'Now ${latest.smoothedLoss.toStringAsFixed(4)}'
              '   ·   '
              'Best ${best == null ? '—' : best.toStringAsFixed(4)}',
            ),
            const SizedBox(height: 12),
            SizedBox(
              height: 180,
              child: CustomPaint(
                painter: _LossChartPainter(
                  losses: losses,
                  lineColor: theme.colorScheme.primary,
                  gridColor: theme.colorScheme.outlineVariant,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Explains, before the first run, what training will and will not do.
  ///
  /// The expectations this card sets are the whole point: a user who expects a
  /// frontier assistant from a phone-sized model trained on a few notes will
  /// read a correct result as a bug.
  Widget _emptyCard(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text('Nothing trained yet', style: theme.textTheme.titleMedium),
            const SizedBox(height: 8),
            const Text(
              'Training teaches this model from the text you gather on the '
              'Corpus tab. A small corpus teaches memorisation rather than '
              'generalisation: the model will recall the text you gave it much '
              'better than it reasons about anything new. Everything stays on '
              'this device — no text and no weights are uploaded. This is a '
              'small on-device model, not a frontier assistant.',
            ),
          ],
        ),
      ),
    );
  }

  /// Reports the controller's last error in a red-tinted card.
  ///
  /// Errors are kept separate from the readiness warning because they mean the
  /// run was attempted and something failed, which needs a different response
  /// than "add more text".
  Widget _errorCard(BuildContext context, String message) {
    final ThemeData theme = Theme.of(context);
    return Card(
      color: theme.colorScheme.errorContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Icon(
              Icons.error_outline,
              color: theme.colorScheme.onErrorContainer,
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                message,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onErrorContainer,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Confirms and then deletes every learned artefact.
  ///
  /// Reset is destructive and cannot be undone, so it is gated behind a dialog;
  /// training itself is the only way back, and that is exactly the point of the
  /// confirmation.
  Future<void> _confirmAndReset() async {
    final bool? confirmed = await showDialog<bool>(
      context: context,
      builder: (BuildContext dialogContext) {
        return AlertDialog(
          title: const Text('Reset the model?'),
          content: const Text(
            'This deletes the learned tokenizer and every trained weight on '
            'this device. The text on the Corpus tab is kept, but training '
            'will start again from a random model.',
          ),
          actions: <Widget>[
            TextButton(
              onPressed: () => Navigator.of(dialogContext).pop(false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.of(dialogContext).pop(true),
              child: const Text('Reset'),
            ),
          ],
        );
      },
    );
    if (confirmed ?? false) {
      await widget.runtime.reset();
      if (!mounted) {
        return;
      }
      // The controller does not forward runtime notifications, so the model
      // card is refreshed explicitly rather than waiting for a training event
      // that may never come.
      setState(() {});
    }
  }
}

/// One labelled number in the live-metrics grid.
///
/// The label sits above the value in a muted style so the eye reads a column of
/// numbers rather than a wall of text.
class _Metric extends StatelessWidget {
  const _Metric({required this.label, required this.value});

  /// The metric's name.
  final String label;

  /// The metric's pre-formatted value.
  final String value;

  @override
  Widget build(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Text(
          label,
          style: theme.textTheme.labelSmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
        const SizedBox(height: 2),
        Text(value, style: theme.textTheme.titleMedium),
      ],
    );
  }
}

/// Lays metrics out in a wrapping grid sized to the available width.
///
/// A fixed column count would either clip on a phone or waste half a tablet, so
/// the width decides how many tiles fit. A tile keeps a sane minimum so a very
/// narrow window wraps to fewer columns instead of shrinking text to nothing.
class _MetricGrid extends StatelessWidget {
  const _MetricGrid({required this.metrics});

  /// The tiles to lay out.
  final List<_Metric> metrics;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (BuildContext context, BoxConstraints constraints) {
        final double available = constraints.maxWidth.isFinite
            ? constraints.maxWidth
            : 360;
        final int columns;
        if (available >= 560) {
          columns = 3;
        } else if (available >= 240) {
          columns = 2;
        } else {
          columns = 1;
        }
        const double spacing = 12;
        final double tileWidth =
            ((available - spacing * (columns - 1)) / columns)
                .clamp(80.0, double.infinity)
                .toDouble();
        return Wrap(
          spacing: spacing,
          runSpacing: spacing,
          children: <Widget>[
            for (final _Metric metric in metrics)
              SizedBox(width: tileWidth, child: metric),
          ],
        );
      },
    );
  }
}

/// A slider that always shows its exact current value as text.
///
/// The thumb alone cannot communicate the difference between, say, 300 and 320
/// steps at phone width, and these two knobs are the ones users reason about as
/// numbers rather than as "more" and "less".
class _LabelledSlider extends StatelessWidget {
  const _LabelledSlider({
    required this.label,
    required this.valueLabel,
    required this.value,
    required this.min,
    required this.max,
    required this.divisions,
    required this.enabled,
    required this.onChanged,
  });

  /// The knob's name.
  final String label;

  /// The current value, already formatted for display.
  final String valueLabel;

  /// The slider position.
  final double value;

  /// Lower bound of the slider.
  final double min;

  /// Upper bound of the slider.
  final double max;

  /// Number of discrete positions between [min] and [max].
  final int divisions;

  /// Whether the slider accepts changes.
  final bool enabled;

  /// Called with the new value while dragging.
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: <Widget>[
            Text(label),
            Text(valueLabel, style: theme.textTheme.labelLarge),
          ],
        ),
        Slider(
          value: value,
          min: min,
          max: max,
          divisions: divisions,
          label: valueLabel,
          onChanged: enabled ? onChanged : null,
        ),
      ],
    );
  }
}

/// Paints the smoothed loss series as a grid plus one polyline.
///
/// The chart is intentionally minimal: the useful signal is the slope, and a
/// richer axis/legend treatment would compete with the metric grid directly
/// above it. It scales to whatever box it is given, so one painter serves a
/// phone and a tablet.
class _LossChartPainter extends CustomPainter {
  const _LossChartPainter({
    required this.losses,
    required this.lineColor,
    required this.gridColor,
  });

  /// Smoothed loss values, oldest first.
  final List<double> losses;

  /// Colour of the loss polyline.
  final Color lineColor;

  /// Colour of the reference grid.
  final Color gridColor;

  @override
  void paint(Canvas canvas, Size size) {
    final Paint grid = Paint()
      ..color = gridColor
      ..strokeWidth = 1
      ..style = PaintingStyle.stroke;
    // Four horizontal guides, including both edges, give the eye a scale
    // without needing labelled axes.
    for (int i = 0; i <= 3; i++) {
      final double y = size.height * i / 3;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), grid);
    }

    final Paint line = Paint()
      ..color = lineColor
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    // Nothing to plot yet: a flat mid-line reads as "waiting", where a blank
    // box reads as a rendering bug.
    if (losses.length < 2) {
      canvas.drawLine(
        Offset(0, size.height / 2),
        Offset(size.width, size.height / 2),
        line,
      );
      return;
    }

    double minValue = losses.first;
    double maxValue = losses.first;
    for (final double value in losses) {
      if (value < minValue) {
        minValue = value;
      }
      if (value > maxValue) {
        maxValue = value;
      }
    }

    // A perfectly flat series has no range to normalise against; drawing it at
    // the middle avoids a divide-by-zero and is the honest picture anyway.
    if (maxValue == minValue) {
      canvas.drawLine(
        Offset(0, size.height / 2),
        Offset(size.width, size.height / 2),
        line,
      );
      return;
    }

    final double span = maxValue - minValue;
    final Path path = Path();
    for (int i = 0; i < losses.length; i++) {
      final double x = size.width * i / (losses.length - 1);
      // Invert y so a smaller loss is drawn lower on the chart, which matches
      // the mental model of "loss goes down over time".
      final double normalized = (losses[i] - minValue) / span;
      final double y = size.height * (1 - normalized);
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    canvas.drawPath(path, line);
  }

  @override
  bool shouldRepaint(covariant _LossChartPainter oldDelegate) {
    if (oldDelegate.losses.length != losses.length) {
      return true;
    }
    if (oldDelegate.lineColor != lineColor ||
        oldDelegate.gridColor != gridColor) {
      return true;
    }
    if (losses.isEmpty) {
      return false;
    }
    return oldDelegate.losses.last != losses.last;
  }
}

/// Formats an integer with thousands separators, e.g. `8303` → `8,303`.
///
/// Parameter counts are read as magnitudes, and an unseparated six-digit number
/// is genuinely hard to compare against its neighbours at a glance.
String _thousands(int value) {
  final bool negative = value < 0;
  final String digits = value.abs().toString();
  final StringBuffer buffer = StringBuffer();
  for (int i = 0; i < digits.length; i++) {
    if (i > 0 && (digits.length - i) % 3 == 0) {
      buffer.write(',');
    }
    buffer.write(digits[i]);
  }
  return negative ? '-$buffer' : buffer.toString();
}

/// Renders a timestamp as a coarse, human "how long ago" phrase.
///
/// The checkpoint age only needs to be accurate enough to answer "is this from
/// today or last week", and a precise timestamp would invite the reader to
/// compute a duration they do not care about.
String _relative(DateTime time) {
  final Duration difference = DateTime.now().difference(time);
  if (difference.inSeconds < 5) {
    return 'just now';
  }
  if (difference.inSeconds < 60) {
    final int seconds = difference.inSeconds;
    return '$seconds ${seconds == 1 ? 'second' : 'seconds'} ago';
  }
  if (difference.inMinutes < 60) {
    final int minutes = difference.inMinutes;
    return '$minutes ${minutes == 1 ? 'minute' : 'minutes'} ago';
  }
  if (difference.inHours < 24) {
    final int hours = difference.inHours;
    return '$hours ${hours == 1 ? 'hour' : 'hours'} ago';
  }
  final int days = difference.inDays;
  return '$days ${days == 1 ? 'day' : 'days'} ago';
}

/// Formats a duration as `mm:ss`, or `h:mm:ss` once it passes an hour.
///
/// Fixed-width fields keep the elapsed and ETA tiles from resizing every second,
/// which is otherwise a distracting jitter in a live display.
String _duration(Duration duration) {
  final int hours = duration.inHours;
  final int minutes = duration.inMinutes.remainder(60);
  final int seconds = duration.inSeconds.remainder(60);
  final String paddedMinutes = minutes.toString().padLeft(2, '0');
  final String paddedSeconds = seconds.toString().padLeft(2, '0');
  if (hours > 0) {
    return '$hours:$paddedMinutes:$paddedSeconds';
  }
  return '$paddedMinutes:$paddedSeconds';
}