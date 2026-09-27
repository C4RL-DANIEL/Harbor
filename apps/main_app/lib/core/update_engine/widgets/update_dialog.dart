// Material 3 update UI: forced modal, dismissible soft banner, and progress.
//
// The distinction the product cares about is enforced structurally here:
//
//   * [ForceUpdateDialog] wraps its content in a `PopScope(canPop: false)` and
//     is only ever shown with `barrierDismissible: false`, so the user cannot
//     back out, tap outside, or swipe it away. There is no "Later" affordance.
//   * [SoftUpdateBanner] always offers "Later", and dismissing it records the
//     version so the same prompt is not shown again on the next launch.
//
// [UpdateFlowController] owns the download/verify/install state machine so both
// surfaces share exactly one implementation and the widgets stay presentational.

import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';

import '../update_service.dart';

/// Stage of the update flow, as presented to the user.
enum UpdateFlowStage {
  idle,
  downloading,
  verifying,
  installing,
  completed,
  failed;

  String get label {
    switch (this) {
      case UpdateFlowStage.idle:
        return 'Ready';
      case UpdateFlowStage.downloading:
        return 'Downloading';
      case UpdateFlowStage.verifying:
        return 'Verifying integrity';
      case UpdateFlowStage.installing:
        return 'Installing';
      case UpdateFlowStage.completed:
        return 'Download complete';
      case UpdateFlowStage.failed:
        return 'Failed';
    }
  }
}

/// Drives download + verification + installation, exposing a listenable state
/// machine that any update surface can render.
class UpdateFlowController extends ChangeNotifier {
  UpdateFlowController({required this.service});

  /// The service performing the network and platform work.
  final UpdateService service;

  UpdateFlowStage _stage = UpdateFlowStage.idle;
  DownloadProgress? _progress;
  InstallProgress? _install;
  String? _error;
  StreamSubscription<InstallProgress>? _installSubscription;
  bool _disposed = false;

  /// Current stage.
  UpdateFlowStage get stage => _stage;

  /// Byte-level progress while [UpdateFlowStage.downloading].
  DownloadProgress? get progress => _progress;

  /// Latest native installer event.
  InstallProgress? get install => _install;

  /// Error text when [UpdateFlowStage.failed].
  String? get error => _error;

  /// Whether work is in flight.
  bool get isBusy =>
      _stage == UpdateFlowStage.downloading ||
      _stage == UpdateFlowStage.verifying ||
      _stage == UpdateFlowStage.installing;

  /// Fraction in `[0, 1]` for determinate progress indicators, or null when the
  /// server did not report a content length.
  double? get fraction {
    if (_stage == UpdateFlowStage.verifying) {
      return 1.0;
    }
    if (_stage == UpdateFlowStage.installing ||
        _stage == UpdateFlowStage.completed) {
      return 1.0;
    }
    return _progress?.fraction;
  }

  void _setStage(UpdateFlowStage stage, {String? error}) {
    _stage = stage;
    _error = error;
    if (!_disposed) {
      notifyListeners();
    }
  }

  /// Runs the full update for [release].
  ///
  /// Returns true when the installer was successfully handed the artifact.
  Future<bool> start(ReleaseInfo release) async {
    if (isBusy) {
      return false;
    }
    _progress = null;
    _install = null;
    _setStage(UpdateFlowStage.downloading);

    try {
      final Stream<InstallProgress> progressStream = await service.performUpdate(
        release: release,
        onDownloadProgress: (DownloadProgress p) {
          _progress = p;
          // A local pre-download reports bytes; the native path reports through
          // the plugin stream instead.
          if (!_disposed) {
            notifyListeners();
          }
        },
      );

      if (service.strategy == InstallStrategy.verifyThenInstall) {
        _setStage(UpdateFlowStage.verifying);
      }

      final Completer<void> done = Completer<void>();
      _installSubscription = progressStream.listen(
        (InstallProgress event) {
          _install = event;
          if (_disposed) {
            return;
          }
          if (event.isInstalling) {
            _stage = UpdateFlowStage.installing;
          } else if (event.isFailure) {
            _stage = UpdateFlowStage.failed;
            _error = event.note ?? event.value ?? 'installer error';
          } else if (event.isCancelled) {
            _stage = UpdateFlowStage.idle;
            _error = event.note;
          } else if (event.isDownloading) {
            _stage = UpdateFlowStage.downloading;
            final int? percent = int.tryParse(event.value ?? '');
            if (percent != null) {
              _progress = DownloadProgress(
                receivedBytes: percent,
                totalBytes: 100,
              );
            }
          }
          notifyListeners();
          if (event.isTerminal && !done.isCompleted) {
            done.complete();
          }
        },
        onError: (Object e) {
          _setStage(UpdateFlowStage.failed, error: e.toString());
          if (!done.isCompleted) {
            done.complete();
          }
        },
        onDone: () {
          if (!done.isCompleted) {
            done.complete();
          }
        },
        cancelOnError: false,
      );

      await done.future;
      return _stage == UpdateFlowStage.installing ||
          _stage == UpdateFlowStage.completed;
    } on ChecksumMismatchException catch (e) {
      _setStage(
        UpdateFlowStage.failed,
        error: 'Integrity check failed — the download was discarded.\n'
            'Expected ${e.expected}, computed ${e.actual}.',
      );
      return false;
    } on ReleaseMetadataException catch (e) {
      _setStage(UpdateFlowStage.failed, error: e.message);
      return false;
    } on UpdateCheckException catch (e) {
      _setStage(UpdateFlowStage.failed, error: e.message);
      return false;
    } on UpdateInstallException catch (e) {
      _setStage(UpdateFlowStage.failed, error: e.message);
      return false;
    } on Object catch (e) {
      _setStage(UpdateFlowStage.failed, error: 'Update failed: $e');
      return false;
    }
  }

  /// Cancels an in-flight native download.
  Future<void> cancel() async {
    if (!isBusy) {
      return;
    }
    try {
      await service.cancelDownload();
    } on UpdateInstallException catch (e) {
      _error = e.message;
    }
    await _installSubscription?.cancel();
    _installSubscription = null;
    _setStage(UpdateFlowStage.idle);
  }

  @override
  void dispose() {
    _disposed = true;
    unawaited(_installSubscription?.cancel());
    super.dispose();
  }
}

/// Non-dismissible modal shown when the update is mandatory.
///
/// There is deliberately no way out of this dialog: `canPop` is false and the
/// call site passes `barrierDismissible: false`.
class ForceUpdateDialog extends StatefulWidget {
  const ForceUpdateDialog({
    super.key,
    required this.release,
    required this.installedVersion,
    required this.controller,
  });

  /// Release the server published.
  final ReleaseInfo release;

  /// Version currently installed, shown so the user sees the delta.
  final String installedVersion;

  /// Shared flow controller.
  final UpdateFlowController controller;

  /// Convenience presentation helper that guarantees the modal cannot be
  /// dismissed by tapping outside it.
  static Future<void> show(
    BuildContext context, {
    required ReleaseInfo release,
    required String installedVersion,
    required UpdateFlowController controller,
  }) {
    return showDialog<void>(
      context: context,
      barrierDismissible: false,
      useRootNavigator: true,
      builder: (BuildContext ctx) => ForceUpdateDialog(
        release: release,
        installedVersion: installedVersion,
        controller: controller,
      ),
    );
  }

  @override
  State<ForceUpdateDialog> createState() => _ForceUpdateDialogState();
}

class _ForceUpdateDialogState extends State<ForceUpdateDialog> {
  @override
  Widget build(BuildContext context) {
    final ColorScheme colors = Theme.of(context).colorScheme;
    final TextTheme text = Theme.of(context).textTheme;

    return PopScope(
      canPop: false,
      child: AnimatedBuilder(
        animation: widget.controller,
        builder: (BuildContext context, Widget? _) {
          final UpdateFlowController c = widget.controller;
          final bool failed = c.stage == UpdateFlowStage.failed;

          return AlertDialog(
            icon: Icon(
              failed ? Icons.error_outline : Icons.system_update_alt,
              color: failed ? colors.error : colors.primary,
              size: 32,
            ),
            title: Text(failed ? 'Update failed' : 'Update required'),
            content: SizedBox(
              width: 420,
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: <Widget>[
                    Text(
                      'This version of the app is no longer supported. '
                      'You must update to continue.',
                      style: text.bodyMedium,
                    ),
                    const SizedBox(height: 16),
                    _VersionRow(
                      installed: widget.installedVersion,
                      latest: widget.release.latestVersion,
                    ),
                    if (widget.release.changelog.trim().isNotEmpty) ...<Widget>[
                      const SizedBox(height: 16),
                      Text("What's new", style: text.titleSmall),
                      const SizedBox(height: 8),
                      _ChangelogText(widget.release.changelog),
                    ],
                    if (c.isBusy) ...<Widget>[
                      const SizedBox(height: 20),
                      _ProgressBlock(controller: c),
                    ],
                    if (failed) ...<Widget>[
                      const SizedBox(height: 16),
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: colors.errorContainer,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          c.error ?? 'An unknown error occurred.',
                          style: text.bodySmall?.copyWith(
                            color: colors.onErrorContainer,
                          ),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
            actions: <Widget>[
              if (c.isBusy && c.stage == UpdateFlowStage.downloading)
                TextButton(
                  onPressed: () => unawaited(c.cancel()),
                  child: const Text('Cancel download'),
                ),
              FilledButton.icon(
                onPressed: c.isBusy
                    ? null
                    : () => unawaited(c.start(widget.release)),
                icon: Icon(failed ? Icons.refresh : Icons.download),
                label: Text(
                  failed
                      ? 'Retry update'
                      : (widget.release.sizeBytes > 0
                          ? 'Update now '
                              '(${formatBytes(widget.release.sizeBytes)})'
                          : 'Update now'),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

/// Dismissible banner shown for optional updates.
class SoftUpdateBanner extends StatelessWidget {
  const SoftUpdateBanner({
    super.key,
    required this.release,
    required this.installedVersion,
    required this.onUpdate,
    required this.onDismiss,
  });

  /// Release the server published.
  final ReleaseInfo release;

  /// Version currently installed.
  final String installedVersion;

  /// Invoked when the user chooses to update.
  final VoidCallback onUpdate;

  /// Invoked when the user defers; the caller records the dismissal.
  final VoidCallback onDismiss;

  @override
  Widget build(BuildContext context) {
    final ColorScheme colors = Theme.of(context).colorScheme;
    final TextTheme text = Theme.of(context).textTheme;

    return Material(
      color: colors.secondaryContainer,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 8, 12),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Icon(Icons.new_releases_outlined, color: colors.onSecondaryContainer),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Text(
                    'Version ${release.latestVersion} is available',
                    style: text.titleSmall?.copyWith(
                      color: colors.onSecondaryContainer,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    'You are on $installedVersion. '
                    'This update is optional.'
                    '${release.sizeBytes > 0 ? ' · ${formatBytes(release.sizeBytes)}' : ''}',
                    style: text.bodySmall?.copyWith(
                      color: colors.onSecondaryContainer,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            TextButton(
              onPressed: onDismiss,
              child: const Text('Later'),
            ),
            FilledButton(
              onPressed: onUpdate,
              child: const Text('Update'),
            ),
          ],
        ),
      ),
    );
  }
}

/// Bottom sheet that visualises download and install progress for a soft
/// update.
class DownloadProgressSheet extends StatelessWidget {
  const DownloadProgressSheet({
    super.key,
    required this.release,
    required this.controller,
  });

  final ReleaseInfo release;
  final UpdateFlowController controller;

  /// Shows the sheet as a modal bottom sheet.
  static Future<void> show(
    BuildContext context, {
    required ReleaseInfo release,
    required UpdateFlowController controller,
  }) {
    return showModalBottomSheet<void>(
      context: context,
      isDismissible: true,
      showDragHandle: true,
      builder: (BuildContext ctx) => DownloadProgressSheet(
        release: release,
        controller: controller,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final TextTheme text = Theme.of(context).textTheme;
    final ColorScheme colors = Theme.of(context).colorScheme;

    return AnimatedBuilder(
      animation: controller,
      builder: (BuildContext context, Widget? _) {
        final bool failed = controller.stage == UpdateFlowStage.failed;
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Text(
                  'Updating to ${release.latestVersion}',
                  style: text.titleLarge,
                ),
                const SizedBox(height: 16),
                _ProgressBlock(controller: controller),
                if (failed) ...<Widget>[
                  const SizedBox(height: 16),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: colors.errorContainer,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      controller.error ?? 'An unknown error occurred.',
                      style: text.bodySmall?.copyWith(
                        color: colors.onErrorContainer,
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  FilledButton.icon(
                    onPressed: () => unawaited(controller.start(release)),
                    icon: const Icon(Icons.refresh),
                    label: const Text('Retry'),
                  ),
                ],
                if (controller.stage == UpdateFlowStage.installing) ...<Widget>[
                  const SizedBox(height: 16),
                  Text(
                    'The system installer has taken over. Follow the on-screen '
                    'prompt to finish updating.',
                    style: text.bodySmall,
                  ),
                ],
                if (controller.isBusy) ...<Widget>[
                  const SizedBox(height: 8),
                  Align(
                    alignment: Alignment.centerRight,
                    child: TextButton(
                      onPressed: () => unawaited(controller.cancel()),
                      child: const Text('Cancel'),
                    ),
                  ),
                ],
              ],
            ),
          ),
        );
      },
    );
  }
}

class _ProgressBlock extends StatelessWidget {
  const _ProgressBlock({required this.controller});

  final UpdateFlowController controller;

  @override
  Widget build(BuildContext context) {
    final TextTheme text = Theme.of(context).textTheme;
    final DownloadProgress? p = controller.progress;
    final double? fraction = controller.fraction;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Row(
          children: <Widget>[
            Expanded(
              child: Text(
                controller.stage.label,
                style: text.labelLarge,
              ),
            ),
            if (p != null && controller.stage == UpdateFlowStage.downloading)
              Text(
                p.totalBytes > 0
                    ? '${formatBytes(p.receivedBytes)} / '
                        '${formatBytes(p.totalBytes)}'
                    : formatBytes(p.receivedBytes),
                style: text.bodySmall,
              ),
          ],
        ),
        const SizedBox(height: 8),
        if (fraction == null)
          const LinearProgressIndicator()
        else
          LinearProgressIndicator(value: fraction),
        if (p?.percent != null &&
            controller.stage == UpdateFlowStage.downloading) ...<Widget>[
          const SizedBox(height: 4),
          Text('${p!.percent}%', style: text.bodySmall),
        ],
        if (controller.install?.note != null) ...<Widget>[
          const SizedBox(height: 8),
          Text(controller.install!.note!, style: text.bodySmall),
        ],
      ],
    );
  }
}

class _VersionRow extends StatelessWidget {
  const _VersionRow({required this.installed, required this.latest});

  final String installed;
  final String latest;

  @override
  Widget build(BuildContext context) {
    final ColorScheme colors = Theme.of(context).colorScheme;
    final TextTheme text = Theme.of(context).textTheme;

    Widget chip(String label, String value, bool highlight) {
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: highlight ? colors.primaryContainer : colors.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(10),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            Text(label, style: text.labelSmall),
            const SizedBox(height: 2),
            Text(
              value,
              style: text.titleSmall?.copyWith(
                color: highlight ? colors.onPrimaryContainer : null,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      );
    }

    return Row(
      children: <Widget>[
        chip('Installed', installed, false),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8),
          child: Icon(Icons.arrow_forward, size: 18, color: colors.outline),
        ),
        chip('Available', latest, true),
      ],
    );
  }
}

class _ChangelogText extends StatelessWidget {
  const _ChangelogText(this.changelog);

  final String changelog;

  @override
  Widget build(BuildContext context) {
    final List<String> lines = changelog
        .split('\n')
        .map((String l) => l.trim())
        .where((String l) => l.isNotEmpty)
        .toList(growable: false);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: lines
            .map(
              (String line) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 2),
                child: Text(
                  line.startsWith('•') || line.startsWith('-')
                      ? line
                      : '• $line',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ),
            )
            .toList(growable: false),
      ),
    );
  }
}

/// Formats a byte count as `B`, `KB`, `MB` or `GB`.
String formatBytes(int bytes) {
  if (bytes < 1024) {
    return '$bytes B';
  }
  const List<String> units = <String>['KB', 'MB', 'GB', 'TB'];
  double value = bytes / 1024;
  int unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return '${value.toStringAsFixed(value >= 100 ? 0 : 1)} ${units[unit]}';
}

/// Whether the running platform can install updates through `ota_update`.
bool get supportsNativeInstall =>
    Platform.isAndroid || Platform.isIOS;