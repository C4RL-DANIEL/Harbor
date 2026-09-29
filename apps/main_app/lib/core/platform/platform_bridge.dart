// Platform bridge — the typed Dart façade over the Android native layer.
//
// The native side (PlatformChannelHandler.kt) exposes a single method-channel
// entry point, `call`, whose argument is `{"method": String, "args": Map}` and
// whose reply is a `Map<String, Object?>`. One entry point keeps the Kotlin
// dispatcher small and makes the contract trivially testable from Dart; this
// file is the only place that knows the method-name strings, and each name is
// wrapped in a typed convenience method so callers never assemble a raw map.
//
// Error policy:
//   * `call` throws [PlatformBridgeException] for a missing handler
//     (`MissingPluginException`, e.g. in a widget test or on desktop) or for a
//     native failure (`PlatformException`, code `platform_error` /
//     `unsupported_method`). Tools surface that message directly.
//   * The map-returning convenience wrappers degrade instead of throwing when
//     the bridge is not supported: they resolve to `{"unavailable": true}`.
//     The UI calls several of these speculatively (battery, thermal, storage)
//     and a thrown exception there would turn a best-effort readout into an
//     error card, so the wrappers return an explicit sentinel the caller can
//     check. Scalar wrappers return the safe default (`false`) and the
//     fire-and-forget training wrappers are no-ops.
//
// This file imports `dart:io` on purpose: it is Android-app-only code and is
// never compiled for web.

import 'dart:io' show Platform;

import 'package:flutter/services.dart';

/// Thrown when a platform-channel call cannot be completed.
///
/// [method] is the bridge method name (`device.battery`, `training.start`, …),
/// [code] is the native error code when the platform reported one, and
/// [message] is a human-readable description suitable for a tool error.
class PlatformBridgeException implements Exception {
  PlatformBridgeException(this.method, this.message, {this.code});

  /// Bridge method that failed, e.g. `device.info`.
  final String method;

  /// Readable failure description.
  final String message;

  /// Native error code (`platform_error`, `unsupported_method`), if any.
  final String? code;

  @override
  String toString() {
    final String codeSuffix = code == null ? '' : ' [$code]';
    return 'PlatformBridgeException ($method)$codeSuffix: $message';
  }
}

/// Typed Dart access to the `com.harbor.main_app/platform` method channel.
class PlatformBridge {
  /// Wraps [channel], or the real platform channel when it is omitted.
  ///
  /// The injectable channel exists so widget tests can install a mock handler.
  PlatformBridge({MethodChannel? channel})
      : _channel = channel ?? defaultChannel;

  /// The channel the native [PlatformChannelHandler] registers itself on.
  static const MethodChannel defaultChannel =
      MethodChannel('com.harbor.main_app/platform');

  /// Computed once, lazily, from the host OS.
  static final bool _isAndroid = Platform.isAndroid;

  final MethodChannel _channel;

  /// Whether a native handler can be expected to answer.
  ///
  /// False on every platform other than Android, and in widget tests (which run
  /// on the host, so `Platform.isAndroid` is false). Tools consult this so they
  /// can return a clear "unsupported on this platform" error instead of
  /// surfacing a [MissingPluginException] from deep inside a call.
  bool get isSupported => _isAndroid;

  /// Invokes the single native `call` entry point with [method] and [args].
  ///
  /// Returns the decoded reply map. Throws [PlatformBridgeException] when the
  /// channel has no handler or the native side reports an error.
  Future<Map<String, Object?>> call(
    String method, [
    Map<String, Object?> args = const <String, Object?>{},
  ]) async {
    try {
      final Map<Object?, Object?>? raw =
          await _channel.invokeMethod<Map<Object?, Object?>>(
        'call',
        <String, Object?>{'method': method, 'args': args},
      );
      if (raw == null) {
        throw PlatformBridgeException(
          method,
          'the platform returned no result for "$method"',
        );
      }
      return <String, Object?>{
        for (final MapEntry<Object?, Object?> entry in raw.entries)
          entry.key.toString(): entry.value,
      };
    } on MissingPluginException catch (e) {
      throw PlatformBridgeException(
        method,
        'no handler is registered for "${defaultChannel.name}": '
        '${e.message ?? 'MissingPluginException'}',
      );
    } on PlatformException catch (e) {
      throw PlatformBridgeException(method, e.message ?? e.code, code: e.code);
    }
  }

  // ---------------------------------------------------------------------------
  // device.*
  // ---------------------------------------------------------------------------

  /// Manufacturer, model, ABI list, SDK level, emulator heuristic and locale.
  Future<Map<String, Object?>> deviceInfo() => _mapCall('device.info');

  /// Battery level/charging state/temperature/health. Degrades to
  /// `{"unavailable": true}` off-device.
  Future<Map<String, Object?>> battery() => _mapCall('device.battery');

  /// Internal/external storage byte counts and this app's files usage.
  Future<Map<String, Object?>> storage() => _mapCall('device.storage');

  /// Active network type, metering and VPN state.
  Future<Map<String, Object?>> connectivity() => _mapCall('device.connectivity');

  /// Width/height in pixels, density, refresh rate and font scale.
  Future<Map<String, Object?>> display() => _mapCall('device.display');

  /// Coarse thermal status (`none`…`shutdown`, or `unknown` below API 29).
  Future<Map<String, Object?>> thermal() => _mapCall('device.thermal');

  /// Language/country, time zone and 12/24-hour preference.
  Future<Map<String, Object?>> locale() => _mapCall('device.locale');

  /// Reads the current clipboard text (empty string when the clip is empty).
  Future<Map<String, Object?>> readClipboard() =>
      _mapCall('device.clipboard.read');

  /// Writes [text] to the clipboard.
  Future<Map<String, Object?>> writeClipboard(String text) =>
      _mapCall('device.clipboard.write', <String, Object?>{'text': text});

  /// Posts a one-shot notification with the platform notification channel.
  Future<Map<String, Object?>> notify({
    required int id,
    required String title,
    required String body,
    bool ongoing = false,
  }) =>
      _mapCall('device.notify', <String, Object?>{
        'id': id,
        'title': title,
        'body': body,
        'ongoing': ongoing,
      });

  /// Vibrates for [millis] milliseconds.
  Future<Map<String, Object?>> vibrate(int millis) =>
      _mapCall('device.vibrate', <String, Object?>{'millis': millis});

  /// Shows a short toast on the main thread.
  Future<Map<String, Object?>> toast(String text) =>
      _mapCall('device.toast', <String, Object?>{'text': text});

  /// Opens the system share chooser for [text] with an optional [subject].
  Future<Map<String, Object?>> share({
    required String text,
    String? subject,
  }) =>
      _mapCall('device.share', <String, Object?>{
        'text': text,
        if (subject != null) 'subject': subject,
      });

  /// Lists launcher activities, optionally including system packages.
  ///
  /// The native side resolves this through `queryIntentActivities` rather than
  /// requesting `QUERY_ALL_PACKAGES`, so only launcher-visible packages appear.
  Future<Map<String, Object?>> installedApps({bool includeSystem = false}) =>
      _mapCall('device.installedApps', <String, Object?>{
        'includeSystem': includeSystem,
      });

  /// Opens [url] with an `ACTION_VIEW` intent.
  Future<Map<String, Object?>> openUrl(String url) =>
      _mapCall('device.openUrl', <String, Object?>{'url': url});

  /// Launches [package] through its launcher intent.
  Future<Map<String, Object?>> openApp(String package) =>
      _mapCall('device.openApp', <String, Object?>{'package': package});

  // ---------------------------------------------------------------------------
  // notifications.*
  // ---------------------------------------------------------------------------

  /// Whether `POST_NOTIFICATIONS` is granted (always true below API 33).
  Future<bool> notificationsGranted() async {
    if (!isSupported) {
      return false;
    }
    final Map<String, Object?> reply = await call('notifications.permission');
    return reply['granted'] == true;
  }

  /// Requests `POST_NOTIFICATIONS` and returns the *current* answer.
  ///
  /// The system dialog is asynchronous and cannot be awaited from a channel
  /// call, so this returns the grant state at the moment of the request; call
  /// [notificationsGranted] again after the dialog is dismissed.
  Future<bool> requestNotifications() async {
    if (!isSupported) {
      return false;
    }
    final Map<String, Object?> reply = await call('notifications.request');
    return reply['granted'] == true;
  }

  // ---------------------------------------------------------------------------
  // training.* — drives TrainingService, which survives backgrounding
  // ---------------------------------------------------------------------------

  /// Starts (or restarts) the foreground training notification.
  Future<void> startTrainingNotification({
    required String title,
    required String text,
  }) async {
    if (!isSupported) {
      return;
    }
    await call('training.start', <String, Object?>{
      'title': title,
      'text': text,
    });
  }

  /// Updates the live progress of the foreground training notification.
  Future<void> updateTrainingNotification({
    required int progress,
    required int max,
    required String text,
  }) async {
    if (!isSupported) {
      return;
    }
    await call('training.update', <String, Object?>{
      'progress': progress,
      'max': max,
      'text': text,
    });
  }

  /// Removes the foreground training notification and stops the service.
  Future<void> stopTrainingNotification() async {
    if (!isSupported) {
      return;
    }
    await call('training.stop');
  }

  /// Shared implementation of the map-returning wrappers.
  ///
  /// Off-platform this returns `{"unavailable": true}` rather than throwing:
  /// the callers are speculative readouts and an explicit sentinel is easier to
  /// render than an exception. Real native failures still propagate as
  /// [PlatformBridgeException] so a broken device API is never hidden.
  Future<Map<String, Object?>> _mapCall(
    String method, [
    Map<String, Object?> args = const <String, Object?>{},
  ]) {
    if (!isSupported) {
      return Future<Map<String, Object?>>.value(
        const <String, Object?>{'unavailable': true},
      );
    }
    return call(method, args);
  }
}