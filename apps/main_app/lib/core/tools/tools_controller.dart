// Tool state for the UI.
//
// The tool screen is the *deterministic* half of the assistant: it runs a
// capability on demand, with arguments the user can see and edit, and shows the
// raw result. Keeping it separate from the chat transcript matters, because when
// the model misbehaves the user needs a way to check the device reading itself
// without going through the model at all.

import 'package:flutter/foundation.dart';
import 'package:harbor_core/harbor_core.dart';

import '../platform/platform_bridge.dart';

/// One recorded invocation.
class ToolInvocation {
  /// Creates a record.
  ToolInvocation({
    required this.call,
    required this.result,
    required this.elapsed,
  });

  /// What was asked for.
  final ToolCall call;

  /// What came back.
  final ToolResult result;

  /// How long it took.
  final Duration elapsed;

  /// When it happened.
  final DateTime at = DateTime.now();

  /// JSON form for the transcript.
  Map<String, Object?> toJson() => <String, Object?>{
        'name': call.name,
        'arguments': call.arguments,
        'ok': result.ok,
        'summary': result.summary,
        'elapsed_ms': elapsed.inMilliseconds,
        'at': at.toUtc().toIso8601String(),
      };
}

/// Builds a [HostCaller] that forwards to the Android platform channel.
///
/// Returning `null` (rather than throwing) when the platform cannot answer is
/// the documented contract the tool layer understands: it becomes an
/// "unsupported" result, which the UI renders differently from a failure. A
/// missing battery gauge on desktop is not an error.
HostCaller hostCallerFrom(PlatformBridge bridge) {
  return (String method, Map<String, Object?> args) async {
    if (!bridge.isSupported) {
      return null;
    }
    return bridge.call(method, args);
  };
}

/// Runs tools and remembers what happened.
class ToolsController extends ChangeNotifier {
  /// Creates a controller over [registry].
  ToolsController({
    required this.registry,
    required this.bridge,
    this.maxHistory = 50,
  });

  /// The available tools.
  final ToolRegistry registry;

  /// The platform channel, for the capability banner.
  final PlatformBridge bridge;

  /// How many invocations to remember.
  final int maxHistory;

  final List<ToolInvocation> _history = <ToolInvocation>[];
  bool _busy = false;
  ToolInvocation? _last;

  /// Known tools, catalogue order.
  List<Tool> get tools => registry.tools;

  /// Recent invocations, newest first.
  List<ToolInvocation> get history => List<ToolInvocation>.unmodifiable(_history);

  /// Whether a call is in flight.
  bool get busy => _busy;

  /// The most recent invocation, if any.
  ToolInvocation? get last => _last;

  /// Whether the host platform can answer device questions.
  bool get platformSupported => bridge.isSupported;

  /// Runs [name] with [arguments] and records the outcome.
  Future<ToolResult> invoke(
    String name, [
    Map<String, Object?> arguments = const <String, Object?>{},
  ]) async {
    final ToolCall call = ToolCall(name, arguments);
    _busy = true;
    notifyListeners();
    final Stopwatch stopwatch = Stopwatch()..start();
    ToolResult result;
    try {
      result = await registry.invoke(call);
    } on Object catch (error) {
      result = ToolResult.failure('$name failed: $error');
    }
    stopwatch.stop();
    _busy = false;
    _last = ToolInvocation(
      call: call,
      result: result,
      elapsed: stopwatch.elapsed,
    );
    _history.insert(0, _last!);
    if (_history.length > maxHistory) {
      _history.removeRange(maxHistory, _history.length);
    }
    notifyListeners();
    return result;
  }

  /// Forgets the invocation history.
  void clearHistory() {
    _history.clear();
    _last = null;
    notifyListeners();
  }
}