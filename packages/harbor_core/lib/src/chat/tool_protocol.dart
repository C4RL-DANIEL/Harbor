// Tool protocol: how the model asks for a capability, and how the app decides
// what the user meant.
//
// Two mechanisms exist because they have different failure modes:
//
//  * `<tool name="...">{...}</tool>` in the model's own output is the intended
//    interface. It is general — any registered tool can be reached — but a small
//    model emits it unreliably, and a malformed call is easy to produce.
//  * [ToolProtocol.routeIntent] is a conservative, deterministic matcher over
//    the user's text. It cannot do anything the model cannot, but when it fires
//    it is always right about *which* tool the user wanted, so a request like
//    "is my phone charging" works on day one, before the model has learned
//    anything.
//
// Both are deliberately narrow. A confident wrong tool call is worse than no
// tool call: the user sees a fabricated fact about their own device, which is
// the one category of answer an on-device assistant must never get wrong.

import 'dart:convert';

import '../tools/tool.dart';
import '../tools/tool_registry.dart';

/// Parsing and routing for the assistant's tool-call syntax.
class ToolProtocol {
  const ToolProtocol._();

  /// The call syntax: `<tool name="device.battery">{}</tool>`.
  ///
  /// The body is matched lazily so two adjacent calls do not swallow each other,
  /// and `[\s\S]` rather than `.` so a pretty-printed multi-line argument object
  /// still parses. `dotAll` is not used because the body may legitimately span
  /// lines and the surrounding text should not.
  static final RegExp _callPattern = RegExp(
    r'<tool\s+name\s*=\s*"([A-Za-z0-9_.\-]+)"\s*>([\s\S]*?)</tool>',
  );

  /// Phrases that mark a question about *how something works* rather than a
  /// request to read the device.
  ///
  /// This single guard is what stops "how does the battery reporting work" from
  /// being answered with the battery level, which is the most annoying possible
  /// failure of a keyword matcher.
  static final RegExp _explanation = RegExp(
    r'\b(why|explain|tutorial|documentation|for example|how (does|do|did|is|are)|'
    r'what is a|what are|what does)\b',
  );

  /// Verbs and openers that make a sentence a request rather than a statement.
  static final RegExp _requestShape = RegExp(
    r'(\?|\b(what|which|how much|how many|is|are|am|do|does|can|could|would|'
    r'show|tell|check|get|give|read|list|copy|open|vibrate|please|share)\b)',
  );

  /// The first well-formed call in [text], or `null` when there is none.
  ///
  /// Never throws. A body that is present but not a JSON object yields a call
  /// with no arguments rather than no call at all: the tool's own validator then
  /// reports exactly which argument was missing, which is far easier to debug
  /// than a silently ignored request.
  static ToolCall? parse(String text) {
    final RegExpMatch? match = _callPattern.firstMatch(text);
    if (match == null) {
      return null;
    }
    final String name = match.group(1)!;
    final String body = match.group(2)!.trim();
    if (body.isEmpty) {
      return ToolCall(name);
    }
    try {
      final Object? decoded = jsonDecode(body);
      if (decoded is Map<String, Object?>) {
        return ToolCall(name, decoded);
      }
      if (decoded is Map) {
        return ToolCall(
          name,
          <String, Object?>{
            for (final MapEntry<Object?, Object?> entry in decoded.entries)
              '${entry.key}': entry.value,
          },
        );
      }
    } on FormatException {
      // Fall through to the empty-argument form.
    }
    return ToolCall(name);
  }

  /// Every call removed from [text], with the leftover whitespace collapsed.
  ///
  /// Collapsing matters because the model usually emits the call on its own line
  /// inside an otherwise complete sentence; leaving the blank line behind makes
  /// the transcript look broken.
  static String stripCalls(String text) {
    return text
        .replaceAll(_callPattern, '')
        .replaceAll(RegExp(r'[ \t]+\n'), '\n')
        .replaceAll(RegExp(r'\n{3,}'), '\n\n')
        .trim();
  }

  /// Whether [text] asks for something the device can answer.
  ///
  /// Returns `null` unless a topic keyword *and* request-shaped phrasing are
  /// both present and the text does not look like a request for an explanation.
  static ToolCall? routeIntent(String text) {
    final String value = text.trim();
    if (value.isEmpty || _explanation.hasMatch(value.toLowerCase())) {
      return null;
    }
    final String lower = value.toLowerCase();
    if (!_requestShape.hasMatch(lower)) {
      return null;
    }

    // A copy command carries its own payload, so it is matched first and by
    // shape rather than by keyword.
    final RegExpMatch? copy = RegExp(
      r'^\s*copy\s+(.+?)(?:\s+to\s+the\s+clipboard)?\s*[.!]?\s*$',
      caseSensitive: false,
    ).firstMatch(value);
    if (copy != null) {
      final String payload = copy.group(1)!.trim();
      if (payload.isNotEmpty) {
        return ToolCall('device.clipboard.write', <String, Object?>{
          'text': payload,
        });
      }
    }

    if (_has(lower, <String>['battery', 'charging', 'charge level', 'battery level'])) {
      return const ToolCall('device.battery');
    }
    if (_has(lower, <String>['storage', 'disk space', 'free space', 'free up space'])) {
      return const ToolCall('device.storage');
    }
    if (_has(lower, <String>['thermal', 'how hot', 'temperature', 'overheat'])) {
      return const ToolCall('device.thermal');
    }
    if (_has(
      lower,
      <String>[
        'device model',
        'device info',
        'model number',
        'which phone',
        'what phone',
        'android version',
      ],
    )) {
      return const ToolCall('device.info');
    }
    if (_has(lower, <String>['online', 'internet', 'connection', 'connected', 'wifi', 'wi-fi'])) {
      return const ToolCall('device.connectivity');
    }
    if (_has(lower, <String>['screen', 'display', 'resolution', 'brightness'])) {
      return const ToolCall('device.display');
    }
    if (_has(lower, <String>['locale', 'timezone', 'time zone', 'what language'])) {
      return const ToolCall('device.locale');
    }
    if (_has(lower, <String>['installed app', 'installed apps', 'list apps', 'what apps'])) {
      return const ToolCall('device.installedApps');
    }
    if (_has(lower, <String>['clipboard', 'paste'])) {
      return const ToolCall('device.clipboard.read');
    }
    if (_has(lower, <String>['vibrate', 'vibration'])) {
      return const ToolCall('device.vibrate');
    }

    final RegExpMatch? url = RegExp(
      r'\b(https?://[^\s<>"\)]+)',
      caseSensitive: false,
    ).firstMatch(value);
    if (url != null && lower.contains('open')) {
      return ToolCall('device.openUrl', <String, Object?>{'url': url.group(1)!});
    }

    final RegExpMatch? app = RegExp(
      r'\bopen\s+(?:the\s+)?([a-z0-9_.]+)\s+app\b',
    ).firstMatch(lower);
    if (app != null) {
      return ToolCall('device.openApp', <String, Object?>{'package': app.group(1)!});
    }

    return null;
  }

  /// A compact catalogue of [tools], one line each, for the prompt.
  ///
  /// [maxTools] bounds the prompt because the catalogue is re-sent every turn:
  /// an unbounded list would crowd out the conversation it is meant to serve.
  static String describeTools(List<Tool> tools, {int maxTools = 24}) {
    final StringBuffer buffer = StringBuffer();
    int written = 0;
    for (final Tool tool in tools) {
      if (written >= maxTools) {
        buffer.writeln('- … ${tools.length - written} more');
        break;
      }
      final StringBuffer arguments = StringBuffer();
      final Map<String, Object?> schema = tool.parameters;
      final Object? properties = schema['properties'];
      final Set<String> required = <String>{
        if (schema['required'] is List)
          for (final Object? entry in schema['required']! as List<Object?>)
            '$entry',
      };
      if (properties is Map) {
        for (final MapEntry<Object?, Object?> entry in properties.entries) {
          final String name = '${entry.key}';
          String type = 'any';
          if (entry.value is Map) {
            final Object? declared = (entry.value! as Map)['type'];
            if (declared is String) {
              type = declared;
            }
          }
          if (arguments.isNotEmpty) {
            arguments.write(', ');
          }
          arguments.write(required.contains(name) ? '$name: $type' : '[$name: $type]');
        }
      }
      buffer.writeln('- ${tool.name}($arguments): ${tool.description}');
      written++;
    }
    return buffer.toString().trimRight();
  }

  static bool _has(String haystack, List<String> needles) {
    for (final String needle in needles) {
      if (haystack.contains(needle)) {
        return true;
      }
    }
    return false;
  }
}