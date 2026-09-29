// The tool abstraction: the unit of "something the assistant can actually do".
//
// A tool is deliberately split into a *declaration* (name, prose description,
// JSON-Schema-style parameters) and an *implementation* (`invoke`). The
// declaration is what the UI renders and what a prompt embeds; the
// implementation is the only part that touches anything outside the process.
// Keeping them in one object means a tool cannot be advertised without also
// being runnable, which is the failure mode that makes an assistant's tool list
// a lie.

import 'dart:convert';

/// Signature a host supplies to execute a platform-backed tool.
///
/// The Android build implements this with a `MethodChannel`; the web build
/// implements it by returning a clear "unsupported" error; tests implement it
/// with a closure that records its calls. Nothing in this package imports
/// `dart:io` or Flutter, so this indirection is what keeps the tool catalogue
/// portable.
typedef HostCaller = Future<Object?> Function(
  String method,
  Map<String, Object?> args,
);

/// The outcome of running a tool.
class ToolResult {
  /// A successful result carrying [data], which must be JSON-encodable.
  const ToolResult.ok(this.data)
      : ok = true,
        error = null;

  /// A failure with a human-readable [error].
  const ToolResult.failure(this.error)
      : ok = false,
        data = null;

  /// A failure that specifically means "this capability does not exist here".
  ///
  /// Distinct from [ToolResult.failure] only in intent: the UI renders it as a
  /// neutral state rather than an error, because a missing accelerometer on a
  /// desktop test host is not a bug.
  const ToolResult.unsupported(this.error)
      : ok = false,
        data = null;

  /// Whether the tool ran successfully.
  final bool ok;

  /// The JSON-encodable result payload, or null on failure.
  final Object? data;

  /// The failure message, or null on success.
  final String? error;

  /// JSON form, used by the chat transcript and the diagnostics panel.
  Map<String, Object?> toJson() => <String, Object?>{
        'ok': ok,
        if (data != null) 'data': data,
        if (error != null) 'error': error,
      };

  /// A single line suitable for a chat transcript or a log row.
  ///
  /// Kept short on purpose: the diagnostics panel shows the full JSON payload,
  /// so the summary only has to be recognisable at a glance.
  String get summary {
    if (!ok) {
      return 'error: $error';
    }
    final Object? payload = data;
    if (payload == null) {
      return 'ok';
    }
    if (payload is Map) {
      final String rendered = payload.entries
          .take(4)
          .map((MapEntry<Object?, Object?> e) => '${e.key}=${_short(e.value)}')
          .join(', ');
      return payload.length > 4 ? 'ok: $rendered, …' : 'ok: $rendered';
    }
    return 'ok: ${_short(payload)}';
  }

  static String _short(Object? value) {
    final String text = value is String ? value : jsonEncode(value);
    final String flat = text.replaceAll('\n', ' ');
    return flat.length <= 48 ? flat : '${flat.substring(0, 45)}…';
  }

  @override
  String toString() => 'ToolResult($summary)';
}

/// A capability the assistant can invoke.
abstract class Tool {
  /// Const constructor so subclasses can be `const` and a tool catalogue can be
  /// built without allocating. Declared here because a const subclass
  /// constructor cannot call a non-const super constructor.
  const Tool();

  /// Dotted, stable identifier, e.g. `device.battery`.
  String get name;

  /// One sentence shown in the UI and embedded in prompts.
  String get description;

  /// JSON-Schema-style description of the accepted arguments.
  ///
  /// Shape: `{'type': 'object', 'properties': {...}, 'required': [...]}`.
  /// Only `type`, `properties`, `required` and `enum` are interpreted, which is
  /// all the validator and the prompt renderer need.
  Map<String, Object?> get parameters;

  /// True when invoking changes state the user can observe.
  ///
  /// Mutating tools are the ones the UI asks about before running, and the ones
  /// an autonomous caller should not fire speculatively.
  bool get mutating;

  /// Runs the tool. Implementations must not throw: every failure mode is a
  /// [ToolResult].
  Future<ToolResult> invoke(Map<String, Object?> args);

  /// Declaration as JSON, for the UI and the prompt builder.
  Map<String, Object?> describe() => <String, Object?>{
        'name': name,
        'description': description,
        'parameters': parameters,
        'mutating': mutating,
      };
}

/// A [Tool] whose only interesting part is `run`.
///
/// [invoke] validates the arguments against [parameters] first, converts any
/// thrown error into a failure result, and guarantees a [ToolResult] is always
/// returned. A tool therefore cannot crash a chat turn or the agent loop by
/// throwing — the worst it can do is report a failure.
abstract class FunctionTool extends Tool {
  /// Const constructor for the subclasses' `const` instances.
  const FunctionTool();

  /// The implementation, called only after [parameters] has been satisfied.
  Future<Object?> run(Map<String, Object?> args);

  @override
  Future<ToolResult> invoke(Map<String, Object?> args) async {
    final String? problem = validateArguments(parameters, args);
    if (problem != null) {
      return ToolResult.failure('$name: $problem');
    }
    try {
      return ToolResult.ok(await run(args));
    } catch (error) {
      return ToolResult.failure('$name: $error');
    }
  }
}

/// Checks [args] against a JSON-Schema-style [parameters] block.
///
/// Returns null when the arguments are acceptable, otherwise a message naming
/// the offending key. Unknown keys are permitted so that a caller passing an
/// extra field is not punished for a schema that has not been updated.
String? validateArguments(
  Map<String, Object?> parameters,
  Map<String, Object?> args,
) {
  final Object? required = parameters['required'];
  if (required is List<Object?>) {
    for (final Object? key in required) {
      if (key is String && args[key] == null) {
        return 'missing required argument "$key"';
      }
    }
  }
  final Object? properties = parameters['properties'];
  if (properties is Map<Object?, Object?>) {
    for (final MapEntry<Object?, Object?> entry in properties.entries) {
      final Object? key = entry.key;
      final Object? value = args[key];
      if (value == null) {
        continue;
      }
      final Object? spec = entry.value;
      if (spec is! Map<Object?, Object?>) {
        continue;
      }
      final Object? expectedType = spec['type'];
      if (expectedType is! String) {
        continue;
      }
      if (!_matchesType(expectedType, value)) {
        return 'argument "$key" must be a $expectedType, got '
            '${value.runtimeType}';
      }
      final Object? allowed = spec['enum'];
      if (allowed is List<Object?> && !allowed.contains(value)) {
        return 'argument "$key" must be one of $allowed';
      }
    }
  }
  return null;
}

bool _matchesType(String expectedType, Object value) {
  switch (expectedType) {
    case 'string':
      return value is String;
    case 'integer':
      return value is int;
    case 'number':
      return value is num;
    case 'boolean':
      return value is bool;
    case 'array':
      return value is List;
    case 'object':
      return value is Map;
    default:
      // An unrecognised type constrains nothing rather than rejecting
      // everything, so a schema written against a newer draft still works.
      return true;
  }
}

/// Reads a required string argument, throwing if it is absent.
///
/// Safe to call inside `run`, because `invoke` has already validated the types.
String requireStringArg(Map<String, Object?> args, String key) {
  final Object? value = args[key];
  if (value is String) {
    return value;
  }
  throw ArgumentError('"$key" must be a string, got ${value.runtimeType}');
}

/// Reads an optional string argument.
String? optionalStringArg(Map<String, Object?> args, String key) {
  final Object? value = args[key];
  if (value == null) {
    return null;
  }
  if (value is String) {
    return value;
  }
  throw ArgumentError('"$key" must be a string, got ${value.runtimeType}');
}

/// Reads a required integer argument.
int requireIntArg(Map<String, Object?> args, String key) {
  final Object? value = args[key];
  if (value is int) {
    return value;
  }
  if (value is num) {
    return value.toInt();
  }
  throw ArgumentError('"$key" must be an integer, got ${value.runtimeType}');
}

/// Reads an optional integer argument, falling back to [fallback].
int optionalIntArg(
  Map<String, Object?> args,
  String key, {
  int fallback = 0,
}) {
  final Object? value = args[key];
  if (value == null) {
    return fallback;
  }
  if (value is int) {
    return value;
  }
  if (value is num) {
    return value.toInt();
  }
  throw ArgumentError('"$key" must be an integer, got ${value.runtimeType}');
}

/// Reads an optional boolean argument, falling back to [fallback].
bool optionalBoolArg(
  Map<String, Object?> args,
  String key, {
  bool fallback = false,
}) {
  final Object? value = args[key];
  if (value == null) {
    return fallback;
  }
  if (value is bool) {
    return value;
  }
  throw ArgumentError('"$key" must be a boolean, got ${value.runtimeType}');
}