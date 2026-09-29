// The tool catalogue: a name-indexed registry plus the call envelope.

import 'tool.dart';

/// A request to run a named tool with arguments.
class ToolCall {
  /// Creates a call.
  const ToolCall(this.name, [this.arguments = const <String, Object?>{}]);

  /// The tool to run.
  final String name;

  /// The arguments to pass it.
  final Map<String, Object?> arguments;

  /// JSON form.
  Map<String, Object?> toJson() => <String, Object?>{
        'name': name,
        'arguments': arguments,
      };

  /// Tolerant parser for [toJson].
  ///
  /// Returns null instead of throwing on any malformed shape, because the input
  /// may be model-generated text that has been through a regex extractor. A
  /// parse failure must degrade to "the model said something that was not a
  /// tool call", never to a crashed turn.
  static ToolCall? fromJson(Object? json) {
    if (json is! Map<Object?, Object?>) {
      return null;
    }
    final Object? name = json['name'];
    if (name is! String || name.trim().isEmpty) {
      return null;
    }
    final Object? rawArguments = json['arguments'];
    final Map<String, Object?> arguments;
    if (rawArguments is Map<Object?, Object?>) {
      arguments = <String, Object?>{
        for (final MapEntry<Object?, Object?> entry in rawArguments.entries)
          if (entry.key is String) entry.key! as String: entry.value,
      };
    } else {
      arguments = const <String, Object?>{};
    }
    return ToolCall(name.trim(), arguments);
  }

  @override
  String toString() => 'ToolCall($name, $arguments)';
}

/// A name-indexed collection of [Tool]s.
class ToolRegistry {
  /// Builds a registry, optionally pre-populated.
  ///
  /// Registering a duplicate name throws rather than silently shadowing: a
  /// shadowed tool is unreachable, and an unreachable advertised tool is exactly
  /// the kind of dead code that makes an assistant look broken.
  ToolRegistry([Iterable<Tool> tools = const <Tool>[]]) {
    for (final Tool tool in tools) {
      register(tool);
    }
  }

  final Map<String, Tool> _byName = <String, Tool>{};

  /// Adds [tool]; throws [ArgumentError] if its name is already taken.
  void register(Tool tool) {
    if (tool.name.trim().isEmpty) {
      throw ArgumentError('a tool must have a non-empty name');
    }
    if (_byName.containsKey(tool.name)) {
      throw ArgumentError('a tool named "${tool.name}" is already registered');
    }
    _byName[tool.name] = tool;
  }

  /// The tool called [name], or null.
  Tool? lookup(String name) => _byName[name];

  /// Every registered tool, in registration order.
  List<Tool> get tools => List<Tool>.unmodifiable(_byName.values);

  /// Number of registered tools.
  int get length => _byName.length;

  /// Declarations for the UI and the prompt builder.
  List<Map<String, Object?>> describe() =>
      <Map<String, Object?>>[for (final Tool tool in _byName.values) tool.describe()];

  /// Runs [call].
  ///
  /// An unknown name is a failure result, not an exception, so a hallucinated
  /// tool name surfaces in the transcript as a normal error.
  Future<ToolResult> invoke(ToolCall call) async {
    final Tool? tool = _byName[call.name];
    if (tool == null) {
      return ToolResult.failure(
        'unknown tool "${call.name}"',
      );
    }
    return tool.invoke(call.arguments);
  }

  /// Runs the tool called [name] with [arguments].
  Future<ToolResult> invokeByName(
    String name, [
    Map<String, Object?> arguments = const <String, Object?>{},
  ]) {
    return invoke(ToolCall(name, arguments));
  }
}