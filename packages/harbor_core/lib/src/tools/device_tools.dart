// Platform-backed device tools: one FunctionTool per host method.
//
// The Android side implements a MethodChannel handler against the same table of
// method strings and argument shapes declared here, so this file is a contract
// as much as a catalogue. Keeping the mapping declarative and boring is the
// point: a typo in a method string would only surface as a runtime failure on a
// real device, never at analysis time.

import 'tool.dart';

/// Builds the device tools, forwarding every call to [caller].
///
/// Each tool validates its arguments, maps them onto the documented host
/// argument map, and wraps the host's answer in a [ToolResult]. A caller that
/// throws becomes [ToolResult.failure]; a caller that answers null becomes
/// [ToolResult.unsupported], because "this platform has no such method" is a
/// neutral fact rather than an error the user should be shown in red.
List<Tool> deviceTools(HostCaller caller) {
  return <Tool>[
    _device(
      caller: caller,
      name: 'device.info',
      method: 'device.info',
      description: 'Reports the device model, manufacturer, platform and OS '
          'version.',
      mutating: false,
      buildArgs: _noArgs,
    ),
    _device(
      caller: caller,
      name: 'device.battery',
      method: 'device.battery',
      description: 'Reports the battery level, charging state and temperature.',
      mutating: false,
      buildArgs: _noArgs,
    ),
    _device(
      caller: caller,
      name: 'device.storage',
      method: 'device.storage',
      description: 'Reports total, free and used bytes for internal storage.',
      mutating: false,
      buildArgs: _noArgs,
    ),
    _device(
      caller: caller,
      name: 'device.connectivity',
      method: 'device.connectivity',
      description:
          'Reports the active network type and whether the device is online.',
      mutating: false,
      buildArgs: _noArgs,
    ),
    _device(
      caller: caller,
      name: 'device.display',
      method: 'device.display',
      description: 'Reports the screen width, height, density and refresh rate.',
      mutating: false,
      buildArgs: _noArgs,
    ),
    _device(
      caller: caller,
      name: 'device.thermal',
      method: 'device.thermal',
      description:
          'Reports the current thermal status and whether the device is '
          'throttling.',
      mutating: false,
      buildArgs: _noArgs,
    ),
    _device(
      caller: caller,
      name: 'device.locale',
      method: 'device.locale',
      description:
          'Reports the device locale, language tag and 24-hour clock '
          'preference.',
      mutating: false,
      buildArgs: _noArgs,
    ),
    _device(
      caller: caller,
      name: 'device.clipboard.read',
      method: 'device.clipboard.read',
      description: 'Reads the current text contents of the system clipboard.',
      mutating: false,
      buildArgs: _noArgs,
    ),
    _device(
      caller: caller,
      name: 'device.clipboard.write',
      method: 'device.clipboard.write',
      description:
          'Replaces the system clipboard contents with the given text.',
      mutating: true,
      buildArgs: _clipboardWriteArgs,
      parameters: _deviceSchema(
        properties: <String, Object?>{'text': _stringType},
        required: <String>['text'],
      ),
    ),
    _device(
      caller: caller,
      name: 'device.notify',
      method: 'device.notify',
      description: 'Posts a system notification with the given title and body.',
      mutating: true,
      buildArgs: _notifyArgs,
      parameters: _deviceSchema(
        properties: <String, Object?>{
          'id': _notificationIdType,
          'title': _stringType,
          'body': _stringType,
          'ongoing': _booleanType,
        },
        required: <String>['title', 'body'],
      ),
    ),
    _device(
      caller: caller,
      name: 'device.vibrate',
      method: 'device.vibrate',
      description: 'Vibrates the device for the given number of milliseconds.',
      mutating: true,
      buildArgs: _vibrateArgs,
      parameters: _deviceSchema(
        properties: <String, Object?>{'millis': _vibrateMillisType},
      ),
    ),
    _device(
      caller: caller,
      name: 'device.toast',
      method: 'device.toast',
      description: 'Shows a brief toast message on the screen.',
      mutating: true,
      buildArgs: _toastArgs,
      parameters: _deviceSchema(
        properties: <String, Object?>{'text': _stringType},
        required: <String>['text'],
      ),
    ),
    _device(
      caller: caller,
      name: 'device.share',
      method: 'device.share',
      description: 'Opens the system share sheet with the given text.',
      mutating: true,
      buildArgs: _shareArgs,
      parameters: _deviceSchema(
        properties: <String, Object?>{
          'text': _stringType,
          'subject': _stringType,
        },
        required: <String>['text'],
      ),
    ),
    _device(
      caller: caller,
      name: 'device.installedApps',
      method: 'device.installedApps',
      description: 'Lists the installed application package names.',
      mutating: false,
      buildArgs: _installedAppsArgs,
      parameters: _deviceSchema(
        properties: <String, Object?>{'includeSystem': _booleanType},
      ),
    ),
    _device(
      caller: caller,
      name: 'device.openUrl',
      method: 'device.openUrl',
      description: 'Opens the given URL in the platform browser.',
      mutating: true,
      buildArgs: _urlArgs,
      parameters: _deviceSchema(
        properties: <String, Object?>{'url': _stringType},
        required: <String>['url'],
      ),
    ),
    _device(
      caller: caller,
      name: 'device.openApp',
      method: 'device.openApp',
      description: 'Launches the application with the given package name.',
      mutating: true,
      buildArgs: _packageArgs,
      parameters: _deviceSchema(
        properties: <String, Object?>{'package': _stringType},
        required: <String>['package'],
      ),
    ),
  ];
}

/// A tool that maps assistant arguments onto one host method.
///
/// Kept as a single class rather than sixteen subclasses: every device tool has
/// exactly the same shape, and the only differences are data. That makes the
/// table above the single source of truth for names, methods and schemas.
class _DeviceTool extends FunctionTool {
  _DeviceTool({
    required this.name,
    required this.method,
    required this.description,
    required this.parameters,
    required this.mutating,
    required this.buildArgs,
    required this.caller,
  });

  @override
  final String name;

  /// The exact method string the host's platform channel expects.
  final String method;

  @override
  final String description;

  @override
  final Map<String, Object?> parameters;

  @override
  final bool mutating;

  /// Maps validated assistant arguments onto the host argument map.
  final Map<String, Object?> Function(Map<String, Object?> args) buildArgs;

  /// The host implementation, supplied by the application.
  final HostCaller caller;

  @override
  Future<Object?> run(Map<String, Object?> args) {
    return caller(method, buildArgs(args));
  }

  @override
  Future<ToolResult> invoke(Map<String, Object?> args) async {
    final ToolResult result = await super.invoke(args);
    if (result.ok && result.data == null) {
      return ToolResult.unsupported('$name is not available on this platform');
    }
    return result;
  }
}

/// Builds one device tool, defaulting to a schema that accepts no arguments.
_DeviceTool _device({
  required HostCaller caller,
  required String name,
  required String method,
  required String description,
  required bool mutating,
  required Map<String, Object?> Function(Map<String, Object?> args) buildArgs,
  Map<String, Object?>? parameters,
}) {
  return _DeviceTool(
    caller: caller,
    name: name,
    method: method,
    description: description,
    mutating: mutating,
    buildArgs: buildArgs,
    parameters: parameters ?? _deviceSchema(),
  );
}

/// The JSON-Schema object wrapper shared by every device tool.
Map<String, Object?> _deviceSchema({
  Map<String, Object?> properties = const <String, Object?>{},
  List<String> required = const <String>[],
}) {
  return <String, Object?>{
    'type': 'object',
    'properties': properties,
    'required': required,
  };
}

const Map<String, Object?> _stringType = <String, Object?>{'type': 'string'};
const Map<String, Object?> _booleanType = <String, Object?>{'type': 'boolean'};
const Map<String, Object?> _notificationIdType = <String, Object?>{
  'type': 'integer',
  'default': 0,
};
const Map<String, Object?> _vibrateMillisType = <String, Object?>{
  'type': 'integer',
  'default': 200,
};

/// No arguments; the host map is always empty for read-only queries.
Map<String, Object?> _noArgs(Map<String, Object?> _) => const <String, Object?>{};

Map<String, Object?> _clipboardWriteArgs(Map<String, Object?> args) {
  return <String, Object?>{'text': requireStringArg(args, 'text')};
}

Map<String, Object?> _notifyArgs(Map<String, Object?> args) {
  return <String, Object?>{
    // Notifications need a stable integer id even when the caller omits it;
    // zero is the conventional "main" notification slot.
    'id': optionalIntArg(args, 'id', fallback: 0),
    'title': requireStringArg(args, 'title'),
    'body': requireStringArg(args, 'body'),
    'ongoing': optionalBoolArg(args, 'ongoing'),
  };
}

Map<String, Object?> _vibrateArgs(Map<String, Object?> args) {
  return <String, Object?>{
    'millis': optionalIntArg(args, 'millis', fallback: 200),
  };
}

Map<String, Object?> _toastArgs(Map<String, Object?> args) {
  return <String, Object?>{'text': requireStringArg(args, 'text')};
}

Map<String, Object?> _shareArgs(Map<String, Object?> args) {
  final String? subject = optionalStringArg(args, 'subject');
  return <String, Object?>{
    'text': requireStringArg(args, 'text'),
    // Omitted rather than sent as null so the host does not have to distinguish
    // "no subject" from an explicit null.
    if (subject != null) 'subject': subject,
  };
}

Map<String, Object?> _installedAppsArgs(Map<String, Object?> args) {
  return <String, Object?>{
    'includeSystem': optionalBoolArg(args, 'includeSystem'),
  };
}

Map<String, Object?> _urlArgs(Map<String, Object?> args) {
  return <String, Object?>{'url': requireStringArg(args, 'url')};
}

Map<String, Object?> _packageArgs(Map<String, Object?> args) {
  return <String, Object?>{'package': requireStringArg(args, 'package')};
}