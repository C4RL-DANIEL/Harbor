// Tests for the device and web tool catalogues.
//
// The device tests never touch a platform channel and the web tests never open
// a socket: both rely on the injection points the tools were designed around.
// What is being pinned down is the contract the Android side is written
// against — exact method strings, exact argument maps — plus the failure
// conversions the agent loop depends on.

import 'dart:convert';

import 'package:harbor_core/src/tools/device_tools.dart';
import 'package:harbor_core/src/tools/tool.dart';
import 'package:harbor_core/src/tools/tool_registry.dart';
import 'package:harbor_core/src/tools/web_tools.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:test/test.dart';

/// Runs every tool test.
void main() {
  group('ToolRegistry', () {
    test('registering a duplicate name throws', () {
      final ToolRegistry registry = ToolRegistry(<Tool>[_EchoTool('echo')]);
      expect(() => registry.register(_EchoTool('echo')), throwsArgumentError);
    });

    test('an unknown call fails and names the tool', () async {
      final ToolRegistry registry = ToolRegistry(<Tool>[_EchoTool('echo')]);
      final ToolResult result = await registry.invoke(const ToolCall('nope'));
      expect(result.ok, isFalse);
      expect(result.error, contains('unknown tool'));
    });

    test('describe returns the four keys for every tool', () {
      final ToolRegistry registry = ToolRegistry(
        <Tool>[_EchoTool('echo'), _EchoTool('echo.two')],
      );
      final List<Map<String, Object?>> described = registry.describe();
      expect(described, hasLength(2));
      for (final Map<String, Object?> entry in described) {
        expect(
          entry.keys,
          containsAll(
            <String>['name', 'description', 'parameters', 'mutating'],
          ),
        );
        expect(entry, hasLength(4));
      }
    });

    test('invokeByName runs the named tool', () async {
      final ToolRegistry registry = ToolRegistry(<Tool>[_EchoTool('echo')]);
      final ToolResult result = await registry.invokeByName(
        'echo',
        <String, Object?>{'value': 'hi'},
      );
      expect(result.ok, isTrue);
      expect(result.data, 'hi');
    });
  });

  group('deviceTools', () {
    test('registers exactly the sixteen documented tools', () {
      final List<_RecordedCall> calls = <_RecordedCall>[];
      final ToolRegistry registry =
          ToolRegistry(deviceTools(_recordingCaller(calls)));
      expect(registry.length, 16);
      final List<String> names =
          registry.tools.map((Tool tool) => tool.name).toList();
      expect(names.toSet(), hasLength(16));
      expect(
        names.toSet(),
        equals(<String>{
          'device.info',
          'device.battery',
          'device.storage',
          'device.connectivity',
          'device.display',
          'device.thermal',
          'device.locale',
          'device.clipboard.read',
          'device.clipboard.write',
          'device.notify',
          'device.vibrate',
          'device.toast',
          'device.share',
          'device.installedApps',
          'device.openUrl',
          'device.openApp',
        }),
      );
    });

    test('forwards the documented method and argument map', () async {
      final List<_RecordedCall> calls = <_RecordedCall>[];
      final ToolRegistry registry = ToolRegistry(
        deviceTools(
          _recordingCaller(calls, result: <String, Object?>{'value': 1}),
        ),
      );

      await registry.invokeByName(
        'device.clipboard.write',
        <String, Object?>{'text': 'hi'},
      );
      expect(calls.last.method, 'device.clipboard.write');
      expect(calls.last.args, <String, Object?>{'text': 'hi'});

      await registry.invokeByName(
        'device.notify',
        <String, Object?>{'title': 'T', 'body': 'B'},
      );
      expect(calls.last.method, 'device.notify');
      expect(calls.last.args, <String, Object?>{
        'id': 0,
        'title': 'T',
        'body': 'B',
        'ongoing': false,
      });

      await registry.invokeByName('device.vibrate');
      expect(calls.last.method, 'device.vibrate');
      expect(calls.last.args, <String, Object?>{'millis': 200});

      await registry.invokeByName(
        'device.openUrl',
        <String, Object?>{'url': 'https://example.com'},
      );
      expect(calls.last.method, 'device.openUrl');
      expect(calls.last.args, <String, Object?>{'url': 'https://example.com'});

      await registry.invokeByName('device.battery');
      expect(calls.last.method, 'device.battery');
      expect(calls.last.args, isEmpty);

      await registry.invokeByName(
        'device.share',
        <String, Object?>{'text': 'hi', 'subject': 'S'},
      );
      expect(calls.last.method, 'device.share');
      expect(calls.last.args, <String, Object?>{'text': 'hi', 'subject': 'S'});
    });

    test('a missing required argument fails without calling the host', () async {
      final List<_RecordedCall> calls = <_RecordedCall>[];
      final ToolRegistry registry =
          ToolRegistry(deviceTools(_recordingCaller(calls)));
      final ToolResult result = await registry.invokeByName('device.openUrl');
      expect(result.ok, isFalse);
      expect(calls, isEmpty);
    });

    test('a wrong-typed argument fails without calling the host', () async {
      final List<_RecordedCall> calls = <_RecordedCall>[];
      final ToolRegistry registry =
          ToolRegistry(deviceTools(_recordingCaller(calls)));
      final ToolResult result = await registry.invokeByName(
        'device.clipboard.write',
        <String, Object?>{'text': 123},
      );
      expect(result.ok, isFalse);
      expect(calls, isEmpty);
    });

    test('a throwing caller becomes a failure carrying its message', () async {
      Future<Object?> throwingCaller(
        String method,
        Map<String, Object?> args,
      ) async {
        throw StateError('boom from the host');
      }

      final ToolRegistry registry = ToolRegistry(deviceTools(throwingCaller));
      final ToolResult result = await registry.invokeByName('device.battery');
      expect(result.ok, isFalse);
      expect(result.error, contains('boom from the host'));
    });

    test('a null-returning caller becomes unsupported', () async {
      Future<Object?> nullCaller(String method, Map<String, Object?> args) async {
        return null;
      }

      final ToolRegistry registry = ToolRegistry(deviceTools(nullCaller));
      final ToolResult result = await registry.invokeByName('device.info');
      expect(result.ok, isFalse);
      expect(result.error, contains('not available'));
    });
  });

  group('webTools', () {
    test('web.fetch counts bytes and strips script content', () async {
      final List<http.Request> requests = <http.Request>[];
      const String html =
          '<html><body><script>alert("x")</script><p>Hello world</p></body></html>';
      final MockClient client = MockClient((http.Request request) async {
        requests.add(request);
        return http.Response(
          html,
          200,
          headers: <String, String>{'content-type': 'text/html; charset=utf-8'},
        );
      });
      final ToolRegistry registry =
          ToolRegistry(webTools(client: client, allow: _allowAll));
      final ToolResult result = await registry.invokeByName(
        'web.fetch',
        <String, Object?>{
          'url': 'https://example.com/page',
          'asText': true,
        },
      );
      expect(result.ok, isTrue);
      final Map<String, Object?> data = _data(result);
      expect(data['status'], 200);
      expect(data['bytes'], utf8.encode(html).length);
      expect(data['truncated'], isFalse);
      final String text = data['text'] as String;
      expect(text, contains('Hello world'));
      expect(text, isNot(contains('alert')));
      expect(requests, hasLength(1));
    });

    test('web.fetch truncates a body larger than maxBytes', () async {
      final List<http.Request> requests = <http.Request>[];
      const String body = 'abcdefghijklmnopqrstuvwxyz';
      final MockClient client = MockClient((http.Request request) async {
        requests.add(request);
        return http.Response(
          body,
          200,
          headers: <String, String>{'content-type': 'text/plain'},
        );
      });
      final ToolRegistry registry = ToolRegistry(
        webTools(client: client, allow: _allowAll, maxBytes: 8),
      );
      final ToolResult result = await registry.invokeByName(
        'web.fetch',
        <String, Object?>{'url': 'https://example.com/big'},
      );
      expect(result.ok, isTrue);
      final Map<String, Object?> data = _data(result);
      expect(data['truncated'], isTrue);
      expect(data['bytes'], 8);
      expect((data['text'] as String).length, lessThanOrEqualTo(8));
    });

    test('web.fetch refuses a host rejected by allow without requesting',
        () async {
      final List<http.Request> requests = <http.Request>[];
      final MockClient client = MockClient((http.Request request) async {
        requests.add(request);
        return http.Response('nope', 200);
      });
      final ToolRegistry registry = ToolRegistry(
        webTools(
          client: client,
          allow: (Uri uri) => uri.host != 'blocked.example',
        ),
      );
      final ToolResult result = await registry.invokeByName(
        'web.fetch',
        <String, Object?>{'url': 'https://blocked.example/secret'},
      );
      expect(result.ok, isFalse);
      expect(result.error, contains('blocked.example'));
      expect(requests, isEmpty);
    });

    test('web.check reports the status and server', () async {
      final MockClient client = MockClient((http.Request request) async {
        return http.Response(
          'ok',
          204,
          headers: <String, String>{
            'content-type': 'text/plain',
            'server': 'mock/1.0',
          },
        );
      });
      final ToolRegistry registry =
          ToolRegistry(webTools(client: client, allow: _allowAll));
      final ToolResult result = await registry.invokeByName(
        'web.check',
        <String, Object?>{'url': 'https://example.com/health'},
      );
      expect(result.ok, isTrue);
      final Map<String, Object?> data = _data(result);
      expect(data['status'], 204);
      expect(data['reachable'], isTrue);
      expect(data['server'], 'mock/1.0');
    });

    test('web.json parses an object and previews it', () async {
      final MockClient client = MockClient((http.Request request) async {
        return http.Response(
          '{"a":1,"b":[true,null]}',
          200,
          headers: <String, String>{'content-type': 'application/json'},
        );
      });
      final ToolRegistry registry =
          ToolRegistry(webTools(client: client, allow: _allowAll));
      final ToolResult result = await registry.invokeByName(
        'web.json',
        <String, Object?>{'url': 'https://example.com/data.json'},
      );
      expect(result.ok, isTrue);
      final Map<String, Object?> data = _data(result);
      expect(data['ok'], isTrue);
      expect(data['type'], 'object');
      expect(data['preview'], contains('"a"'));
    });

    test('web.json fails on malformed JSON', () async {
      final MockClient client = MockClient((http.Request request) async {
        return http.Response(
          '{not json',
          200,
          headers: <String, String>{'content-type': 'application/json'},
        );
      });
      final ToolRegistry registry =
          ToolRegistry(webTools(client: client, allow: _allowAll));
      final ToolResult result = await registry.invokeByName(
        'web.json',
        <String, Object?>{'url': 'https://example.com/bad.json'},
      );
      expect(result.ok, isFalse);
      expect(result.error, contains('invalid JSON'));
    });

    test('web.fetch rejects ftp and relative URLs without requesting', () async {
      final List<http.Request> requests = <http.Request>[];
      final MockClient client = MockClient((http.Request request) async {
        requests.add(request);
        return http.Response('never', 200);
      });
      final ToolRegistry registry =
          ToolRegistry(webTools(client: client, allow: _allowAll));
      final ToolResult ftp = await registry.invokeByName(
        'web.fetch',
        <String, Object?>{'url': 'ftp://x'},
      );
      expect(ftp.ok, isFalse);
      final ToolResult relative = await registry.invokeByName(
        'web.fetch',
        <String, Object?>{'url': 'relative/path'},
      );
      expect(relative.ok, isFalse);
      expect(requests, isEmpty);
    });
  });
}

/// A minimal tool used to exercise [ToolRegistry] without device or web I/O.
class _EchoTool extends FunctionTool {
  _EchoTool(this.name);

  @override
  final String name;

  @override
  String get description => 'Echoes the value argument back.';

  @override
  Map<String, Object?> get parameters => <String, Object?>{
        'type': 'object',
        'properties': <String, Object?>{
          'value': <String, Object?>{'type': 'string'},
        },
        'required': <String>['value'],
      };

  @override
  bool get mutating => false;

  @override
  Future<Object?> run(Map<String, Object?> args) async => args['value'];
}

/// Records every `(method, args)` pair a device tool hands to the host.
class _RecordedCall {
  _RecordedCall(this.method, this.args);

  final String method;
  final Map<String, Object?> args;
}

/// A [HostCaller] that records its calls and returns a canned [result].
HostCaller _recordingCaller(List<_RecordedCall> calls, {Object? result}) {
  return (String method, Map<String, Object?> args) async {
    calls.add(_RecordedCall(method, args));
    return result;
  };
}

/// An allow predicate that permits everything, for tests about other behaviour.
bool _allowAll(Uri uri) => true;

/// Extracts the data map from a successful [ToolResult], failing otherwise.
Map<String, Object?> _data(ToolResult result) {
  final Object? data = result.data;
  if (data is! Map<String, Object?>) {
    fail('expected a map payload, got $data');
  }
  return data;
}