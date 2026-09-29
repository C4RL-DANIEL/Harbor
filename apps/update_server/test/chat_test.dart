// Tests for the Harbor update server chat API.
//
// Like server_test.dart these depend on `package:test`, not `flutter_test`, so
// they run on the plain Dart VM via `dart test`. That is what keeps the chat
// service and the router Flutter-free.
//
// The model is hosted for real: the tests build a default (browser-preset)
// service over a small seed corpus and drive it through HTTP, because the point
// of these endpoints is that a request maps onto a real generation and a real
// stream, not onto a stub.

import 'dart:convert';
import 'dart:io';

import 'package:harbor_core/harbor_core.dart';
import 'package:http/http.dart' as http;
import 'package:shelf/shelf.dart';
import 'package:shelf/shelf_io.dart' as shelf_io;
import 'package:test/test.dart';
import 'package:update_server/src/api_router.dart';
import 'package:update_server/src/chat_service.dart';
import 'package:update_server/src/release_store.dart';

/// Replies are generated from a model that has seen no training data at all, so
/// fast, deterministic HTTP assertions need a short token budget.
const int _testMaxTokens = 8;

/// A few hundred characters of prose with enough repetition for [BpeTrainer] to
/// learn real merges, which is what makes the tokenizer path (rather than the
/// bare byte vocabulary) the one under test.
const String _seedCorpus = '''
The harbor keeps a small language model on the device. The model reads the
text the user lets it read, and it writes short answers. A small model cannot
know everything, so it says when it does not know. The harbor keeps the model
on the device so the user's text never leaves the phone. A model kept on the
device is a model the user can trust, and a model the user can trust is a
model the user will keep.
''';

Map<String, Object?> _json(http.Response response) {
  final Object? decoded = jsonDecode(response.body);
  return decoded! as Map<String, Object?>;
}

List<Object?> _messages(int count) {
  return <Object?>[
    for (int i = 0; i < count; i++)
      <String, Object?>{'role': 'user', 'content': 'message $i'},
  ];
}

class _ChatServer {
  _ChatServer(this.server, this.service);

  final HttpServer server;
  final HarborChatService service;

  Uri uri(String path, [Map<String, String>? query]) {
    final Uri base = Uri.parse('http://127.0.0.1:${server.port}$path');
    return query == null ? base : base.replace(queryParameters: query);
  }

  Future<void> close() async {
    await server.close(force: true);
  }
}

Future<_ChatServer> _start() async {
  final ReleaseStore store = ReleaseStore.inMemory();
  await store.load();
  final HarborChatService service = HarborChatService(
    seedCorpus: _seedCorpus,
    maxNewTokens: _testMaxTokens,
  );
  await service.initialize();
  final Handler handler = buildApiHandler(
    store: store,
    adminToken: '',
    chatService: service,
  );
  final HttpServer server =
      await shelf_io.serve(handler, InternetAddress.loopbackIPv4, 0);
  return _ChatServer(server, service);
}

void main() {
  group('chat API', () {
    late _ChatServer server;

    setUp(() async {
      server = await _start();
    });
    tearDown(() async {
      await server.close();
    });

    test('status reports the untrained model shape', () async {
      final http.Response response =
          await http.get(server.uri('/api/v1/chat/status'));
      expect(response.statusCode, 200);
      expect(response.headers['content-type'], contains('application/json'));

      final Map<String, Object?> body = _json(response);
      expect(body['ready'], true);
      expect(body['stage'], 'untrained');
      // The seed corpus must have taught the tokenizer at least one merge.
      expect(body['vocabulary'], greaterThan(260));
      expect(body['parameters'], greaterThan(0));
      expect(body['layers'], TinyLmConfig.browserPreset.nLayers);
      expect(body['context_length'], TinyLmConfig.browserPreset.contextLength);
      expect(body['seed_documents'], 1);
      expect(body['seed_chars'], _seedCorpus.length);
      expect(body['max_new_tokens'], _testMaxTokens);
    });

    test('non-streaming chat returns generated text', () async {
      final http.Response response = await http.post(
        server.uri('/api/v1/chat'),
        headers: <String, String>{'content-type': 'application/json'},
        body: jsonEncode(<String, Object?>{
          'messages': _messages(1),
          'max_tokens': _testMaxTokens,
        }),
      );
      expect(response.statusCode, 200);
      expect(response.headers['content-type'], contains('application/json'));

      final Map<String, Object?> body = _json(response);
      expect(body['text'], isA<String>());
      expect((body['text']! as String).isNotEmpty, isTrue);
      expect(body['generated_tokens'], isA<int>());
      expect(body['stop_reason'], isA<String>());

      final Object? model = body['model'];
      expect(model, isA<Map<String, Object?>>());
      final Map<String, Object?> status = model! as Map<String, Object?>;
      expect(status['ready'], true);
      expect(status['stage'], 'untrained');
    });

    test('streaming chat emits data frames and ends with [DONE]', () async {
      final http.Response response = await http.post(
        server.uri('/api/v1/chat', <String, String>{'stream': 'true'}),
        headers: <String, String>{
          'content-type': 'application/json',
          'accept': 'text/event-stream',
        },
        body: jsonEncode(<String, Object?>{
          'messages': _messages(1),
          'max_tokens': 4,
        }),
      );
      expect(response.statusCode, 200);
      expect(response.headers['content-type'], contains('text/event-stream'));

      final String payload = utf8.decode(response.bodyBytes);
      final List<String> dataLines = payload
          .split('\n')
          .where((String line) => line.startsWith('data: '))
          .toList();
      expect(dataLines, isNotEmpty);
      expect(dataLines.last, 'data: [DONE]');

      final Object? first =
          jsonDecode(dataLines.first.substring('data: '.length));
      expect(first, isA<Map<String, Object?>>());
      expect((first! as Map<String, Object?>)['type'], isA<String>());
    });

    test('empty body is rejected with 400', () async {
      final http.Response response = await http.post(
        server.uri('/api/v1/chat'),
        headers: <String, String>{'content-type': 'application/json'},
        body: '',
      );
      expect(response.statusCode, 400);
      expect(_json(response)['error'], isA<String>());
    });

    test('unknown role is rejected with 400', () async {
      final http.Response response = await http.post(
        server.uri('/api/v1/chat'),
        headers: <String, String>{'content-type': 'application/json'},
        body: jsonEncode(<String, Object?>{
          'messages': <Object?>[
            <String, Object?>{'role': 'robot', 'content': 'hello'},
          ],
        }),
      );
      expect(response.statusCode, 400);
      expect(_json(response)['error'], contains('robot'));
    });

    test('more than $kChatMessageLimit messages is rejected with 400', () async {
      final http.Response response = await http.post(
        server.uri('/api/v1/chat'),
        headers: <String, String>{'content-type': 'application/json'},
        body: jsonEncode(<String, Object?>{
          'messages': _messages(kChatMessageLimit + 1),
        }),
      );
      expect(response.statusCode, 400);
      expect(_json(response)['error'], contains('$kChatMessageLimit'));
    });

    test('non-integer max_tokens is rejected with 400', () async {
      final http.Response response = await http.post(
        server.uri('/api/v1/chat'),
        headers: <String, String>{'content-type': 'application/json'},
        body: jsonEncode(<String, Object?>{
          'messages': _messages(1),
          'max_tokens': 'lots',
        }),
      );
      expect(response.statusCode, 400);
      expect(_json(response)['error'], contains('max_tokens'));
    });

    test('tools lists the device and web catalogue', () async {
      final http.Response response =
          await http.get(server.uri('/api/v1/tools'));
      expect(response.statusCode, 200);

      final Map<String, Object?> body = _json(response);
      final Object? rawTools = body['tools'];
      expect(rawTools, isA<List<Object?>>());
      final List<Object?> tools = rawTools! as List<Object?>;

      final Set<String> names = <String>{
        for (final Map<String, Object?> tool
            in tools.whereType<Map<String, Object?>>())
          tool['name']! as String,
      };
      expect(
        names,
        containsAll(<String>['device.battery', 'device.notify', 'web.fetch']),
      );

      final Map<String, Object?> battery = tools
          .whereType<Map<String, Object?>>()
          .firstWhere((Map<String, Object?> tool) =>
              tool['name'] == 'device.battery');
      expect(battery['description'], isA<String>());
      expect(battery['mutating'], false);
      expect(battery['parameters'], isA<Map<String, Object?>>());
    });
  });
}