// HTTP routing for the Harbor update server.
//
// This library is Flutter-free: it runs on the pure Dart VM.

import 'dart:convert';

import 'package:harbor_core/harbor_core.dart';
import 'package:pub_semver/pub_semver.dart';
import 'package:shelf/shelf.dart';
import 'package:shelf_router/shelf_router.dart';

import 'chat_service.dart';
import 'models.dart';
import 'release_store.dart';

/// Default no-op logger for [buildRouter] and [buildApiHandler].
void _noopLog(String message) {}

const Map<String, String> _jsonHeaders = <String, String>{
  'content-type': 'application/json; charset=utf-8',
};

/// Serializes [body] as a JSON response.
Response jsonResponse(
  Object? body, {
  int statusCode = 200,
  Map<String, String>? headers,
}) {
  return Response(
    statusCode,
    body: jsonEncode(body),
    headers: <String, String>{
      ..._jsonHeaders,
      ...?headers,
    },
  );
}

/// Serializes the frozen `{"error": "<message>"}` error envelope.
Response jsonError(int statusCode, String message) =>
    jsonResponse(<String, Object?>{'error': message}, statusCode: statusCode);

/// Translates [ApiException]s into their JSON status codes and swallows any
/// other failure into a logged HTTP 500.
Middleware apiErrorMiddleware(void Function(String) log) {
  return (Handler inner) {
    return (Request request) async {
      try {
        return await inner(request);
      } on ApiException catch (error) {
        return jsonError(error.statusCode, error.message);
      } catch (error, stackTrace) {
        log(
          'unhandled error on ${request.method} /${request.url.path}: '
          '$error\n$stackTrace',
        );
        return jsonError(500, 'internal server error');
      }
    };
  };
}

/// Builds the router with every route in the frozen contract.
///
/// [adminToken] is the token required by every mutating admin route. An empty
/// token disables admin mutations with HTTP 503 rather than accepting anything.
///
/// [chatService] is the optional model host behind the `/api/v1/chat` routes.
/// When omitted a default, lazily-initialised service is used, so existing call
/// sites and tests keep working without knowing about chat at all.
Router buildRouter({
  required ReleaseStore store,
  required String adminToken,
  void Function(String) log = _noopLog,
  HarborChatService? chatService,
}) {
  final Router router = Router();
  final HarborChatService service = chatService ?? HarborChatService();

  router.get('/health', (Request request) => _health(store));
  router.get(
    '/api/v1/update-check',
    (Request request) => _updateCheck(request, store),
  );
  router.get('/api/v1/flags', (Request request) => _flags(store));
  router.put(
    '/api/v1/flags',
    (Request request) => _putFlags(request, store, adminToken, log),
  );
  router.get('/api/v1/releases', (Request request) => _listReleases(store));
  router.post(
    '/api/v1/releases',
    (Request request) => _publishRelease(request, store, adminToken, log),
  );
  router.put(
    '/api/v1/config/min-supported',
    (Request request) => _setMinSupported(request, store, adminToken, log),
  );
  router.post(
    '/api/v1/config/force',
    (Request request) => _setForceUpdate(request, store, adminToken, log),
  );
  router.get('/admin/state', (Request request) => _adminState(store));
  router.get(
    '/latest_version.json',
    (Request request) => _latestVersionJson(store),
  );
  router.get(
    '/api/v1/chat/status',
    (Request request) => jsonResponse(service.status()),
  );
  router.get(
    '/api/v1/tools',
    (Request request) => jsonResponse(<String, Object?>{
      'tools': service.toolRegistry.describe(),
    }),
  );
  router.post(
    '/api/v1/chat',
    (Request request) => _chat(request, service, log),
  );

  // ---- Memory ----------------------------------------------------------
  // Read is unauthenticated because it returns only what the operator has
  // already shown to the dashboard, and the chat pane needs it to render the
  // memory strip. Every mutation is admin-gated: forgetting on someone else's
  // behalf is not a public operation.
  router.get('/api/v1/memory', (Request request) => _listMemory(service));
  router.post(
    '/api/v1/memory',
    (Request request) => _addMemory(request, service, adminToken, log),
  );
  router.post(
    '/api/v1/memory/clear',
    (Request request) => _clearMemory(request, service, adminToken, log),
  );
  router.delete(
    '/api/v1/memory/<id>',
    (Request request, String id) => _forgetMemory(request, id, service, adminToken, log),
  );

  // Catch-all so unknown routes answer with the JSON error envelope.
  router.all('/<ignored|.*>', (Request request) {
    throw NotFoundException(
      'no route for ${request.method} /${request.url.path}',
    );
  });

  return router;
}

/// The CORS headers applied to every response, including errors.
const Map<String, String> corsHeaders = <String, String>{
  'access-control-allow-origin': '*',
  'access-control-allow-methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'access-control-allow-headers': 'X-Admin-Token, Content-Type',
  'access-control-max-age': '86400',
};

/// Answers preflight `OPTIONS` with 204 and decorates every response with the
/// permissive CORS headers the cross-origin admin dashboard needs.
Middleware corsMiddleware() {
  return (Handler inner) {
    return (Request request) async {
      if (request.method == 'OPTIONS') {
        return Response(204, headers: corsHeaders);
      }
      final Response response = await inner(request);
      return response.change(headers: corsHeaders);
    };
  };
}

/// Adds a stable `ETag` to the cacheable GET endpoints and honors
/// `If-None-Match` with a bodyless 304.
Middleware etagMiddleware() {
  return (Handler inner) {
    return (Request request) async {
      final Response response = await inner(request);
      if (request.method != 'GET') {
        return response;
      }
      final String path = request.url.path;
      if (path != 'api/v1/flags' && path != 'api/v1/update-check') {
        return response;
      }
      final String body = await response.readAsString();
      final String etag = '"${etagHash(body)}"';
      if (request.headers['if-none-match'] == etag) {
        return Response(304, headers: <String, String>{'etag': etag});
      }
      return response.change(
        body: body,
        headers: <String, String>{'etag': etag},
      );
    };
  };
}

/// Stable, dependency-free 32-bit FNV-1a hash rendered as eight hex digits.
String etagHash(String body) {
  int hash = 0x811c9dc5;
  for (final int unit in body.codeUnits) {
    hash ^= unit;
    hash = (hash * 0x01000193) & 0xFFFFFFFF;
  }
  return hash.toRadixString(16).padLeft(8, '0');
}

/// The router wrapped in the error + ETag middleware: the exact handler the
/// server and the tests both serve.
Handler buildApiHandler({
  required ReleaseStore store,
  required String adminToken,
  void Function(String) log = _noopLog,
  bool includeEtag = true,
  HarborChatService? chatService,
}) {
  final Router router = buildRouter(
    store: store,
    adminToken: adminToken,
    log: log,
    chatService: chatService,
  );
  Pipeline pipeline = const Pipeline().addMiddleware(apiErrorMiddleware(log));
  if (includeEtag) {
    pipeline = pipeline.addMiddleware(etagMiddleware());
  }
  return pipeline.addHandler(router.call);
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

Future<Response> _health(ReleaseStore store) async {
  final ServerState state = await store.load();
  return jsonResponse(<String, Object?>{
    'ok': true,
    'status': 'healthy',
    'time': DateTime.now().toUtc().toIso8601String(),
    'version_count': state.releasesByPlatform.length,
  });
}

Future<Response> _updateCheck(Request request, ReleaseStore store) async {
  final Map<String, String> params = request.url.queryParameters;

  final String? versionRaw = params['version'];
  if (versionRaw == null || versionRaw.trim().isEmpty) {
    throw const ValidationException(
      'query parameter "version" is required',
    );
  }
  final Version installed = parseSemver(versionRaw, field: 'version');

  final String platform = (params['platform'] ?? '').trim();
  if (platform.isEmpty) {
    throw const ValidationException(
      'query parameter "platform" is required',
    );
  }

  final ServerState state = await store.load();
  final UpdateCheckResponse response = UpdateCheckResponse.evaluate(
    release: state.releaseFor(platform),
    installed: installed,
    platform: platform,
  );
  return jsonResponse(response.toJson());
}

Future<Response> _flags(ReleaseStore store) async {
  final ServerState state = await store.load();
  return jsonResponse(state.flags.toJson());
}

Future<Response> _putFlags(
  Request request,
  ReleaseStore store,
  String adminToken,
  void Function(String) log,
) async {
  _requireAdmin(request, adminToken);
  final Map<String, Object?> body = await _readJsonObject(request);

  final bool merge = _optionalBool(body, 'merge') ?? true;
  final Map<String, Object?>? flags = _optionalMap(body, 'flags');
  final Map<String, Object?>? defaults =
      _optionalMap(body, 'remote_defaults');
  final Map<String, Object?>? layout = _optionalMap(body, 'layout');
  FeatureFlagMatrix.validateLayout(body['layout']);

  final ServerState state = await store.load();
  final FeatureFlagMatrix patch = FeatureFlagMatrix(
    version: state.flags.version,
    updatedAt: DateTime.now().toUtc(),
    flags: flags ?? const <String, Object?>{},
    remoteDefaults: defaults ?? const <String, Object?>{},
    layout: layout ?? const <String, Object?>{},
  );
  final FeatureFlagMatrix updated = state.flags.merge(patch, merge: merge);
  await store.save(state.copyWith(flags: updated));
  log('admin: PUT /api/v1/flags merge=$merge -> v${updated.version}');
  return jsonResponse(<String, Object?>{'ok': true, 'version': updated.version});
}

Future<Response> _listReleases(ReleaseStore store) async {
  final ServerState state = await store.load();
  return jsonResponse(<String, Object?>{
    'releases': <String, Object?>{
      for (final MapEntry<String, ReleaseInfo> entry
          in state.releasesByPlatform.entries)
        entry.key: entry.value.toJson(),
    },
  });
}

Future<Response> _publishRelease(
  Request request,
  ReleaseStore store,
  String adminToken,
  void Function(String) log,
) async {
  _requireAdmin(request, adminToken);
  final Map<String, Object?> body = await _readJsonObject(request);
  final ReleaseInfo release = ReleaseInfo.fromJson(
    body,
    defaultPublishedAt: DateTime.now().toUtc(),
  );

  final ServerState state = await store.load();
  await store.save(state.withRelease(release));
  log(
    'admin: published ${release.platform} ${release.version} '
    '(min ${release.minSupportedVersion})',
  );
  return jsonResponse(
    <String, Object?>{'ok': true, 'release': release.toJson()},
    statusCode: 201,
  );
}

Future<Response> _setMinSupported(
  Request request,
  ReleaseStore store,
  String adminToken,
  void Function(String) log,
) async {
  _requireAdmin(request, adminToken);
  final Map<String, Object?> body = await _readJsonObject(request);

  final String platform = _requireString(body, 'platform');
  final String minRaw = _requireString(body, 'min_supported_version');
  final Version min = parseSemver(minRaw, field: 'min_supported_version');

  final ServerState state = await store.load();
  final ReleaseInfo? current = state.releaseFor(platform);
  if (current == null) {
    throw NotFoundException('no release stored for platform "$platform"');
  }
  if (min > current.semver) {
    throw ValidationException(
      'min_supported_version "$minRaw" must not exceed the latest version '
      '"${current.version}" for platform "$platform"',
    );
  }
  final bool forceUpdate =
      _optionalBool(body, 'force_update') ?? current.forceUpdate;

  final ReleaseInfo updated = current.copyWith(
    minSupportedVersion: minRaw,
    forceUpdate: forceUpdate,
  );
  await store.save(state.withRelease(updated));
  log('admin: $platform min_supported_version -> $minRaw '
      '(force=$forceUpdate)');
  return jsonResponse(<String, Object?>{'ok': true});
}

Future<Response> _setForceUpdate(
  Request request,
  ReleaseStore store,
  String adminToken,
  void Function(String) log,
) async {
  _requireAdmin(request, adminToken);
  final Map<String, Object?> body = await _readJsonObject(request);

  final String platform = _requireString(body, 'platform');
  final bool force = _requireBool(body, 'force_update');

  final ServerState state = await store.load();
  final ReleaseInfo? current = state.releaseFor(platform);
  if (current == null) {
    throw NotFoundException('no release stored for platform "$platform"');
  }
  await store.save(state.withRelease(current.copyWith(forceUpdate: force)));
  log('admin: $platform force_update -> $force');
  return jsonResponse(<String, Object?>{'ok': true});
}

Future<Response> _adminState(ReleaseStore store) async {
  final ServerState state = await store.load();
  return jsonResponse(<String, Object?>{
    'releases': <String, Object?>{
      for (final MapEntry<String, ReleaseInfo> entry
          in state.releasesByPlatform.entries)
        entry.key: entry.value.toJson(),
    },
    'flags': state.flags.toJson(),
  });
}

Future<Response> _latestVersionJson(ReleaseStore store) async {
  final ServerState state = await store.load();
  final ReleaseInfo? release = state.releaseFor('android') ??
      (state.releasesByPlatform.isEmpty
          ? null
          : state.releasesByPlatform.values.first);
  final DateTime generatedAt = DateTime.now().toUtc();

  if (release == null) {
    return jsonResponse(
      UpdateCheckResponse.empty('android').toJson(generatedAt: generatedAt),
    );
  }
  final UpdateCheckResponse response = UpdateCheckResponse.evaluate(
    release: release,
    installed: release.semver,
    platform: release.platform,
  );
  return jsonResponse(response.toJson(generatedAt: generatedAt));
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

/// Serves one `POST /api/v1/chat`, streaming or buffered by request.
Future<Response> _chat(
  Request request,
  HarborChatService service,
  void Function(String) log,
) async {
  final _ChatRequest parsed = await _readChatRequest(request);
  if (_wantsEventStream(request)) {
    return _streamingChat(parsed, service, log);
  }
  return _bufferedChat(parsed, service);
}

/// Parses and validates the chat request body.
///
/// Every failure is a [ValidationException], so a malformed request reaches the
/// client as a 400 with the same `{"error": ...}` envelope as the rest of the
/// API rather than as an unhandled exception.
Future<_ChatRequest> _readChatRequest(Request request) async {
  final String raw = await request.readAsString();
  if (raw.trim().isEmpty) {
    throw const ValidationException('request body must not be empty');
  }
  final Object? decoded;
  try {
    decoded = jsonDecode(raw);
  } on FormatException catch (error) {
    throw ValidationException(
      'request body is not valid JSON: ${error.message}',
    );
  }
  if (decoded is! Map<String, Object?>) {
    throw const ValidationException('request body must be a JSON object');
  }

  final Object? rawMessages = decoded['messages'];
  if (rawMessages is! List<Object?>) {
    throw const ValidationException(
      '"messages" is required and must be an array',
    );
  }
  if (rawMessages.isEmpty) {
    throw const ValidationException(
      '"messages" must contain at least one message',
    );
  }
  if (rawMessages.length > kChatMessageLimit) {
    throw ValidationException(
      '"messages" must contain at most $kChatMessageLimit messages, got '
      '${rawMessages.length}',
    );
  }

  final List<ChatMessage> messages = <ChatMessage>[];
  for (int i = 0; i < rawMessages.length; i++) {
    final Object? entry = rawMessages[i];
    if (entry is! Map<String, Object?>) {
      throw ValidationException('message $i must be a JSON object');
    }
    try {
      messages.add(ChatMessage.fromJson(entry));
    } on FormatException catch (error) {
      throw ValidationException('message $i: ${error.message}');
    }
  }

  return _ChatRequest(
    messages: messages,
    maxTokens: _readMaxTokens(decoded['max_tokens']),
    useMemory: _optionalBool(decoded, 'memory'),
    thinking: _readThinking(decoded['thinking']),
  );
}

/// The reasoning depth requested for one turn.
///
/// `auto` (and an absent field) both return null, meaning "let the engine
/// choose per turn" — the two spellings collapse to the same behaviour because
/// an omitted field must not silently mean something different from the value
/// that documents the default.
String? _readThinking(Object? value) {
  if (value == null) {
    return null;
  }
  if (value is! String) {
    throw const ValidationException('"thinking" must be a string');
  }
  final String mode = value.trim().toLowerCase();
  const List<String> allowed = <String>['auto', 'none', 'concise', 'thorough'];
  if (!allowed.contains(mode)) {
    throw ValidationException(
      '"thinking" must be one of ${allowed.join(', ')}; got "$value"',
    );
  }
  return mode == 'auto' ? null : mode;
}

/// Reads `max_tokens`, defaulting it and capping it at [kChatMaxTokensCap].
///
/// The cap is applied rather than rejected because a client asking for "as much
/// as possible" should still get a bounded answer; only a value that is not an
/// integer, or one that asks for no output at all, is a client error.
int _readMaxTokens(Object? value) {
  if (value == null) {
    return kChatDefaultMaxTokens;
  }
  if (value is! int) {
    throw const ValidationException('"max_tokens" must be an integer');
  }
  if (value < 1) {
    throw const ValidationException('"max_tokens" must be at least 1');
  }
  return value > kChatMaxTokensCap ? kChatMaxTokensCap : value;
}

/// Whether the client asked for server-sent events.
///
/// Both `?stream=true` and `Accept: text/event-stream` are honored because a
/// browser `EventSource` cannot set headers while a hand-rolled client usually
/// prefers to.
bool _wantsEventStream(Request request) {
  final String? stream = request.url.queryParameters['stream'];
  if (stream != null && stream.toLowerCase() == 'true') {
    return true;
  }
  final String? accept = request.headers['accept'];
  return accept != null && accept.toLowerCase().contains('text/event-stream');
}

/// Streams the turn as `text/event-stream`.
Response _streamingChat(
  _ChatRequest parsed,
  HarborChatService service,
  void Function(String) log,
) {
  final Stream<ChatEvent> events = service.respond(
    parsed.messages,
    maxNewTokens: parsed.maxTokens,
    useMemory: parsed.useMemory,
    thinking: parsed.thinking,
  );
  return Response.ok(
    _sseStream(events, log),
    headers: const <String, String>{
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      'connection': 'keep-alive',
    },
  );
}

/// Renders [events] as SSE frames and terminates the stream with `[DONE]`.
///
/// The stream is already committed to HTTP 200 by the time events arrive, so a
/// failure is reported in-band as a `failed` event instead of being allowed to
/// tear down the connection without explanation.
Stream<List<int>> _sseStream(
  Stream<ChatEvent> events,
  void Function(String) log,
) async* {
  try {
    await for (final ChatEvent event in events) {
      yield utf8.encode('data: ${jsonEncode(_ssePayload(event))}\n\n');
    }
  } on Object catch (error) {
    log('chat stream failed: $error');
    yield utf8.encode(
      'data: ${jsonEncode(<String, Object?>{
        'type': 'failed',
        'message': '$error',
      })}\n\n',
    );
  }
  yield utf8.encode('data: [DONE]\n\n');
}

/// The wire form of one [ChatEvent].
Map<String, Object?> _ssePayload(ChatEvent event) => switch (event) {
      ChatToken(:final String text) =>
        <String, Object?>{'type': 'token', 'text': text},
      ChatToolStarted(:final ToolCall call) =>
        <String, Object?>{'type': 'tool_started', 'name': call.name},
      ChatToolFinished(:final ToolCall call, :final ToolResult result) =>
        <String, Object?>{
          'type': 'tool_finished',
          'name': call.name,
          'ok': result.ok,
          'summary': result.summary,
        },
      ChatThinking(:final ThinkingPlan plan, :final ThinkingTrace? trace) =>
        <String, Object?>{
          'type': 'thinking',
          'plan': plan.toJson(),
          'trace': trace?.toJson(),
        },
      ChatMemoryUpdated(
        :final List<MemoryEntry> entries,
        :final bool explicit,
      ) =>
        <String, Object?>{
          'type': 'memory_updated',
          'entries': <Object?>[
            for (final MemoryEntry entry in entries) entry.toJson(),
          ],
          'explicit': explicit,
        },
      ChatFinished(
        :final String text,
        :final int generatedTokens,
        :final String? stopReason,
      ) =>
        <String, Object?>{
          'type': 'finished',
          'text': text,
          'generated_tokens': generatedTokens,
          'stop_reason': stopReason,
        },
      ChatFailed(:final String message) =>
        <String, Object?>{'type': 'failed', 'message': message},
    };

/// Runs the turn to completion and answers with the whole reply.
Future<Response> _bufferedChat(
  _ChatRequest parsed,
  HarborChatService service,
) async {
  final StringBuffer streamed = StringBuffer();
  ChatFinished? finished;
  String? failure;
  await for (final ChatEvent event in service.respond(
    parsed.messages,
    maxNewTokens: parsed.maxTokens,
    useMemory: parsed.useMemory,
    thinking: parsed.thinking,
  )) {
    switch (event) {
      case ChatToken(:final String text):
        streamed.write(text);
      case ChatFinished():
        finished = event;
      case ChatFailed(:final String message):
        failure = message;
      case ChatToolStarted():
      case ChatToolFinished():
      case ChatThinking():
      case ChatMemoryUpdated():
        break;
    }
  }

  final String? problem = failure;
  if (problem != null) {
    throw ApiException(500, 'chat failed: $problem');
  }
  final ChatFinished? result = finished;
  if (result == null) {
    throw const ApiException(500, 'chat produced no result');
  }
  return jsonResponse(<String, Object?>{
    // A tool-only turn can finish with its markup stripped away to nothing; the
    // concatenated tokens are then the most faithful thing left to return.
    'text': result.text.isEmpty ? streamed.toString() : result.text,
    'generated_tokens': result.generatedTokens,
    'stop_reason': result.stopReason,
    'model': service.status(),
  });
}

/// A validated chat request.
class _ChatRequest {
  const _ChatRequest({
    required this.messages,
    required this.maxTokens,
    this.useMemory,
    this.thinking,
  });

  /// The conversation, oldest first.
  final List<ChatMessage> messages;

  /// The token budget for this turn.
  final int maxTokens;

  /// Per-turn memory opt-out, or null for the server default.
  final bool? useMemory;

  /// Per-turn reasoning depth, or null to let the engine choose.
  final String? thinking;
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

/// The kinds a client may name in `POST /api/v1/memory`.
///
/// Listed from [MemoryKind] rather than hard-coded so a new kind cannot be
/// added to the runtime without the API accepting it. A built list rather than
/// a const one, because the comprehension over the enum is not a constant
/// expression — the immutability the message needs is the enum's own.
final List<String> _memoryKinds = List<String>.unmodifiable(<String>[
  for (final MemoryKind kind in MemoryKind.values) kind.wire,
]);

/// `GET /api/v1/memory` — every stored memory, newest first.
Future<Response> _listMemory(HarborChatService service) async {
  await service.initialize();
  return jsonResponse(<String, Object?>{
    'entries': <Object?>[
      for (final MemoryEntry entry in service.memory.entries) entry.toJson(),
    ],
    'stats': service.memory.describe(),
  });
}

/// `POST /api/v1/memory` — remember one thing on the operator's behalf.
///
/// Importance defaults to 1.0 rather than the extractor's softer values: an
/// explicit API write is as deliberate as a user saying "remember this", and
/// the store prunes on importance, so a half-weighted manual entry would be
/// the first thing lost.
Future<Response> _addMemory(
  Request request,
  HarborChatService service,
  String adminToken,
  void Function(String) log,
) async {
  _requireAdmin(request, adminToken);
  final Map<String, Object?> body = await _readJsonObject(request);
  final String text = _requireString(body, 'text');

  final Object? rawKind = body['kind'];
  MemoryKind kind = MemoryKind.fact;
  if (rawKind != null) {
    if (rawKind is! String || MemoryKind.tryParse(rawKind) == null) {
      throw ValidationException(
        '"kind" must be one of ${_memoryKinds.join(', ')}; got "$rawKind"',
      );
    }
    kind = MemoryKind.tryParse(rawKind)!;
  }

  final Object? rawImportance = body['importance'];
  double importance = 1.0;
  if (rawImportance != null) {
    if (rawImportance is! num ||
        rawImportance < 0 ||
        rawImportance > 1) {
      throw const ValidationException(
        '"importance" must be a number between 0 and 1',
      );
    }
    importance = rawImportance.toDouble();
  }

  final Object? rawSubject = body['subject'];
  if (rawSubject != null && rawSubject is! String) {
    throw const ValidationException('"subject" must be a string');
  }

  await service.initialize();
  final MemoryEntry entry = service.memory.remember(
    text,
    kind: kind,
    subject: rawSubject is String ? rawSubject : null,
    importance: importance,
    source: 'api',
  );
  await service.memory.flush();
  log('memory: stored ${entry.kind.wire} ${entry.id}');
  return jsonResponse(<String, Object?>{'entry': entry.toJson()});
}

/// `DELETE /api/v1/memory/<id>` — forget one memory.
///
/// Reports `deleted: false` for an unknown id rather than a 404: the caller's
/// intent (this must not be remembered) is satisfied either way, and a boolean
/// keeps the dashboard's row removal idempotent under a double tap.
Future<Response> _forgetMemory(
  Request request,
  String id,
  HarborChatService service,
  String adminToken,
  void Function(String) log,
) async {
  _requireAdmin(request, adminToken);
  await service.initialize();
  final bool deleted = service.memory.forget(id);
  if (deleted) {
    await service.memory.flush();
    log('memory: forgot $id');
  }
  return jsonResponse(<String, Object?>{'deleted': deleted});
}

/// `POST /api/v1/memory/clear` — drop every memory.
Future<Response> _clearMemory(
  Request request,
  HarborChatService service,
  String adminToken,
  void Function(String) log,
) async {
  _requireAdmin(request, adminToken);
  await service.initialize();
  final int count = service.memory.length;
  await service.memory.clear();
  log('memory: cleared $count entries');
  return jsonResponse(<String, Object?>{'cleared': true, 'count': count});
}

// ---------------------------------------------------------------------------
// Admin auth
// ---------------------------------------------------------------------------

void _requireAdmin(Request request, String adminToken) {
  if (adminToken.isEmpty) {
    throw const ApiException(503, 'admin token not configured');
  }
  final String provided = request.headers['x-admin-token'] ?? '';
  if (!_constantTimeEquals(provided, adminToken)) {
    throw const ApiException(401, 'unauthorized');
  }
}

/// Compares two strings without an early exit on the first differing byte.
bool _constantTimeEquals(String a, String b) {
  if (a.length != b.length) {
    return false;
  }
  int difference = 0;
  for (int i = 0; i < a.length; i++) {
    difference |= a.codeUnitAt(i) ^ b.codeUnitAt(i);
  }
  return difference == 0;
}

// ---------------------------------------------------------------------------
// Body helpers
// ---------------------------------------------------------------------------

Future<Map<String, Object?>> _readJsonObject(Request request) async {
  final String raw = await request.readAsString();
  if (raw.trim().isEmpty) {
    throw const ValidationException('request body must not be empty');
  }
  final Object? decoded;
  try {
    decoded = jsonDecode(raw);
  } on FormatException catch (error) {
    throw ValidationException(
      'request body is not valid JSON: ${error.message}',
    );
  }
  if (decoded is! Map<String, Object?>) {
    throw const ValidationException('request body must be a JSON object');
  }
  return decoded;
}

String _requireString(Map<String, Object?> body, String key) {
  final Object? value = body[key];
  if (value is! String || value.trim().isEmpty) {
    throw ValidationException('"$key" is required and must be a non-empty '
        'string');
  }
  return value;
}

bool _requireBool(Map<String, Object?> body, String key) {
  final Object? value = body[key];
  if (value is! bool) {
    throw ValidationException('"$key" is required and must be a boolean');
  }
  return value;
}

bool? _optionalBool(Map<String, Object?> body, String key) {
  final Object? value = body[key];
  if (value == null) {
    return null;
  }
  if (value is! bool) {
    throw ValidationException('"$key" must be a boolean');
  }
  return value;
}

Map<String, Object?>? _optionalMap(Map<String, Object?> body, String key) {
  final Object? value = body[key];
  if (value == null) {
    return null;
  }
  if (value is! Map<Object?, Object?>) {
    throw ValidationException('"$key" must be a JSON object');
  }
  return <String, Object?>{
    for (final MapEntry<Object?, Object?> entry in value.entries)
      entry.key.toString(): entry.value,
  };
}