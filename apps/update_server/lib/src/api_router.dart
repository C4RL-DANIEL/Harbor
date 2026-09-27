// HTTP routing for the Harbor update server.
//
// This library is Flutter-free: it runs on the pure Dart VM.

import 'dart:convert';

import 'package:pub_semver/pub_semver.dart';
import 'package:shelf/shelf.dart';
import 'package:shelf_router/shelf_router.dart';

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
Router buildRouter({
  required ReleaseStore store,
  required String adminToken,
  void Function(String) log = _noopLog,
}) {
  final Router router = Router();

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
  'access-control-allow-methods': 'GET, POST, PUT, OPTIONS',
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
}) {
  final Router router =
      buildRouter(store: store, adminToken: adminToken, log: log);
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