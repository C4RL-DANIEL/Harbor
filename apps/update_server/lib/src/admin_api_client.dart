// Typed HTTP client for every Harbor update-server route.
//
// This library is Flutter-free so it can also be driven from CLI tooling; the
// Flutter Web dashboard (lib/main.dart) is its primary consumer.

import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import 'models.dart';

/// Thrown when the client cannot reach the server at all.
class NetworkException extends ApiException {
  const NetworkException(String message) : super(0, message);
}

/// A small, typed wrapper around the update-server admin API.
class AdminApiClient {
  AdminApiClient({
    required String baseUrl,
    String adminToken = '',
    http.Client? httpClient,
    Duration timeout = const Duration(seconds: 10),
  })  : baseUrl = _normalizeBaseUrl(baseUrl),
        adminToken = adminToken.trim(),
        _http = httpClient ?? http.Client(),
        _timeout = timeout;

  /// Base origin, without a trailing slash (for example
  /// `http://localhost:8080`).
  final String baseUrl;

  /// Value sent as `X-Admin-Token` on mutating routes.
  final String adminToken;

  final http.Client _http;
  final Duration _timeout;

  static String _normalizeBaseUrl(String value) {
    String normalized = value.trim();
    while (normalized.endsWith('/')) {
      normalized = normalized.substring(0, normalized.length - 1);
    }
    return normalized;
  }

  Uri _uri(String path, [Map<String, String>? query]) {
    final Uri uri = Uri.parse('$baseUrl$path');
    if (query == null || query.isEmpty) {
      return uri;
    }
    return uri.replace(queryParameters: query);
  }

  // ---- Non-admin reads ----------------------------------------------------

  /// `GET /health`.
  Future<Map<String, Object?>> health() => _send('GET', '/health');

  /// Reads [path], falling back to [staticPath] when the live server is not
  /// hosted (a 404), so the dashboard still works against a static Pages site.
  Future<Map<String, Object?>> _readWithStaticFallback({
    required String path,
    required String staticPath,
    Map<String, String>? query,
  }) async {
    try {
      return await _send('GET', path, query: query);
    } on ApiException catch (error) {
      if (error.statusCode != 404) {
        rethrow;
      }
      return _send('GET', staticPath);
    }
  }

  /// `GET /api/v1/flags`, falling back to the static `/api/v1/flags.json`.
  Future<FeatureFlagMatrix> fetchFlags() async {
    final Map<String, Object?> json = await _readWithStaticFallback(
      path: '/api/v1/flags',
      staticPath: '/api/v1/flags.json',
    );
    return FeatureFlagMatrix.fromJson(json);
  }

  /// `GET /api/v1/update-check`, falling back to the static
  /// `/latest_version.json`. The static document is release-wide, so it ignores
  /// the installed version; the caller compares versions itself.
  Future<UpdateCheckResponse> updateCheck({
    required String installedVersion,
    required String platform,
  }) async {
    final Map<String, Object?> json = await _readWithStaticFallback(
      path: '/api/v1/update-check',
      staticPath: '/latest_version.json',
      query: <String, String>{
        'version': installedVersion,
        'platform': platform,
      },
    );
    return UpdateCheckResponse.fromJson(json);
  }

  /// `GET /api/v1/releases`.
  Future<Map<String, ReleaseInfo>> listReleases() async {
    final Map<String, Object?> json = await _send('GET', '/api/v1/releases');
    final Object? raw = json['releases'];
    if (raw is! Map<Object?, Object?>) {
      throw const ApiException(500, 'release list response is malformed');
    }
    return <String, ReleaseInfo>{
      for (final MapEntry<Object?, Object?> entry in raw.entries)
        entry.key.toString():
            ReleaseInfo.fromJson(_stringMap(entry.value)),
    };
  }

  /// `GET /admin/state`.
  Future<ServerState> adminState() async {
    final Map<String, Object?> json = await _send('GET', '/admin/state');
    return ServerState.fromJson(json);
  }

  /// `GET /latest_version.json`.
  Future<Map<String, Object?>> latestVersionJson() =>
      _send('GET', '/latest_version.json');

  // ---- Admin mutations ----------------------------------------------------

  /// `PUT /api/v1/flags`; returns the new matrix revision.
  Future<int> putFlags({
    required Map<String, Object?> flags,
    required Map<String, Object?> remoteDefaults,
    required Map<String, Object?> layout,
    required bool merge,
  }) async {
    final Map<String, Object?> json = await _send(
      'PUT',
      '/api/v1/flags',
      admin: true,
      body: <String, Object?>{
        'flags': flags,
        'remote_defaults': remoteDefaults,
        'layout': layout,
        'merge': merge,
      },
    );
    final Object? version = json['version'];
    if (version is int) {
      return version;
    }
    if (version is num) {
      return version.toInt();
    }
    throw const ApiException(500, 'flag update response is missing "version"');
  }

  /// `POST /api/v1/releases`; returns the stored release record.
  Future<ReleaseInfo> publishRelease(ReleaseInfo release) async {
    final Map<String, Object?> json = await _send(
      'POST',
      '/api/v1/releases',
      admin: true,
      body: release.toJson(),
    );
    final Object? stored = json['release'];
    if (stored is! Map<Object?, Object?>) {
      throw const ApiException(
        500,
        'publish response is missing the "release" object',
      );
    }
    return ReleaseInfo.fromJson(_stringMap(stored));
  }

  /// `PUT /api/v1/config/min-supported`.
  Future<void> setMinSupported({
    required String platform,
    required String minSupportedVersion,
    bool? forceUpdate,
  }) async {
    await _send(
      'PUT',
      '/api/v1/config/min-supported',
      admin: true,
      body: <String, Object?>{
        'platform': platform,
        'min_supported_version': minSupportedVersion,
        if (forceUpdate != null) 'force_update': forceUpdate,
      },
    );
  }

  /// `POST /api/v1/config/force`.
  Future<void> setForceUpdate({
    required String platform,
    required bool forceUpdate,
  }) async {
    await _send(
      'POST',
      '/api/v1/config/force',
      admin: true,
      body: <String, Object?>{
        'platform': platform,
        'force_update': forceUpdate,
      },
    );
  }

  /// Releases the underlying HTTP connection pool.
  void close() => _http.close();

  // ---- Internals ----------------------------------------------------------

  Future<Map<String, Object?>> _send(
    String method,
    String path, {
    Map<String, String>? query,
    Object? body,
    bool admin = false,
  }) async {
    final Uri uri = _uri(path, query);
    final Map<String, String> headers = <String, String>{
      'accept': 'application/json',
    };
    String? encoded;
    if (body != null) {
      headers['content-type'] = 'application/json; charset=utf-8';
      encoded = jsonEncode(body);
    }
    if (admin && adminToken.isNotEmpty) {
      headers['x-admin-token'] = adminToken;
    }

    final http.Response response;
    try {
      final Future<http.Response> pending;
      switch (method) {
        case 'GET':
          pending = _http.get(uri, headers: headers);
          break;
        case 'PUT':
          pending = _http.put(uri, headers: headers, body: encoded);
          break;
        case 'POST':
          pending = _http.post(uri, headers: headers, body: encoded);
          break;
        default:
          throw ArgumentError.value(method, 'method', 'unsupported HTTP verb');
      }
      response = await pending.timeout(_timeout);
    } on TimeoutException {
      throw NetworkException(
        '$method $uri timed out after ${_timeout.inSeconds}s',
      );
    } on http.ClientException catch (error) {
      throw NetworkException('$method $uri failed: ${error.message}');
    } on FormatException catch (error) {
      throw NetworkException('$method $uri returned an invalid URI: $error');
    }

    return _decode(response, method: method, uri: uri);
  }

  Map<String, Object?> _decode(
    http.Response response, {
    required String method,
    required Uri uri,
  }) {
    final String raw = utf8.decode(response.bodyBytes, allowMalformed: true);
    Object? decoded;
    if (raw.trim().isNotEmpty) {
      try {
        decoded = jsonDecode(raw);
      } on FormatException {
        decoded = null;
      }
    }

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw ApiException(
        response.statusCode,
        _errorMessage(response.statusCode, decoded, raw),
      );
    }
    if (decoded == null) {
      throw ApiException(
        response.statusCode,
        '$method $uri returned an empty or non-JSON body',
      );
    }
    if (decoded is! Map<Object?, Object?>) {
      throw ApiException(
        response.statusCode,
        '$method $uri returned ${decoded.runtimeType}, expected a JSON object',
      );
    }
    return _stringMap(decoded);
  }

  static String _errorMessage(int statusCode, Object? decoded, String raw) {
    if (decoded is Map<Object?, Object?>) {
      final Object? error = decoded['error'];
      if (error is String && error.trim().isNotEmpty) {
        return error;
      }
    }
    final String trimmed = raw.trim();
    if (trimmed.isNotEmpty) {
      return trimmed;
    }
    return 'HTTP $statusCode';
  }

  static Map<String, Object?> _stringMap(Object? value) {
    if (value is! Map<Object?, Object?>) {
      return <String, Object?>{};
    }
    return <String, Object?>{
      for (final MapEntry<Object?, Object?> entry in value.entries)
        entry.key.toString(): entry.value,
    };
  }
}