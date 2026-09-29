// In-process web tools: fetching, reachability checks and JSON inspection.
//
// These tools are the only place in the package that performs outbound network
// I/O. The transport and the allow-list predicate are injected rather than
// constructed here, so the tests exercise every branch with zero sockets and an
// embedding application can point the tools at whatever client it already owns.

import 'dart:convert';

import 'package:http/http.dart' as http;

import 'tool.dart';

/// Builds the in-process web tools.
///
/// Tools that run in-process. [allow] gates every outbound request; [maxBytes]
/// caps the response body. Injected client and predicate keep this testable
/// with zero real network access.
List<Tool> webTools({
  required http.Client client,
  required bool Function(Uri uri) allow,
  int maxBytes = 256 * 1024,
}) {
  return <Tool>[
    _WebTool(
      name: 'web.fetch',
      description:
          'Fetches a URL and returns its status, byte count and decoded text.',
      parameters: _urlSchema(
        extra: <String, Object?>{'asText': _booleanType},
      ),
      body: (Map<String, Object?> args) => _fetch(client, allow, maxBytes, args),
    ),
    _WebTool(
      name: 'web.check',
      description: 'Checks whether a URL is reachable and reports its HTTP '
          'status.',
      parameters: _urlSchema(),
      body: (Map<String, Object?> args) => _check(client, allow, args),
    ),
    _WebTool(
      name: 'web.json',
      description:
          'Fetches a URL and reports the JSON type and a short preview.',
      parameters: _urlSchema(),
      body: (Map<String, Object?> args) => _json(client, allow, args),
    ),
  ];
}

/// A tool whose entire implementation is an injected async body.
///
/// [FunctionTool] still owns argument validation and exception containment, so
/// a body can signal an ordinary failure by throwing [_WebFailure] and never
/// has to remember to catch anything.
class _WebTool extends FunctionTool {
  _WebTool({
    required this.name,
    required this.description,
    required this.parameters,
    required this.body,
  });

  @override
  final String name;

  @override
  final String description;

  @override
  final Map<String, Object?> parameters;

  @override
  bool get mutating => false;

  final Future<Object?> Function(Map<String, Object?> args) body;

  @override
  Future<Object?> run(Map<String, Object?> args) => body(args);
}

/// A controlled failure whose text is the whole message.
///
/// [FunctionTool.invoke] renders a thrown error as `"$name: $error"`, so a
/// dedicated exception type keeps that prefix while avoiding the noisy
/// `Bad state:`-style decorations of the core exception classes.
class _WebFailure implements Exception {
  _WebFailure(this.message);

  final String message;

  @override
  String toString() => message;
}

/// The shared `{url: string}` schema, optionally extended with extra fields.
Map<String, Object?> _urlSchema({
  Map<String, Object?> extra = const <String, Object?>{},
}) {
  return <String, Object?>{
    'type': 'object',
    'properties': <String, Object?>{
      'url': _stringType,
      ...extra,
    },
    'required': <String>['url'],
  };
}

const Map<String, Object?> _stringType = <String, Object?>{'type': 'string'};
const Map<String, Object?> _booleanType = <String, Object?>{'type': 'boolean'};

/// Downloads the target URL and reports its status, size and body.
Future<Object?> _fetch(
  http.Client client,
  bool Function(Uri uri) allow,
  int maxBytes,
  Map<String, Object?> args,
) async {
  final Uri uri = _allowedUri(args, allow);
  final http.Response response = await _get(client, uri);
  final List<int> received = response.bodyBytes;
  final bool truncated = received.length > maxBytes;
  final List<int> kept =
      truncated ? received.sublist(0, maxBytes) : received;
  final String? contentType = _header(response.headers, 'content-type');
  final bool asText = optionalBoolArg(args, 'asText');
  final String decoded = utf8.decode(kept, allowMalformed: true);
  final String text =
      asText && _isHtml(contentType) ? _htmlToText(decoded) : decoded;
  return <String, Object?>{
    'status': response.statusCode,
    'contentType': contentType,
    // `bytes` counts what was actually returned, so `bytes <= maxBytes` always
    // holds; `truncated` says whether the origin sent more.
    'bytes': kept.length,
    'truncated': truncated,
    'text': text,
  };
}

/// Issues a request and reports whether the host answered at all.
Future<Object?> _check(
  http.Client client,
  bool Function(Uri uri) allow,
  Map<String, Object?> args,
) async {
  final Uri uri = _allowedUri(args, allow);
  try {
    final http.Response response = await _get(client, uri);
    return <String, Object?>{
      'status': response.statusCode,
      'reachable': true,
      'contentType': _header(response.headers, 'content-type'),
      'server': _header(response.headers, 'server'),
    };
  } catch (_) {
    // A connection failure is the answer to "is it reachable?", not an error in
    // the tool, so it is reported in-band rather than as a failure result.
    return <String, Object?>{
      'status': null,
      'reachable': false,
      'contentType': null,
      'server': null,
    };
  }
}

/// Fetches and classifies a JSON document.
Future<Object?> _json(
  http.Client client,
  bool Function(Uri uri) allow,
  Map<String, Object?> args,
) async {
  final Uri uri = _allowedUri(args, allow);
  final http.Response response = await _get(client, uri);
  if (response.statusCode < 200 || response.statusCode >= 300) {
    throw _WebFailure('HTTP ${response.statusCode}');
  }
  final Object? decoded;
  try {
    decoded = jsonDecode(utf8.decode(response.bodyBytes, allowMalformed: true));
  } on FormatException catch (error) {
    throw _WebFailure('invalid JSON: ${error.message}');
  }
  return <String, Object?>{
    'ok': true,
    'type': _jsonTypeName(decoded),
    'preview': _truncate(jsonEncode(decoded), 2000),
  };
}

/// Resolves and vets the `url` argument, throwing [_WebFailure] when unusable.
///
/// Centralised so `web.fetch`, `web.check` and `web.json` cannot drift into
/// different ideas of what "a URL the host allows" means.
Uri _allowedUri(Map<String, Object?> args, bool Function(Uri uri) allow) {
  final String raw = requireStringArg(args, 'url');
  final Uri? uri = Uri.tryParse(raw);
  if (uri == null ||
      !uri.isAbsolute ||
      (uri.scheme != 'http' && uri.scheme != 'https')) {
    throw _WebFailure('"$raw" is not an absolute http(s) URL');
  }
  if (!allow(uri)) {
    // The message names the host because that is the value the user has to
    // change in the allow-list to make the call succeed.
    throw _WebFailure('host "${uri.host}" is not allowed');
  }
  return uri;
}

/// Performs the GET, converting any transport error into a [_WebFailure].
Future<http.Response> _get(http.Client client, Uri uri) async {
  try {
    return await client.get(uri);
  } catch (error) {
    throw _WebFailure('request failed: $error');
  }
}

/// Case-insensitive header lookup, since header casing is not guaranteed.
String? _header(Map<String, String> headers, String name) {
  final String lowered = name.toLowerCase();
  for (final MapEntry<String, String> entry in headers.entries) {
    if (entry.key.toLowerCase() == lowered) {
      return entry.value;
    }
  }
  return null;
}

bool _isHtml(String? contentType) {
  return contentType != null && contentType.toLowerCase().contains('html');
}

/// Flattens HTML into readable text by dropping scripts, styles and tags.
///
/// Deliberately a small regex pass rather than a real parser: the goal is to
/// make a page legible to a language model, not to preserve document structure,
/// and this package must stay dependency-free.
String _htmlToText(String html) {
  final String stripped = html
      .replaceAll(
        RegExp(
          r'<script\b[^>]*>.*?</script>',
          caseSensitive: false,
          dotAll: true,
        ),
        ' ',
      )
      .replaceAll(
        RegExp(
          r'<style\b[^>]*>.*?</style>',
          caseSensitive: false,
          dotAll: true,
        ),
        ' ',
      )
      .replaceAll(RegExp(r'<!--.*?-->', dotAll: true), ' ')
      .replaceAll(RegExp(r'<[^>]+>'), ' ');
  final String decoded = stripped
      .replaceAll('&nbsp;', ' ')
      .replaceAll('&amp;', '&')
      .replaceAll('&lt;', '<')
      .replaceAll('&gt;', '>')
      .replaceAll('&quot;', '"')
      .replaceAll('&#39;', "'");
  return decoded.replaceAll(RegExp(r'\s+'), ' ').trim();
}

/// Shortens [text] to at most [maxChars] characters, ellipsis included.
String _truncate(String text, int maxChars) {
  if (maxChars <= 0) {
    return '';
  }
  if (text.length <= maxChars) {
    return text;
  }
  if (maxChars == 1) {
    return '…';
  }
  return '${text.substring(0, maxChars - 1)}…';
}

/// The JSON type name of [value], matching `jsonDecode`'s possible results.
String _jsonTypeName(Object? value) {
  if (value == null) {
    return 'null';
  }
  if (value is bool) {
    return 'boolean';
  }
  if (value is num) {
    return 'number';
  }
  if (value is String) {
    return 'string';
  }
  if (value is List) {
    return 'array';
  }
  if (value is Map) {
    return 'object';
  }
  return 'unknown';
}