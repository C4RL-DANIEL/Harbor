// Harbor update server entry point.
//
// Pure Dart VM executable: `dart run bin/server.dart --port 8080`.
// It serves the JSON API (lib/src/api_router.dart) and, when the Flutter Web
// admin dashboard has been built, the static `build/web` bundle at `/`.

import 'dart:async';
import 'dart:io';

import 'package:args/args.dart';
import 'package:shelf/shelf.dart';
import 'package:shelf/shelf_io.dart' as shelf_io;
import 'package:shelf_router/shelf_router.dart';
import 'package:shelf_static/shelf_static.dart';

import 'package:update_server/src/api_router.dart';
import 'package:update_server/src/lifecycle.dart';
import 'package:update_server/src/models.dart';
import 'package:update_server/src/release_store.dart';

const String _serviceName = 'harbor-update-server';

/// The token accepted when `HARBOR_ALLOW_DEV_TOKEN=true` and no real token is
/// configured. Never enable this in production.
const String _devAdminToken = 'dev';

/// Upper bound on each step of a SIGTERM/SIGINT shutdown: how long stopping the
/// listener may take, and how long in-flight requests are given to finish
/// before their sockets are destroyed. A shutdown therefore completes in at
/// most roughly twice this value.
const Duration _shutdownGrace = Duration(seconds: 5);

void main(List<String> arguments) async {
  final ArgParser parser = _buildArgParser();
  final ArgResults options;
  try {
    options = parser.parse(arguments);
  } on FormatException catch (error) {
    stderr.writeln('${error.message}\n');
    stderr.writeln(parser.usage);
    exitCode = 64;
    return;
  }

  if (options.flag('help')) {
    stdout.writeln('Harbor update server\n');
    stdout.writeln(parser.usage);
    return;
  }

  final int? port = int.tryParse(options.option('port') ?? '');
  final String host = options.option('host') ?? '0.0.0.0';
  final String stateFile =
      options.option('state-file') ?? './data/state.json';

  if (port == null || port < 0 || port > 65535) {
    stderr.writeln(
      'Invalid --port ${options.option('port')} (expected an integer '
      'between 0 and 65535).',
    );
    exitCode = 64;
    return;
  }

  final String adminToken = _resolveAdminToken();
  final ReleaseStore store = ReleaseStore(stateFile);

  try {
    await store.load();
  } on StateStoreException catch (error) {
    stderr.writeln('Fatal: ${error.message}');
    exitCode = 1;
    return;
  }

  final Directory? webDirectory = _findWebDirectory();
  final Router router = buildRouter(
    store: store,
    adminToken: adminToken,
    log: _log,
  );
  final InFlightTracker tracker = InFlightTracker();
  final Handler handler = const Pipeline()
      .addMiddleware(_logMiddleware(_log))
      .addMiddleware(corsMiddleware())
      .addMiddleware(apiErrorMiddleware(_log))
      .addMiddleware(etagMiddleware())
      .addMiddleware(tracker.wrap)
      .addHandler(_composeHandler(router, webDirectory));

  final HttpServer server;
  try {
    server = await shelf_io.serve(
      handler,
      host,
      port,
      poweredByHeader: _serviceName,
    );
  } on SocketException catch (error) {
    stderr.writeln(
      'Fatal: unable to bind $host:$port (${error.osError?.message ?? error.message}). '
      'Is another process already listening on that port?',
    );
    exitCode = 1;
    return;
  }

  _log('listening on http://${server.address.host}:${server.port}');
  _log('state file: ${store.path ?? '<in-memory>'}');
  _log(
    adminToken.isEmpty
        ? 'admin mutations disabled: set ADMIN_TOKEN (or HARBOR_ALLOW_DEV_TOKEN=true for the "dev" token)'
        : 'admin token configured (${adminToken.length} chars)',
  );
  if (webDirectory != null) {
    _log('serving admin dashboard from ${webDirectory.path}');
  } else {
    _log('no build/web found; serving the JSON landing page at /');
  }

  await _awaitShutdown(server, store, tracker);

  // Signal subscriptions and the HTTP server keep the Dart event loop alive
  // after main() returns, so a process that only awaits its shutdown work
  // would linger until the supervisor SIGKILLs it. Terminate explicitly once
  // the state has been flushed.
  await stdout.flush();
  exit(0);
}

ArgParser _buildArgParser() {
  final String defaultHost = Platform.environment['HARBOR_HOST'] ?? '0.0.0.0';
  final String defaultPort = Platform.environment['HARBOR_PORT'] ??
      Platform.environment['PORT'] ??
      '8080';
  final String defaultStateFile =
      Platform.environment['HARBOR_STATE_FILE'] ?? './data/state.json';

  return ArgParser()
    ..addOption(
      'port',
      abbr: 'p',
      defaultsTo: defaultPort,
      help: 'TCP port to listen on.',
    )
    ..addOption(
      'host',
      defaultsTo: defaultHost,
      help: 'Interface to bind (0.0.0.0 for all).',
    )
    ..addOption(
      'state-file',
      defaultsTo: defaultStateFile,
      help: 'JSON file used to persist releases and flags.',
    )
    ..addFlag(
      'help',
      abbr: 'h',
      negatable: false,
      help: 'Show this usage information.',
    );
}

String _resolveAdminToken() {
  final Map<String, String> env = Platform.environment;
  final String configured =
      (env['HARBOR_ADMIN_TOKEN'] ?? env['ADMIN_TOKEN'] ?? '').trim();
  if (configured.isNotEmpty) {
    return configured;
  }
  if ((env['HARBOR_ALLOW_DEV_TOKEN'] ?? '').toLowerCase() == 'true') {
    _log('HARBOR_ALLOW_DEV_TOKEN=true: accepting the documented dev token');
    return _devAdminToken;
  }
  return '';
}

// ---------------------------------------------------------------------------
// Static dashboard / landing page
// ---------------------------------------------------------------------------

Directory? _findWebDirectory() {
  final Directory scriptDirectory = _scriptDirectory();
  final List<Directory> candidates = <Directory>[
    Directory('${Directory.current.path}/build/web'),
    Directory('${scriptDirectory.parent.path}/build/web'),
    Directory('${scriptDirectory.path}/build/web'),
  ];
  for (final Directory candidate in candidates) {
    if (File('${candidate.path}/index.html').existsSync()) {
      return candidate;
    }
  }
  return null;
}

Directory _scriptDirectory() {
  final Uri script = Platform.script;
  if (script.scheme == 'file') {
    return File.fromUri(script).parent;
  }
  return Directory.current;
}

Handler _composeHandler(Router router, Directory? webDirectory) {
  if (webDirectory != null) {
    final Handler staticHandler = createStaticHandler(
      webDirectory.path,
      defaultDocument: 'index.html',
      listDirectories: false,
    );
    return (Request request) async {
      final Response apiResponse = await router.call(request);
      if (apiResponse.statusCode != 404) {
        return apiResponse;
      }
      final Response staticResponse = await staticHandler(request);
      if (staticResponse.statusCode != 404) {
        return staticResponse;
      }
      // Single-page-app fallback: extension-less paths render index.html.
      final String path = request.url.path;
      if (request.method == 'GET' &&
          !path.contains('.') &&
          !path.startsWith('api/') &&
          !path.startsWith('admin/')) {
        final File index = File('${webDirectory.path}/index.html');
        if (await index.exists()) {
          return Response.ok(
            await index.readAsBytes(),
            headers: <String, String>{
              'content-type': 'text/html; charset=utf-8',
            },
          );
        }
      }
      return apiResponse;
    };
  }

  return (Request request) async {
    final String path = request.url.path;
    if (request.method == 'GET' && (path.isEmpty || path == '/')) {
      return _landingResponse();
    }
    return router.call(request);
  };
}

Response _landingResponse() {
  const List<Map<String, Object?>> routes = <Map<String, Object?>>[
    <String, Object?>{'method': 'GET', 'path': '/health', 'auth': false},
    <String, Object?>{
      'method': 'GET',
      'path': '/api/v1/update-check?version=&platform=',
      'auth': false,
    },
    <String, Object?>{'method': 'GET', 'path': '/api/v1/flags', 'auth': false},
    <String, Object?>{'method': 'PUT', 'path': '/api/v1/flags', 'auth': true},
    <String, Object?>{'method': 'GET', 'path': '/api/v1/releases', 'auth': false},
    <String, Object?>{'method': 'POST', 'path': '/api/v1/releases', 'auth': true},
    <String, Object?>{
      'method': 'PUT',
      'path': '/api/v1/config/min-supported',
      'auth': true,
    },
    <String, Object?>{
      'method': 'POST',
      'path': '/api/v1/config/force',
      'auth': true,
    },
    <String, Object?>{'method': 'GET', 'path': '/admin/state', 'auth': false},
    <String, Object?>{
      'method': 'GET',
      'path': '/latest_version.json',
      'auth': false,
    },
  ];
  return jsonResponse(<String, Object?>{
    'service': _serviceName,
    'status': 'ok',
    'dashboard': 'not built (run "flutter build web" to serve /admin)',
    'routes': routes,
  });
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

Middleware _logMiddleware(void Function(String) log) {
  return (Handler inner) {
    return (Request request) async {
      final Stopwatch stopwatch = Stopwatch()..start();
      try {
        final Response response = await inner(request);
        log(
          '${request.method} /${request.url.path} -> ${response.statusCode} '
          '(${stopwatch.elapsedMilliseconds}ms)',
        );
        return response;
      } on Object catch (error) {
        log(
          '${request.method} /${request.url.path} -> exception after '
          '${stopwatch.elapsedMilliseconds}ms: $error',
        );
        rethrow;
      }
    };
  };
}

// CORS + ETag middleware live in lib/src/api_router.dart so the exact same
// implementations can be exercised by the HTTP tests.

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

Future<void> _awaitShutdown(
  HttpServer server,
  ReleaseStore store,
  InFlightTracker tracker,
) async {
  final Completer<void> finished = Completer<void>();
  bool started = false;

  Future<void> shutdown(String signal) async {
    if (started) {
      _log('ignoring $signal: shutdown is already in progress');
      return;
    }
    started = true;
    _log('received $signal; stopping the listener');
    try {
      // 1. Stop accepting connections. Requests already being served continue.
      try {
        await server.close(force: false).timeout(_shutdownGrace);
      } on TimeoutException {
        _log('stopping the listener exceeded ${_shutdownGrace.inSeconds}s');
      }

      // 2. Give those requests a bounded window to finish, so a client is not
      //    cut off by the exit that follows.
      try {
        await tracker.drained.timeout(_shutdownGrace);
      } on TimeoutException {
        _log('${tracker.active} request(s) still in flight after '
            '${_shutdownGrace.inSeconds}s; forcing connections closed');
      }

      // 3. Destroy whatever is left, then persist state.
      try {
        await server.close(force: true);
      } on Object catch (error) {
        stderr.writeln('Force-closing connections failed: $error');
      }
      await store.flush();
      _log('state flushed; shutdown complete');
    } on Object catch (error, stackTrace) {
      stderr.writeln('Shutdown failed: $error\n$stackTrace');
    } finally {
      finished.complete();
    }
  }

  ProcessSignal.sigint.watch().listen((_) {
    unawaited(shutdown('SIGINT'));
  });
  if (!Platform.isWindows) {
    ProcessSignal.sigterm.watch().listen((_) {
      unawaited(shutdown('SIGTERM'));
    });
  }

  await finished.future;
}

void _log(String message) {
  stdout.writeln(
    '[$_serviceName] ${DateTime.now().toUtc().toIso8601String()} $message',
  );
}