// End-to-end and unit tests for the Harbor update server.
//
// The tests import `package:flutter_test` so they run under `flutter test` in
// the real package; the pure-Dart verification mirror rewrites that import to
// `package:test/test.dart`.

import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:pub_semver/pub_semver.dart';
import 'package:shelf/shelf_io.dart' as shelf_io;
import 'package:update_server/src/admin_api_client.dart';
import 'package:update_server/src/api_router.dart';
import 'package:update_server/src/models.dart';
import 'package:update_server/src/release_store.dart';

const String _token = 'test-admin-token';
const String _shaA =
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
const String _shaB =
    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';

ReleaseInfo _release({
  String platform = 'android',
  String version = '1.2.0',
  String minSupported = '1.1.0',
  bool forceUpdate = false,
  String downloadUrl =
      'https://github.com/OWNER/REPO/releases/download/v1.2.0/app-release.apk',
  String sha256 = _shaA,
  int sizeBytes = 24891342,
}) {
  return ReleaseInfo(
    platform: platform,
    version: version,
    minSupportedVersion: minSupported,
    downloadUrl: downloadUrl,
    sha256: sha256,
    sizeBytes: sizeBytes,
    changelog: '• release $version',
    forceUpdate: forceUpdate,
    releaseNotesUrl:
        'https://github.com/OWNER/REPO/releases/tag/v$version',
    publishedAt: DateTime.utc(2026, 2, 1, 10),
  );
}

class _TestServer {
  _TestServer(this.server, this.store, this.client);

  final HttpServer server;
  final ReleaseStore store;
  final AdminApiClient client;

  Uri uri(String path, [Map<String, String>? query]) {
    final Uri base =
        Uri.parse('http://127.0.0.1:${server.port}$path');
    return query == null ? base : base.replace(queryParameters: query);
  }

  Future<void> close() async {
    client.close();
    await server.close(force: true);
  }
}

Future<_TestServer> _start({
  String token = _token,
  ServerState? seed,
}) async {
  final ReleaseStore store = ReleaseStore.inMemory();
  if (seed != null) {
    await store.save(seed);
  } else {
    await store.load();
  }
  final handler = buildApiHandler(
    store: store,
    adminToken: token,
    log: (String _) {},
  );
  final HttpServer server =
      await shelf_io.serve(handler, InternetAddress.loopbackIPv4, 0);
  final AdminApiClient client = AdminApiClient(
    baseUrl: 'http://127.0.0.1:${server.port}',
    adminToken: token,
  );
  return _TestServer(server, store, client);
}

Map<String, Object?> _json(http.Response response) {
  final Object? decoded = jsonDecode(response.body);
  return decoded! as Map<String, Object?>;
}

void main() {
  group('empty server', () {
    late _TestServer server;

    setUp(() async {
      server = await _start(seed: ServerState.empty());
    });
    tearDown(() async => server.close());

    test('update-check returns the documented zero payload', () async {
      final http.Response response = await http.get(
        server.uri('/api/v1/update-check',
            <String, String>{'version': '1.0.0', 'platform': 'android'}),
      );
      expect(response.statusCode, 200);
      expect(response.headers['content-type'], contains('application/json'));
      final Map<String, Object?> body = _json(response);
      expect(body['latest_version'], '0.0.0');
      expect(body['min_supported_version'], '0.0.0');
      expect(body['download_url'], '');
      expect(body['sha256'], '');
      expect(body['size_bytes'], 0);
      expect(body['changelog'], '');
      expect(body['force_update'], false);
      expect(body['update_available'], false);
      expect(body['update_required'], false);
      expect(body['platform'], 'android');
      expect(body['release_notes_url'], '');
    });

    test('health reports a healthy empty store', () async {
      final Map<String, Object?> body = await server.client.health();
      expect(body['ok'], true);
      expect(body['status'], 'healthy');
      expect(body['version_count'], 0);
      expect(DateTime.tryParse(body['time']! as String), isNotNull);
    });
  });

  group('update decision math', () {
    test('equal version: no update', () {
      final UpdateCheckResponse response = UpdateCheckResponse.evaluate(
        release: _release(),
        installed: Version.parse('1.2.0'),
        platform: 'android',
      );
      expect(response.updateAvailable, false);
      expect(response.updateRequired, false);
    });

    test('patch bump: available but not required', () {
      final UpdateCheckResponse response = UpdateCheckResponse.evaluate(
        release: _release(),
        installed: Version.parse('1.1.5'),
        platform: 'android',
      );
      expect(response.updateAvailable, true);
      expect(response.updateRequired, false);
    });

    test('below minimum: available and required', () {
      final UpdateCheckResponse response = UpdateCheckResponse.evaluate(
        release: _release(),
        installed: Version.parse('1.0.0'),
        platform: 'android',
      );
      expect(response.updateAvailable, true);
      expect(response.updateRequired, true);
    });

    test('newer than latest: neither flag', () {
      final UpdateCheckResponse response = UpdateCheckResponse.evaluate(
        release: _release(),
        installed: Version.parse('2.0.0'),
        platform: 'android',
      );
      expect(response.updateAvailable, false);
      expect(response.updateRequired, false);
    });

    test('pre-release ordering uses pub_semver', () {
      final UpdateCheckResponse response = UpdateCheckResponse.evaluate(
        release: _release(version: '1.2.0', minSupported: '1.0.0'),
        installed: Version.parse('1.2.0-beta.1'),
        platform: 'android',
      );
      expect(response.updateAvailable, true);
      expect(response.updateRequired, false);
    });

    test('force flag forces update_required even when current', () {
      final UpdateCheckResponse response = UpdateCheckResponse.evaluate(
        release: _release(
          version: '1.0.0',
          minSupported: '1.0.0',
          forceUpdate: true,
        ),
        installed: Version.parse('1.0.0'),
        platform: 'android',
      );
      expect(response.updateAvailable, false);
      expect(response.updateRequired, true);
      expect(response.forceUpdate, true);
    });
  });

  group('validation', () {
    test('sha256 must be 64 hex chars', () {
      expect(
        () => _release(sha256: 'abc'),
        throwsA(isA<ValidationException>()),
      );
    });

    test('sha256 is normalized to lowercase', () {
      final ReleaseInfo release = _release(sha256: _shaA.toUpperCase());
      expect(release.sha256, _shaA);
    });

    test('unsupported platform is rejected', () {
      expect(
        () => _release(platform: 'symbian'),
        throwsA(isA<ValidationException>()),
      );
    });

    test('negative size is rejected', () {
      expect(
        () => _release(sizeBytes: -1),
        throwsA(isA<ValidationException>()),
      );
    });

    test('relative download URL is rejected', () {
      expect(
        () => _release(downloadUrl: '/releases/app.apk'),
        throwsA(isA<ValidationException>()),
      );
    });

    test('malformed semver is rejected with 400', () {
      expect(
        () => parseSemver('not-a-version', field: 'version'),
        throwsA(
          isA<ValidationException>()
              .having((ValidationException e) => e.statusCode, 'status', 400),
        ),
      );
    });
  });

  group('flag matrix merge', () {
    FeatureFlagMatrix base() => FeatureFlagMatrix(
          version: 1,
          updatedAt: DateTime.utc(2026, 1, 1),
          flags: const <String, Object?>{'dynamic_ui': true, 'labs': false},
          remoteDefaults: const <String, Object?>{'dynamic_ui': true},
          layout: const <String, Object?>{
            'sections': <Object?>[
              <String, Object?>{
                'id': 'home',
                'title': 'Home',
                'order': 0,
                'modules': <Object?>[
                  <String, Object?>{
                    'id': 'm1',
                    'type': 'thinking_panel',
                    'flag': 'agent.thinking',
                    'order': 5,
                  },
                ],
              },
            ],
          },
        );

    test('merge=true deep-merges maps and bumps the version', () {
      final FeatureFlagMatrix merged = base().merge(
        FeatureFlagMatrix(
          version: 1,
          updatedAt: DateTime.utc(2026, 2, 1),
          flags: const <String, Object?>{'labs': true, 'new_flag': 3},
          remoteDefaults: const <String, Object?>{'extra': 'x'},
          layout: const <String, Object?>{},
        ),
        merge: true,
      );
      expect(merged.version, 2);
      expect(merged.flags, <String, Object?>{
        'dynamic_ui': true,
        'labs': true,
        'new_flag': 3,
      });
      expect(merged.remoteDefaults, <String, Object?>{
        'dynamic_ui': true,
        'extra': 'x',
      });
      expect(merged.updatedAt, DateTime.utc(2026, 2, 1));
    });

    test('merge=false replaces flags and layout', () {
      final FeatureFlagMatrix replaced = base().merge(
        FeatureFlagMatrix(
          version: 1,
          updatedAt: DateTime.utc(2026, 2, 1),
          flags: const <String, Object?>{'only': true},
          remoteDefaults: const <String, Object?>{},
          layout: const <String, Object?>{
            'sections': <Object?>[
              <String, Object?>{'id': 'labs', 'title': 'Labs', 'order': 1},
            ],
          },
        ),
        merge: false,
      );
      expect(replaced.version, 2);
      expect(replaced.flags, <String, Object?>{'only': true});
      expect(replaced.remoteDefaults, isEmpty);
      final List<Object?> sections =
          replaced.layout['sections']! as List<Object?>;
      expect(sections.length, 1);
      expect((sections.first! as Map<String, Object?>)['id'], 'labs');
    });

    test('layout merge keys sections by id and sorts modules by order', () {
      final FeatureFlagMatrix merged = base().merge(
        FeatureFlagMatrix(
          version: 1,
          updatedAt: DateTime.utc(2026, 2, 1),
          flags: const <String, Object?>{},
          remoteDefaults: const <String, Object?>{},
          layout: const <String, Object?>{
            'sections': <Object?>[
              <String, Object?>{
                'id': 'home',
                'title': 'Home',
                'order': 0,
                'modules': <Object?>[
                  <String, Object?>{
                    'id': 'm0',
                    'type': 'update_status',
                    'flag': '',
                    'order': 1,
                  },
                ],
              },
            ],
          },
        ),
        merge: true,
      );
      final List<Object?> sections =
          merged.layout['sections']! as List<Object?>;
      expect(sections.length, 1);
      final Map<String, Object?> home = sections.first! as Map<String, Object?>;
      final List<Object?> modules = home['modules']! as List<Object?>;
      expect(modules.length, 2);
      expect((modules[0]! as Map<String, Object?>)['id'], 'm0');
      expect((modules[1]! as Map<String, Object?>)['id'], 'm1');
    });

    test('invalid layout is rejected', () {
      expect(
        () => FeatureFlagMatrix.validateLayout(<String, Object?>{
          'sections': <Object?>[
            <String, Object?>{'title': 'missing id'},
          ],
        }),
        throwsA(isA<ValidationException>()),
      );
    });
  });

  group('admin authentication', () {
    test('missing token is 401 with the error envelope', () async {
      final _TestServer server = await _start();
      addTearDown(server.close);
      final http.Response response = await http.put(
        server.uri('/api/v1/flags'),
        headers: <String, String>{'content-type': 'application/json'},
        body: jsonEncode(<String, Object?>{'flags': <String, Object?>{}}),
      );
      expect(response.statusCode, 401);
      expect(_json(response)['error'], 'unauthorized');
    });

    test('wrong token is 401', () async {
      final _TestServer server = await _start();
      addTearDown(server.close);
      final http.Response response = await http.put(
        server.uri('/api/v1/flags'),
        headers: <String, String>{
          'content-type': 'application/json',
          'x-admin-token': 'nope',
        },
        body: jsonEncode(<String, Object?>{'flags': <String, Object?>{}}),
      );
      expect(response.statusCode, 401);
      expect(_json(response)['error'], 'unauthorized');
    });

    test('unconfigured token is 503', () async {
      final _TestServer server = await _start(token: '');
      addTearDown(server.close);
      final http.Response response = await http.post(
        server.uri('/api/v1/config/force'),
        headers: <String, String>{
          'content-type': 'application/json',
          'x-admin-token': '',
        },
        body: jsonEncode(<String, Object?>{
          'platform': 'android',
          'force_update': true,
        }),
      );
      expect(response.statusCode, 503);
      expect(_json(response)['error'], 'admin token not configured');
    });
  });

  group('release publishing', () {
    test('publish 201 then update-check reflects it, then force', () async {
      final _TestServer server = await _start(seed: ServerState.empty());
      addTearDown(server.close);

      final ReleaseInfo published = await server.client.publishRelease(
        _release(version: '1.2.0', minSupported: '1.1.0'),
      );
      expect(published.version, '1.2.0');
      expect(published.sha256, _shaA);

      final UpdateCheckResponse below = await server.client.updateCheck(
        installedVersion: '1.0.0',
        platform: 'android',
      );
      expect(below.latestVersion, '1.2.0');
      expect(below.minSupportedVersion, '1.1.0');
      expect(below.updateAvailable, true);
      expect(below.updateRequired, true);
      expect(below.downloadUrl, contains('app-release.apk'));

      final UpdateCheckResponse current = await server.client.updateCheck(
        installedVersion: '1.2.0',
        platform: 'android',
      );
      expect(current.updateAvailable, false);
      expect(current.updateRequired, false);

      await server.client.setForceUpdate(
        platform: 'android',
        forceUpdate: true,
      );
      final UpdateCheckResponse forced = await server.client.updateCheck(
        installedVersion: '1.2.0',
        platform: 'android',
      );
      expect(forced.updateAvailable, false);
      expect(forced.updateRequired, true);

      await server.client.setMinSupported(
        platform: 'android',
        minSupportedVersion: '1.2.0',
        forceUpdate: false,
      );
      final Map<String, ReleaseInfo> releases =
          await server.client.listReleases();
      expect(releases['android']!.minSupportedVersion, '1.2.0');
      expect(releases['android']!.forceUpdate, false);
    });

    test('min-supported above latest is a 400', () async {
      final _TestServer server = await _start(seed: ServerState.seed());
      addTearDown(server.close);
      final http.Response response = await http.put(
        server.uri('/api/v1/config/min-supported'),
        headers: <String, String>{
          'content-type': 'application/json',
          'x-admin-token': _token,
        },
        body: jsonEncode(<String, Object?>{
          'platform': 'android',
          'min_supported_version': '9.9.9',
        }),
      );
      expect(response.statusCode, 400);
      expect(_json(response)['error'], contains('must not exceed'));
    });

    test('malformed version query is a 400', () async {
      final _TestServer server = await _start();
      addTearDown(server.close);
      final http.Response response = await http.get(
        server.uri('/api/v1/update-check',
            <String, String>{'version': 'banana', 'platform': 'android'}),
      );
      expect(response.statusCode, 400);
      final Map<String, Object?> body = _json(response);
      expect(body['error'], contains('not a valid semantic version'));
    });
  });

  group('flags over HTTP', () {
    test('GET /api/v1/flags exposes the seeded matrix', () async {
      final _TestServer server = await _start();
      addTearDown(server.close);
      final FeatureFlagMatrix flags = await server.client.fetchFlags();
      expect(flags.version, 1);
      expect(flags.flags['dynamic_ui'], true);
      expect(flags.flags['agent.thinking'], true);
      expect(flags.flags['labs.voice_mode'], false);
      expect(flags.layout['sections'], isA<List<Object?>>());
    });

    test('PUT /api/v1/flags merges and returns the new version', () async {
      final _TestServer server = await _start();
      addTearDown(server.close);
      final int version = await server.client.putFlags(
        flags: const <String, Object?>{'labs.voice_mode': true},
        remoteDefaults: const <String, Object?>{},
        layout: const <String, Object?>{},
        merge: true,
      );
      expect(version, 2);
      final FeatureFlagMatrix flags = await server.client.fetchFlags();
      expect(flags.version, 2);
      expect(flags.flags['labs.voice_mode'], true);
      expect(flags.flags['dynamic_ui'], true);
    });
  });

  group('latest_version.json', () {
    test('mirrors the update-check payload plus generated_at', () async {
      final _TestServer server = await _start(seed: ServerState.seed());
      addTearDown(server.close);
      final http.Response response =
          await http.get(server.uri('/latest_version.json'));
      expect(response.statusCode, 200);
      final Map<String, Object?> body = _json(response);
      for (final String key in <String>[
        'latest_version',
        'min_supported_version',
        'download_url',
        'sha256',
        'size_bytes',
        'changelog',
        'force_update',
        'published_at',
        'platform',
        'update_available',
        'update_required',
        'release_notes_url',
        'generated_at',
      ]) {
        expect(body.containsKey(key), true, reason: 'missing key $key');
      }
      expect(body['platform'], 'android');
      expect(body['latest_version'], '1.0.0');
      expect(DateTime.tryParse(body['generated_at']! as String), isNotNull);
    });
  });

  group('unknown routes', () {
    test('answer with a JSON 404', () async {
      final _TestServer server = await _start();
      addTearDown(server.close);
      final http.Response response = await http.get(server.uri('/nope'));
      expect(response.statusCode, 404);
      expect(_json(response)['error'], isA<String>());
    });
  });

  group('release store', () {
    test('seeds defaults when the file is absent and round-trips', () async {
      final Directory dir =
          await Directory.systemTemp.createTemp('harbor_store_test');
      addTearDown(() async {
        if (await dir.exists()) {
          await dir.delete(recursive: true);
        }
      });
      final File file = File('${dir.path}/state.json');

      final ReleaseStore store = ReleaseStore.file(file);
      final ServerState seeded = await store.load();
      expect(seeded.releasesByPlatform['android']!.version, '1.0.0');
      expect(seeded.flags.flags['agent.thinking'], true);
      while (store.cached.releasesByPlatform.isEmpty) {
        await Future<void>.delayed(Duration.zero);
      }
      expect(await file.exists(), true);

      final ServerState updated = seeded.withRelease(
        _release(version: '2.0.0', minSupported: '2.0.0', sha256: _shaB),
      ).copyWith(
        flags: seeded.flags.merge(
          FeatureFlagMatrix(
            version: seeded.flags.version,
            updatedAt: DateTime.utc(2026, 3, 1),
            flags: const <String, Object?>{'labs.lora_training': true},
            remoteDefaults: const <String, Object?>{},
            layout: const <String, Object?>{},
          ),
          merge: true,
        ),
      );
      await store.save(updated);
      await store.flush();
      expect(await File('${file.path}.tmp').exists(), false);

      final ReleaseStore reopened = ReleaseStore.file(file);
      final ServerState loaded = await reopened.load();
      expect(loaded.releasesByPlatform['android']!.version, '2.0.0');
      expect(loaded.flags.version, 2);
      expect(loaded.flags.flags['labs.lora_training'], true);
      expect(loaded.flags.flags['dynamic_ui'], true);
    });
  });

  group('HTTP server end to end', () {
    late _TestServer server;

    setUp(() async {
      server = await _start(seed: ServerState.seed());
    });
    tearDown(() async => server.close());

    test('serves health, update-check and flags on loopback', () async {
      final http.Response health =
          await http.get(server.uri('/health'));
      expect(health.statusCode, 200);
      expect(_json(health)['status'], 'healthy');

      final http.Response check = await http.get(server.uri(
        '/api/v1/update-check',
        <String, String>{'version': '0.9.0', 'platform': 'android'},
      ));
      expect(check.statusCode, 200);
      final Map<String, Object?> checkBody = _json(check);
      expect(checkBody['latest_version'], '1.0.0');
      expect(checkBody['update_available'], true);
      expect(checkBody['update_required'], true);

      final http.Response flags = await http.get(server.uri('/api/v1/flags'));
      expect(flags.statusCode, 200);
      expect(flags.headers['etag'], isNotNull);
      expect(_json(flags)['version'], 1);

      final http.Response cached = await http.get(
        server.uri('/api/v1/flags'),
        headers: <String, String>{'if-none-match': flags.headers['etag']!},
      );
      expect(cached.statusCode, 304);
    });
  });
}