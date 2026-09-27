// Unit tests for the OTA update engine (`lib/core/update_engine/update_service.dart`).
//
// AUTHORING NOTE: these tests could NOT be executed while they were written —
// the host is aarch64 and the Linux Flutter SDK is x86-64 only, so `flutter
// pub get` / `flutter test` cannot run here. They are written against the exact
// public surface of the source under test and must be run in CI.
//
// Hermetic: every test injects a fake `http.Client` (MockClient) and, where a
// download happens, a temporary directory provider. No real network or device
// path is ever touched. `SharedPreferences` is faked in `setUp`.

import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:main_app/core/update_engine/update_service.dart';
import 'package:pub_semver/pub_semver.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// A valid 64-character hex digest used across the tests.
const String _hex64 =
    '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef';

UpdateService _makeService({
  required http.Client client,
  String currentVersion = '1.0.0',
  String platform = 'android',
  InstallStrategy strategy = InstallStrategy.nativeStreaming,
  SupportDirectoryProvider? supportDirectoryProvider,
}) {
  final UpdateService service = UpdateService(
    baseUrl: 'https://updates.test',
    currentVersion: currentVersion,
    platform: platform,
    strategy: strategy,
    client: client,
    supportDirectoryProvider:
        supportDirectoryProvider ?? () async => Directory.systemTemp,
  );
  addTearDown(service.dispose);
  return service;
}

ReleaseInfo _release({
  String latest = '1.2.0',
  String min = '1.0.0',
  String url = 'https://cdn.test/app-release.apk',
  String sha = _hex64,
  int size = 0,
  String changelog = '',
  bool force = false,
}) =>
    ReleaseInfo(
      latestVersion: latest,
      minSupportedVersion: min,
      downloadUrl: url,
      sha256: sha,
      sizeBytes: size,
      changelog: changelog,
      forceUpdate: force,
      updateAvailable: true,
      updateRequired: force,
      platform: 'android',
    );

/// Builds a `GET /api/v1/update-check` body.
String _checkBody({
  String latest = '1.2.0',
  String min = '1.0.0',
  String url = 'https://cdn.test/app-release.apk',
  String sha = _hex64,
  int size = 0,
  bool force = false,
  bool includeLatest = true,
}) {
  final Map<String, Object?> body = <String, Object?>{
    'min_supported_version': min,
    'download_url': url,
    'sha256': sha,
    'size_bytes': size,
    'changelog': '',
    'force_update': force,
    'published_at': '2026-02-01T10:00:00.000Z',
    'platform': 'android',
    'update_available': true,
    'update_required': force,
    'release_notes_url': 'https://cdn.test/notes',
  };
  if (includeLatest) {
    body['latest_version'] = latest;
  }
  return jsonEncode(body);
}

Directory _tempDir() {
  final Directory dir = Directory.systemTemp.createTempSync('harbor_ota_test_');
  addTearDown(() {
    if (dir.existsSync()) {
      dir.deleteSync(recursive: true);
    }
  });
  return dir;
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    SharedPreferences.setMockInitialValues(<String, Object>{});
  });

  // ---------------------------------------------------------------------------
  // 1. ReleaseInfo.fromJson
  // ---------------------------------------------------------------------------

  group('ReleaseInfo.fromJson', () {
    test('parses the full frozen contract payload', () {
      final String payload = jsonEncode(<String, Object?>{
        'latest_version': '1.2.0',
        'min_supported_version': '1.1.0',
        'download_url':
            'https://github.com/OWNER/REPO/releases/download/v1.2.0/app-release.apk',
        'sha256': _hex64,
        'size_bytes': 24891342,
        'changelog': '• Fixes\n• Speed',
        'force_update': false,
        'published_at': '2026-02-01T10:00:00.000Z',
        'platform': 'android',
        'update_available': true,
        'update_required': false,
        'release_notes_url':
            'https://github.com/OWNER/REPO/releases/tag/v1.2.0',
      });

      final ReleaseInfo release =
          ReleaseInfo.parse(payload, platform: 'android');

      expect(release.latestVersion, '1.2.0');
      expect(release.minSupportedVersion, '1.1.0');
      expect(release.downloadUrl,
          'https://github.com/OWNER/REPO/releases/download/v1.2.0/app-release.apk');
      expect(release.sha256, _hex64);
      expect(release.sizeBytes, 24891342);
      expect(release.changelog, '• Fixes\n• Speed');
      expect(release.forceUpdate, isFalse);
      expect(release.platform, 'android');
      expect(release.updateAvailable, isTrue);
      expect(release.updateRequired, isFalse);
      expect(release.releaseNotesUrl,
          'https://github.com/OWNER/REPO/releases/tag/v1.2.0');
      expect(release.publishedAt, isNotNull);
      expect(release.publishedAt!.isUtc, isTrue);
      expect(release.publishedAt, DateTime.utc(2026, 2, 1, 10));
      expect(release.hasRelease, isTrue);
      expect(release.latestSemver, Version.parse('1.2.0'));
      expect(release.minSupportedSemver, Version.parse('1.1.0'));
    });

    test('the zero payload reports hasRelease false', () {
      final ReleaseInfo release = ReleaseInfo.fromJson(<String, Object?>{
        'latest_version': '0.0.0',
        'min_supported_version': '0.0.0',
        'download_url': '',
        'sha256': '',
        'size_bytes': 0,
        'changelog': '',
        'force_update': false,
        'platform': 'android',
      }, platform: 'android');

      expect(release.hasRelease, isFalse);
      expect(release.digest, isNull);
      expect(release.publishedAt, isNull);
      expect(release.releaseNotesUrl, isNull);
    });

    test('missing latest_version throws ReleaseMetadataException', () {
      expect(
        () => ReleaseInfo.fromJson(
          <String, Object?>{'min_supported_version': '1.0.0'},
          platform: 'android',
        ),
        throwsA(isA<ReleaseMetadataException>()
            .having((ReleaseMetadataException e) => e.field, 'field',
                'latest_version')),
      );
    });

    test('non-semver latest_version throws ReleaseMetadataException', () {
      expect(
        () => ReleaseInfo.fromJson(
          <String, Object?>{'latest_version': 'not-a-version'},
          platform: 'android',
        ),
        throwsA(isA<ReleaseMetadataException>()),
      );
    });

    test('a leading v is accepted', () {
      final ReleaseInfo release = ReleaseInfo.fromJson(<String, Object?>{
        'latest_version': 'v1.2.3',
        'min_supported_version': 'v1.0.0',
        'download_url': 'https://cdn.test/app.apk',
      }, platform: 'android');

      expect(release.latestVersion, 'v1.2.3');
      expect(release.latestSemver, Version.parse('1.2.3'));
      expect(release.minSupportedSemver, Version.parse('1.0.0'));
      expect(release.hasRelease, isTrue);
    });

    test('digest is non-null for 64 hex chars and null for the wrong length',
        () {
      final ReleaseInfo good = _release();
      expect(good.digest, isNotNull);
      expect(good.digest!.toString(), _hex64);

      expect(_release(sha: 'abc').digest, isNull);
      expect(_release(sha: '').digest, isNull);
      expect(_release(sha: '${_hex64}0').digest, isNull);
    });

    test('coerces scalar types and defaults optional fields', () {
      final ReleaseInfo release = ReleaseInfo.fromJson(<String, Object?>{
        'latest_version': '1.2.0',
        'size_bytes': '123',
        'force_update': 1,
        'update_available': 'true',
        'platform': 'ios',
      }, platform: 'android');

      expect(release.sizeBytes, 123);
      expect(release.forceUpdate, isTrue);
      expect(release.updateAvailable, isTrue);
      expect(release.updateRequired, isFalse);
      expect(release.platform, 'ios');
      expect(release.minSupportedVersion, '1.2.0');
    });
  });

  // ---------------------------------------------------------------------------
  // 2. ReleaseInfo.parse
  // ---------------------------------------------------------------------------

  group('ReleaseInfo.parse', () {
    test('invalid JSON throws ReleaseMetadataException', () {
      expect(
        () => ReleaseInfo.parse('{not json', platform: 'android'),
        throwsA(isA<ReleaseMetadataException>()),
      );
    });

    test('a JSON array throws ReleaseMetadataException', () {
      expect(
        () => ReleaseInfo.parse('[]', platform: 'android'),
        throwsA(isA<ReleaseMetadataException>()),
      );
    });

    test('toJson round-trip keeps every contract key', () {
      final ReleaseInfo release = ReleaseInfo.fromJson(<String, Object?>{
        'latest_version': '1.2.0',
        'min_supported_version': '1.1.0',
        'download_url': 'https://cdn.test/app.apk',
        'sha256': _hex64,
        'size_bytes': 42,
        'changelog': 'hello',
        'force_update': true,
        'published_at': '2026-02-01T10:00:00.000Z',
        'platform': 'android',
        'update_available': true,
        'update_required': true,
        'release_notes_url': 'https://cdn.test/notes',
      }, platform: 'android');

      final Map<String, Object?> json = release.toJson();
      expect(
        json.keys.toSet(),
        <String>{
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
        },
      );
      expect(json['published_at'], '2026-02-01T10:00:00.000Z');

      final ReleaseInfo back =
          ReleaseInfo.fromJson(json, platform: 'android');
      expect(back.latestVersion, release.latestVersion);
      expect(back.minSupportedVersion, release.minSupportedVersion);
      expect(back.downloadUrl, release.downloadUrl);
      expect(back.sha256, release.sha256);
      expect(back.sizeBytes, release.sizeBytes);
      expect(back.changelog, release.changelog);
      expect(back.forceUpdate, release.forceUpdate);
      expect(back.updateAvailable, release.updateAvailable);
      expect(back.updateRequired, release.updateRequired);
      expect(back.publishedAt, release.publishedAt);
      expect(back.releaseNotesUrl, release.releaseNotesUrl);
    });
  });

  // ---------------------------------------------------------------------------
  // 3. UpdateService.decide (pure)
  // ---------------------------------------------------------------------------

  group('UpdateService.decide', () {
    UpdateDecision decide(String installed, String latest, String min,
            {bool forceFlag = false}) =>
        UpdateService.decide(
          installed: Version.parse(installed),
          latest: Version.parse(latest),
          minSupported: Version.parse(min),
          forceFlag: forceFlag,
        );

    test('installed == latest and installed >= min is upToDate', () {
      expect(decide('1.2.0', '1.2.0', '1.0.0'), UpdateDecision.upToDate);
    });

    test('installed < latest but >= min is softUpdate', () {
      expect(decide('1.1.0', '1.2.0', '1.0.0'), UpdateDecision.softUpdate);
    });

    test('installed < min is forceUpdate', () {
      expect(decide('0.9.0', '1.2.0', '1.0.0'), UpdateDecision.forceUpdate);
    });

    test('forceFlag forces even when installed == latest', () {
      expect(
        decide('1.2.0', '1.2.0', '1.0.0', forceFlag: true),
        UpdateDecision.forceUpdate,
      );
    });

    test('1.9.0 vs 1.10.0 is softUpdate, not upToDate', () {
      expect(decide('1.9.0', '1.10.0', '1.0.0'), UpdateDecision.softUpdate);
    });

    test('1.0.0 vs 1.0.0+1 (build metadata) is softUpdate', () {
      expect(decide('1.0.0', '1.0.0+1', '1.0.0'), UpdateDecision.softUpdate);
    });

    test('a pre-release installed 1.0.0-rc.1 vs 1.0.0 is softUpdate', () {
      expect(
        decide('1.0.0-rc.1', '1.0.0', '0.9.0'),
        UpdateDecision.softUpdate,
      );
    });

    test('installed above latest (rollback) is upToDate', () {
      expect(decide('2.0.0', '1.5.0', '1.0.0'), UpdateDecision.upToDate);
    });
  });

  // ---------------------------------------------------------------------------
  // 4. checkForUpdate with a mocked client
  // ---------------------------------------------------------------------------

  group('checkForUpdate', () {
    test('200 + valid body yields the right decision and metadata', () async {
      final UpdateService service = _makeService(
        client: MockClient((http.Request request) async =>
            http.Response(_checkBody(latest: '1.2.0', min: '1.0.0'), 200)),
      );

      final DateTime before = DateTime.now().toUtc();
      final UpdateCheckResult result = await service.checkForUpdate();
      final DateTime after = DateTime.now().toUtc();

      expect(result.decision, UpdateDecision.softUpdate);
      expect(result.release, isNotNull);
      expect(result.release!.latestVersion, '1.2.0');
      expect(result.installedVersion, Version.parse('1.0.0'));
      expect(result.error, isNull);
      expect(result.dismissed, isFalse);
      expect(result.isSoft, isTrue);
      expect(result.shouldPrompt, isTrue);
      expect(result.checkedAt.isBefore(before), isFalse);
      expect(
        result.checkedAt.isAfter(after.add(const Duration(seconds: 1))),
        isFalse,
      );
      expect(await service.lastCheckAt(), isNotNull);
    });

    test('the request URL and query params are exact', () async {
      late http.Request captured;
      final UpdateService service = _makeService(
        client: MockClient((http.Request request) async {
          captured = request;
          return http.Response(_checkBody(), 200);
        }),
        platform: 'android',
      );

      await service.checkForUpdate();

      expect(captured.method, 'GET');
      expect(captured.url.path, '/api/v1/update-check');
      expect(captured.url.queryParameters['version'], '1.0.0');
      expect(captured.url.queryParameters['platform'], 'android');
      expect(
        captured.url.toString(),
        'https://updates.test/api/v1/update-check?version=1.0.0&platform=android',
      );
      expect(captured.headers['Accept'], 'application/json');
    });

    test('silent: true with a 500 does not throw and records the error',
        () async {
      final UpdateService service = _makeService(
        client: MockClient(
            (http.Request request) async => http.Response('boom', 500)),
      );

      final UpdateCheckResult result = await service.checkForUpdate();

      expect(result.decision, UpdateDecision.upToDate);
      expect(result.error, isNotNull);
      expect(result.error, contains('500'));
      expect(result.release, isNull);
      expect(result.shouldPrompt, isFalse);
    });

    test('silent: true with a malformed body does not throw', () async {
      final UpdateService service = _makeService(
        client: MockClient(
            (http.Request request) async => http.Response('{bad json', 200)),
      );

      final UpdateCheckResult result = await service.checkForUpdate();

      expect(result.decision, UpdateDecision.upToDate);
      expect(result.error, isNotNull);
      expect(result.release, isNull);
    });

    test('silent: false with a 500 throws UpdateCheckException(500)', () async {
      final UpdateService service = _makeService(
        client: MockClient(
            (http.Request request) async => http.Response('boom', 500)),
      );

      await expectLater(
        service.checkForUpdate(silent: false),
        throwsA(isA<UpdateCheckException>().having(
            (UpdateCheckException e) => e.statusCode, 'statusCode', 500)),
      );
    });

    test('a force_update payload yields forceUpdate', () async {
      final UpdateService service = _makeService(
        client: MockClient((http.Request request) async =>
            http.Response(_checkBody(latest: '1.2.0', force: true), 200)),
      );

      final UpdateCheckResult result = await service.checkForUpdate();

      expect(result.decision, UpdateDecision.forceUpdate);
      expect(result.isForced, isTrue);
      expect(result.isSoft, isFalse);
      expect(result.shouldPrompt, isTrue);
      expect(result.dismissed, isFalse);
    });

    test('the empty/zero payload is upToDate with release null', () async {
      final UpdateService service = _makeService(
        client: MockClient((http.Request request) async => http.Response(
            _checkBody(latest: '0.0.0', min: '0.0.0', url: '', sha: ''),
            200)),
      );

      final UpdateCheckResult result = await service.checkForUpdate();

      expect(result.decision, UpdateDecision.upToDate);
      expect(result.release, isNull);
      expect(result.shouldPrompt, isFalse);
    });

    test('a malformed version query is surfaced as a silent error', () async {
      final UpdateService service = _makeService(
        client: MockClient((http.Request request) async =>
            http.Response('{"error":"bad version"}', 400)),
      );

      final UpdateCheckResult result = await service.checkForUpdate();
      expect(result.error, contains('400'));
      expect(result.decision, UpdateDecision.upToDate);
    });
  });

  // ---------------------------------------------------------------------------
  // 5. Dismissal persistence
  // ---------------------------------------------------------------------------

  group('dismissal persistence', () {
    test('dismiss / isDismissed / clearDismissal round-trip', () async {
      final UpdateService service = _makeService(
        client: MockClient(
            (http.Request request) async => http.Response('{}', 500)),
      );

      expect(await service.isDismissed('1.2.0'), isFalse);

      await service.dismiss('1.2.0');
      expect(await service.isDismissed('1.2.0'), isTrue);
      expect(await service.isDismissed('1.3.0'), isFalse);

      await service.clearDismissal();
      expect(await service.isDismissed('1.2.0'), isFalse);
    });

    test('a dismissed soft update is not soft but the decision stands',
        () async {
      final UpdateService service = _makeService(
        client: MockClient((http.Request request) async =>
            http.Response(_checkBody(latest: '1.2.0', min: '1.0.0'), 200)),
      );

      await service.dismiss('1.2.0');
      final UpdateCheckResult result = await service.checkForUpdate();

      expect(result.decision, UpdateDecision.softUpdate);
      expect(result.dismissed, isTrue);
      expect(result.isSoft, isFalse);
      expect(result.shouldPrompt, isFalse);
    });

    test('a forced update is never suppressed by a dismissal', () async {
      final UpdateService service = _makeService(
        client: MockClient((http.Request request) async => http.Response(
            _checkBody(latest: '1.2.0', min: '1.0.0', force: true), 200)),
      );

      await service.dismiss('1.2.0');
      final UpdateCheckResult result = await service.checkForUpdate();

      expect(result.decision, UpdateDecision.forceUpdate);
      expect(result.isForced, isTrue);
      expect(result.dismissed, isFalse);
      expect(result.shouldPrompt, isTrue);
    });
  });

  // ---------------------------------------------------------------------------
  // 6. downloadAndVerify
  // ---------------------------------------------------------------------------

  group('downloadAndVerify', () {
    test('happy path verifies the digest and renames .part into place',
        () async {
      final Directory dir = _tempDir();
      final List<int> bytes =
          List<int>.generate(300, (int i) => i % 256);
      final List<List<int>> chunks = <List<int>>[
        bytes.sublist(0, 100),
        bytes.sublist(100, 250),
        bytes.sublist(250),
      ];
      final String expected = sha256.convert(bytes).toString();

      final UpdateService service = _makeService(
        supportDirectoryProvider: () async => dir,
        client: MockClient.streaming(
          (http.BaseRequest request, http.ByteStream body) async =>
              http.StreamedResponse(
            Stream<List<int>>.fromIterable(chunks),
            200,
            contentLength: bytes.length,
          ),
        ),
      );

      final List<DownloadProgress> progress = <DownloadProgress>[];
      final VerifiedArtifact artifact = await service.downloadAndVerify(
        url: Uri.parse('https://cdn.test/app-release.apk'),
        expectedSha256: expected,
        fileName: 'app-release.apk',
        onProgress: progress.add,
      );

      expect(artifact.sha256, expected);
      expect(artifact.expectedSha256, expected);
      expect(artifact.bytes, bytes.length);
      expect(artifact.file.existsSync(), isTrue);
      expect(artifact.file.lengthSync(), bytes.length);
      expect(artifact.file.path, '${dir.path}/ota/app-release.apk');
      expect(File('${artifact.file.path}.part').existsSync(), isFalse);
      expect(artifact.duration, isA<Duration>());

      expect(progress, isNotEmpty);
      expect(
        progress.map((DownloadProgress p) => p.receivedBytes).toList(),
        <int>[100, 250, 300],
      );
      expect(
        progress.every((DownloadProgress p) => p.totalBytes == bytes.length),
        isTrue,
      );
      for (int i = 1; i < progress.length; i++) {
        expect(progress[i].receivedBytes >= progress[i - 1].receivedBytes,
            isTrue);
      }
      expect(progress.last.receivedBytes, bytes.length);
      expect(progress.last.percent, 100);
    });

    test('falls back to expectedBytes when contentLength is absent', () async {
      final Directory dir = _tempDir();
      final List<int> bytes = List<int>.generate(64, (int i) => i);
      final String expected = sha256.convert(bytes).toString();

      final UpdateService service = _makeService(
        supportDirectoryProvider: () async => dir,
        client: MockClient.streaming(
          (http.BaseRequest request, http.ByteStream body) async =>
              http.StreamedResponse(
            Stream<List<int>>.fromIterable(<List<int>>[bytes]),
            200,
          ),
        ),
      );

      final List<DownloadProgress> progress = <DownloadProgress>[];
      await service.downloadAndVerify(
        url: Uri.parse('https://cdn.test/app.apk'),
        expectedSha256: expected,
        fileName: 'app.apk',
        expectedBytes: bytes.length,
        onProgress: progress.add,
      );

      expect(progress.last.totalBytes, bytes.length);
    });

    test('checksum mismatch throws and leaves no partial or target file',
        () async {
      final Directory dir = _tempDir();
      final List<int> bytes =
          List<int>.generate(128, (int i) => (i * 7) % 256);
      final String wrong = 'f' * 64;

      final UpdateService service = _makeService(
        supportDirectoryProvider: () async => dir,
        client: MockClient.streaming(
          (http.BaseRequest request, http.ByteStream body) async =>
              http.StreamedResponse(
            Stream<List<int>>.fromIterable(<List<int>>[bytes]),
            200,
            contentLength: bytes.length,
          ),
        ),
      );

      await expectLater(
        service.downloadAndVerify(
          url: Uri.parse('https://cdn.test/app.apk'),
          expectedSha256: wrong,
          fileName: 'app.apk',
        ),
        throwsA(isA<ChecksumMismatchException>()
            .having((ChecksumMismatchException e) => e.expected, 'expected',
                wrong)),
      );

      final File target = File('${dir.path}/ota/app.apk');
      expect(target.existsSync(), isFalse);
      expect(File('${target.path}.part').existsSync(), isFalse);
    });

    test('a short digest is rejected before any download', () async {
      final Directory dir = _tempDir();
      final UpdateService service = _makeService(
        supportDirectoryProvider: () async => dir,
        client: MockClient.streaming(
          (http.BaseRequest request, http.ByteStream body) async =>
              http.StreamedResponse(
            const Stream<List<int>>.empty(),
            200,
          ),
        ),
      );

      await expectLater(
        service.downloadAndVerify(
          url: Uri.parse('https://cdn.test/app.apk'),
          expectedSha256: 'abc',
          fileName: 'app.apk',
        ),
        throwsA(isA<ReleaseMetadataException>()
            .having((ReleaseMetadataException e) => e.field, 'field',
                'sha256')),
      );
    });

    test('a non-200 response throws UpdateCheckException', () async {
      final Directory dir = _tempDir();
      final UpdateService service = _makeService(
        supportDirectoryProvider: () async => dir,
        client: MockClient.streaming(
          (http.BaseRequest request, http.ByteStream body) async =>
              http.StreamedResponse(
            const Stream<List<int>>.empty(),
            404,
            contentLength: 0,
          ),
        ),
      );

      await expectLater(
        service.downloadAndVerify(
          url: Uri.parse('https://cdn.test/missing.apk'),
          expectedSha256: _hex64,
          fileName: 'missing.apk',
        ),
        throwsA(isA<UpdateCheckException>().having(
            (UpdateCheckException e) => e.statusCode, 'statusCode', 404)),
      );
    });
  });

  // ---------------------------------------------------------------------------
  // 7. performUpdate policy (verifyThenInstall) — no plugin invocation
  // ---------------------------------------------------------------------------

  group('performUpdate policy', () {
    test('verifyThenInstall with an empty sha256 rejects', () async {
      final UpdateService service = _makeService(
        strategy: InstallStrategy.verifyThenInstall,
        client: MockClient(
            (http.Request request) async => http.Response('{}', 500)),
      );

      await expectLater(
        service.performUpdate(release: _release(sha: '')),
        throwsA(isA<ReleaseMetadataException>()
            .having((ReleaseMetadataException e) => e.field, 'field',
                'sha256')),
      );
    });

    test('a release with hasRelease == false rejects', () async {
      final UpdateService service = _makeService(
        strategy: InstallStrategy.verifyThenInstall,
        client: MockClient(
            (http.Request request) async => http.Response('{}', 500)),
      );

      await expectLater(
        service.performUpdate(release: _release(latest: '0.0.0', url: '')),
        throwsA(isA<ReleaseMetadataException>()
            .having((ReleaseMetadataException e) => e.field, 'field',
                'download_url')),
      );
    });

    test('a non-absolute download_url rejects', () async {
      final UpdateService service = _makeService(
        strategy: InstallStrategy.verifyThenInstall,
        client: MockClient(
            (http.Request request) async => http.Response('{}', 500)),
      );

      await expectLater(
        service.performUpdate(release: _release(url: 'not-a-url')),
        throwsA(isA<ReleaseMetadataException>()
            .having((ReleaseMetadataException e) => e.field, 'field',
                'download_url')),
      );
    });
  });

  // ---------------------------------------------------------------------------
  // 8. describe() and constructor validation
  // ---------------------------------------------------------------------------

  group('describe and construction', () {
    test('describe() reports the diagnostic keys', () {
      final UpdateService service = _makeService(
        strategy: InstallStrategy.verifyThenInstall,
        client: MockClient(
            (http.Request request) async => http.Response('{}', 500)),
      );

      final Map<String, Object?> described = service.describe();
      expect(described['base_url'], 'https://updates.test');
      expect(described['current_version'], '1.0.0');
      expect(described['platform'], 'android');
      expect(described['strategy'], 'verifyThenInstall');
      expect(described['request_timeout_ms'], 15000);
      expect(service.installedVersion, Version.parse('1.0.0'));
    });

    test('a non-semver currentVersion throws ReleaseMetadataException', () {
      expect(
        () => UpdateService(
          baseUrl: 'https://updates.test',
          currentVersion: 'not-semver',
          client: MockClient(
              (http.Request request) async => http.Response('{}', 500)),
        ),
        throwsA(isA<ReleaseMetadataException>()),
      );
    });
  });

  // ---------------------------------------------------------------------------
  // 9. Computed getters
  // ---------------------------------------------------------------------------

  group('computed getters', () {
    test('DownloadProgress fraction/percent for a known total', () {
      const DownloadProgress half =
          DownloadProgress(receivedBytes: 50, totalBytes: 200);
      expect(half.fraction, 0.25);
      expect(half.percent, 25);

      const DownloadProgress over =
          DownloadProgress(receivedBytes: 300, totalBytes: 200);
      expect(over.fraction, 1.0);
      expect(over.percent, 100);
    });

    test('DownloadProgress is unknown when totalBytes == 0', () {
      const DownloadProgress unknown =
          DownloadProgress(receivedBytes: 512, totalBytes: 0);
      expect(unknown.fraction, isNull);
      expect(unknown.percent, isNull);
    });

    test('UpdateCheckResult.shouldPrompt across decisions', () {
      UpdateCheckResult result(
        UpdateDecision decision, {
        bool dismissed = false,
      }) =>
          UpdateCheckResult(
            decision: decision,
            installedVersion: Version.parse('1.0.0'),
            release: null,
            checkedAt: DateTime.utc(2026, 2, 1),
            dismissed: dismissed,
          );

      expect(result(UpdateDecision.forceUpdate).shouldPrompt, isTrue);
      expect(result(UpdateDecision.forceUpdate).isForced, isTrue);
      expect(result(UpdateDecision.softUpdate).shouldPrompt, isTrue);
      expect(
        result(UpdateDecision.softUpdate, dismissed: true).shouldPrompt,
        isFalse,
      );
      expect(result(UpdateDecision.upToDate).shouldPrompt, isFalse);
      expect(result(UpdateDecision.upToDate).isSoft, isFalse);
    });

    test('UpdateCheckException.toString carries the status code', () {
      final UpdateCheckException e =
          UpdateCheckException('nope', statusCode: 503);
      expect(e.toString(), contains('503'));
      expect(e.toString(), contains('nope'));
    });
  });
}