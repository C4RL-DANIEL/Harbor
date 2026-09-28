// OTA update engine.
//
// Responsibilities:
//
//   1. Ask the update server what the latest release is, using package:pub_semver
//      for every comparison (never string ordering, which gets `1.10.0` vs
//      `1.9.0` wrong).
//   2. Decide between three states: up to date, a *soft* update the user may
//      dismiss, and a *forced* update that must be applied before the app is
//      usable. A force is triggered either by the server's `force_update` flag
//      or by the installed version falling below `min_supported_version`.
//   3. Download the APK directly and verify its SHA256 against the published
//      digest *while streaming*, then hand off to the native installer through
//      `ota_update`.
//
// Integrity is checked twice on purpose. The streaming check in this file lets
// the app fail fast with a precise byte-level error and never write a corrupt
// artifact to disk; the native plugin then re-verifies the same digest as part
// of the platform install flow.

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:convert/convert.dart' show AccumulatorSink;
import 'package:crypto/crypto.dart' show Digest, sha256;
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show PlatformException;
import 'package:http/http.dart' as http;
import 'package:ota_update/ota_update.dart';
import 'package:path_provider/path_provider.dart';
import 'package:pub_semver/pub_semver.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Thrown when the update server cannot be reached or answers unexpectedly.
class UpdateCheckException implements Exception {
  UpdateCheckException(this.message, {this.statusCode, this.cause});

  final String message;
  final int? statusCode;
  final Object? cause;

  @override
  String toString() => 'UpdateCheckException: $message'
      '${statusCode == null ? '' : ' (HTTP $statusCode)'}'
      '${cause == null ? '' : ' ($cause)'}';
}

/// Thrown when release metadata is missing or malformed.
class ReleaseMetadataException implements Exception {
  ReleaseMetadataException(this.message, {this.field});

  final String message;
  final String? field;

  @override
  String toString() => 'ReleaseMetadataException: $message'
      '${field == null ? '' : ' [field: $field]'}';
}

/// Thrown when a downloaded artifact fails SHA256 verification.
class ChecksumMismatchException implements Exception {
  ChecksumMismatchException({
    required this.expected,
    required this.actual,
    this.path,
  });

  final String expected;
  final String actual;
  final String? path;

  /// Whether the mismatch is only a formatting/prefix difference.
  bool get sameDigestDifferentFormat =>
      expected.toLowerCase() == actual.toLowerCase();

  @override
  String toString() =>
      'ChecksumMismatchException: expected $expected but computed $actual'
      '${path == null ? '' : ' for $path'}';
}

/// Thrown when the native installation step fails.
class UpdateInstallException implements Exception {
  UpdateInstallException(this.message, {this.otaStatus});

  final String message;
  final OtaStatus? otaStatus;

  @override
  String toString() => 'UpdateInstallException: $message'
      '${otaStatus == null ? '' : ' (${otaStatus!.name})'}';
}

/// What the client should do about the current release.
enum UpdateDecision {
  /// Installed version is the latest.
  upToDate,

  /// A newer version exists but the user may defer it.
  softUpdate,

  /// The app must be updated before it can be used.
  forceUpdate,
}

/// Parsed payload of `GET /api/v1/update-check`, mirroring the frozen contract.
@immutable
class ReleaseInfo {
  const ReleaseInfo({
    required this.latestVersion,
    required this.minSupportedVersion,
    required this.downloadUrl,
    required this.sha256,
    required this.sizeBytes,
    required this.changelog,
    required this.forceUpdate,
    required this.updateAvailable,
    required this.updateRequired,
    required this.platform,
    this.publishedAt,
    this.releaseNotesUrl,
  });

  final String latestVersion;
  final String minSupportedVersion;
  final String downloadUrl;
  final String sha256;
  final int sizeBytes;
  final String changelog;
  final bool forceUpdate;
  final bool updateAvailable;
  final bool updateRequired;
  final String platform;
  final DateTime? publishedAt;
  final String? releaseNotesUrl;

  /// Whether the server published an actual release (as opposed to the
  /// documented "no release yet" zero payload).
  bool get hasRelease => latestVersion != '0.0.0' && downloadUrl.isNotEmpty;

  /// Parsed latest version, or null when unparseable.
  Version? get latestSemver => _tryVersion(latestVersion);

  /// Parsed minimum supported version, or null when unparseable.
  Version? get minSupportedSemver => _tryVersion(minSupportedVersion);

  /// The parsed digest, or null when the server omitted or malformed it.
  ///
  /// `package:crypto`'s [Digest] has no hex constructor, so the 64-character
  /// hex string is decoded here rather than assumed to be parseable.
  Digest? get digest {
    if (sha256.length != 64) {
      return null;
    }
    final List<int>? bytes = _tryHexDecode(sha256);
    return bytes == null ? null : Digest(bytes);
  }

  factory ReleaseInfo.fromJson(Map<String, Object?> json, {required String platform}) {
    Object? pick(String key) => json[key];

    final String latest = pick('latest_version')?.toString() ?? '';
    if (latest.isEmpty) {
      throw ReleaseMetadataException(
        'server response is missing "latest_version"',
        field: 'latest_version',
      );
    }
    if (_tryVersion(latest) == null) {
      throw ReleaseMetadataException(
        '"latest_version" ("$latest") is not a valid semantic version',
        field: 'latest_version',
      );
    }
    final String minSupported =
        pick('min_supported_version')?.toString() ?? latest;
    final String url = pick('download_url')?.toString() ?? '';
    final String digest = pick('sha256')?.toString() ?? '';

    return ReleaseInfo(
      latestVersion: latest,
      minSupportedVersion: minSupported,
      downloadUrl: url,
      sha256: digest,
      sizeBytes: _asInt(pick('size_bytes')) ?? 0,
      changelog: pick('changelog')?.toString() ?? '',
      forceUpdate: _asBool(pick('force_update')) ?? false,
      updateAvailable: _asBool(pick('update_available')) ?? false,
      updateRequired: _asBool(pick('update_required')) ?? false,
      platform: pick('platform')?.toString() ?? platform,
      publishedAt: _asDateTime(pick('published_at')),
      releaseNotesUrl: pick('release_notes_url')?.toString(),
    );
  }

  static ReleaseInfo parse(String body, {required String platform}) {
    final Object? decoded;
    try {
      decoded = jsonDecode(body);
    } on FormatException catch (e) {
      throw ReleaseMetadataException(
        'update-check response is not valid JSON: ${e.message}',
      );
    }
    if (decoded is! Map) {
      throw ReleaseMetadataException('update-check response must be a JSON object');
    }
    return ReleaseInfo.fromJson(decoded.cast<String, Object?>(), platform: platform);
  }

  Map<String, Object?> toJson() => <String, Object?>{
        'latest_version': latestVersion,
        'min_supported_version': minSupportedVersion,
        'download_url': downloadUrl,
        'sha256': sha256,
        'size_bytes': sizeBytes,
        'changelog': changelog,
        'force_update': forceUpdate,
        'published_at': publishedAt?.toUtc().toIso8601String(),
        'platform': platform,
        'update_available': updateAvailable,
        'update_required': updateRequired,
        'release_notes_url': releaseNotesUrl,
      };
}

/// Full outcome of a single update check.
@immutable
class UpdateCheckResult {
  const UpdateCheckResult({
    required this.decision,
    required this.installedVersion,
    required this.release,
    required this.checkedAt,
    required this.dismissed,
    this.error,
  });

  final UpdateDecision decision;

  /// The version installed on this device.
  final Version installedVersion;

  /// Server metadata, or null when the server had nothing published.
  final ReleaseInfo? release;

  /// When the check ran.
  final DateTime checkedAt;

  /// Whether the user previously dismissed this exact version.
  final bool dismissed;

  /// Error text when the check could not complete (the decision then reports
  /// the safe default of [UpdateDecision.upToDate]).
  final String? error;

  /// Whether the app should show a non-dismissible modal.
  bool get isForced => decision == UpdateDecision.forceUpdate;

  /// Whether the app should show a dismissible banner.
  bool get isSoft =>
      decision == UpdateDecision.softUpdate && !dismissed;

  /// Whether anything should be shown at all.
  bool get shouldPrompt => isForced || isSoft;

  Map<String, Object?> toJson() => <String, Object?>{
        'decision': decision.name,
        'installed_version': installedVersion.toString(),
        'checked_at': checkedAt.toIso8601String(),
        'dismissed': dismissed,
        'error': error,
        'release': release?.toJson(),
      };
}

/// Byte-level download progress.
@immutable
class DownloadProgress {
  const DownloadProgress({
    required this.receivedBytes,
    required this.totalBytes,
  });

  final int receivedBytes;
  final int totalBytes;

  /// Fraction in `[0, 1]`, or null when the total is unknown.
  double? get fraction =>
      totalBytes > 0 ? (receivedBytes / totalBytes).clamp(0.0, 1.0) : null;

  /// Percentage in `[0, 100]`, or null when the total is unknown.
  int? get percent => fraction == null ? null : (fraction! * 100).round();

  @override
  String toString() =>
      'DownloadProgress($receivedBytes/$totalBytes, ${percent ?? '?'}%)';
}

/// A downloaded artifact whose digest has been verified.
@immutable
class VerifiedArtifact {
  const VerifiedArtifact({
    required this.file,
    required this.sha256,
    required this.bytes,
    required this.duration,
    required this.expectedSha256,
  });

  final File file;

  /// Digest actually computed from the bytes on disk.
  final String sha256;

  /// Size of the artifact in bytes.
  final int bytes;

  /// How long the download took.
  final Duration duration;

  /// Digest the server published.
  final String expectedSha256;

  /// Average throughput in bytes per second.
  double get bytesPerSecond => duration.inMilliseconds == 0
      ? bytes.toDouble()
      : bytes / (duration.inMilliseconds / 1000.0);

  @override
  String toString() =>
      'VerifiedArtifact(${file.path}, $bytes bytes, sha256=$sha256)';
}

/// Progress of the native install hand-off.
@immutable
class InstallProgress {
  const InstallProgress({
    required this.status,
    this.value,
    this.note,
  });

  final OtaStatus status;

  /// Plugin-reported value: percent string while downloading, or an error
  /// message on failure.
  final String? value;

  /// Human-readable interpretation.
  final String? note;

  /// Whether the platform has taken over and is installing.
  bool get isInstalling => status == OtaStatus.INSTALLING;

  /// Whether the download is still in flight.
  bool get isDownloading => status == OtaStatus.DOWNLOADING;

  /// Whether the flow ended in an error.
  bool get isFailure =>
      status == OtaStatus.ALREADY_RUNNING_ERROR ||
      status == OtaStatus.PERMISSION_NOT_GRANTED_ERROR ||
      status == OtaStatus.INTERNAL_ERROR ||
      status == OtaStatus.DOWNLOAD_ERROR ||
      status == OtaStatus.CHECKSUM_ERROR;

  /// Whether the user or the app cancelled the download.
  bool get isCancelled => status == OtaStatus.CANCELED;

  /// Whether the flow reached a terminal state.
  bool get isTerminal => isInstalling || isFailure || isCancelled;

  @override
  String toString() =>
      'InstallProgress(${status.name}${value == null ? '' : ', $value'})';
}

/// How the artifact should be obtained before installation.
enum InstallStrategy {
  /// Hand the URL and digest to the native plugin and let it download, verify
  /// and install. Lowest memory use and the canonical Android path.
  nativeStreaming,

  /// Download and verify in Dart first, then ask the platform to install the
  /// already-verified local file. Costs disk space but gives a precise,
  /// app-visible integrity error before the installer is ever invoked.
  verifyThenInstall,
}

/// Directory provider injection point, so tests never touch the real filesystem
/// layout of a device.
typedef SupportDirectoryProvider = Future<Directory> Function();

/// Fetches release metadata, verifies artifacts and triggers installation.
class UpdateService {
  UpdateService({
    required this.baseUrl,
    required this.currentVersion,
    this.platform = 'android',
    http.Client? client,
    this.requestTimeout = const Duration(seconds: 15),
    this.strategy = InstallStrategy.nativeStreaming,
    SupportDirectoryProvider? supportDirectoryProvider,
    this.dismissedVersionKey = 'harbor.dismissed_update_version',
    this.lastCheckKey = 'harbor.last_update_check',
    OtaUpdate? otaUpdate,
  })  : _client = client ?? http.Client(),
        _ownsClient = client == null,
        _supportDirectoryProvider =
            supportDirectoryProvider ?? getApplicationSupportDirectory,
        _otaUpdate = otaUpdate ?? OtaUpdate(),
        _installedVersion = _parseVersionOrThrow(currentVersion);

  /// Base URL of the update server.
  final String baseUrl;

  /// Version string of the running app (e.g. `1.0.0`).
  final String currentVersion;

  /// Platform identifier sent to the server (`android`, `ios`, ...).
  final String platform;

  /// Per-request timeout for metadata calls.
  final Duration requestTimeout;

  /// How artifacts are fetched before installation.
  final InstallStrategy strategy;

  /// SharedPreferences key holding the dismissed version.
  final String dismissedVersionKey;

  /// SharedPreferences key holding the last check timestamp.
  final String lastCheckKey;

  final http.Client _client;
  final bool _ownsClient;
  final SupportDirectoryProvider _supportDirectoryProvider;
  final OtaUpdate _otaUpdate;
  final Version _installedVersion;

  /// The installed version as a parsed semver.
  Version get installedVersion => _installedVersion;

  static Version _parseVersionOrThrow(String raw) {
    final Version? parsed = _tryVersion(raw);
    if (parsed == null) {
      throw ReleaseMetadataException(
        'installed version "$raw" is not a valid semantic version',
        field: 'current_version',
      );
    }
    return parsed;
  }

  /// Decides what the client should do, given installed and published versions.
  ///
  /// Extracted as a pure function so the policy is unit-testable without any
  /// network or platform involvement.
  static UpdateDecision decide({
    required Version installed,
    required Version latest,
    required Version minSupported,
    required bool forceFlag,
  }) {
    // A force is unconditional when the server demands it, or when the
    // installed build is older than the oldest build still allowed to run.
    if (forceFlag) {
      return UpdateDecision.forceUpdate;
    }
    if (installed < minSupported) {
      return UpdateDecision.forceUpdate;
    }
    if (installed < latest) {
      return UpdateDecision.softUpdate;
    }
    return UpdateDecision.upToDate;
  }

  /// Performs a silent check against `GET /api/v1/update-check`.
  ///
  /// When the server is not hosted, the same Pages site can publish a static
  /// `latest_version.json`; if the live endpoint returns 404, this method falls
  /// back to that file.
  ///
  /// Never throws when [silent] is true: the failure is captured in
  /// [UpdateCheckResult.error] and the decision degrades to
  /// [UpdateDecision.upToDate] so app startup is never blocked by a network
  /// problem. With `silent: false` the underlying typed exception propagates.
  Future<UpdateCheckResult> checkForUpdate({bool silent = true}) async {
    final DateTime checkedAt = DateTime.now().toUtc();
    try {
      final ReleaseInfo release = await _fetchReleaseInfo();

      final Version latest = release.latestSemver ?? _installedVersion;
      final Version minSupported =
          release.minSupportedSemver ?? release.latestSemver ?? _installedVersion;

      final UpdateDecision decision = decide(
        installed: _installedVersion,
        latest: latest,
        minSupported: minSupported,
        forceFlag: release.forceUpdate,
      );

      final bool dismissed =
          decision == UpdateDecision.softUpdate && await isDismissed(release.latestVersion);

      await _recordCheck(checkedAt);

      return UpdateCheckResult(
        decision: decision,
        installedVersion: _installedVersion,
        release: release.hasRelease ? release : null,
        checkedAt: checkedAt,
        dismissed: dismissed,
      );
    } on UpdateCheckException catch (e) {
      if (!silent) {
        rethrow;
      }
      return UpdateCheckResult(
        decision: UpdateDecision.upToDate,
        installedVersion: _installedVersion,
        release: null,
        checkedAt: checkedAt,
        dismissed: false,
        error: e.toString(),
      );
    } on ReleaseMetadataException catch (e) {
      if (!silent) {
        rethrow;
      }
      return UpdateCheckResult(
        decision: UpdateDecision.upToDate,
        installedVersion: _installedVersion,
        release: null,
        checkedAt: checkedAt,
        dismissed: false,
        error: e.toString(),
      );
    } on TimeoutException {
      const String message = 'update check timed out';
      if (!silent) {
        throw UpdateCheckException(
          '$message after ${requestTimeout.inSeconds}s',
        );
      }
      return UpdateCheckResult(
        decision: UpdateDecision.upToDate,
        installedVersion: _installedVersion,
        release: null,
        checkedAt: checkedAt,
        dismissed: false,
        error: '$message after ${requestTimeout.inSeconds}s',
      );
    } on Object catch (e) {
      if (!silent) {
        throw UpdateCheckException('update check failed', cause: e);
      }
      return UpdateCheckResult(
        decision: UpdateDecision.upToDate,
        installedVersion: _installedVersion,
        release: null,
        checkedAt: checkedAt,
        dismissed: false,
        error: 'update check failed: $e',
      );
    }
  }

  /// Tries the live update-check endpoint, then the static manifest fallback.
  Future<ReleaseInfo> _fetchReleaseInfo() async {
    final Uri liveUri = Uri.parse('$baseUrl/api/v1/update-check').replace(
      queryParameters: <String, String>{
        'version': currentVersion,
        'platform': platform,
      },
    );

    final http.Response liveResponse = await _client.get(
      liveUri,
      headers: const <String, String>{'Accept': 'application/json'},
    ).timeout(requestTimeout);

    if (liveResponse.statusCode == 200) {
      return ReleaseInfo.parse(
        utf8.decode(liveResponse.bodyBytes),
        platform: platform,
      );
    }

    if (liveResponse.statusCode == 404) {
      final Uri staticUri = Uri.parse('$baseUrl/latest_version.json');
      final http.Response staticResponse = await _client.get(
        staticUri,
        headers: const <String, String>{'Accept': 'application/json'},
      ).timeout(requestTimeout);

      if (staticResponse.statusCode == 200) {
        return ReleaseInfo.parse(
          utf8.decode(staticResponse.bodyBytes),
          platform: platform,
        );
      }
    }

    throw UpdateCheckException(
      'update server returned an error',
      statusCode: liveResponse.statusCode,
    );
  }

  /// Streams the APK to disk while computing its SHA256, and verifies the
  /// result against [expectedSha256].
  ///
  /// Throws [ChecksumMismatchException] (and deletes the partial file) when the
  /// digest does not match, so a corrupt artifact can never reach the installer.
  Future<VerifiedArtifact> downloadAndVerify({
    required Uri url,
    required String expectedSha256,
    required String fileName,
    void Function(DownloadProgress progress)? onProgress,
    int? expectedBytes,
  }) async {
    if (expectedSha256.length != 64) {
      throw ReleaseMetadataException(
        'expected sha256 must be 64 hex characters, got '
        '${expectedSha256.length}',
        field: 'sha256',
      );
    }
    final String expected = expectedSha256.toLowerCase();

    final Directory directory = await _supportDirectoryProvider();
    final Directory updatesDir = Directory('${directory.path}/ota');
    if (!await updatesDir.exists()) {
      await updatesDir.create(recursive: true);
    }
    final File target = File('${updatesDir.path}/$fileName');
    final File partial = File('${target.path}.part');
    if (await partial.exists()) {
      await partial.delete();
    }

    final Stopwatch sw = Stopwatch()..start();
    final AccumulatorSink<Digest> digestSink = AccumulatorSink<Digest>();
    final ByteConversionSink hashInput = sha256.startChunkedConversion(digestSink);
    int received = 0;

    final http.Request request = http.Request('GET', url)
      ..headers['Accept'] = 'application/octet-stream';
    final http.StreamedResponse response =
        await _client.send(request).timeout(requestTimeout);

    if (response.statusCode != 200) {
      throw UpdateCheckException(
        'artifact download failed',
        statusCode: response.statusCode,
      );
    }

    final int total = response.contentLength ?? expectedBytes ?? 0;
    final IOSink sink = partial.openWrite();

    try {
      await for (final List<int> chunk
          in response.stream.timeout(requestTimeout)) {
        sink.add(chunk);
        hashInput.add(chunk);
        received += chunk.length;
        onProgress?.call(
          DownloadProgress(receivedBytes: received, totalBytes: total),
        );
      }
      await sink.flush();
    } on Object {
      await sink.close();
      if (await partial.exists()) {
        await partial.delete();
      }
      rethrow;
    } finally {
      await sink.close();
      hashInput.close();
    }

    final String actual = digestSink.events.single.toString().toLowerCase();
    if (actual != expected) {
      if (await partial.exists()) {
        await partial.delete();
      }
      throw ChecksumMismatchException(
        expected: expected,
        actual: actual,
        path: partial.path,
      );
    }

    if (await target.exists()) {
      await target.delete();
    }
    await partial.rename(target.path);
    sw.stop();

    return VerifiedArtifact(
      file: target,
      sha256: actual,
      bytes: received,
      duration: sw.elapsed,
      expectedSha256: expected,
    );
  }

  /// Hands the update to the native installer via `ota_update`.
  ///
  /// [localFile] should only be supplied for [InstallStrategy.verifyThenInstall];
  /// the plugin is then asked to install an already-verified artifact.
  Stream<InstallProgress> install({
    required Uri url,
    String? sha256Checksum,
    File? localFile,
    String? destinationFilename,
  }) {
    final String target = localFile != null && localFile.existsSync()
        ? localFile.uri.toString()
        : url.toString();

    final Stream<OtaEvent> events = _otaUpdate.execute(
      target,
      sha256checksum: sha256Checksum,
      destinationFilename: destinationFilename,
    );

    return events.map(_mapOtaEvent).handleError((Object error) {
      throw UpdateInstallException(
        'native installer reported an error: $error',
      );
    });
  }

  InstallProgress _mapOtaEvent(OtaEvent event) {
    String? note;
    switch (event.status) {
      case OtaStatus.DOWNLOADING:
        note = 'Downloading update…';
        break;
      case OtaStatus.INSTALLING:
        note = 'Installation started — follow the system prompt to finish.';
        break;
      case OtaStatus.ALREADY_RUNNING_ERROR:
        note = 'A download is already in progress.';
        break;
      case OtaStatus.PERMISSION_NOT_GRANTED_ERROR:
        note = 'Install permission was not granted. Allow installs from this '
            'source and retry.';
        break;
      case OtaStatus.INTERNAL_ERROR:
        note = 'The installer hit an internal error.';
        break;
      case OtaStatus.DOWNLOAD_ERROR:
        note = 'The update could not be downloaded.';
        break;
      case OtaStatus.CHECKSUM_ERROR:
        note = 'The download failed its SHA256 integrity check and was '
            'discarded.';
        break;
      case OtaStatus.CANCELED:
        note = 'The update was cancelled.';
        break;
    }
    return InstallProgress(status: event.status, value: event.value, note: note);
  }

  /// End-to-end update: fetch metadata, verify the artifact where the strategy
  /// calls for it, then trigger installation.
  ///
  /// Returns the stream of installer progress, or throws a typed exception
  /// before installation begins if verification fails.
  Future<Stream<InstallProgress>> performUpdate({
    required ReleaseInfo release,
    void Function(DownloadProgress progress)? onDownloadProgress,
  }) async {
    if (!release.hasRelease) {
      throw ReleaseMetadataException(
        'the server has not published a downloadable release',
        field: 'download_url',
      );
    }
    final Uri? url = Uri.tryParse(release.downloadUrl);
    if (url == null || !url.hasScheme) {
      throw ReleaseMetadataException(
        '"${release.downloadUrl}" is not an absolute URL',
        field: 'download_url',
      );
    }
    final String fileName = url.pathSegments.isEmpty
        ? 'harbor-update.apk'
        : url.pathSegments.last;

    switch (strategy) {
      case InstallStrategy.nativeStreaming:
        // The plugin downloads and verifies natively; the digest still travels
        // with the request so the platform performs the integrity check.
        return install(
          url: url,
          sha256Checksum: release.sha256.isEmpty ? null : release.sha256,
          destinationFilename: fileName,
        );

      case InstallStrategy.verifyThenInstall:
        if (release.sha256.isEmpty) {
          throw ReleaseMetadataException(
            'verifyThenInstall requires a sha256 digest from the server',
            field: 'sha256',
          );
        }
        final VerifiedArtifact artifact = await downloadAndVerify(
          url: url,
          expectedSha256: release.sha256,
          fileName: fileName,
          expectedBytes: release.sizeBytes > 0 ? release.sizeBytes : null,
          onProgress: onDownloadProgress,
        );
        return install(
          url: url,
          sha256Checksum: artifact.sha256,
          localFile: artifact.file,
          destinationFilename: fileName,
        );
    }
  }

  /// Requests cancellation of an in-flight native download.
  Future<void> cancelDownload() async {
    try {
      await _otaUpdate.cancel();
    } on PlatformException catch (e) {
      throw UpdateInstallException(
        'could not cancel the download: ${e.message}',
      );
    }
  }

  /// The most-preferred ABI, useful when the server publishes split APKs.
  Future<String?> preferredAbi() async {
    try {
      return await _otaUpdate.getAbi();
    } on PlatformException catch (e) {
      throw UpdateInstallException('could not read the device ABI: ${e.message}');
    }
  }

  /// Whether [version] was previously dismissed by the user.
  Future<bool> isDismissed(String version) async {
    final SharedPreferences? prefs = await _prefs();
    if (prefs == null) {
      return false;
    }
    final String? stored = prefs.getString(dismissedVersionKey);
    if (stored == null) {
      return false;
    }
    final Version? dismissed = _tryVersion(stored);
    final Version? candidate = _tryVersion(version);
    if (dismissed == null || candidate == null) {
      return stored == version;
    }
    return dismissed == candidate;
  }

  /// Records that the user dismissed [version]'s soft update prompt.
  ///
  /// A forced update is never dismissible, so callers must not invoke this for
  /// a [UpdateDecision.forceUpdate].
  Future<void> dismiss(String version) async {
    final SharedPreferences? prefs = await _prefs();
    if (prefs == null) {
      return;
    }
    await prefs.setString(dismissedVersionKey, version);
  }

  /// Clears the dismissal, e.g. after the user taps "check again".
  Future<void> clearDismissal() async {
    final SharedPreferences? prefs = await _prefs();
    await prefs?.remove(dismissedVersionKey);
  }

  /// When the last successful check ran.
  Future<DateTime?> lastCheckAt() async {
    final SharedPreferences? prefs = await _prefs();
    final String? raw = prefs?.getString(lastCheckKey);
    return raw == null ? null : DateTime.tryParse(raw)?.toUtc();
  }

  Future<void> _recordCheck(DateTime at) async {
    final SharedPreferences? prefs = await _prefs();
    await prefs?.setString(lastCheckKey, at.toIso8601String());
  }

  /// SharedPreferences is best-effort: without it, dismissal simply does not
  /// persist rather than the update flow failing.
  Future<SharedPreferences?> _prefs() async {
    try {
      return await SharedPreferences.getInstance();
    } on Object {
      return null;
    }
  }

  /// Releases the HTTP client when this service owns it.
  void dispose() {
    if (_ownsClient) {
      _client.close();
    }
  }

  /// Diagnostic snapshot for the settings screen.
  Map<String, Object?> describe() => <String, Object?>{
        'base_url': baseUrl,
        'current_version': currentVersion,
        'platform': platform,
        'strategy': strategy.name,
        'request_timeout_ms': requestTimeout.inMilliseconds,
      };
}

Version? _tryVersion(String raw) {
  final String trimmed = raw.trim();
  if (trimmed.isEmpty) {
    return null;
  }
  try {
    // Accept a leading "v" as GitHub tags are published that way.
    final String normalised =
        trimmed.startsWith('v') ? trimmed.substring(1) : trimmed;
    return Version.parse(normalised);
  } on FormatException {
    return null;
  }
}

int? _asInt(Object? value) {
  if (value is int) {
    return value;
  }
  if (value is num) {
    return value.toInt();
  }
  if (value is String) {
    return int.tryParse(value);
  }
  return null;
}

bool? _asBool(Object? value) {
  if (value is bool) {
    return value;
  }
  if (value is num) {
    return value != 0;
  }
  if (value is String) {
    switch (value.trim().toLowerCase()) {
      case 'true':
        return true;
      case 'false':
        return false;
    }
  }
  return null;
}

DateTime? _asDateTime(Object? value) {
  if (value is DateTime) {
    return value.toUtc();
  }
  if (value is String) {
    return DateTime.tryParse(value)?.toUtc();
  }
  if (value is int) {
    return DateTime.fromMillisecondsSinceEpoch(value, isUtc: true);
  }
  return null;
}

/// Decodes a hexadecimal string into bytes, or null when any character is not
/// a hex digit or the string has an odd length.
List<int>? _tryHexDecode(String hex) {
  if (hex.length.isOdd) {
    return null;
  }
  final List<int> out = List<int>.filled(hex.length ~/ 2, 0);
  for (int i = 0; i < out.length; i++) {
    final int hi = _hexDigit(hex.codeUnitAt(i * 2));
    final int lo = _hexDigit(hex.codeUnitAt(i * 2 + 1));
    if (hi < 0 || lo < 0) {
      return null;
    }
    out[i] = (hi << 4) | lo;
  }
  return out;
}

/// Numeric value of a single ASCII hex digit, or -1 when [codeUnit] is not one.
int _hexDigit(int codeUnit) {
  if (codeUnit >= 0x30 && codeUnit <= 0x39) {
    return codeUnit - 0x30; // '0'-'9'
  }
  if (codeUnit >= 0x61 && codeUnit <= 0x66) {
    return codeUnit - 0x61 + 10; // 'a'-'f'
  }
  if (codeUnit >= 0x41 && codeUnit <= 0x46) {
    return codeUnit - 0x41 + 10; // 'A'-'F'
  }
  return -1;
}