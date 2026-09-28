// Immutable, strongly typed domain models for the Harbor update server.
//
// This library is deliberately Flutter-free so that `bin/server.dart` and the
// whole `lib/src/**` tree run on the pure Dart VM.

import 'package:pub_semver/pub_semver.dart';

/// Every failure that must be surfaced to an HTTP client as a JSON error.
class ApiException implements Exception {
  const ApiException(this.statusCode, this.message);

  /// HTTP status code to return.
  final int statusCode;

  /// Human readable, client-safe message.
  final String message;

  @override
  String toString() => 'ApiException($statusCode): $message';
}

/// A request that failed validation; always maps to HTTP 400.
class ValidationException extends ApiException {
  const ValidationException(String message) : super(400, message);
}

/// A referenced entity (for example a release for a platform) does not exist;
/// always maps to HTTP 404.
class NotFoundException extends ApiException {
  const NotFoundException(String message) : super(404, message);
}

/// Thrown when on-disk state cannot be read back or decoded.
class StateStoreException implements Exception {
  const StateStoreException(this.message);

  final String message;

  @override
  String toString() => 'StateStoreException: $message';
}

const Set<String> kSupportedPlatforms = <String>{
  'android',
  'ios',
  'web',
  'windows',
  'macos',
  'linux',
};

final RegExp _sha256Pattern = RegExp(r'^[0-9a-f]{64}$');

/// The documented "no release published yet" version.
const String kZeroVersion = '0.0.0';

String? _asString(Object? value) {
  if (value == null) {
    return null;
  }
  if (value is String) {
    return value;
  }
  return value.toString();
}

int? _asInt(Object? value) {
  if (value is int) {
    return value;
  }
  if (value is num) {
    return value.toInt();
  }
  if (value is String) {
    return int.tryParse(value.trim());
  }
  return null;
}

bool? _asBool(Object? value) {
  if (value is bool) {
    return value;
  }
  if (value is String) {
    final String v = value.trim().toLowerCase();
    if (v == 'true' || v == '1') {
      return true;
    }
    if (v == 'false' || v == '0') {
      return false;
    }
  }
  if (value is num) {
    return value != 0;
  }
  return null;
}

Object? _deepCopy(Object? value) {
  if (value is Map<Object?, Object?>) {
    return <String, Object?>{
      for (final MapEntry<Object?, Object?> entry in value.entries)
        entry.key.toString(): _deepCopy(entry.value),
    };
  }
  if (value is List<Object?>) {
    return <Object?>[for (final Object? item in value) _deepCopy(item)];
  }
  return value;
}

/// Returns an unmodifiable deep copy of [value] as a JSON object map.
Map<String, Object?> deepCopyMap(Map<String, Object?> value) {
  final Object? copied = _deepCopy(value);
  if (copied is! Map<String, Object?>) {
    throw const ValidationException('value must be a JSON object');
  }
  return Map<String, Object?>.unmodifiable(copied);
}

/// Parses [raw] with `package:pub_semver`, translating failures into a
/// [ValidationException] that names the offending [field].
Version parseSemver(String raw, {required String field}) {
  final String trimmed = raw.trim();
  if (trimmed.isEmpty) {
    throw ValidationException('$field must not be empty');
  }
  try {
    return Version.parse(trimmed);
  } on FormatException catch (error) {
    throw ValidationException(
      '$field "$raw" is not a valid semantic version: ${error.message}',
    );
  }
}

Version? tryParseSemver(String raw) {
  try {
    return Version.parse(raw.trim());
  } on FormatException {
    return null;
  }
}

// ---------------------------------------------------------------------------
// ReleaseInfo
// ---------------------------------------------------------------------------

/// A single published release for one platform.
class ReleaseInfo {
  ReleaseInfo({
    required this.platform,
    required this.version,
    required this.minSupportedVersion,
    required this.downloadUrl,
    required this.sizeBytes,
    required this.changelog,
    required this.forceUpdate,
    required String sha256,
    this.releaseNotesUrl = '',
    this.publishedAt,
  }) : sha256 = sha256.trim().toLowerCase() {
    _validate();
  }

  factory ReleaseInfo.fromJson(
    Map<String, Object?> json, {
    DateTime? defaultPublishedAt,
  }) {
    final String platform = _asString(json['platform']) ?? '';
    final String version =
        _asString(json['version'] ?? json['latest_version']) ?? '';
    final String minSupported =
        _asString(json['min_supported_version']) ?? version;
    final String downloadUrl = _asString(json['download_url']) ?? '';
    final String sha256 = _asString(json['sha256']) ?? '';
    final int sizeBytes = _asInt(json['size_bytes']) ?? 0;
    final String changelog = _asString(json['changelog']) ?? '';
    final bool forceUpdate = _asBool(json['force_update']) ?? false;
    final String releaseNotesUrl =
        _asString(json['release_notes_url']) ?? '';
    final DateTime? publishedAt = _parseDateTime(json['published_at']) ??
        defaultPublishedAt;

    return ReleaseInfo(
      platform: platform,
      version: version,
      minSupportedVersion: minSupported,
      downloadUrl: downloadUrl,
      sha256: sha256,
      sizeBytes: sizeBytes,
      changelog: changelog,
      forceUpdate: forceUpdate,
      releaseNotesUrl: releaseNotesUrl,
      publishedAt: publishedAt,
    );
  }

  final String platform;
  final String version;
  final String minSupportedVersion;
  final String downloadUrl;
  final String sha256;
  final int sizeBytes;
  final String changelog;
  final bool forceUpdate;
  final String releaseNotesUrl;
  final DateTime? publishedAt;

  bool get isPlaceholder => version == kZeroVersion;

  Version get semver => parseSemver(version, field: 'version');

  Version get minSupportedSemver =>
      parseSemver(minSupportedVersion, field: 'min_supported_version');

  void _validate() {
    if (!kSupportedPlatforms.contains(platform)) {
      throw ValidationException(
        'platform "$platform" is not supported; expected one of '
        '${kSupportedPlatforms.join(', ')}',
      );
    }
    parseSemver(version, field: 'version');
    parseSemver(minSupportedVersion, field: 'min_supported_version');

    if (sizeBytes < 0) {
      throw ValidationException('size_bytes must be >= 0 (got $sizeBytes)');
    }

    if (isPlaceholder) {
      // The documented zero payload may omit the artifact fields entirely.
      return;
    }

    if (!_sha256Pattern.hasMatch(sha256)) {
      throw ValidationException(
        'sha256 must be exactly 64 lowercase hexadecimal characters '
        '(got "${sha256.length} chars")',
      );
    }

    if (downloadUrl.trim().isEmpty) {
      throw const ValidationException(
        'download_url must not be empty for a published release',
      );
    }
    final Uri? uri = Uri.tryParse(downloadUrl);
    if (uri == null || !uri.isAbsolute || !uri.hasScheme) {
      throw ValidationException(
        'download_url "$downloadUrl" is not an absolute URI',
      );
    }
    if (uri.scheme != 'http' && uri.scheme != 'https') {
      throw ValidationException(
        'download_url must use the http or https scheme (got "${uri.scheme}")',
      );
    }

    if (releaseNotesUrl.trim().isNotEmpty) {
      final Uri? notes = Uri.tryParse(releaseNotesUrl);
      if (notes == null || !notes.isAbsolute) {
        throw ValidationException(
          'release_notes_url "$releaseNotesUrl" is not an absolute URI',
        );
      }
    }
  }

  ReleaseInfo copyWith({
    String? platform,
    String? version,
    String? minSupportedVersion,
    String? downloadUrl,
    String? sha256,
    int? sizeBytes,
    String? changelog,
    bool? forceUpdate,
    String? releaseNotesUrl,
    DateTime? publishedAt,
  }) {
    return ReleaseInfo(
      platform: platform ?? this.platform,
      version: version ?? this.version,
      minSupportedVersion: minSupportedVersion ?? this.minSupportedVersion,
      downloadUrl: downloadUrl ?? this.downloadUrl,
      sha256: sha256 ?? this.sha256,
      sizeBytes: sizeBytes ?? this.sizeBytes,
      changelog: changelog ?? this.changelog,
      forceUpdate: forceUpdate ?? this.forceUpdate,
      releaseNotesUrl: releaseNotesUrl ?? this.releaseNotesUrl,
      publishedAt: publishedAt ?? this.publishedAt,
    );
  }

  /// Stored-release representation (note: `version`, not `latest_version`).
  Map<String, Object?> toJson() => <String, Object?>{
        'platform': platform,
        'version': version,
        'min_supported_version': minSupportedVersion,
        'download_url': downloadUrl,
        'sha256': sha256,
        'size_bytes': sizeBytes,
        'changelog': changelog,
        'force_update': forceUpdate,
        'release_notes_url': releaseNotesUrl,
        'published_at': publishedAt?.toUtc().toIso8601String(),
      };

  @override
  String toString() =>
      'ReleaseInfo($platform $version min=$minSupportedVersion '
      'force=$forceUpdate)';
}

DateTime? _parseDateTime(Object? value) {
  if (value == null) {
    return null;
  }
  if (value is DateTime) {
    return value.toUtc();
  }
  final String? text = _asString(value);
  if (text == null || text.trim().isEmpty) {
    return null;
  }
  return DateTime.tryParse(text)?.toUtc();
}

// ---------------------------------------------------------------------------
// FeatureFlagMatrix
// ---------------------------------------------------------------------------

/// The server-owned feature flag matrix plus the dynamic layout document.
class FeatureFlagMatrix {
  FeatureFlagMatrix({
    required this.version,
    required this.updatedAt,
    required Map<String, Object?> flags,
    required Map<String, Object?> remoteDefaults,
    required Map<String, Object?> layout,
  })  : flags = deepCopyMap(flags),
        remoteDefaults = deepCopyMap(remoteDefaults),
        layout = _normalizeLayout(layout);

  factory FeatureFlagMatrix.empty() => FeatureFlagMatrix(
        version: 0,
        updatedAt: null,
        flags: const <String, Object?>{},
        remoteDefaults: const <String, Object?>{},
        layout: const <String, Object?>{'sections': <Object?>[]},
      );

  factory FeatureFlagMatrix.fromJson(Map<String, Object?> json) {
    return FeatureFlagMatrix(
      version: _asInt(json['version']) ?? 0,
      updatedAt: _parseDateTime(json['updated_at']),
      flags: _asMap(json['flags']),
      remoteDefaults: _asMap(json['remote_defaults']),
      layout: _asMap(json['layout']),
    );
  }

  final int version;
  final DateTime? updatedAt;
  final Map<String, Object?> flags;
  final Map<String, Object?> remoteDefaults;
  final Map<String, Object?> layout;

  FeatureFlagMatrix copyWith({
    int? version,
    DateTime? updatedAt,
    Map<String, Object?>? flags,
    Map<String, Object?>? remoteDefaults,
    Map<String, Object?>? layout,
  }) {
    return FeatureFlagMatrix(
      version: version ?? this.version,
      updatedAt: updatedAt ?? this.updatedAt,
      flags: flags ?? this.flags,
      remoteDefaults: remoteDefaults ?? this.remoteDefaults,
      layout: layout ?? this.layout,
    );
  }

  /// Applies [other] on top of this matrix and bumps the revision by one.
  ///
  /// With `merge == true` the scalar maps are key-wise merged and layout
  /// sections/modules are merged by id; with `merge == false` both the scalar
  /// maps and the layout are replaced wholesale.
  FeatureFlagMatrix merge(FeatureFlagMatrix other, {required bool merge}) {
    final Map<String, Object?> mergedFlags;
    final Map<String, Object?> mergedDefaults;
    final Map<String, Object?> mergedLayout;

    if (merge) {
      mergedFlags = <String, Object?>{...flags, ...other.flags};
      mergedDefaults = <String, Object?>{
        ...remoteDefaults,
        ...other.remoteDefaults,
      };
      mergedLayout = mergeLayouts(layout, other.layout);
    } else {
      mergedFlags = Map<String, Object?>.of(other.flags);
      mergedDefaults = Map<String, Object?>.of(other.remoteDefaults);
      mergedLayout = _normalizeLayout(other.layout);
    }

    return FeatureFlagMatrix(
      version: version + 1,
      updatedAt: other.updatedAt ?? DateTime.now().toUtc(),
      flags: mergedFlags,
      remoteDefaults: mergedDefaults,
      layout: mergedLayout,
    );
  }

  Map<String, Object?> toJson() => <String, Object?>{
        'version': version,
        'updated_at': updatedAt?.toUtc().toIso8601String(),
        'flags': flags,
        'remote_defaults': remoteDefaults,
        'layout': layout,
      };

  /// Merges [patch] into [base] by section id, then module id within a section.
  ///
  /// Output is normalized: sections are sorted by `order` then `id`, and the
  /// modules of every section are sorted by `order` then `id`.
  static Map<String, Object?> mergeLayouts(
    Map<String, Object?> base,
    Map<String, Object?> patch,
  ) {
    final Map<String, Map<String, Object?>> byId =
        <String, Map<String, Object?>>{};
    for (final Map<String, Object?> section in _sectionsOf(base)) {
      byId[section['id']! as String] = section;
    }
    for (final Map<String, Object?> incoming in _sectionsOf(patch)) {
      final String id = incoming['id']! as String;
      final Map<String, Object?>? existing = byId[id];
      if (existing == null) {
        byId[id] = incoming;
        continue;
      }
      final Map<String, Object?> merged = Map<String, Object?>.of(existing)
        ..addAll(incoming);
      merged['modules'] = _mergeModules(
        _modulesOf(existing),
        _modulesOf(incoming),
      );
      byId[id] = merged;
    }
    return _normalizeLayout(<String, Object?>{
      'sections': byId.values.toList(growable: false),
    });
  }

  static List<Map<String, Object?>> _mergeModules(
    List<Map<String, Object?>> base,
    List<Map<String, Object?>> patch,
  ) {
    final Map<String, Map<String, Object?>> byId =
        <String, Map<String, Object?>>{};
    for (final Map<String, Object?> module in base) {
      byId[module['id']! as String] = module;
    }
    for (final Map<String, Object?> incoming in patch) {
      byId[incoming['id']! as String] = incoming;
    }
    final List<Map<String, Object?>> modules = byId.values.toList();
    modules.sort(_byOrderThenId);
    return modules;
  }

  /// Validates the shape of an admin-supplied layout document.
  static void validateLayout(Object? layout) {
    if (layout == null) {
      return;
    }
    if (layout is! Map<Object?, Object?>) {
      throw const ValidationException('layout must be a JSON object');
    }
    final Object? sections = layout['sections'];
    if (sections == null) {
      return;
    }
    if (sections is! List<Object?>) {
      throw const ValidationException('layout.sections must be a JSON array');
    }
    for (int i = 0; i < sections.length; i++) {
      final Object? rawSection = sections[i];
      if (rawSection is! Map<Object?, Object?>) {
        throw ValidationException('layout.sections[$i] must be a JSON object');
      }
      final Object? id = rawSection['id'];
      if (id is! String || id.trim().isEmpty) {
        throw ValidationException(
          'layout.sections[$i] is missing a non-empty "id"',
        );
      }
      final Object? modules = rawSection['modules'];
      if (modules == null) {
        continue;
      }
      if (modules is! List<Object?>) {
        throw ValidationException(
          'layout.sections[$i] ("$id").modules must be a JSON array',
        );
      }
      for (int j = 0; j < modules.length; j++) {
        final Object? rawModule = modules[j];
        if (rawModule is! Map<Object?, Object?>) {
          throw ValidationException(
            'layout.sections[$i] ("$id").modules[$j] must be a JSON object',
          );
        }
        final Object? moduleId = rawModule['id'];
        if (moduleId is! String || moduleId.trim().isEmpty) {
          throw ValidationException(
            'layout.sections[$i] ("$id").modules[$j] is missing a '
            'non-empty "id"',
          );
        }
        final Object? type = rawModule['type'];
        if (type == null || (type is String && type.trim().isEmpty)) {
          throw ValidationException(
            'layout module "$moduleId" is missing a non-empty "type"',
          );
        }
      }
    }
  }

  static Map<String, Object?> _normalizeLayout(Map<String, Object?> layout) {
    final List<Map<String, Object?>> sections = _sectionsOf(layout).toList();
    for (final Map<String, Object?> section in sections) {
      final List<Map<String, Object?>> modules = _modulesOf(section).toList();
      modules.sort(_byOrderThenId);
      section['modules'] = modules;
    }
    sections.sort(_byOrderThenId);
    return deepCopyMap(<String, Object?>{'sections': sections});
  }

  static List<Map<String, Object?>> _sectionsOf(Map<String, Object?> layout) {
    final Object? raw = layout['sections'];
    if (raw is! List<Object?>) {
      return <Map<String, Object?>>[];
    }
    final List<Map<String, Object?>> sections = <Map<String, Object?>>[];
    for (final Object? entry in raw) {
      if (entry is Map<Object?, Object?>) {
        sections.add(_asMap(entry));
      }
    }
    return sections;
  }

  static List<Map<String, Object?>> _modulesOf(Map<String, Object?> section) {
    final Object? raw = section['modules'];
    if (raw is! List<Object?>) {
      return <Map<String, Object?>>[];
    }
    final List<Map<String, Object?>> modules = <Map<String, Object?>>[];
    for (final Object? entry in raw) {
      if (entry is Map<Object?, Object?>) {
        modules.add(_asMap(entry));
      }
    }
    return modules;
  }

  static int _byOrderThenId(Map<String, Object?> a, Map<String, Object?> b) {
    final int orderA = _asInt(a['order']) ?? 0;
    final int orderB = _asInt(b['order']) ?? 0;
    final int cmp = orderA.compareTo(orderB);
    if (cmp != 0) {
      return cmp;
    }
    return (_asString(a['id']) ?? '').compareTo(_asString(b['id']) ?? '');
  }

  @override
  String toString() => 'FeatureFlagMatrix(v$version, '
      '${flags.length} flags, ${_sectionsOf(layout).length} sections)';
}

Map<String, Object?> _asMap(Object? value) {
  if (value is Map<Object?, Object?>) {
    return <String, Object?>{
      for (final MapEntry<Object?, Object?> entry in value.entries)
        entry.key.toString(): entry.value,
    };
  }
  return <String, Object?>{};
}

// ---------------------------------------------------------------------------
// ServerState
// ---------------------------------------------------------------------------

/// The complete persisted server state.
class ServerState {
  ServerState({
    required Map<String, ReleaseInfo> releasesByPlatform,
    required this.flags,
  }) : releasesByPlatform =
            Map<String, ReleaseInfo>.unmodifiable(releasesByPlatform);

  factory ServerState.empty() => ServerState(
        releasesByPlatform: const <String, ReleaseInfo>{},
        flags: FeatureFlagMatrix.empty(),
      );

  factory ServerState.fromJson(Map<String, Object?> json) {
    final Map<String, ReleaseInfo> releases = <String, ReleaseInfo>{};
    final Object? rawReleases = json['releases'];
    if (rawReleases is Map<Object?, Object?>) {
      for (final MapEntry<Object?, Object?> entry in rawReleases.entries) {
        final String platform = entry.key.toString();
        final Object? value = entry.value;
        if (value is! Map<Object?, Object?>) {
          throw ValidationException(
            'release entry "$platform" must be a JSON object',
          );
        }
        final Map<String, Object?> record = _asMap(value);
        releases[platform] =
            ReleaseInfo.fromJson(<String, Object?>{
          'platform': record['platform'] ?? platform,
          ...record,
        });
      }
    }
    final Object? rawFlags = json['flags'];
    return ServerState(
      releasesByPlatform: releases,
      flags: rawFlags is Map<Object?, Object?>
          ? FeatureFlagMatrix.fromJson(_asMap(rawFlags))
          : FeatureFlagMatrix.empty(),
    );
  }

  final Map<String, ReleaseInfo> releasesByPlatform;
  final FeatureFlagMatrix flags;

  ReleaseInfo? releaseFor(String platform) => releasesByPlatform[platform];

  ServerState copyWith({
    Map<String, ReleaseInfo>? releasesByPlatform,
    FeatureFlagMatrix? flags,
  }) {
    return ServerState(
      releasesByPlatform: releasesByPlatform ?? this.releasesByPlatform,
      flags: flags ?? this.flags,
    );
  }

  /// Returns a copy with [release] stored under its own platform key.
  ServerState withRelease(ReleaseInfo release) {
    return copyWith(
      releasesByPlatform: <String, ReleaseInfo>{
        ...releasesByPlatform,
        release.platform: release,
      },
    );
  }

  Map<String, Object?> toJson() => <String, Object?>{
        'releases': <String, Object?>{
          for (final MapEntry<String, ReleaseInfo> entry
              in releasesByPlatform.entries)
            entry.key: entry.value.toJson(),
        },
        'flags': flags.toJson(),
      };

  /// The defaults written when no state file exists yet.
  ///
  /// The layout below must stay in step with two other copies of the same
  /// document: the client's compiled-in `kDefaultFlagMatrix` (apps/main_app) and
  /// `apps/update_server/flags/default_flags.json`, which the Pages builder
  /// publishes as `/api/v1/flags.json`. All three describe the same first-run UI,
  /// so a hosted server, a static Pages deployment and a fully offline app
  /// render the same thing. Every `type` here must have a renderer in the
  /// client's `DynamicModuleRegistry`; both test suites assert that.
  static ServerState seed({DateTime? now}) {
    final DateTime publishedAt =
        DateTime.utc(2026, 2, 1, 10, 0, 0);
    return ServerState(
      releasesByPlatform: <String, ReleaseInfo>{
        'android': ReleaseInfo(
          platform: 'android',
          version: '1.0.0',
          minSupportedVersion: '1.0.0',
          downloadUrl: 'https://github.com/OWNER/REPO/releases/download/'
              'v1.0.0/app-release.apk',
          sha256: 'a1b2c3d4e5f60718293a4b5c6d7e8f90'
              'a1b2c3d4e5f60718293a4b5c6d7e8f90',
          sizeBytes: 24117248,
          changelog: '• Initial Harbor release.\n'
              '• On-device agent core and OTA update engine.',
          forceUpdate: false,
          releaseNotesUrl:
              'https://github.com/OWNER/REPO/releases/tag/v1.0.0',
          publishedAt: publishedAt,
        ),
      },
      flags: FeatureFlagMatrix(
        version: 1,
        updatedAt: publishedAt,
        flags: const <String, Object?>{
          'dynamic_ui': true,
          'agent.thinking': true,
          'agent.subagents': true,
          'labs.voice_mode': false,
          'labs.lora_training': false,
        },
        remoteDefaults: const <String, Object?>{
          'dynamic_ui': true,
          'agent.thinking': true,
          'agent.subagents': true,
          'labs.voice_mode': false,
          'labs.lora_training': false,
        },
        layout: const <String, Object?>{
          'sections': <Object?>[
            <String, Object?>{
              'id': 'home',
              'title': 'Home',
              'order': 0,
              'modules': <Object?>[
                <String, Object?>{
                  'id': 'engine_status',
                  'type': 'engine_status',
                  'flag': '',
                  'order': 0,
                },
                <String, Object?>{
                  'id': 'thinking_panel',
                  'type': 'thinking_panel',
                  'flag': 'agent.thinking',
                  'order': 1,
                  'props': <String, Object?>{'collapsedByDefault': true},
                },
                <String, Object?>{
                  'id': 'agent_console',
                  'type': 'agent_console',
                  'flag': 'agent.subagents',
                  'order': 2,
                },
                <String, Object?>{
                  'id': 'file_inspector',
                  'type': 'file_inspector',
                  'flag': '',
                  'order': 3,
                },
                <String, Object?>{
                  'id': 'update_status',
                  'type': 'update_status',
                  'flag': '',
                  'order': 4,
                },
                <String, Object?>{
                  'id': 'feature_flags',
                  'type': 'feature_flags',
                  'flag': 'dynamic_ui',
                  'order': 5,
                },
              ],
            },
            <String, Object?>{
              'id': 'labs',
              'title': 'Labs',
              'order': 1,
              'modules': <Object?>[
                <String, Object?>{
                  'id': 'labs_announcement',
                  'type': 'banner',
                  'flag': 'labs.voice_mode',
                  'order': 0,
                  'props': <String, Object?>{
                    'message': 'Voice mode is enabled for this account.',
                    'severity': 'info',
                  },
                },
              ],
            },
          ],
        },
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// UpdateCheckResponse
// ---------------------------------------------------------------------------

/// The exact JSON payload returned by `GET /api/v1/update-check`.
class UpdateCheckResponse {
  const UpdateCheckResponse({
    required this.latestVersion,
    required this.minSupportedVersion,
    required this.downloadUrl,
    required this.sha256,
    required this.sizeBytes,
    required this.changelog,
    required this.forceUpdate,
    required this.platform,
    required this.updateAvailable,
    required this.updateRequired,
    this.publishedAt,
    this.releaseNotesUrl = '',
  });

  /// The documented payload for a platform with no published release.
  factory UpdateCheckResponse.empty(String platform) => UpdateCheckResponse(
        latestVersion: kZeroVersion,
        minSupportedVersion: kZeroVersion,
        downloadUrl: '',
        sha256: '',
        sizeBytes: 0,
        changelog: '',
        forceUpdate: false,
        platform: platform,
        updateAvailable: false,
        updateRequired: false,
      );

  factory UpdateCheckResponse.fromJson(Map<String, Object?> json) {
    final String latest = _asString(json['latest_version']) ?? kZeroVersion;
    return UpdateCheckResponse(
      latestVersion: latest,
      minSupportedVersion:
          _asString(json['min_supported_version']) ?? latest,
      downloadUrl: _asString(json['download_url']) ?? '',
      sha256: _asString(json['sha256']) ?? '',
      sizeBytes: _asInt(json['size_bytes']) ?? 0,
      changelog: _asString(json['changelog']) ?? '',
      forceUpdate: _asBool(json['force_update']) ?? false,
      platform: _asString(json['platform']) ?? '',
      updateAvailable: _asBool(json['update_available']) ?? false,
      updateRequired: _asBool(json['update_required']) ?? false,
      publishedAt: _parseDateTime(json['published_at']),
      releaseNotesUrl: _asString(json['release_notes_url']) ?? '',
    );
  }

  /// Computes the payload for [installed] from an optional stored [release].
  factory UpdateCheckResponse.evaluate({
    required ReleaseInfo? release,
    required Version installed,
    required String platform,
  }) {
    if (release == null || release.isPlaceholder) {
      return UpdateCheckResponse.empty(platform);
    }
    final Version latest = release.semver;
    final Version minSupported = release.minSupportedSemver;
    return UpdateCheckResponse(
      latestVersion: release.version,
      minSupportedVersion: release.minSupportedVersion,
      downloadUrl: release.downloadUrl,
      sha256: release.sha256,
      sizeBytes: release.sizeBytes,
      changelog: release.changelog,
      forceUpdate: release.forceUpdate,
      platform: platform,
      updateAvailable: installed < latest,
      updateRequired: installed < minSupported || release.forceUpdate,
      publishedAt: release.publishedAt,
      releaseNotesUrl: release.releaseNotesUrl,
    );
  }

  final String latestVersion;
  final String minSupportedVersion;
  final String downloadUrl;
  final String sha256;
  final int sizeBytes;
  final String changelog;
  final bool forceUpdate;
  final String platform;
  final bool updateAvailable;
  final bool updateRequired;
  final DateTime? publishedAt;
  final String releaseNotesUrl;

  /// Serializes the contract payload, optionally adding `generated_at` for the
  /// GitHub-Pages-compatible `latest_version.json` artifact.
  Map<String, Object?> toJson({DateTime? generatedAt}) => <String, Object?>{
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
        if (generatedAt != null)
          'generated_at': generatedAt.toUtc().toIso8601String(),
      };

  @override
  String toString() => 'UpdateCheckResponse($platform $latestVersion '
      'available=$updateAvailable required=$updateRequired)';
}