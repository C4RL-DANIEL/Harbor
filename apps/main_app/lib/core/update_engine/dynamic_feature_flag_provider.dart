// Dynamic feature-flag provider and dynamic-layout schema.
//
// The update server owns a JSON "flag matrix" that the app fetches on launch
// and re-fetches on demand. Nothing about the UI is hard-coded: the matrix also
// carries a *layout* document describing sections and modules, and each module
// declares the flag that gates it. The result is that the server can add,
// remove or reorder UI modules without shipping a new build.
//
// Resolution order for any key:
//
//     local override  >  remote flags  >  remote defaults  >  caller default
//
// Local overrides are persisted so a developer (or an A/B assignment) survives
// restarts. Every failure is typed: a network problem never clears the last
// known-good matrix, so a flaky connection degrades to cached flags rather than
// to a broken UI.

import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

/// Thrown when the flag matrix cannot be retrieved.
class FlagFetchException implements Exception {
  FlagFetchException(this.message, {this.statusCode, this.cause});

  final String message;
  final int? statusCode;
  final Object? cause;

  @override
  String toString() => 'FlagFetchException: $message'
      '${statusCode == null ? '' : ' (HTTP $statusCode)'}'
      '${cause == null ? '' : ' ($cause)'}';
}

/// Thrown when the flag matrix payload is structurally invalid.
class FlagParseException implements Exception {
  FlagParseException(this.message, {this.key});

  final String message;
  final String? key;

  @override
  String toString() =>
      'FlagParseException: $message${key == null ? '' : ' [key: $key]'}';
}

/// The JSON type a flag currently resolves to.
enum FlagValueType {
  boolean,
  integer,
  doubleValue,
  string,
  list,
  map,
  nullValue;

  /// Classifies a decoded JSON value.
  static FlagValueType of(Object? value) {
    if (value == null) {
      return FlagValueType.nullValue;
    }
    if (value is bool) {
      return FlagValueType.boolean;
    }
    if (value is int) {
      return FlagValueType.integer;
    }
    if (value is double) {
      return FlagValueType.doubleValue;
    }
    if (value is String) {
      return FlagValueType.string;
    }
    if (value is List) {
      return FlagValueType.list;
    }
    if (value is Map) {
      return FlagValueType.map;
    }
    return FlagValueType.nullValue;
  }

  String get label {
    switch (this) {
      case FlagValueType.boolean:
        return 'boolean';
      case FlagValueType.integer:
        return 'integer';
      case FlagValueType.doubleValue:
        return 'number';
      case FlagValueType.string:
        return 'string';
      case FlagValueType.list:
        return 'list';
      case FlagValueType.map:
        return 'object';
      case FlagValueType.nullValue:
        return 'null';
    }
  }
}

/// A single injectable UI module declared by the server.
@immutable
class DynamicModule {
  const DynamicModule({
    required this.id,
    required this.type,
    required this.flag,
    this.order = 0,
    this.props = const <String, Object?>{},
    this.title,
  });

  /// Stable identifier, unique within its section.
  final String id;

  /// Renderer key the client maps to a concrete widget.
  final String type;

  /// Feature flag that gates this module's visibility. An empty string means
  /// "always visible".
  final String flag;

  /// Sort position within the section (ascending).
  final int order;

  /// Arbitrary server-provided configuration for the renderer.
  final Map<String, Object?> props;

  /// Optional display title.
  final String? title;

  factory DynamicModule.fromJson(Map<String, Object?> json) {
    final Object? id = json['id'];
    final Object? type = json['type'];
    if (id is! String || id.isEmpty) {
      throw FlagParseException('module is missing a non-empty "id"');
    }
    if (type is! String || type.isEmpty) {
      throw FlagParseException('module "$id" is missing a non-empty "type"',
          key: id);
    }
    return DynamicModule(
      id: id,
      type: type,
      flag: json['flag'] as String? ?? '',
      order: _asInt(json['order']) ?? 0,
      props: _asStringObjectMap(json['props']) ?? const <String, Object?>{},
      title: json['title'] as String?,
    );
  }

  /// Reads a typed prop with a fallback.
  T prop<T>(String key, T fallback) {
    final Object? value = props[key];
    if (value is T) {
      return value;
    }
    if (fallback is bool && value is String) {
      final String v = value.toLowerCase();
      if (v == 'true') {
        return true as T;
      }
      if (v == 'false') {
        return false as T;
      }
    }
    if (fallback is int && value is num) {
      return value.toInt() as T;
    }
    if (fallback is double && value is num) {
      return value.toDouble() as T;
    }
    if (fallback is String && value != null) {
      return value.toString() as T;
    }
    return fallback;
  }

  Map<String, Object?> toJson() => <String, Object?>{
        'id': id,
        'type': type,
        'flag': flag,
        'order': order,
        'props': props,
        if (title != null) 'title': title,
      };
}

/// A named group of modules.
@immutable
class DynamicSection {
  const DynamicSection({
    required this.id,
    required this.title,
    required this.modules,
    this.order = 0,
    this.flag = '',
  });

  final String id;
  final String title;
  final List<DynamicModule> modules;
  final int order;

  /// Optional flag gating the entire section.
  final String flag;

  factory DynamicSection.fromJson(Map<String, Object?> json) {
    final Object? id = json['id'];
    if (id is! String || id.isEmpty) {
      throw FlagParseException('section is missing a non-empty "id"');
    }
    final List<DynamicModule> modules = <DynamicModule>[];
    final Object? rawModules = json['modules'];
    if (rawModules is List) {
      for (int i = 0; i < rawModules.length; i++) {
        final Map<String, Object?>? m = _asStringObjectMap(rawModules[i]);
        if (m == null) {
          throw FlagParseException(
            'section "$id" module at index $i is not a JSON object',
            key: id,
          );
        }
        modules.add(DynamicModule.fromJson(m));
      }
    }
    modules.sort((DynamicModule a, DynamicModule b) {
      final int cmp = a.order.compareTo(b.order);
      return cmp != 0 ? cmp : a.id.compareTo(b.id);
    });
    return DynamicSection(
      id: id,
      title: json['title'] as String? ?? id,
      modules: List<DynamicModule>.unmodifiable(modules),
      order: _asInt(json['order']) ?? 0,
      flag: json['flag'] as String? ?? '',
    );
  }

  Map<String, Object?> toJson() => <String, Object?>{
        'id': id,
        'title': title,
        'order': order,
        if (flag.isNotEmpty) 'flag': flag,
        'modules':
            modules.map((DynamicModule m) => m.toJson()).toList(growable: false),
      };
}

/// The full dynamic layout document.
@immutable
class DynamicLayout {
  const DynamicLayout({required this.sections});

  final List<DynamicSection> sections;

  /// An empty layout: the client then falls back to its built-in UI.
  static const DynamicLayout empty = DynamicLayout(sections: <DynamicSection>[]);

  factory DynamicLayout.fromJson(Map<String, Object?>? json) {
    if (json == null) {
      return DynamicLayout.empty;
    }
    final Object? rawSections = json['sections'];
    if (rawSections is! List) {
      return DynamicLayout.empty;
    }
    final List<DynamicSection> sections = <DynamicSection>[];
    for (int i = 0; i < rawSections.length; i++) {
      final Map<String, Object?>? s = _asStringObjectMap(rawSections[i]);
      if (s == null) {
        throw FlagParseException('layout section at index $i is not an object');
      }
      sections.add(DynamicSection.fromJson(s));
    }
    sections.sort((DynamicSection a, DynamicSection b) {
      final int cmp = a.order.compareTo(b.order);
      return cmp != 0 ? cmp : a.id.compareTo(b.id);
    });
    return DynamicLayout(sections: List<DynamicSection>.unmodifiable(sections));
  }

  /// All modules across all sections, in layout order.
  List<DynamicModule> get allModules => sections
      .expand((DynamicSection s) => s.modules)
      .toList(growable: false);

  Map<String, Object?> toJson() => <String, Object?>{
        'sections':
            sections.map((DynamicSection s) => s.toJson()).toList(growable: false),
      };
}

/// An immutable snapshot of the server's flag matrix.
@immutable
class FeatureFlagMatrix {
  const FeatureFlagMatrix({
    required this.version,
    required this.updatedAt,
    required this.flags,
    required this.remoteDefaults,
    required this.layout,
  });

  /// Monotonically increasing revision issued by the server. `0` means "never
  /// synced".
  final int version;

  /// Server-side modification timestamp; null when unknown.
  final DateTime? updatedAt;

  /// Explicit flag values.
  final Map<String, Object?> flags;

  /// Fallback values used when [flags] has no entry for a key.
  final Map<String, Object?> remoteDefaults;

  /// Dynamic layout document.
  final DynamicLayout layout;

  /// The state before any successful sync.
  static const FeatureFlagMatrix empty = FeatureFlagMatrix(
    version: 0,
    updatedAt: null,
    flags: <String, Object?>{},
    remoteDefaults: <String, Object?>{},
    layout: DynamicLayout.empty,
  );

  factory FeatureFlagMatrix.fromJson(Map<String, Object?> json) {
    final Object? rawFlags = json['flags'];
    if (rawFlags != null && rawFlags is! Map) {
      throw FlagParseException('"flags" must be a JSON object');
    }
    return FeatureFlagMatrix(
      version: _asInt(json['version']) ?? 0,
      updatedAt: _asDateTime(json['updated_at']),
      flags: _asStringObjectMap(rawFlags) ?? const <String, Object?>{},
      remoteDefaults:
          _asStringObjectMap(json['remote_defaults']) ?? const <String, Object?>{},
      layout: DynamicLayout.fromJson(_asStringObjectMap(json['layout'])),
    );
  }

  /// Parses a raw JSON string, throwing [FlagParseException] on bad input.
  static FeatureFlagMatrix parse(String body) {
    final Object? decoded;
    try {
      decoded = jsonDecode(body);
    } on FormatException catch (e) {
      throw FlagParseException('flag matrix is not valid JSON: ${e.message}');
    }
    final Map<String, Object?>? map = _asStringObjectMap(decoded);
    if (map == null) {
      throw FlagParseException('flag matrix must be a JSON object');
    }
    return FeatureFlagMatrix.fromJson(map);
  }

  Map<String, Object?> toJson() => <String, Object?>{
        'version': version,
        'updated_at': updatedAt?.toUtc().toIso8601String(),
        'flags': flags,
        'remote_defaults': remoteDefaults,
        'layout': layout.toJson(),
      };

  /// Every key known to this matrix.
  Set<String> get knownKeys => <String>{...flags.keys, ...remoteDefaults.keys};

  FeatureFlagMatrix copyWith({
    int? version,
    DateTime? updatedAt,
    Map<String, Object?>? flags,
    Map<String, Object?>? remoteDefaults,
    DynamicLayout? layout,
  }) =>
      FeatureFlagMatrix(
        version: version ?? this.version,
        updatedAt: updatedAt ?? this.updatedAt,
        flags: flags ?? this.flags,
        remoteDefaults: remoteDefaults ?? this.remoteDefaults,
        layout: layout ?? this.layout,
      );
}

/// Outcome of a [FeatureFlagProvider.refresh] call.
@immutable
class FlagSyncResult {
  const FlagSyncResult({
    required this.updated,
    required this.version,
    required this.source,
    this.error,
  });

  /// Whether the matrix actually changed.
  final bool updated;

  /// Version now in effect.
  final int version;

  /// Where the matrix came from: `network`, `cache` or `built-in`.
  final String source;

  /// Populated when the refresh failed but a previous matrix is still in use.
  final String? error;

  bool get ok => error == null;

  Map<String, Object?> toJson() => <String, Object?>{
        'updated': updated,
        'version': version,
        'source': source,
        'error': error,
      };
}

/// Reactive feature-flag store.
///
/// Register it with the widget tree (e.g. via an `InheritedNotifier` or a
/// `ListenableBuilder`) and the UI rebuilds whenever the server's matrix
/// changes, hiding and showing modules automatically.
class FeatureFlagProvider extends ChangeNotifier {
  FeatureFlagProvider({
    required this.baseUrl,
    http.Client? client,
    this.cacheKey = 'harbor.feature_flag_matrix',
    this.overridesKey = 'harbor.feature_flag_overrides',
    this.requestTimeout = const Duration(seconds: 10),
    FeatureFlagMatrix? seed,
  })  : _client = client ?? http.Client(),
        _ownsClient = client == null,
        _matrix = seed ?? FeatureFlagMatrix.empty,
        _seed = seed;

  /// Base URL of the update server, e.g. `https://updates.example.com`.
  final String baseUrl;

  final http.Client _client;
  final bool _ownsClient;
  final FeatureFlagMatrix? _seed;

  /// SharedPreferences key holding the last good matrix JSON.
  final String cacheKey;

  /// SharedPreferences key holding local overrides.
  final String overridesKey;

  /// Per-request timeout.
  final Duration requestTimeout;

  FeatureFlagMatrix _matrix;
  final Map<String, Object?> _overrides = <String, Object?>{};
  final StreamController<FeatureFlagMatrix> _changes =
      StreamController<FeatureFlagMatrix>.broadcast();

  SharedPreferences? _prefs;
  bool _initialised = false;
  bool _refreshing = false;
  bool _disposed = false;
  String? _lastError;
  String _source = 'built-in';
  DateTime? _lastSyncedAt;

  /// The matrix currently in effect.
  FeatureFlagMatrix get matrix => _matrix;

  /// Version of the matrix in effect.
  int get version => _matrix.version;

  /// Where the current matrix came from.
  String get source => _source;

  /// The last error encountered, if any.
  String? get lastError => _lastError;

  /// When the matrix was last successfully synced from the network.
  DateTime? get lastSyncedAt => _lastSyncedAt;

  /// Whether a refresh is in flight.
  bool get isRefreshing => _refreshing;

  /// Whether [initialize] has run.
  bool get isInitialised => _initialised;

  /// Local overrides currently applied.
  Map<String, Object?> get overrides => Map<String, Object?>.unmodifiable(_overrides);

  /// Broadcast stream of matrix changes.
  Stream<FeatureFlagMatrix> get changes => _changes.stream;

  /// The dynamic layout in effect.
  DynamicLayout get layout => _matrix.layout;

  /// Loads the cached matrix and persisted overrides. Safe to call twice.
  Future<void> initialize() async {
    if (_initialised) {
      return;
    }
    try {
      _prefs ??= await SharedPreferences.getInstance();
    } on Object catch (e) {
      // Persistence is best-effort: without it the app still works, it just
      // re-fetches on every launch.
      _lastError = 'preferences unavailable: $e';
    }
    final SharedPreferences? prefs = _prefs;
    if (prefs != null) {
      final String? cached = prefs.getString(cacheKey);
      if (cached != null && cached.isNotEmpty) {
        try {
          _matrix = FeatureFlagMatrix.parse(cached);
          _source = 'cache';
        } on FlagParseException catch (e) {
          _lastError = 'discarded corrupt cached matrix: ${e.message}';
          await prefs.remove(cacheKey);
        }
      }
      final String? rawOverrides = prefs.getString(overridesKey);
      if (rawOverrides != null && rawOverrides.isNotEmpty) {
        try {
          final Map<String, Object?>? decoded =
              _asStringObjectMap(jsonDecode(rawOverrides));
          if (decoded != null) {
            _overrides.addAll(decoded);
          }
        } on FormatException catch (e) {
          _lastError = 'discarded corrupt overrides: ${e.message}';
          await prefs.remove(overridesKey);
        }
      }
    }
    if (_matrix.version == 0 && _seed != null) {
      _matrix = _seed;
      _source = 'built-in';
    }
    _initialised = true;
    _notify();
  }

  /// Fetches `GET /api/v1/flags` and swaps in the new matrix.
  ///
  /// On failure the previous matrix is retained and the error is recorded in
  /// [lastError]; the returned result reports `ok: false` rather than throwing,
  /// so callers can degrade silently.
  Future<FlagSyncResult> refresh({bool force = false}) async {
    if (_refreshing) {
      return FlagSyncResult(
        updated: false,
        version: _matrix.version,
        source: _source,
        error: 'a refresh is already in progress',
      );
    }
    _refreshing = true;
    _notify();

    final Uri uri = Uri.parse('$baseUrl/api/v1/flags');
    try {
      final http.Response response =
          await _client.get(uri, headers: const <String, String>{
        'Accept': 'application/json',
      }).timeout(requestTimeout);

      if (response.statusCode != 200) {
        throw FlagFetchException(
          'update server rejected the flag request',
          statusCode: response.statusCode,
        );
      }

      final FeatureFlagMatrix fetched =
          FeatureFlagMatrix.parse(utf8.decode(response.bodyBytes));

      final bool changed =
          force || fetched.version != _matrix.version || _matrix.version == 0;
      if (changed) {
        _matrix = fetched;
        _source = 'network';
        _lastSyncedAt = DateTime.now().toUtc();
        await _persistMatrix(fetched);
      }
      _lastError = null;
      return FlagSyncResult(
        updated: changed,
        version: _matrix.version,
        source: _source,
      );
    } on FlagFetchException catch (e) {
      _lastError = e.toString();
      return FlagSyncResult(
        updated: false,
        version: _matrix.version,
        source: _source,
        error: _lastError,
      );
    } on FlagParseException catch (e) {
      _lastError = e.toString();
      return FlagSyncResult(
        updated: false,
        version: _matrix.version,
        source: _source,
        error: _lastError,
      );
    } on TimeoutException {
      _lastError = 'flag request timed out after '
          '${requestTimeout.inSeconds}s';
      return FlagSyncResult(
        updated: false,
        version: _matrix.version,
        source: _source,
        error: _lastError,
      );
    } on Object catch (e) {
      _lastError = 'flag request failed: $e';
      return FlagSyncResult(
        updated: false,
        version: _matrix.version,
        source: _source,
        error: _lastError,
      );
    } finally {
      _refreshing = false;
      _notify();
    }
  }

  Future<void> _persistMatrix(FeatureFlagMatrix matrix) async {
    final SharedPreferences? prefs = _prefs;
    if (prefs == null) {
      return;
    }
    try {
      await prefs.setString(cacheKey, jsonEncode(matrix.toJson()));
    } on Object catch (e) {
      _lastError = 'could not cache the flag matrix: $e';
    }
  }

  Future<void> _persistOverrides() async {
    final SharedPreferences? prefs = _prefs;
    if (prefs == null) {
      return;
    }
    try {
      if (_overrides.isEmpty) {
        await prefs.remove(overridesKey);
      } else {
        await prefs.setString(overridesKey, jsonEncode(_overrides));
      }
    } on Object catch (e) {
      _lastError = 'could not persist overrides: $e';
    }
  }

  /// Resolves the raw value for [key] through the override/flag/default chain.
  Object? rawValue(String key, {Object? defaultValue}) {
    if (_overrides.containsKey(key)) {
      return _overrides[key];
    }
    if (_matrix.flags.containsKey(key)) {
      return _matrix.flags[key];
    }
    if (_matrix.remoteDefaults.containsKey(key)) {
      return _matrix.remoteDefaults[key];
    }
    return defaultValue;
  }

  /// Whether [key] is currently forced by a local override.
  bool isOverridden(String key) => _overrides.containsKey(key);

  /// The JSON type [key] currently resolves to.
  FlagValueType typeOf(String key) =>
      FlagValueType.of(rawValue(key));

  /// Reads a boolean flag.
  bool getFlag(String key, {bool defaultValue = false}) {
    final Object? value = rawValue(key);
    final bool? coerced = _coerceBool(value);
    return coerced ?? defaultValue;
  }

  /// Reads a string flag.
  String? getString(String key, {String? defaultValue}) {
    final Object? value = rawValue(key);
    if (value == null) {
      return defaultValue;
    }
    return value is String ? value : value.toString();
  }

  /// Reads an integer flag.
  int getInt(String key, {int defaultValue = 0}) {
    final Object? value = rawValue(key);
    if (value is int) {
      return value;
    }
    if (value is num) {
      return value.toInt();
    }
    if (value is String) {
      return int.tryParse(value) ?? defaultValue;
    }
    return defaultValue;
  }

  /// Reads a double flag.
  double getDouble(String key, {double defaultValue = 0.0}) {
    final Object? value = rawValue(key);
    if (value is num) {
      return value.toDouble();
    }
    if (value is String) {
      return double.tryParse(value) ?? defaultValue;
    }
    return defaultValue;
  }

  /// Reads a list-of-strings flag.
  List<String> getStringList(String key, {List<String>? defaultValue}) {
    final Object? value = rawValue(key);
    if (value is List) {
      return value.map((Object? v) => v.toString()).toList(growable: false);
    }
    return defaultValue ?? const <String>[];
  }

  /// Reads a nested object flag.
  Map<String, Object?> getMap(String key) {
    final Object? value = rawValue(key);
    return _asStringObjectMap(value) ?? const <String, Object?>{};
  }

  /// Sets a local override and persists it.
  Future<void> setOverride(String key, Object? value) async {
    if (value == null) {
      _overrides.remove(key);
    } else {
      _overrides[key] = value;
    }
    await _persistOverrides();
    _notify();
  }

  /// Removes a single override.
  Future<void> clearOverride(String key) async {
    if (_overrides.remove(key) != null) {
      await _persistOverrides();
      _notify();
    }
  }

  /// Removes every override.
  Future<void> clearAllOverrides() async {
    if (_overrides.isEmpty) {
      return;
    }
    _overrides.clear();
    await _persistOverrides();
    _notify();
  }

  /// Whether a module gated by [module]'s flag should be visible.
  ///
  /// A module with no flag is always visible; a module whose flag is unknown
  /// falls back to `defaultWhenMissing` so a server typo cannot blank the app.
  bool isModuleEnabled(DynamicModule module, {bool defaultWhenMissing = true}) {
    if (module.flag.isEmpty) {
      return true;
    }
    return getFlag(module.flag, defaultValue: defaultWhenMissing);
  }

  /// Sections that pass their own gate, in layout order.
  List<DynamicSection> get visibleSections {
    final List<DynamicSection> out = <DynamicSection>[];
    for (final DynamicSection section in layout.sections) {
      if (section.flag.isNotEmpty && !getFlag(section.flag)) {
        continue;
      }
      final List<DynamicModule> modules = section.modules
          .where((DynamicModule m) => isModuleEnabled(m))
          .toList(growable: false);
      if (modules.isEmpty) {
        continue;
      }
      out.add(
        DynamicSection(
          id: section.id,
          title: section.title,
          modules: List<DynamicModule>.unmodifiable(modules),
          order: section.order,
          flag: section.flag,
        ),
      );
    }
    return List<DynamicSection>.unmodifiable(out);
  }

  /// Modules visible for [sectionId], or an empty list when unknown.
  List<DynamicModule> visibleModulesIn(String sectionId) {
    for (final DynamicSection section in visibleSections) {
      if (section.id == sectionId) {
        return section.modules;
      }
    }
    return const <DynamicModule>[];
  }

  void _notify() {
    if (!_disposed) {
      notifyListeners();
    }
  }

  @override
  void dispose() {
    _disposed = true;
    unawaited(_changes.close());
    if (_ownsClient) {
      _client.close();
    }
    super.dispose();
  }

  /// Diagnostic snapshot for the settings screen.
  Map<String, Object?> describe() => <String, Object?>{
        'base_url': baseUrl,
        'matrix_version': _matrix.version,
        'source': _source,
        'last_synced_at': _lastSyncedAt?.toIso8601String(),
        'last_error': _lastError,
        'flag_count': _matrix.flags.length,
        'remote_default_count': _matrix.remoteDefaults.length,
        'override_count': _overrides.length,
        'section_count': layout.sections.length,
        'module_count': layout.allModules.length,
        'visible_section_count': visibleSections.length,
        'known_keys': _matrix.knownKeys.toList(growable: false)..sort(),
        'overrides': _overrides,
      };
}

bool? _coerceBool(Object? value) {
  if (value is bool) {
    return value;
  }
  if (value is num) {
    return value != 0;
  }
  if (value is String) {
    switch (value.trim().toLowerCase()) {
      case 'true':
      case 'yes':
      case '1':
      case 'on':
        return true;
      case 'false':
      case 'no':
      case '0':
      case 'off':
        return false;
    }
  }
  return null;
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

DateTime? _asDateTime(Object? value) {
  if (value is DateTime) {
    return value;
  }
  if (value is String) {
    return DateTime.tryParse(value)?.toUtc();
  }
  if (value is int) {
    return DateTime.fromMillisecondsSinceEpoch(value, isUtc: true);
  }
  return null;
}

Map<String, Object?>? _asStringObjectMap(Object? value) {
  if (value is Map<String, Object?>) {
    return value;
  }
  if (value is Map) {
    return value.map<String, Object?>(
      (Object? k, Object? v) => MapEntry<String, Object?>(k.toString(), v),
    );
  }
  return null;
}