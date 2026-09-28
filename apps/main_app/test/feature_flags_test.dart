// Unit tests for the dynamic feature-flag layer
// (`lib/core/update_engine/dynamic_feature_flag_provider.dart`).
//
// AUTHORING NOTE: these tests could NOT be executed while they were written —
// the host is aarch64 and the Linux Flutter SDK is x86-64 only, so `flutter
// pub get` / `flutter test` cannot run here. They are written against the exact
// public surface of the source under test and must be run in CI.
//
// Hermetic: a fake `http.Client` is injected everywhere and `SharedPreferences`
// is faked in `setUp`.

import 'dart:async';
import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:main_app/core/update_engine/dynamic_feature_flag_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

const String _cacheKey = 'test.feature_flag_matrix';
const String _overridesKey = 'test.feature_flag_overrides';

FeatureFlagProvider _provider({
  required http.Client client,
  FeatureFlagMatrix? seed,
}) {
  final FeatureFlagProvider provider = FeatureFlagProvider(
    baseUrl: 'https://flags.test',
    client: client,
    cacheKey: _cacheKey,
    overridesKey: _overridesKey,
    seed: seed,
  );
  addTearDown(provider.dispose);
  return provider;
}

http.Client _deadClient() =>
    MockClient((http.Request request) async => http.Response('{}', 500));

String _matrixJson({
  int version = 1,
  Map<String, Object?> flags = const <String, Object?>{},
  Map<String, Object?> remoteDefaults = const <String, Object?>{},
  Map<String, Object?>? layout,
}) =>
    jsonEncode(<String, Object?>{
      'version': version,
      'updated_at': '2026-02-01T10:00:00.000Z',
      'flags': flags,
      'remote_defaults': remoteDefaults,
      if (layout != null) 'layout': layout,
    });

DynamicModule _module(String id, String flag, {int order = 0}) =>
    DynamicModule(id: id, type: 'banner', flag: flag, order: order);

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    SharedPreferences.setMockInitialValues(<String, Object>{});
  });

  // ---------------------------------------------------------------------------
  // 1. FeatureFlagMatrix.parse
  // ---------------------------------------------------------------------------

  group('FeatureFlagMatrix.parse', () {
    test('parses version, flags, remote defaults and layout', () {
      final FeatureFlagMatrix matrix = FeatureFlagMatrix.parse(_matrixJson(
        version: 7,
        flags: <String, Object?>{
          'dynamic_ui': true,
          'labs.voice_mode': false,
        },
        remoteDefaults: <String, Object?>{'dynamic_ui': true},
        layout: <String, Object?>{
          'sections': <Object?>[
            <String, Object?>{
              'id': 'home',
              'title': 'Home',
              'order': 0,
              'modules': <Object?>[
                <String, Object?>{
                  'id': 'thinking_panel',
                  'type': 'thinking_panel',
                  'flag': 'agent.thinking',
                  'order': 0,
                  'props': <String, Object?>{'collapsedByDefault': true},
                },
              ],
            },
          ],
        },
      ));

      expect(matrix.version, 7);
      expect(matrix.updatedAt, DateTime.utc(2026, 2, 1, 10));
      expect(matrix.flags['dynamic_ui'], isTrue);
      expect(matrix.flags['labs.voice_mode'], isFalse);
      expect(matrix.remoteDefaults['dynamic_ui'], isTrue);
      expect(matrix.layout.sections.length, 1);

      final DynamicSection home = matrix.layout.sections.single;
      expect(home.id, 'home');
      expect(home.title, 'Home');
      expect(home.modules.single.id, 'thinking_panel');
      expect(home.modules.single.type, 'thinking_panel');
      expect(home.modules.single.flag, 'agent.thinking');
      expect(home.modules.single.props['collapsedByDefault'], isTrue);
      expect(matrix.knownKeys,
          containsAll(<String>{'dynamic_ui', 'labs.voice_mode'}));
    });

    test('sorts layout sections and modules by order', () {
      final FeatureFlagMatrix matrix = FeatureFlagMatrix.parse(
        _matrixJson(
          layout: <String, Object?>{
            'sections': <Object?>[
              <String, Object?>{
                'id': 'second',
                'order': 2,
                'modules': <Object?>[
                  <String, Object?>{'id': 'late', 'type': 't', 'order': 5},
                  <String, Object?>{'id': 'early', 'type': 't', 'order': 1},
                ],
              },
              <String, Object?>{'id': 'first', 'order': 0, 'modules': <Object?>[]},
            ],
          },
        ),
      );

      expect(
        matrix.layout.sections.map((DynamicSection s) => s.id).toList(),
        <String>['first', 'second'],
      );
      expect(
        matrix.layout.sections[1].modules
            .map((DynamicModule m) => m.id)
            .toList(),
        <String>['early', 'late'],
      );
    });

    test('breaks module order ties by id', () {
      final DynamicSection section = DynamicSection.fromJson(const <String, Object?>{
        'id': 's',
        'modules': <Object?>[
          <String, Object?>{'id': 'b', 'type': 't', 'order': 1},
          <String, Object?>{'id': 'a', 'type': 't', 'order': 1},
        ],
      });

      expect(
        section.modules.map((DynamicModule m) => m.id).toList(),
        <String>['a', 'b'],
      );
    });

    test('invalid JSON throws FlagParseException', () {
      expect(
        () => FeatureFlagMatrix.parse('{not json'),
        throwsA(isA<FlagParseException>()),
      );
    });

    test('a non-object flags field throws FlagParseException', () {
      expect(
        () => FeatureFlagMatrix.parse('{"version":1,"flags":[]}'),
        throwsA(isA<FlagParseException>()),
      );
      expect(
        () => FeatureFlagMatrix.parse('{"version":1,"flags":"nope"}'),
        throwsA(isA<FlagParseException>()),
      );
    });

    test('a non-object top level throws FlagParseException', () {
      expect(
        () => FeatureFlagMatrix.parse('[]'),
        throwsA(isA<FlagParseException>()),
      );
    });
  });

  // ---------------------------------------------------------------------------
  // 2. DynamicModule
  // ---------------------------------------------------------------------------

  group('DynamicModule', () {
    test('missing id throws FlagParseException', () {
      expect(
        () => DynamicModule.fromJson(const <String, Object?>{'type': 't'}),
        throwsA(isA<FlagParseException>()),
      );
    });

    test('missing type throws FlagParseException', () {
      expect(
        () => DynamicModule.fromJson(const <String, Object?>{'id': 'm'}),
        throwsA(isA<FlagParseException>()),
      );
    });

    test('prop<T> reads typed values and coerces', () {
      const DynamicModule module = DynamicModule(
        id: 'm',
        type: 't',
        flag: '',
        title: 'Title',
        props: <String, Object?>{
          'boolTrue': 'true',
          'boolFalse': 'false',
          'boolReal': true,
          'intVal': 4,
          'numForInt': 7.9,
          'doubleVal': 2.5,
          'strVal': 'hello',
          'numForStr': 12,
        },
      );

      expect(module.prop<bool>('boolTrue', false), isTrue);
      expect(module.prop<bool>('boolFalse', true), isFalse);
      expect(module.prop<bool>('boolReal', false), isTrue);
      expect(module.prop<bool>('missing', true), isTrue);
      expect(module.prop<bool>('intVal', false), isFalse);
      expect(module.prop<int>('intVal', 0), 4);
      expect(module.prop<int>('numForInt', 0), 7);
      expect(module.prop<int>('strVal', 0), 0);
      expect(module.prop<int>('missing', 9), 9);
      expect(module.prop<double>('doubleVal', 0.0), 2.5);
      expect(module.prop<double>('intVal', 0.0), 4.0);
      expect(module.prop<double>('missing', 1.5), 1.5);
      expect(module.prop<String>('strVal', 'x'), 'hello');
      expect(module.prop<String>('numForStr', 'x'), '12');
      expect(module.prop<String>('missing', 'fallback'), 'fallback');
    });

    test('toJson round-trips every field', () {
      const DynamicModule module = DynamicModule(
        id: 'm',
        type: 'banner',
        flag: 'x',
        order: 3,
        title: 'Title',
        props: <String, Object?>{'message': 'hi', 'count': 2},
      );

      final Map<String, Object?> json = module.toJson();
      expect(json['id'], 'm');
      expect(json['type'], 'banner');
      expect(json['flag'], 'x');
      expect(json['order'], 3);
      expect(json['title'], 'Title');
      expect(json['props'], <String, Object?>{'message': 'hi', 'count': 2});

      final DynamicModule back = DynamicModule.fromJson(json);
      expect(back.id, module.id);
      expect(back.type, module.type);
      expect(back.flag, module.flag);
      expect(back.order, module.order);
      expect(back.title, module.title);
      expect(back.props['message'], 'hi');
      expect(back.props['count'], 2);
    });

    test('toJson omits a null title', () {
      final DynamicModule module = _module('m', '');
      expect(module.toJson().containsKey('title'), isFalse);
    });
  });

  // ---------------------------------------------------------------------------
  // 3. DynamicSection
  // ---------------------------------------------------------------------------

  group('DynamicSection', () {
    test('missing id throws FlagParseException', () {
      expect(
        () => DynamicSection.fromJson(const <String, Object?>{}),
        throwsA(isA<FlagParseException>()),
      );
    });

    test('a non-object module entry throws FlagParseException', () {
      expect(
        () => DynamicSection.fromJson(const <String, Object?>{
          'id': 's',
          'modules': <Object?>[42],
        }),
        throwsA(isA<FlagParseException>()),
      );
    });

    test('title defaults to the id and flag defaults to empty', () {
      final DynamicSection section = DynamicSection.fromJson(
          const <String, Object?>{'id': 's', 'modules': <Object?>[]});
      expect(section.title, 's');
      expect(section.flag, '');
      expect(section.order, 0);
      expect(section.modules, isEmpty);
    });
  });

  // ---------------------------------------------------------------------------
  // 4. FeatureFlagProvider resolution order
  // ---------------------------------------------------------------------------

  group('FeatureFlagProvider resolution', () {
    test('local override > remote flags > remote defaults > caller default',
        () async {
      final FeatureFlagProvider provider = _provider(
        client: MockClient((http.Request request) async => http.Response(
              _matrixJson(
                version: 3,
                flags: <String, Object?>{
                  'both': 'remote',
                  'remote.only': true,
                },
                remoteDefaults: <String, Object?>{
                  'both': 'default',
                  'default.only': 7,
                },
              ),
              200,
            )),
      );

      await provider.initialize();
      final FlagSyncResult result = await provider.refresh();
      expect(result.ok, isTrue);

      expect(provider.rawValue('both'), 'remote');
      expect(provider.rawValue('default.only'), 7);
      expect(provider.rawValue('remote.only'), isTrue);
      expect(provider.rawValue('nope', defaultValue: 42), 42);
      expect(provider.getFlag('remote.only'), isTrue);
      expect(provider.isOverridden('both'), isFalse);

      await provider.setOverride('both', 'local');
      expect(provider.rawValue('both'), 'local');
      expect(provider.isOverridden('both'), isTrue);
      expect(provider.overrides['both'], 'local');

      await provider.clearOverride('both');
      expect(provider.rawValue('both'), 'remote');
      expect(provider.isOverridden('both'), isFalse);

      await provider.setOverride('remote.only', false);
      expect(provider.getFlag('remote.only'), isFalse);

      await provider.clearAllOverrides();
      expect(provider.getFlag('remote.only'), isTrue);
      expect(provider.overrides, isEmpty);
    });

    test('typed getters read and coerce every JSON type', () async {
      final FeatureFlagProvider provider = _provider(
        client: MockClient((http.Request request) async => http.Response(
              _matrixJson(
                version: 1,
                flags: <String, Object?>{
                  'b.true': 'true',
                  'b.false': 'false',
                  'b.yes': 'yes',
                  'b.off': 'off',
                  'b.one': 1,
                  'b.zero': 0,
                  'b.real': true,
                  'i.int': 3,
                  'i.double': 4.9,
                  'i.str': '7',
                  'd.num': 2.5,
                  'd.str': '3.25',
                  'd.int': 4,
                  's.str': 'hi',
                  's.num': 9,
                  'l.list': <Object?>['a', 1, true],
                  'm.map': <String, Object?>{'k': 'v'},
                },
              ),
              200,
            )),
      );

      await provider.initialize();
      await provider.refresh();

      expect(provider.getFlag('b.true'), isTrue);
      expect(provider.getFlag('b.false'), isFalse);
      expect(provider.getFlag('b.yes'), isTrue);
      expect(provider.getFlag('b.off'), isFalse);
      expect(provider.getFlag('b.one'), isTrue);
      expect(provider.getFlag('b.zero'), isFalse);
      expect(provider.getFlag('b.real'), isTrue);
      expect(provider.getFlag('missing.bool', defaultValue: true), isTrue);

      expect(provider.getInt('i.int'), 3);
      expect(provider.getInt('i.double'), 4);
      expect(provider.getInt('i.str'), 7);
      expect(provider.getInt('missing.int', defaultValue: -1), -1);

      expect(provider.getDouble('d.num'), 2.5);
      expect(provider.getDouble('d.str'), 3.25);
      expect(provider.getDouble('d.int'), 4.0);
      expect(provider.getDouble('missing.double', defaultValue: 1.5), 1.5);

      expect(provider.getString('s.str'), 'hi');
      expect(provider.getString('s.num'), '9');
      expect(provider.getString('missing.str', defaultValue: 'fb'), 'fb');
      expect(provider.getString('missing.str'), isNull);

      expect(provider.getStringList('l.list'), <String>['a', '1', 'true']);
      expect(provider.getStringList('missing.list'), isEmpty);
      expect(
        provider.getStringList('missing.list', defaultValue: <String>['x']),
        <String>['x'],
      );

      expect(provider.getMap('m.map'), <String, Object?>{'k': 'v'});
      expect(provider.getMap('missing.map'), isEmpty);
    });

    test('typeOf classifies every JSON type', () async {
      final FeatureFlagProvider provider = _provider(
        client: MockClient((http.Request request) async => http.Response(
              _matrixJson(
                version: 1,
                flags: <String, Object?>{
                  't.bool': true,
                  't.int': 1,
                  't.double': 1.5,
                  't.string': 's',
                  't.list': <Object?>[1],
                  't.map': <String, Object?>{'a': 1},
                },
              ),
              200,
            )),
      );
      await provider.initialize();
      await provider.refresh();

      expect(provider.typeOf('t.bool'), FlagValueType.boolean);
      expect(provider.typeOf('t.int'), FlagValueType.integer);
      expect(provider.typeOf('t.double'), FlagValueType.doubleValue);
      expect(provider.typeOf('t.string'), FlagValueType.string);
      expect(provider.typeOf('t.list'), FlagValueType.list);
      expect(provider.typeOf('t.map'), FlagValueType.map);
      expect(provider.typeOf('t.missing'), FlagValueType.nullValue);
      expect(FlagValueType.boolean.label, 'boolean');
      expect(FlagValueType.doubleValue.label, 'number');
      expect(FlagValueType.map.label, 'object');
    });
  });

  // ---------------------------------------------------------------------------
  // 5. refresh()
  // ---------------------------------------------------------------------------

  group('refresh', () {
    test('a higher version updates, becomes network source and is cached',
        () async {
      final FeatureFlagProvider provider = _provider(
        client: MockClient((http.Request request) async => http.Response(
              _matrixJson(
                version: 7,
                flags: <String, Object?>{'a': true},
              ),
              200,
            )),
      );
      await provider.initialize();

      final FlagSyncResult result = await provider.refresh();

      expect(result.updated, isTrue);
      expect(result.ok, isTrue);
      expect(result.source, 'network');
      expect(result.version, 7);
      expect(provider.version, 7);
      expect(provider.source, 'network');
      expect(provider.lastError, isNull);
      expect(provider.lastSyncedAt, isNotNull);

      final SharedPreferences prefs = await SharedPreferences.getInstance();
      final String? cached = prefs.getString(_cacheKey);
      expect(cached, isNotNull);
      final FeatureFlagMatrix restored = FeatureFlagMatrix.parse(cached!);
      expect(restored.version, 7);
      expect(restored.flags['a'], isTrue);
    });

    test('a 500 keeps the previous matrix and records an error', () async {
      int status = 200;
      String body = _matrixJson(
        version: 5,
        flags: <String, Object?>{'a': true},
      );
      final FeatureFlagProvider provider = _provider(
        client: MockClient((http.Request request) async =>
            http.Response(status == 200 ? body : 'boom', status)),
      );
      await provider.initialize();
      await provider.refresh();
      expect(provider.version, 5);
      expect(provider.source, 'network');

      status = 500;
      final FlagSyncResult result = await provider.refresh();

      expect(result.ok, isFalse);
      expect(result.updated, isFalse);
      expect(result.error, contains('500'));
      expect(provider.version, 5);
      expect(provider.source, 'network');
      expect(provider.lastError, isNotNull);
      expect(provider.getFlag('a'), isTrue);
    });

    test('a malformed body keeps the previous matrix', () async {
      String body = _matrixJson(version: 5);
      final FeatureFlagProvider provider = _provider(
        client: MockClient(
            (http.Request request) async => http.Response(body, 200)),
      );
      await provider.initialize();
      await provider.refresh();

      body = '{not json';
      final FlagSyncResult result = await provider.refresh();

      expect(result.ok, isFalse);
      expect(result.error, isNotNull);
      expect(provider.version, 5);
      expect(provider.source, 'network');
    });

    test('a concurrent refresh reports an in-progress refresh', () async {
      final Completer<http.Response> completer = Completer<http.Response>();
      final FeatureFlagProvider provider = _provider(
        client: MockClient((http.Request request) => completer.future),
      );
      await provider.initialize();

      final Future<FlagSyncResult> first = provider.refresh();
      expect(provider.isRefreshing, isTrue);

      final FlagSyncResult second = await provider.refresh();
      expect(second.ok, isFalse);
      expect(second.error, contains('in progress'));

      completer.complete(
          http.Response(_matrixJson(version: 2), 200));
      final FlagSyncResult firstResult = await first;
      expect(firstResult.updated, isTrue);
      expect(firstResult.version, 2);
      expect(provider.isRefreshing, isFalse);
    });
  });

  // ---------------------------------------------------------------------------
  // 6. initialize()
  // ---------------------------------------------------------------------------

  group('initialize', () {
    test('restores the cached matrix and persisted overrides', () async {
      final FeatureFlagMatrix cached = FeatureFlagMatrix(
        version: 9,
        updatedAt: DateTime.utc(2026, 1, 1),
        flags: const <String, Object?>{'flag.x': false},
        remoteDefaults: const <String, Object?>{},
        layout: DynamicLayout.empty,
      );
      SharedPreferences.setMockInitialValues(<String, Object>{
        _cacheKey: jsonEncode(cached.toJson()),
        _overridesKey: jsonEncode(<String, Object?>{'flag.x': true}),
      });

      final FeatureFlagProvider provider = _provider(client: _deadClient());
      await provider.initialize();

      expect(provider.isInitialised, isTrue);
      expect(provider.source, 'cache');
      expect(provider.version, 9);
      expect(provider.isOverridden('flag.x'), isTrue);
      expect(provider.rawValue('flag.x'), isTrue);
      expect(provider.getFlag('flag.x'), isTrue);
    });

    test('a corrupt cached matrix is discarded without throwing', () async {
      SharedPreferences.setMockInitialValues(<String, Object>{
        _cacheKey: '{not json',
      });

      final FeatureFlagProvider provider = _provider(client: _deadClient());
      await provider.initialize();

      expect(provider.isInitialised, isTrue);
      expect(provider.lastError, contains('discarded corrupt cached matrix'));
      final SharedPreferences prefs = await SharedPreferences.getInstance();
      expect(prefs.getString(_cacheKey), isNull);
    });

    test('a corrupt overrides blob is discarded without throwing', () async {
      SharedPreferences.setMockInitialValues(<String, Object>{
        _overridesKey: 'not json',
      });

      final FeatureFlagProvider provider = _provider(client: _deadClient());
      await provider.initialize();

      expect(provider.isInitialised, isTrue);
      expect(provider.lastError, contains('discarded corrupt overrides'));
      expect(provider.overrides, isEmpty);
    });

    test('initialize is idempotent', () async {
      SharedPreferences.setMockInitialValues(<String, Object>{
        _cacheKey: jsonEncode(const FeatureFlagMatrix(
          version: 4,
          updatedAt: null,
          flags: <String, Object?>{'a': true},
          remoteDefaults: <String, Object?>{},
          layout: DynamicLayout.empty,
        ).toJson()),
      });

      final FeatureFlagProvider provider = _provider(client: _deadClient());
      await provider.initialize();
      final int version = provider.version;
      final String source = provider.source;

      await provider.initialize();

      expect(provider.version, version);
      expect(provider.source, source);
      expect(provider.version, 4);
      expect(provider.source, 'cache');
    });

    test('a seed is used when there is no cache', () async {
      const FeatureFlagMatrix seed = FeatureFlagMatrix(
        version: 0,
        updatedAt: null,
        flags: <String, Object?>{'seeded': true},
        remoteDefaults: <String, Object?>{},
        layout: DynamicLayout.empty,
      );
      final FeatureFlagProvider provider =
          _provider(client: _deadClient(), seed: seed);

      await provider.initialize();

      expect(provider.getFlag('seeded'), isTrue);
      expect(provider.source, 'built-in');
    });
  });

  // ---------------------------------------------------------------------------
  // 7. UI gating
  // ---------------------------------------------------------------------------

  group('UI gating', () {
    FeatureFlagProvider gatedProvider() {
      final DynamicModule always = _module('m1', '');
      final DynamicModule on = _module('m2', 'on');
      final DynamicModule off = _module('m3', 'off');
      final DynamicModule unknown = _module('m4', 'unknown.flag');
      final DynamicModule inHiddenSection = _module('m5', '');

      final FeatureFlagMatrix matrix = FeatureFlagMatrix(
        version: 1,
        updatedAt: null,
        flags: const <String, Object?>{'on': true, 'off': false, 'sec.off': false},
        remoteDefaults: const <String, Object?>{},
        layout: DynamicLayout(
          sections: <DynamicSection>[
            DynamicSection(
              id: 's1',
              title: 'S1',
              order: 0,
              modules: <DynamicModule>[always, off],
            ),
            DynamicSection(
              id: 's2',
              title: 'S2',
              order: 1,
              flag: 'sec.off',
              modules: <DynamicModule>[inHiddenSection],
            ),
            DynamicSection(
              id: 's3',
              title: 'S3',
              order: 2,
              modules: <DynamicModule>[on],
            ),
            DynamicSection(
              id: 's4',
              title: 'S4',
              order: 3,
              modules: <DynamicModule>[off],
            ),
            DynamicSection(
              id: 's5',
              title: 'S5',
              order: 4,
              modules: <DynamicModule>[unknown],
            ),
          ],
        ),
      );

      return _provider(client: _deadClient(), seed: matrix);
    }

    test('isModuleEnabled honours flags and defaultWhenMissing', () {
      final FeatureFlagProvider provider = gatedProvider();

      expect(provider.isModuleEnabled(_module('a', '')), isTrue);
      expect(provider.isModuleEnabled(_module('b', 'on')), isTrue);
      expect(provider.isModuleEnabled(_module('c', 'off')), isFalse);
      expect(provider.isModuleEnabled(_module('d', 'unknown.flag')), isTrue);
      expect(
        provider.isModuleEnabled(_module('e', 'unknown.flag'),
            defaultWhenMissing: false),
        isFalse,
      );
    });

    test('visibleSections filters modules, empty sections and section flags',
        () {
      final FeatureFlagProvider provider = gatedProvider();
      final List<DynamicSection> visible = provider.visibleSections;

      expect(
        visible.map((DynamicSection s) => s.id).toList(),
        <String>['s1', 's3', 's5'],
      );
      expect(
        visible.first.modules.map((DynamicModule m) => m.id).toList(),
        <String>['m1'],
      );
      expect(visible[1].modules.single.id, 'm2');
      expect(visible[2].modules.single.id, 'm4');
    });

    test('visibleModulesIn returns the right list and empty for unknown', () {
      final FeatureFlagProvider provider = gatedProvider();

      expect(
        provider
            .visibleModulesIn('s1')
            .map((DynamicModule m) => m.id)
            .toList(),
        <String>['m1'],
      );
      expect(
        provider.visibleModulesIn('s3').single.id,
        'm2',
      );
      expect(provider.visibleModulesIn('s4'), isEmpty);
      expect(provider.visibleModulesIn('s2'), isEmpty);
      expect(provider.visibleModulesIn('no.such.section'), isEmpty);
    });
  });

  // ---------------------------------------------------------------------------
  // 8. describe() and changes stream
  // ---------------------------------------------------------------------------

  group('describe and changes', () {
    test('describe exposes the expected diagnostic keys', () async {
      final FeatureFlagProvider provider = _provider(
        client: MockClient((http.Request request) async =>
            http.Response(_matrixJson(version: 1), 200)),
      );
      await provider.initialize();
      // initialize() only restores the cache, overrides and seed by design; the
      // network snapshot arrives from refresh(), which is what the app calls
      // next at startup (see HarborServices in main.dart).
      await provider.refresh();

      final Map<String, Object?> described = provider.describe();
      expect(
        described.keys,
        containsAll(<String>[
          'base_url',
          'matrix_version',
          'source',
          'last_synced_at',
          'last_error',
          'flag_count',
          'remote_default_count',
          'override_count',
          'section_count',
          'module_count',
          'visible_section_count',
          'known_keys',
          'overrides',
        ]),
      );
      expect(described['base_url'], 'https://flags.test');
      expect(described['matrix_version'], 1);
      expect(described['source'], 'network');
    });

    test('changes emits the new matrix after a successful refresh', () async {
      final FeatureFlagProvider provider = _provider(
        client: MockClient((http.Request request) async =>
            http.Response(_matrixJson(version: 6), 200)),
      );
      await provider.initialize();

      final List<FeatureFlagMatrix> events = <FeatureFlagMatrix>[];
      final StreamSubscription<FeatureFlagMatrix> subscription =
          provider.changes.listen(events.add);
      addTearDown(subscription.cancel);

      final FlagSyncResult result = await provider.refresh();
      await pumpEventQueue();

      expect(result.updated, isTrue);
      expect(events, isNotEmpty);
      expect(events.last.version, 6);
    });
  });

  // ---------------------------------------------------------------------------
  // 9. FlagValueType.of
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // 10. Static Pages fallback
  // ---------------------------------------------------------------------------

  group('static fallback', () {
    test('falls back to /api/v1/flags.json when the live endpoint 404s',
        () async {
      final List<String> requested = <String>[];
      final FeatureFlagProvider provider = _provider(
        client: MockClient((http.Request request) async {
          requested.add(request.url.path);
          if (request.url.path == '/api/v1/flags') {
            return http.Response('not found', 404);
          }
          return http.Response(
            _matrixJson(
              version: 7,
              flags: <String, Object?>{'labs': true},
              layout: <String, Object?>{
                'sections': <Object?>[
                  <String, Object?>{
                    'id': 'labs',
                    'title': 'Labs',
                    'order': 0,
                    'modules': <Object?>[
                      <String, Object?>{
                        'id': 'lora_lab',
                        'type': 'lora_lab',
                        'flag': 'labs',
                      },
                    ],
                  },
                ],
              },
            ),
            200,
            headers: <String, String>{'content-type': 'application/json'},
          );
        }),
      );

      final FlagSyncResult result = await provider.refresh();

      expect(result.ok, isTrue);
      expect(result.updated, isTrue);
      expect(result.source, 'network');
      expect(provider.version, 7);
      expect(provider.matrix.flags['labs'], isTrue);
      expect(requested, <String>['/api/v1/flags', '/api/v1/flags.json']);
    });

    test('does not fall back on a non-404 error', () async {
      final List<String> requested = <String>[];
      final FeatureFlagProvider provider = _provider(
        client: MockClient((http.Request request) async {
          requested.add(request.url.path);
          return http.Response('boom', 500);
        }),
      );

      final FlagSyncResult result = await provider.refresh();

      expect(result.ok, isFalse);
      expect(requested, <String>['/api/v1/flags']);
    });

    test('keeps the built-in seed when both endpoints fail', () async {
      const FeatureFlagMatrix seed = FeatureFlagMatrix(
        version: 1,
        updatedAt: null,
        flags: <String, Object?>{'labs': false},
        remoteDefaults: <String, Object?>{},
        layout: DynamicLayout(sections: <DynamicSection>[]),
      );
      final FeatureFlagProvider provider = _provider(
        client: MockClient((http.Request request) async =>
            http.Response('offline', 404)),
        seed: seed,
      );

      await provider.initialize();
      final FlagSyncResult result = await provider.refresh();

      expect(result.ok, isFalse);
      expect(result.source, 'built-in');
      expect(provider.version, 1);
      expect(provider.matrix.flags['labs'], isFalse);
      expect(provider.lastError, isNotNull);
    });
  });

  group('FlagValueType.of', () {
    test('classifies every JSON type including null', () {
      expect(FlagValueType.of(null), FlagValueType.nullValue);
      expect(FlagValueType.of(true), FlagValueType.boolean);
      expect(FlagValueType.of(1), FlagValueType.integer);
      expect(FlagValueType.of(1.5), FlagValueType.doubleValue);
      expect(FlagValueType.of('s'), FlagValueType.string);
      expect(FlagValueType.of(<Object?>[]), FlagValueType.list);
      expect(
          FlagValueType.of(<String, Object?>{}), FlagValueType.map);
      expect(FlagValueType.of(Object()), FlagValueType.nullValue);
    });
  });
}