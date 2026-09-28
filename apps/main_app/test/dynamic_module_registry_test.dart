// Guards the built-in feature-flag matrix that ships inside the APK.
//
// `kDefaultFlagMatrix` is what the app renders before (or instead of) a
// successful network sync, so a mistake in it is invisible to every other test
// but immediately obvious to a user: an unregistered module `type` renders the
// "unknown module" placeholder card, and a module gated on an undeclared flag
// is silently dropped. Both are checked here.
//
// The same document is published to GitHub Pages as /api/v1/flags.json (see
// apps/update_server/flags/default_flags.json), and the server suite validates
// that copy; this suite validates the compiled-in copy and its agreement with
// the registry.

import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:main_app/core/update_engine/dynamic_feature_flag_provider.dart';
import 'package:main_app/features/dynamic_module_registry.dart';
import 'package:main_app/main.dart';

/// Every module in the built-in matrix, with its section id for diagnostics.
List<(String, DynamicModule)> _modules() {
  final List<(String, DynamicModule)> out = <(String, DynamicModule)>[];
  for (final DynamicSection section in kDefaultFlagMatrix.layout.sections) {
    for (final DynamicModule module in section.modules) {
      out.add((section.id, module));
    }
  }
  return out;
}

/// A provider seeded with the shipped defaults and no working network, which is
/// exactly the state the app is in before the first successful sync.
FeatureFlagProvider _providerWithDefaults() {
  final FeatureFlagProvider provider = FeatureFlagProvider(
    baseUrl: 'https://flags.test',
    client: MockClient(
      (http.Request request) async => http.Response('offline', 404),
    ),
    seed: kDefaultFlagMatrix,
  );
  addTearDown(provider.dispose);
  return provider;
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('kDefaultFlagMatrix', () {
    test('is non-empty and reachable from the registry', () {
      expect(kDefaultFlagMatrix.version, greaterThan(0));
      expect(kDefaultFlagMatrix.layout.sections, isNotEmpty);
      expect(_modules(), isNotEmpty);
    });

    test('declares every flag its own layout gates on', () {
      final Set<String> declared = <String>{
        ...kDefaultFlagMatrix.flags.keys,
        ...kDefaultFlagMatrix.remoteDefaults.keys,
      };
      for (final (String sectionId, DynamicModule module) in _modules()) {
        if (module.flag.isEmpty) {
          continue;
        }
        expect(
          declared,
          contains(module.flag),
          reason: 'module "${module.id}" in section "$sectionId" gates on an '
              'undeclared flag "${module.flag}"',
        );
      }
      for (final DynamicSection section in kDefaultFlagMatrix.layout.sections) {
        if (section.flag.isEmpty) {
          continue;
        }
        expect(
          declared,
          contains(section.flag),
          reason: 'section "${section.id}" gates on an undeclared flag',
        );
      }
    });

    test('every module type has a registered renderer', () {
      final DynamicModuleRegistry registry = DynamicModuleRegistry();
      for (final (String sectionId, DynamicModule module) in _modules()) {
        expect(
          registry.supports(module.type),
          isTrue,
          reason: 'module "${module.id}" in section "$sectionId" uses type '
              '"${module.type}", which is not in '
              '${registry.supportedTypes.toList()..sort()}',
        );
      }
    });

    test('module and section ids are unique', () {
      final Set<String> sectionIds = <String>{};
      for (final DynamicSection section in kDefaultFlagMatrix.layout.sections) {
        expect(sectionIds.add(section.id), isTrue,
            reason: 'duplicate section id "${section.id}"');
        final Set<String> moduleIds = <String>{};
        for (final DynamicModule module in section.modules) {
          expect(moduleIds.add(module.id), isTrue,
              reason: 'duplicate module id "${module.id}"');
        }
      }
    });

    test('renders at least one section with the shipped defaults', () {
      // A visible section proves the shipped defaults do not gate the whole UI
      // off; a blank first launch would look like a broken app.
      final FeatureFlagProvider provider = _providerWithDefaults();
      expect(provider.matrix.version, kDefaultFlagMatrix.version);
      expect(provider.source, 'built-in');
      final List<DynamicSection> visible = provider.visibleSections;
      expect(visible, isNotEmpty);
      expect(visible.expand((DynamicSection s) => s.modules), isNotEmpty);
    });

    test('hidden-by-default labs module is the only gated-off module', () {
      // labs.voice_mode is false in the shipped defaults, so the Labs section
      // must disappear entirely rather than render an empty header.
      final List<String> visibleIds = _providerWithDefaults()
          .visibleSections
          .expand((DynamicSection s) => s.modules)
          .map((DynamicModule m) => m.id)
          .toList();
      expect(visibleIds, isNot(contains('labs_announcement')));
      expect(visibleIds, contains('engine_status'));
      expect(visibleIds, contains('thinking_panel'));
    });

    test('survives the real serializer round trip', () {
      final FeatureFlagMatrix parsed =
          FeatureFlagMatrix.parse(jsonEncode(kDefaultFlagMatrix.toJson()));

      expect(parsed.version, kDefaultFlagMatrix.version);
      expect(parsed.flags, kDefaultFlagMatrix.flags);
      expect(parsed.remoteDefaults, kDefaultFlagMatrix.remoteDefaults);

      // The *unfiltered* layout must round-trip exactly, including the props a
      // renderer reads (e.g. thinking_panel's collapsedByDefault).
      String describeModules(FeatureFlagMatrix matrix) => matrix.layout.sections
          .map((DynamicSection s) => '${s.id}[${s.flag}]'
              '${s.modules.map((DynamicModule m) => '${m.id}/${m.type}/'
                  '${m.flag}/${m.order}/${jsonEncode(m.props)}').join(',')}')
          .join('|');
      expect(describeModules(parsed), describeModules(kDefaultFlagMatrix));

      // And the *gated* view must agree, which is what the user actually sees.
      final FeatureFlagProvider fromJson = FeatureFlagProvider(
        baseUrl: 'https://flags.test',
        client: MockClient(
          (http.Request request) async => http.Response('offline', 404),
        ),
        seed: parsed,
      );
      addTearDown(fromJson.dispose);
      List<String> visibleIds(FeatureFlagProvider provider) => provider
          .visibleSections
          .expand((DynamicSection s) => s.modules)
          .map((DynamicModule m) => m.id)
          .toList();
      expect(visibleIds(fromJson), visibleIds(_providerWithDefaults()));
    });
  });
}
