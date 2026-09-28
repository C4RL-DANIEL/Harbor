// Widget tests for the OTA update UI and the dynamic module registry.
//
// AUTHORING NOTE: these tests could NOT be executed while they were written —
// the host is aarch64 and the Linux Flutter SDK is x86-64 only, so `flutter
// test` cannot run here. They are written against the exact public surface of
// the source under test and must be run in CI.
//
// Hermetic: the native `ota_update` plugin is never invoked. `UpdateService` is
// subclassed with a hand-written fake that overrides `performUpdate`, so no
// platform channel or network call is ever made.

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:main_app/core/agent/extended_thinking.dart';
import 'package:main_app/core/agent/file_readers.dart';
import 'package:main_app/core/agent/subagent_runner.dart';
import 'package:main_app/core/update_engine/dynamic_feature_flag_provider.dart';
import 'package:main_app/core/update_engine/update_service.dart';
import 'package:main_app/core/update_engine/widgets/update_dialog.dart';
import 'package:main_app/features/dynamic_module_registry.dart';
import 'package:main_app/main.dart' as app;

/// Signature of the overridable work in [UpdateService].
typedef _PerformUpdate = Future<Stream<InstallProgress>> Function(
  ReleaseInfo release,
  void Function(DownloadProgress progress)? onDownloadProgress,
);

/// A hand-written fake: no mocking framework, no plugin, no network.
class _FakeUpdateService extends UpdateService {
  _FakeUpdateService({
    _PerformUpdate? onPerform,
    // Kept as a super parameter rather than dropped: the fake must stay able
    // to inject a non-default strategy. The default is inherited from
    // UpdateService (InstallStrategy.nativeStreaming).
    super.strategy,
  })  : _onPerform = onPerform,
        super(
          baseUrl: 'https://updates.test',
          currentVersion: '1.0.0',
          client: MockClient(
              (http.Request request) async => http.Response('{}', 500)),
        );

  final _PerformUpdate? _onPerform;

  @override
  Future<Stream<InstallProgress>> performUpdate({
    required ReleaseInfo release,
    void Function(DownloadProgress progress)? onDownloadProgress,
  }) {
    final _PerformUpdate? fn = _onPerform;
    if (fn != null) {
      return fn(release, onDownloadProgress);
    }
    return Future<Stream<InstallProgress>>.value(
        const Stream<InstallProgress>.empty());
  }
}

/// A controller whose presentation state is set directly, for the progress
/// surface (the real controller only reaches `downloading` through the fake
/// service stream).
class _StubFlowController extends UpdateFlowController {
  _StubFlowController({
    required super.service,
    required UpdateFlowStage stage,
    DownloadProgress? progress,
  })  : stubStage = stage,
        stubProgress = progress;

  final UpdateFlowStage stubStage;
  final DownloadProgress? stubProgress;

  @override
  UpdateFlowStage get stage => stubStage;

  @override
  DownloadProgress? get progress => stubProgress;

  @override
  double? get fraction => stubProgress?.fraction;
}

ReleaseInfo _release({
  String latest = '1.2.0',
  String min = '1.0.0',
  String url = 'https://cdn.test/app-release.apk',
  String sha = '',
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

FeatureFlagProvider _flagProvider({FeatureFlagMatrix? seed}) {
  final FeatureFlagProvider provider = FeatureFlagProvider(
    baseUrl: 'https://flags.test',
    client:
        MockClient((http.Request request) async => http.Response('{}', 500)),
    seed: seed,
  );
  addTearDown(provider.dispose);
  return provider;
}

DynamicModuleContext _context(FeatureFlagProvider flags) {
  final _FakeUpdateService service = _FakeUpdateService();
  addTearDown(service.dispose);
  return DynamicModuleContext(
    flags: flags,
    reader: UniversalFileReader(),
    agent: ExtendedThinkingAgent(runner: SubagentRunner()),
    updateService: service,
    currentUpdateResult: null,
  );
}

void main() {
  // ---------------------------------------------------------------------------
  // 1. ForceUpdateDialog
  // ---------------------------------------------------------------------------

  group('ForceUpdateDialog', () {
    testWidgets('renders title, version chips, changelog and update button',
        (WidgetTester tester) async {
      final _FakeUpdateService service = _FakeUpdateService();
      addTearDown(service.dispose);
      final UpdateFlowController controller =
          UpdateFlowController(service: service);
      addTearDown(controller.dispose);

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: ForceUpdateDialog(
            release: _release(changelog: '• Fixed a crash\nFaster startup'),
            installedVersion: '1.0.0',
            controller: controller,
          ),
        ),
      ));

      expect(find.text('Update required'), findsOneWidget);
      expect(find.text('Installed'), findsOneWidget);
      expect(find.text('Available'), findsOneWidget);
      expect(find.text('1.0.0'), findsOneWidget);
      expect(find.text('1.2.0'), findsOneWidget);
      expect(find.text("What's new"), findsOneWidget);
      expect(find.text('• Fixed a crash'), findsOneWidget);
      expect(find.text('• Faster startup'), findsOneWidget);
      expect(find.text('Update now'), findsOneWidget);
      expect(find.byIcon(Icons.system_update_alt), findsOneWidget);
    });

    testWidgets('cannot be popped once shown', (WidgetTester tester) async {
      final _FakeUpdateService service = _FakeUpdateService();
      addTearDown(service.dispose);
      final UpdateFlowController controller =
          UpdateFlowController(service: service);
      addTearDown(controller.dispose);

      final GlobalKey<NavigatorState> navKey = GlobalKey<NavigatorState>();
      await tester.pumpWidget(MaterialApp(
        navigatorKey: navKey,
        home: const Scaffold(body: SizedBox.shrink()),
      ));

      unawaited(ForceUpdateDialog.show(
        navKey.currentContext!,
        release: _release(),
        installedVersion: '1.0.0',
        controller: controller,
      ));
      await tester.pumpAndSettle();
      expect(find.text('Update required'), findsOneWidget);

      // Navigator.maybePop returns whether the pop request was *handled*, not
      // whether the route was popped: a PopScope that vetoes the pop yields
      // RoutePopDisposition.doNotPop, which returns true "but does not do
      // anything beyond that" (see the maybePop docs in navigator.dart). So the
      // meaningful assertion is that the dialog survived, not the return value.
      final bool handled = await navKey.currentState!.maybePop();
      await tester.pumpAndSettle();

      expect(handled, isTrue);
      expect(find.text('Update required'), findsOneWidget);
    });

    testWidgets('a failed stage shows the error and Retry update',
        (WidgetTester tester) async {
      final _FakeUpdateService service = _FakeUpdateService(
        onPerform: (ReleaseInfo release,
                void Function(DownloadProgress progress)? onProgress) =>
            Future<Stream<InstallProgress>>.error(
                ReleaseMetadataException('disk full')),
      );
      addTearDown(service.dispose);
      final UpdateFlowController controller =
          UpdateFlowController(service: service);
      addTearDown(controller.dispose);

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: ForceUpdateDialog(
            release: _release(),
            installedVersion: '1.0.0',
            controller: controller,
          ),
        ),
      ));

      final bool started = await controller.start(_release());
      await tester.pump();

      expect(started, isFalse);
      expect(controller.stage, UpdateFlowStage.failed);
      expect(find.text('Update failed'), findsOneWidget);
      expect(find.text('disk full'), findsOneWidget);
      expect(find.text('Retry update'), findsOneWidget);
      expect(find.byIcon(Icons.refresh), findsOneWidget);
    });

    testWidgets('verifyThenInstall routes the flow through the verifying stage',
        (WidgetTester tester) async {
      final _FakeUpdateService service = _FakeUpdateService(
        strategy: InstallStrategy.verifyThenInstall,
        onPerform: (ReleaseInfo release,
                void Function(DownloadProgress progress)? onProgress) =>
            Future<Stream<InstallProgress>>.value(
                const Stream<InstallProgress>.empty()),
      );
      addTearDown(service.dispose);
      final UpdateFlowController controller =
          UpdateFlowController(service: service);
      addTearDown(controller.dispose);

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: ForceUpdateDialog(
            release: _release(),
            installedVersion: '1.0.0',
            controller: controller,
          ),
        ),
      ));

      // The empty stream ends immediately, so no install is ever reported and
      // start() returns false. The point of the test is the stage the
      // controller sits in: UpdateFlowController.start branches on
      // service.strategy, and verifyThenInstall is the only strategy that
      // routes the flow through `verifying` instead of jumping straight from
      // downloading to the terminal idle end state.
      final bool started = await controller.start(_release());
      await tester.pump();

      expect(started, isFalse);
      expect(controller.stage, UpdateFlowStage.verifying);
    });
  });

  // ---------------------------------------------------------------------------
  // 2. SoftUpdateBanner
  // ---------------------------------------------------------------------------

  group('SoftUpdateBanner', () {
    testWidgets('renders versions and forwards Later/Update once each',
        (WidgetTester tester) async {
      int updates = 0;
      int dismissals = 0;

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: SoftUpdateBanner(
            release: _release(),
            installedVersion: '1.0.0',
            onUpdate: () => updates++,
            onDismiss: () => dismissals++,
          ),
        ),
      ));

      expect(find.text('Version 1.2.0 is available'), findsOneWidget);
      expect(find.textContaining('You are on 1.0.0'), findsOneWidget);
      expect(find.text('Later'), findsOneWidget);
      expect(find.text('Update'), findsOneWidget);

      await tester.tap(find.text('Later'));
      await tester.pump();
      expect(dismissals, 1);
      expect(updates, 0);

      await tester.tap(find.text('Update'));
      await tester.pump();
      expect(updates, 1);
      expect(dismissals, 1);
    });

    testWidgets('shows a formatted size when known',
        (WidgetTester tester) async {
      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: SoftUpdateBanner(
            release: _release(size: 2 * 1024 * 1024),
            installedVersion: '1.0.0',
            onUpdate: () {},
            onDismiss: () {},
          ),
        ),
      ));

      expect(find.textContaining('2.0 MB'), findsOneWidget);
    });
  });

  // ---------------------------------------------------------------------------
  // 3. DownloadProgressSheet
  // ---------------------------------------------------------------------------

  group('DownloadProgressSheet', () {
    testWidgets('shows determinate progress with a known total',
        (WidgetTester tester) async {
      final _FakeUpdateService service = _FakeUpdateService();
      addTearDown(service.dispose);
      final _StubFlowController controller = _StubFlowController(
        service: service,
        stage: UpdateFlowStage.downloading,
        progress:
            const DownloadProgress(receivedBytes: 512, totalBytes: 1024),
      );
      addTearDown(controller.dispose);

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: DownloadProgressSheet(
            release: _release(),
            controller: controller,
          ),
        ),
      ));

      expect(find.text('Updating to 1.2.0'), findsOneWidget);
      expect(find.text('Downloading'), findsOneWidget);
      expect(find.text('512 B / 1.0 KB'), findsOneWidget);
      expect(find.text('50%'), findsOneWidget);

      final LinearProgressIndicator indicator =
          tester.widget<LinearProgressIndicator>(
              find.byType(LinearProgressIndicator));
      expect(indicator.value, 0.5);
    });

    testWidgets('is indeterminate when totalBytes == 0',
        (WidgetTester tester) async {
      final _FakeUpdateService service = _FakeUpdateService();
      addTearDown(service.dispose);
      final _StubFlowController controller = _StubFlowController(
        service: service,
        stage: UpdateFlowStage.downloading,
        progress: const DownloadProgress(receivedBytes: 512, totalBytes: 0),
      );
      addTearDown(controller.dispose);

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: DownloadProgressSheet(
            release: _release(),
            controller: controller,
          ),
        ),
      ));

      expect(controller.fraction, isNull);
      expect(find.text('512 B'), findsOneWidget);

      final LinearProgressIndicator indicator =
          tester.widget<LinearProgressIndicator>(
              find.byType(LinearProgressIndicator));
      expect(indicator.value, isNull);
    });
  });

  // ---------------------------------------------------------------------------
  // 4. formatBytes
  // ---------------------------------------------------------------------------

  group('formatBytes', () {
    test('formats exact byte counts', () {
      expect(formatBytes(0), '0 B');
      expect(formatBytes(512), '512 B');
      expect(formatBytes(1024), '1.0 KB');
      expect(formatBytes(1536), '1.5 KB');
      expect(formatBytes(1048576), '1.0 MB');
      expect(formatBytes(2 * 1024 * 1024 * 1024), '2.0 GB');
      expect(formatBytes(150 * 1024 * 1024), '150 MB');
    });
  });

  // ---------------------------------------------------------------------------
  // 5. UpdateFlowStage.label
  // ---------------------------------------------------------------------------

  group('UpdateFlowStage.label', () {
    test('every stage has its label', () {
      expect(UpdateFlowStage.idle.label, 'Ready');
      expect(UpdateFlowStage.downloading.label, 'Downloading');
      expect(UpdateFlowStage.verifying.label, 'Verifying integrity');
      expect(UpdateFlowStage.installing.label, 'Installing');
      expect(UpdateFlowStage.completed.label, 'Download complete');
      expect(UpdateFlowStage.failed.label, 'Failed');
    });
  });

  // ---------------------------------------------------------------------------
  // 6. DynamicModuleRegistry
  // ---------------------------------------------------------------------------

  group('DynamicModuleRegistry', () {
    test('supports every shipped module type and nothing else', () {
      final DynamicModuleRegistry registry = DynamicModuleRegistry();
      for (final String type in <String>[
        'engine_status',
        'feature_flags',
        'update_status',
        'file_inspector',
        'agent_console',
        'thinking_panel',
        'banner',
      ]) {
        expect(registry.supports(type), isTrue, reason: type);
      }
      expect(registry.supports('not_a_module'), isFalse);
    });

    testWidgets('an unknown type renders the unsupported card and lists types',
        (WidgetTester tester) async {
      final FeatureFlagProvider flags = _flagProvider();
      final DynamicModuleRegistry registry = DynamicModuleRegistry();
      final DynamicModuleContext context = _context(flags);
      const DynamicModule module =
          DynamicModule(id: 'mystery1', type: 'mystery', flag: '');

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: Builder(
            builder: (BuildContext c) => registry.build(c, module, context),
          ),
        ),
      ));

      expect(find.textContaining('Unsupported module type'), findsOneWidget);
      expect(find.textContaining('Supported types:'), findsOneWidget);
      expect(find.textContaining('engine_status'), findsOneWidget);
      expect(find.textContaining('banner'), findsOneWidget);
    });

    testWidgets('register overrides a builder', (WidgetTester tester) async {
      final FeatureFlagProvider flags = _flagProvider();
      final DynamicModuleRegistry registry = DynamicModuleRegistry();
      registry.register('engine_status',
          (BuildContext c, DynamicModule m, DynamicModuleContext ctx) =>
              const Text('CUSTOM ENGINE'));
      expect(registry.supports('engine_status'), isTrue);

      const DynamicModule module =
          DynamicModule(id: 'engine1', type: 'engine_status', flag: '');

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: Builder(
            builder: (BuildContext c) =>
                registry.build(c, module, _context(flags)),
          ),
        ),
      ));

      expect(find.text('CUSTOM ENGINE'), findsOneWidget);
    });

    testWidgets('buildLayout with everything hidden shows the empty notice',
        (WidgetTester tester) async {
      const FeatureFlagMatrix matrix = FeatureFlagMatrix(
        version: 1,
        updatedAt: null,
        flags: <String, Object?>{'off': false},
        remoteDefaults: <String, Object?>{},
        layout: DynamicLayout(
          sections: <DynamicSection>[
            DynamicSection(
              id: 's',
              title: 'S',
              order: 0,
              modules: <DynamicModule>[
                DynamicModule(id: 'm', type: 'banner', flag: 'off'),
              ],
            ),
          ],
        ),
      );
      final FeatureFlagProvider flags = _flagProvider(seed: matrix);
      final DynamicModuleRegistry registry = DynamicModuleRegistry();
      final DynamicModuleContext context = _context(flags);

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: Builder(
            builder: (BuildContext c) =>
                registry.buildLayout(c, matrix.layout, context),
          ),
        ),
      ));

      expect(
        find.text('No dynamic modules are currently enabled.'),
        findsOneWidget,
      );
    });
  });

  // ---------------------------------------------------------------------------
  // 7. SliderListTile (public widget in main.dart)
  // ---------------------------------------------------------------------------

  group('SliderListTile', () {
    testWidgets('renders its label and forwards slider changes',
        (WidgetTester tester) async {
      double? changed;
      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: app.SliderListTile(
            label: 'Min supported',
            value: 0.25,
            onChanged: (double v) => changed = v,
          ),
        ),
      ));

      expect(find.text('Min supported'), findsOneWidget);
      final Slider slider = tester.widget<Slider>(find.byType(Slider));
      expect(slider.value, 0.25);
      slider.onChanged!(0.75);
      expect(changed, 0.75);
    });
  });
}