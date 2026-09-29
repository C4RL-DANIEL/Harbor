// Widget tests for the four top-level Harbor screens.
//
// AUTHORING NOTE: these tests could not be executed where they were written —
// the host is aarch64 and the bundled Flutter SDK is x86-64 only, so
// `flutter test` cannot run here. They are written against the exact public
// surface of the screens under test and must be run in CI.
//
// Hermetic: every collaborator is either built from `harbor_core` primitives
// with deterministic behaviour or is a hand-written fake. No platform channel,
// no network call, and no real corpus/model file is read or written; the only
// file-system touch is a throwaway temp directory that the model runtime never
// bootstraps into.

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:harbor_core/harbor_core.dart';
import 'package:main_app/core/chat/chat_controller.dart';
import 'package:main_app/core/corpus/corpus_controller.dart';
import 'package:main_app/core/corpus/corpus_persistence.dart';
import 'package:main_app/core/corpus/device_collector.dart';
import 'package:main_app/core/model/harbor_model_runtime.dart';
import 'package:main_app/core/platform/platform_bridge.dart';
import 'package:main_app/core/tools/tools_controller.dart';
import 'package:main_app/core/training/training_controller.dart';
import 'package:main_app/features/screens/chat_screen.dart';
import 'package:main_app/features/screens/corpus_screen.dart';
import 'package:main_app/features/screens/tools_screen.dart';
import 'package:main_app/features/screens/training_screen.dart';

/// A deterministic [LanguageModelRuntime] that always prefers token id 0.
///
/// The chat engine samples greedily when the sampler temperature is zero, so
/// this stub makes a turn run for exactly `maxNewTokens` steps and then stop; no
/// real weights and no randomness are involved.
class _StubModelRuntime implements LanguageModelRuntime {
  @override
  int get vocabSize => 260;

  @override
  int get contextLength => 256;

  @override
  int get parameterCount => 260;

  @override
  List<double> logitsFor(List<int> tokens) {
    return List<double>.filled(vocabSize, 0.0)..[0] = 1.0;
  }

  @override
  void resetCache() {}
}

/// An in-memory [CorpusStorage], so the corpus never touches disk.
class _MemoryCorpusStorage implements CorpusStorage {
  String? _payload;

  @override
  Future<String?> read() async => _payload;

  @override
  Future<void> write(String payload) async {
    _payload = payload;
  }

  @override
  Future<void> delete() async {
    _payload = null;
  }
}

/// The first hand-written tool: echoes its optional `text` argument back.
class _EchoTool extends FunctionTool {
  const _EchoTool();

  @override
  String get name => 'demo.echo';

  @override
  String get description => 'Echoes the supplied text back.';

  @override
  Map<String, Object?> get parameters => const <String, Object?>{
        'type': 'object',
        'properties': <String, Object?>{
          'text': <String, Object?>{'type': 'string'},
        },
      };

  @override
  bool get mutating => false;

  @override
  Future<Object?> run(Map<String, Object?> args) async {
    return <String, Object?>{'echo': optionalStringArg(args, 'text') ?? ''};
  }
}

/// The second hand-written tool: a boolean switch the filter should exclude.
class _FlagTool extends FunctionTool {
  const _FlagTool();

  @override
  String get name => 'demo.flag';

  @override
  String get description => 'Flips a boolean demonstration switch.';

  @override
  Map<String, Object?> get parameters => const <String, Object?>{
        'type': 'object',
        'properties': <String, Object?>{
          'enabled': <String, Object?>{'type': 'boolean'},
        },
      };

  @override
  bool get mutating => false;

  @override
  Future<Object?> run(Map<String, Object?> args) async {
    return <String, Object?>{'enabled': args['enabled'] == true};
  }
}

/// Builds a chat controller whose engine answers deterministically.
ChatController _buildChatController() {
  final ByteTokenizer tokenizer = ByteTokenizer();
  final ChatEngine engine = ChatEngine(
    model: _StubModelRuntime(),
    tokenizer: tokenizer,
    sampler: Sampler(temperature: 0.0),
    routeIntents: false,
    maxNewTokens: 8,
  );
  return ChatController(engine: engine, tokenizer: tokenizer);
}

/// Every collaborator the four screens need, wired with the cheapest viable
/// arguments. The model runtime is deliberately never bootstrapped, so it stays
/// un-ready and the training screen renders its readiness warning.
/// Scrolls the first scrollable until [finder] matches, then stops.
///
/// `scrollUntilVisible` checks the current tree before scrolling, so this is a
/// no-op for content that is already on screen.
Future<void> _scrollTo(WidgetTester tester, Finder finder) async {
  await tester.scrollUntilVisible(
    finder,
    400,
    scrollable: find.byType(Scrollable).first,
    maxScrolls: 80,
  );
}

class _Harness {
  _Harness._({
    required this.runtime,
    required this.corpus,
    required this.training,
    required this.tools,
    required this.chat,
  });

  factory _Harness() {
    final Directory directory =
        Directory.systemTemp.createTempSync('harbor_screen_test');
    final HarborStorageLayout layout = HarborStorageLayout(directory);
    final HarborModelRuntime runtime = HarborModelRuntime(layout: layout);
    final CorpusStore store = CorpusStore(storage: _MemoryCorpusStorage());
    final CorpusController corpus = CorpusController(
      store: store,
      collector: DeviceCorpusCollector(store: store),
      deviceRoots: const <Directory>[],
    );
    final PlatformBridge bridge = PlatformBridge();
    return _Harness._(
      runtime: runtime,
      corpus: corpus,
      training: TrainingController(
        runtime: runtime,
        corpus: corpus,
        bridge: bridge,
        log: TrainingLog(layout.trainingLogFile),
      ),
      tools: ToolsController(
        registry: ToolRegistry(const <Tool>[_EchoTool(), _FlagTool()]),
        bridge: bridge,
      ),
      chat: _buildChatController(),
    );
  }

  final HarborModelRuntime runtime;
  final CorpusController corpus;
  final TrainingController training;
  final ToolsController tools;
  final ChatController chat;
}

/// Wraps [child] in the minimal Material host a screen needs.
Widget _app(Widget child) {
  return MaterialApp(home: Scaffold(body: child));
}

void main() {
  testWidgets('every screen builds and pumps without exception', (
    WidgetTester tester,
  ) async {
    final _Harness harness = _Harness();

    await tester.pumpWidget(_app(ChatScreen(controller: harness.chat)));
    await tester.pump();
    expect(tester.takeException(), isNull);

    await tester.pumpWidget(
      _app(
        TrainingScreen(
          controller: harness.training,
          runtime: harness.runtime,
        ),
      ),
    );
    await tester.pump();
    expect(tester.takeException(), isNull);

    await tester.pumpWidget(_app(ToolsScreen(controller: harness.tools)));
    await tester.pump();
    expect(tester.takeException(), isNull);

    await tester.pumpWidget(_app(CorpusScreen(controller: harness.corpus)));
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
  });

  testWidgets('chat seeds a greeting and adds a sent user bubble', (
    WidgetTester tester,
  ) async {
    final _Harness harness = _Harness();
    await tester.pumpWidget(_app(ChatScreen(controller: harness.chat)));
    await tester.pump();

    expect(
      find.textContaining('Harbor answers with a small language model'),
      findsOneWidget,
    );

    await tester.enterText(find.byType(TextField).first, 'hello there');
    await tester.pump();
    await tester.tap(find.text('Send'));
    await tester.pump();

    expect(find.text('hello there'), findsOneWidget);
  });

  testWidgets('chat Prompt button opens a dialog with the rendered prompt', (
    WidgetTester tester,
  ) async {
    final _Harness harness = _Harness();
    await tester.pumpWidget(_app(ChatScreen(controller: harness.chat)));
    await tester.pump();

    final String prompt = harness.chat.renderedPrompt;

    await tester.tap(find.text('Prompt'));
    await tester.pumpAndSettle();

    expect(find.text('Prompt sent to the model'), findsOneWidget);
    expect(find.text(prompt), findsOneWidget);
  });

  testWidgets('tools filter narrows the visible list', (
    WidgetTester tester,
  ) async {
    final _Harness harness = _Harness();
    await tester.pumpWidget(_app(ToolsScreen(controller: harness.tools)));
    await tester.pump();

    expect(find.text('demo.echo'), findsOneWidget);
    expect(find.text('demo.flag'), findsOneWidget);
    expect(find.text('2 of 2 tools'), findsOneWidget);

    await tester.enterText(find.byType(TextField).first, 'echo');
    await tester.pump();

    expect(find.text('demo.echo'), findsOneWidget);
    expect(find.text('demo.flag'), findsNothing);
    expect(find.text('1 of 2 tools'), findsOneWidget);
  });

  testWidgets('tools tile expands and runs, showing the result summary', (
    WidgetTester tester,
  ) async {
    final _Harness harness = _Harness();
    await tester.pumpWidget(_app(ToolsScreen(controller: harness.tools)));
    await tester.pump();

    await tester.tap(find.text('demo.echo'));
    await tester.pumpAndSettle();

    expect(find.text('Echoes the supplied text back.'), findsOneWidget);
    expect(find.text('Run demo.echo'), findsOneWidget);

    await tester.enterText(find.byType(TextFormField), 'hi');
    await tester.tap(find.text('Run demo.echo'));
    await tester.pumpAndSettle();

    expect(find.textContaining('echo=hi'), findsOneWidget);
  });

  testWidgets('corpus shows the empty state, then counts an added document', (
    WidgetTester tester,
  ) async {
    final _Harness harness = _Harness();
    await tester.pumpWidget(_app(CorpusScreen(controller: harness.corpus)));
    await tester.pumpAndSettle();

    // The document list sits below the stats and the collection controls, and a
    // ListView does not build what is off-screen, so the assertions have to
    // scroll to it first.
    await _scrollTo(tester, find.text('Documents (0)'));
    expect(find.text('Documents (0)'), findsOneWidget);
    await _scrollTo(tester, find.text('No documents yet'));
    expect(find.text('No documents yet'), findsOneWidget);

    final bool added = await harness.corpus.addText(
      'some sufficiently long text for the corpus to accept it',
    );
    await tester.pump();

    expect(added, isTrue);
    await _scrollTo(tester, find.text('user-input'));
    expect(find.text('user-input'), findsOneWidget);
    expect(find.text('Documents (1)'), findsOneWidget);
  });

  testWidgets('corpus rejects a too-short text and renders the error', (
    WidgetTester tester,
  ) async {
    final _Harness harness = _Harness();
    await tester.pumpWidget(_app(CorpusScreen(controller: harness.corpus)));
    await tester.pumpAndSettle();

    final bool added = await harness.corpus.addText('too short');
    await tester.pump();

    expect(added, isFalse);
    expect(harness.corpus.error, isNotNull);
    expect(find.textContaining('too short to be useful'), findsOneWidget);
  });

  testWidgets('training shows the readiness warning and the reset control', (
    WidgetTester tester,
  ) async {
    final _Harness harness = _Harness();
    await tester.pumpWidget(
      _app(
        TrainingScreen(
          controller: harness.training,
          runtime: harness.runtime,
        ),
      ),
    );
    await tester.pump();

    expect(find.textContaining('The model is not ready'), findsOneWidget);
    expect(find.text('Reset model'), findsOneWidget);
  });
}