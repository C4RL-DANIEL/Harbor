// Controller-level tests for the five `lib/core` controllers.
//
// These exercise the state machines the UI drives: the model runtime that owns
// the tokenizer and weights on disk, the corpus store front end, the tool
// runner, the chat transcript, and the training runner. Everything is
// deterministic: the model config is the tiny test preset, the corpus is a
// fixed block of prose, the chat model is a stub that always wants to stop, and
// every training run is four steps. No test touches the network, and every file
// written lives under a freshly created `Directory.systemTemp` root that is
// removed on teardown.

import 'dart:async';
import 'dart:io';

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

/// A little under a kilobyte of ordinary English prose.
///
/// The exact content does not matter, only that it is long enough that its
/// token count clears the default training window and comfortably exercises the
/// runtime's tokenizer and model paths. It deliberately contains no e-mail
/// address, phone number or long opaque token, so `TextRedactor.redact` is the
/// identity on it and an assertion such as `trainingText().contains(_prose)`
/// is meaningful.
const String _prose = '''
Harbor learns from the words a person already owns. A small model, trained on
a phone, does not need a library of books to be useful; it needs a steady supply
of ordinary sentences, the kind people write in notes, messages and diaries.
Every paragraph here is deliberately plain. The point of the corpus is not
literary quality but repetition: common words, common endings, and the small
grammatical turns that a byte level tokenizer can learn to compress.

When the device gathers text, it redacts secrets first, hashes what remains,
and keeps each document once. Duplicate pages cost storage without teaching
anything new, so the store refuses them and counts the refusal. The trainer
then walks random windows over the joined text, predicting the next token from
the ones before it. That single objective, repeated a few thousand times, is
enough to make a model that finishes sentences in the style of its corpus.
''';

/// A document long enough to pass the controller's minimum-length check.
const String _longText = 'Harbor keeps a copy of every accepted document so that '
    'the corpus screen can show exactly what the trainer will read.';

/// A second document, used to prove removal and clearing change the count.
const String _otherText = 'A second document proves that removal and clearing '
    'change the count in both directions.';

/// The small architecture every test builds, with the byte vocabulary floor.
///
/// Note that `HarborModelRuntime` derives its BPE merge budget from
/// `baseConfig.vocabSize - 260`, so a vocabulary of 260 requests zero merges
/// and the tokenizer stays the raw byte vocabulary. The model is sized to the
/// tokenizer's *actual* vocabulary regardless, which is the property under test.
TinyLmConfig _testConfig() => TinyLmConfig.testPreset.copyWith(vocabSize: 260);

/// Creates a temp directory and registers its removal.
Directory _tempDir() {
  final Directory dir = Directory.systemTemp.createTempSync('harbor_ctrl');
  addTearDown(() {
    if (dir.existsSync()) {
      dir.deleteSync(recursive: true);
    }
  });
  return dir;
}

/// Bootstraps a runtime over a fresh layout rooted at [root].
Future<HarborModelRuntime> _bootstrappedRuntime(Directory root) async {
  final HarborModelRuntime runtime = HarborModelRuntime(
    layout: HarborStorageLayout(root),
    baseConfig: _testConfig(),
  );
  await runtime.bootstrap(_prose);
  return runtime;
}

/// An in-memory [CorpusStorage], the seam the core package leaves open for
/// tests and for browser storage.
class _MemoryCorpusStorage implements CorpusStorage {
  String? payload;

  @override
  Future<String?> read() async => payload;

  @override
  Future<void> write(String value) async {
    payload = value;
  }

  @override
  Future<void> delete() async {
    payload = null;
  }
}

/// Builds a [CorpusController] backed by [storage] and no web collector.
CorpusController _corpusController(_MemoryCorpusStorage storage) {
  final CorpusStore store = CorpusStore(storage: storage);
  return CorpusController(
    store: store,
    collector: DeviceCorpusCollector(store: store),
    webCollector: null,
  );
}

/// A tool that echoes its `text` argument back inside a map.
class _EchoTool extends FunctionTool {
  const _EchoTool();

  @override
  String get name => 'demo.echo';

  @override
  String get description => 'Echoes a text argument back to the caller.';

  @override
  Map<String, Object?> get parameters => const <String, Object?>{
        'type': 'object',
        'properties': <String, Object?>{
          'text': <String, Object?>{'type': 'string'},
        },
        'required': <String>['text'],
      };

  @override
  bool get mutating => false;

  @override
  Future<Object?> run(Map<String, Object?> args) async =>
      <String, Object?>{'echo': requireStringArg(args, 'text')};
}

/// A tool whose only argument is a boolean.
class _ConfirmTool extends FunctionTool {
  const _ConfirmTool();

  @override
  String get name => 'demo.confirm';

  @override
  String get description => 'Reports whether a boolean flag was supplied.';

  @override
  Map<String, Object?> get parameters => const <String, Object?>{
        'type': 'object',
        'properties': <String, Object?>{
          'flag': <String, Object?>{'type': 'boolean'},
        },
        'required': <String>['flag'],
      };

  @override
  bool get mutating => false;

  @override
  Future<Object?> run(Map<String, Object?> args) async =>
      <String, Object?>{'flag': optionalBoolArg(args, 'flag')};
}

/// A [LanguageModelRuntime] that always wants to stop.
///
/// End-of-sequence is the highest-scoring id on every call, so a turn ends as
/// soon as the engine's minimum-length guard allows it. Byte 104 (`h`) is the
/// runner-up, which gives that guard a printable token to fall back on while
/// end-of-sequence is still suppressed. The logit gap is large enough that the
/// sampler's softmax weight for the losing candidates underflows to exactly
/// zero, which makes the choice deterministic for any random draw.
class _EosBiasedModel implements LanguageModelRuntime {
  _EosBiasedModel({required this.vocabSize, required this.eosId});

  @override
  final int vocabSize;

  /// The id that ends generation.
  final int eosId;

  @override
  int get contextLength => 256;

  @override
  int get parameterCount => 0;

  @override
  void resetCache() {
    // Stateless stub: there is no incremental cache to discard.
  }

  @override
  List<double> logitsFor(List<int> prefix) {
    final List<double> logits = List<double>.filled(vocabSize, 0.0);
    logits[eosId] = 1000.0;
    logits[104] = 500.0;
    return logits;
  }
}

/// Builds a [ChatController] over the stub model and a bare byte tokenizer.
///
/// The engine's context window is deliberately much larger than the rendered
/// prompt: at the default 256 tokens the system instruction alone fills the
/// window and the prompt renderer trims the user's message away before
/// generation, which would make the transcript assertion meaningless.
ChatController _chatController() {
  final ByteTokenizer tokenizer = ByteTokenizer();
  final ChatEngine engine = ChatEngine(
    model: _EosBiasedModel(
      vocabSize: tokenizer.vocabSize,
      eosId: tokenizer.eosId,
    ),
    tokenizer: tokenizer,
    contextLength: 4096,
  );
  final ChatController controller = ChatController(
    engine: engine,
    tokenizer: tokenizer,
  );
  addTearDown(controller.dispose);
  return controller;
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('HarborModelRuntime', () {
    test('bootstraps a fresh layout and reloads a saved checkpoint', () async {
      final Directory root = _tempDir();
      final HarborStorageLayout layout = HarborStorageLayout(root);
      final TinyLmConfig baseConfig = _testConfig();
      final HarborModelRuntime runtime = HarborModelRuntime(
        layout: layout,
        baseConfig: baseConfig,
      );

      expect(runtime.ready, isFalse);
      expect(runtime.hasCheckpoint, isFalse);

      await runtime.bootstrap(_prose);
      expect(runtime.ready, isTrue);
      expect(runtime.tokenizer, isNotNull);
      expect(runtime.model, isNotNull);
      expect(runtime.model!.config.vocabSize, runtime.tokenizer!.vocabSize);
      expect(runtime.status.contains('not initialised'), isFalse);

      final int parameterCount = runtime.model!.totalParameterCount;
      await runtime.save();
      expect(layout.tokenizerFile.existsSync(), isTrue);
      expect(layout.modelFile.existsSync(), isTrue);

      final HarborModelRuntime reloaded = HarborModelRuntime(
        layout: layout,
        baseConfig: baseConfig,
      );
      await reloaded.bootstrap(_prose);
      expect(reloaded.ready, isTrue);
      expect(reloaded.model!.totalParameterCount, parameterCount);
    });

    test('round-trips a training checkpoint and resets to empty', () async {
      final Directory root = _tempDir();
      final HarborStorageLayout layout = HarborStorageLayout(root);
      final HarborModelRuntime runtime = HarborModelRuntime(
        layout: layout,
        baseConfig: _testConfig(),
      );
      await runtime.bootstrap(_prose);
      expect(runtime.ready, isTrue);

      await runtime.save();
      await runtime.saveCheckpoint(runtime.model!.encode());
      expect(await runtime.restoreTrainingCheckpoint(), isTrue);
      expect(runtime.model, isNotNull);
      expect(
        runtime.model!.logitsFor(<int>[1, 2, 3]).length,
        runtime.tokenizer!.vocabSize,
      );

      await runtime.reset();
      expect(runtime.ready, isFalse);
      expect(runtime.hasCheckpoint, isFalse);
      expect(layout.modelFile.existsSync(), isFalse);
      expect(layout.tokenizerFile.existsSync(), isFalse);
    });
  });

  group('CorpusController', () {
    test('initialises empty and deduplicates by content hash', () async {
      final CorpusController controller =
          _corpusController(_MemoryCorpusStorage());
      await controller.initialize();
      expect(controller.stats.documentCount, 0);

      expect(await controller.addText(_longText), isTrue);
      expect(controller.stats.documentCount, 1);

      // The identical text hashes to the same id and must be refused.
      expect(await controller.addText(_longText), isFalse);
      expect(controller.stats.documentCount, 1);

      expect(await controller.addText('short'), isFalse);
      expect(controller.error, isNotNull);
    });

    test('removes and clears documents and exposes the training text',
        () async {
      final CorpusController controller =
          _corpusController(_MemoryCorpusStorage());
      await controller.initialize();
      expect(await controller.addText(_longText), isTrue);
      expect(controller.trainingText(), contains(_longText));

      final String id = controller.documents.single.id;
      await controller.remove(id);
      expect(controller.stats.documentCount, 0);

      expect(await controller.addText(_otherText), isTrue);
      expect(controller.stats.documentCount, 1);
      await controller.clear();
      expect(controller.stats.documentCount, 0);
    });

    test('refuses a malformed URL without any network access', () async {
      final CorpusController controller =
          _corpusController(_MemoryCorpusStorage());
      await controller.initialize();
      expect(await controller.fetchUrl('not a url'), isFalse);
      expect(controller.error, isNotNull);
    });
  });

  group('ToolsController', () {
    test('invokes tools, records history newest first and clears it', () async {
      final ToolRegistry registry = ToolRegistry(<Tool>[
        const _EchoTool(),
        const _ConfirmTool(),
      ]);
      final ToolsController controller = ToolsController(
        registry: registry,
        bridge: PlatformBridge(),
      );
      expect(controller.platformSupported, isFalse);

      final ToolResult echoed = await controller.invoke(
        'demo.echo',
        <String, Object?>{'text': 'hi'},
      );
      expect(echoed.ok, isTrue);
      expect((echoed.data! as Map<String, Object?>)['echo'], 'hi');

      final ToolResult confirmed = await controller.invoke(
        'demo.confirm',
        <String, Object?>{'flag': true},
      );
      expect(confirmed.ok, isTrue);

      expect(controller.history.length, 2);
      expect(controller.history.first.call.name, 'demo.confirm');
      expect(controller.history.last.call.name, 'demo.echo');
      expect(controller.last!.call.name, 'demo.confirm');

      controller.clearHistory();
      expect(controller.history, isEmpty);
      expect(controller.last, isNull);
    });

    test('returns a failure result for an unknown tool name', () async {
      final ToolRegistry registry = ToolRegistry(<Tool>[const _EchoTool()]);
      final ToolsController controller = ToolsController(
        registry: registry,
        bridge: PlatformBridge(),
      );
      final ToolResult unknown = await controller.invoke('demo.missing');
      expect(unknown.ok, isFalse);
      expect(controller.last!.call.name, 'demo.missing');
    });

    test('reports device tools as unsupported off-platform', () async {
      final PlatformBridge bridge = PlatformBridge();
      final ToolRegistry registry =
          ToolRegistry(deviceTools(hostCallerFrom(bridge)));
      final ToolsController controller = ToolsController(
        registry: registry,
        bridge: bridge,
      );
      expect(controller.platformSupported, isFalse);

      final ToolResult result = await controller.invoke('device.battery');
      expect(result.ok, isFalse);
    });
  });

  group('ChatController', () {
    test('completes a turn with a user and an assistant message', () async {
      final ChatController controller = _chatController();
      await controller.send('hello there');

      expect(controller.busy, isFalse);
      expect(controller.error, isNull);
      expect(controller.messages.length, 2);
      expect(controller.messages.first.role, ChatRole.user);
      expect(controller.messages.first.content, 'hello there');
      expect(controller.messages.last.role, ChatRole.assistant);
      expect(controller.messages.last.content, isNotEmpty);
      expect(controller.renderedPrompt, contains('hello there'));
    });

    test('stop cancels an in-flight turn', () async {
      final ChatController controller = _chatController();
      final Future<void> pending = controller.send('hello there');
      expect(controller.busy, isTrue);

      controller.stop();
      expect(controller.busy, isFalse);
      expect(controller.stopReason, 'cancelled');

      // Cancelling the subscription abandons the turn's completion future, so
      // it is intentionally not awaited; the controller is already settled.
      unawaited(pending);
    });

    test('seedGreeting is idempotent and clear empties the transcript',
        () async {
      final ChatController controller = _chatController();
      controller.seedGreeting('Welcome');
      expect(controller.messages.length, 1);
      expect(controller.messages.single.role, ChatRole.assistant);

      controller.seedGreeting('Again');
      expect(controller.messages.length, 1);
      expect(controller.messages.single.content, 'Welcome');

      await controller.send('hello there');
      expect(controller.messages.length, 3);
      controller.clear();
      expect(controller.messages, isEmpty);
    });

    test('ignores a second send while busy', () async {
      final ChatController controller = _chatController();
      final Future<void> first = controller.send('one');
      await controller.send('two');
      await first;

      int userMessages = 0;
      for (final ChatMessage message in controller.messages) {
        if (message.role == ChatRole.user) {
          userMessages += 1;
        }
      }
      expect(userMessages, 1);
      expect(controller.messages.length, 2);
    });
  });

  group('TrainingController', () {
    test('reports readiness and fails cleanly on an empty corpus', () async {
      final Directory root = _tempDir();
      final HarborModelRuntime runtime = await _bootstrappedRuntime(root);
      final CorpusController corpus = _corpusController(_MemoryCorpusStorage());
      await corpus.initialize();

      final TrainingController controller = TrainingController(
        runtime: runtime,
        corpus: corpus,
        bridge: PlatformBridge(),
        log: TrainingLog(File('${root.path}/training_log.jsonl')),
      );
      expect(controller.readiness, isNotNull);

      await controller.start();
      expect(controller.error, isNotNull);
    });

    test('completes a short run and writes the model', () async {
      final Directory root = _tempDir();
      final HarborModelRuntime runtime = await _bootstrappedRuntime(root);
      final CorpusController corpus = _corpusController(_MemoryCorpusStorage());
      await corpus.initialize();
      expect(await corpus.addText(_prose), isTrue);

      final TrainingController controller = TrainingController(
        runtime: runtime,
        corpus: corpus,
        bridge: PlatformBridge(),
        log: TrainingLog(File('${root.path}/training_log.jsonl')),
      );
      expect(controller.readiness, isNull);

      await controller
          .start(
            override: const TrainingConfig(
              totalSteps: 4,
              windowLength: 8,
              batchSize: 2,
              warmupSteps: 1,
              checkpointEvery: 2,
            ),
          )
          .timeout(const Duration(seconds: 120));

      expect(controller.latest, isNotNull);
      // The controller's `latest` only reaches `completed` if the session's
      // terminal progress event is delivered before `start` cancels its
      // subscription; the model save below hangs off that same event.
      expect(controller.latest!.phase, TrainingPhase.completed);
      expect(controller.lossHistory, isNotEmpty);
      // Do not assert that the loss fell: four steps on a byte model is far too
      // little to prove learning. Only that every point is a real number.
      expect(
        controller.lossHistory.every((double value) => value.isFinite),
        isTrue,
      );
      expect(controller.latest!.smoothedLoss.isFinite, isTrue);

      // The completion handler saves off the progress stream, so the file can
      // appear a few microtasks after `start` resolves; wait for it bounded.
      for (int i = 0; i < 100 && !runtime.layout.modelFile.existsSync(); i++) {
        await Future<void>.delayed(const Duration(milliseconds: 20));
      }
      expect(runtime.layout.modelFile.existsSync(), isTrue);
    });
  });
}