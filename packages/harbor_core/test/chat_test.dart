// Tests for the chat engine, the chat message model, and the tool protocol.
//
// The engine is exercised with a hand-written deterministic model stub and a
// recording tool so every assertion is about the engine's own behaviour rather
// than about a real transformer's output. Nothing here touches a socket, a
// platform channel, or the clock.

import 'package:harbor_core/harbor_core.dart';
import 'package:test/test.dart';

/// Runs every chat test.
void main() {
  group('promptBudget', () {
    test('reserves maxNewTokens when it is small', () {
      final Tokenizer tokenizer = ByteTokenizer();
      final ChatEngine engine = _engine(
        model: _StubModel(vocabSize: tokenizer.vocabSize),
        tokenizer: tokenizer,
        contextLength: 400,
        maxNewTokens: 8,
      );
      expect(engine.promptBudget, 392);
    });

    test('caps the reserve at half the window', () {
      final Tokenizer tokenizer = ByteTokenizer();
      final ChatEngine engine = _engine(
        model: _StubModel(vocabSize: tokenizer.vocabSize),
        tokenizer: tokenizer,
        contextLength: 400,
        maxNewTokens: 300,
      );
      expect(engine.promptBudget, 200);
    });

    test('never drops below one token', () {
      final Tokenizer tokenizer = ByteTokenizer();
      final ChatEngine engine = _engine(
        model: _StubModel(vocabSize: tokenizer.vocabSize),
        tokenizer: tokenizer,
        contextLength: 2,
        maxNewTokens: 8,
      );
      expect(engine.promptBudget, 1);
    });
  });

  group('empty reply regression', () {
    // THE IMPORTANT ONE: a freshly initialised model often puts its whole
    // probability mass on EOS, which used to produce a completely empty reply
    // that looks like a crash. The engine must fall back to a real token.
    test('an EOS-favouring model still produces text, then stops after minNewTokens', () async {
      final Tokenizer tokenizer = ByteTokenizer();
      final _StubModel model = _StubModel(
        vocabSize: tokenizer.vocabSize,
        alwaysEos: true,
      );
      final ChatEngine engine = _engine(
        model: model,
        tokenizer: tokenizer,
        maxNewTokens: 8,
        minNewTokens: 4,
        contextLength: 400,
      );

      final List<ChatEvent> events = await engine
          .respond(<ChatMessage>[ChatMessage.user('say something')])
          .toList();

      final String text = events
          .whereType<ChatToken>()
          .map((ChatToken token) => token.text)
          .join();
      expect(text, isNotEmpty);

      final ChatFinished finished = events.whereType<ChatFinished>().single;
      expect(finished.text, isNotEmpty);
      expect(finished.generatedTokens, greaterThanOrEqualTo(1));

      // Once the floor has been reached, EOS is allowed to win.
      expect(finished.stopReason, 'eos');
      expect(finished.generatedTokens, lessThanOrEqualTo(8));

      // One call per produced token, plus the final call that sampled EOS.
      expect(model.logitsForCalls, finished.generatedTokens + 1);
    });
  });

  group('catalogue fitting', () {
    test('shortens the catalogue, not the conversation', () {
      final Tokenizer tokenizer = ByteTokenizer();
      final ToolRegistry registry = ToolRegistry(<Tool>[
        _largeTool('alpha'),
        _largeTool('beta'),
        _largeTool('gamma'),
      ]);
      final ChatEngine engine = _engine(
        model: _StubModel(vocabSize: tokenizer.vocabSize),
        tokenizer: tokenizer,
        tools: registry,
        contextLength: 64,
        maxNewTokens: 8,
        systemPrompt: '',
      );

      final String prompt = engine.renderPrompt(<ChatMessage>[
        ChatMessage.user('hi'),
      ]);
      expect(
        tokenizer.encode(prompt).length,
        lessThanOrEqualTo(engine.promptBudget),
      );
      expect(prompt, contains('User: hi'));
      expect(prompt, isNot(contains('quasar')));
    });

    test('keeps full tool descriptions when the window is generous', () {
      final Tokenizer tokenizer = ByteTokenizer();
      final ToolRegistry registry = ToolRegistry(<Tool>[
        _largeTool('alpha'),
        _largeTool('beta'),
        _largeTool('gamma'),
      ]);
      final ChatEngine engine = _engine(
        model: _StubModel(vocabSize: tokenizer.vocabSize),
        tokenizer: tokenizer,
        tools: registry,
        contextLength: 4096,
        maxNewTokens: 16,
        systemPrompt: '',
      );

      final String prompt = engine.renderPrompt(<ChatMessage>[
        ChatMessage.user('hi'),
      ]);
      expect(prompt, contains('quasar'));
      expect(
        tokenizer.encode(prompt).length,
        lessThanOrEqualTo(engine.promptBudget),
      );
    });
  });

  group('routed intent', () {
    test('runs the registered tool before the model speaks', () async {
      final Tokenizer tokenizer = ByteTokenizer();
      final _RecordingTool tool = _RecordingTool(
        name: 'device.clipboard.write',
        description: 'Writes text to the clipboard.',
        parameters: <String, Object?>{
          'type': 'object',
          'properties': <String, Object?>{
            'text': <String, Object?>{'type': 'string'},
          },
          'required': <String>['text'],
        },
      );
      final ChatEngine engine = _engine(
        model: _StubModel(vocabSize: tokenizer.vocabSize),
        tokenizer: tokenizer,
        tools: ToolRegistry(<Tool>[tool]),
        maxNewTokens: 8,
        contextLength: 400,
      );

      final List<ChatEvent> events = await engine
          .respond(<ChatMessage>[
            ChatMessage.user('copy hello to the clipboard'),
          ])
          .toList();

      expect(events.first, isA<ChatToolStarted>());
      final ChatToolStarted started =
          events.whereType<ChatToolStarted>().single;
      expect(started.call.name, 'device.clipboard.write');

      final ChatToolFinished finished =
          events.whereType<ChatToolFinished>().single;
      expect(finished.result.ok, isTrue);
      expect(finished.result.data, <String, Object?>{'value': 'hello'});

      expect(tool.runs, 1);
      expect(tool.lastArgs, <String, Object?>{'text': 'hello'});
      expect(engine.lastTurnUsedTool, isTrue);
      expect(events.last, isA<ChatFinished>());
    });
  });

  group('model-emitted tool call', () {
    test('executes a call found in the generated text', () async {
      final Tokenizer tokenizer = ByteTokenizer();
      const String markup = '<tool name="test.echo">{"value":"hi"}</tool>';
      final List<int> script = tokenizer.encode(markup);
      final _StubModel model = _StubModel(
        vocabSize: tokenizer.vocabSize,
        tokens: script,
      );
      final _RecordingTool tool = _RecordingTool(
        name: 'test.echo',
        description: 'Echoes the value argument back.',
        parameters: <String, Object?>{
          'type': 'object',
          'properties': <String, Object?>{
            'value': <String, Object?>{'type': 'string'},
          },
          'required': <String>['value'],
        },
      );
      final ChatEngine engine = _engine(
        model: model,
        tokenizer: tokenizer,
        tools: ToolRegistry(<Tool>[tool]),
        maxNewTokens: script.length,
        contextLength: 400,
        routeIntents: false,
      );

      final List<ChatEvent> events = await engine
          .respond(<ChatMessage>[ChatMessage.user('please echo hi')])
          .toList();

      final ChatToolStarted started =
          events.whereType<ChatToolStarted>().single;
      expect(started.call.name, 'test.echo');

      final ChatToolFinished finished =
          events.whereType<ChatToolFinished>().single;
      expect(finished.result.ok, isTrue);
      expect(finished.result.data, <String, Object?>{'value': 'hi'});

      expect(tool.runs, 1);
      expect(engine.lastTurnUsedTool, isTrue);
      expect(model.logitsForCalls, script.length);
      expect(events.last, isA<ChatFinished>());
    });
  });

  group('no tools', () {
    test('a routed intent reports not-available and the turn still ends', () async {
      final Tokenizer tokenizer = ByteTokenizer();
      final ChatEngine engine = _engine(
        model: _StubModel(vocabSize: tokenizer.vocabSize),
        tokenizer: tokenizer,
        maxNewTokens: 8,
        contextLength: 400,
      );

      final List<ChatEvent> events = await engine
          .respond(<ChatMessage>[ChatMessage.user('is my battery charging')])
          .toList();

      final ChatToolFinished finished =
          events.whereType<ChatToolFinished>().single;
      expect(finished.result.ok, isFalse);
      expect(finished.result.error, contains('not available'));
      expect(events.whereType<ChatFailed>(), isEmpty);
      expect(events.last, isA<ChatFinished>());
    });
  });

  group('context limit', () {
    test('a prompt longer than the window still generates tokens', () async {
      final Tokenizer tokenizer = ByteTokenizer();
      final String long = 'question ' * 400;
      final ChatEngine engine = _engine(
        model: _StubModel(vocabSize: tokenizer.vocabSize),
        tokenizer: tokenizer,
        maxNewTokens: 100,
        minNewTokens: 1,
        contextLength: 32,
      );

      final List<ChatEvent> events = await engine
          .respond(<ChatMessage>[ChatMessage.user(long)])
          .toList();

      final ChatFinished finished = events.whereType<ChatFinished>().single;
      expect(finished.generatedTokens, greaterThanOrEqualTo(1));
      expect(finished.stopReason, anyOf('max_tokens', 'context_limit'));
    });
  });

  group('failure paths', () {
    test('an empty history yields a single ChatFailed', () async {
      final Tokenizer tokenizer = ByteTokenizer();
      final ChatEngine engine = _engine(
        model: _StubModel(vocabSize: tokenizer.vocabSize),
        tokenizer: tokenizer,
      );

      final List<ChatEvent> events =
          await engine.respond(const <ChatMessage>[]).toList();

      expect(events, hasLength(1));
      expect(events.single, isA<ChatFailed>());
      final ChatFailed failed = events.whereType<ChatFailed>().single;
      expect(failed.message, isNotEmpty);
    });

    test('a throwing tool does not crash the turn', () async {
      final Tokenizer tokenizer = ByteTokenizer();
      const String markup = '<tool name="test.boom">{}</tool>';
      final List<int> script = tokenizer.encode(markup);
      final ChatEngine engine = _engine(
        model: _StubModel(vocabSize: tokenizer.vocabSize, tokens: script),
        tokenizer: tokenizer,
        tools: ToolRegistry(<Tool>[_ThrowingTool('test.boom')]),
        maxNewTokens: script.length,
        contextLength: 400,
        routeIntents: false,
      );

      final List<ChatEvent> events = await engine
          .respond(<ChatMessage>[ChatMessage.user('boom please')])
          .toList();

      expect(events.whereType<ChatToolFinished>(), hasLength(1));
      expect(
        events.last,
        anyOf(isA<ChatFinished>(), isA<ChatFailed>()),
      );
    });
  });

  group('ChatMessage JSON', () {
    test('round-trips every role', () {
      final DateTime at = DateTime.utc(2024, 1, 2, 3, 4, 5);
      final List<ChatMessage> messages = <ChatMessage>[
        const ChatMessage(role: ChatRole.system, content: 'sys'),
        ChatMessage(role: ChatRole.user, content: 'usr', at: at),
        ChatMessage(role: ChatRole.assistant, content: 'asst'),
        const ChatMessage(
          role: ChatRole.tool,
          content: 'tool out',
          name: 'device.battery',
        ),
      ];

      for (final ChatMessage message in messages) {
        final ChatMessage restored = ChatMessage.fromJson(message.toJson());
        expect(restored.role, message.role);
        expect(restored.content, message.content);
        expect(restored.name, message.name);
        expect(restored.at, message.at);
      }
    });

    test('a message missing its role throws FormatException', () {
      expect(
        () => ChatMessage.fromJson(<String, Object?>{'content': 'hi'}),
        throwsFormatException,
      );
    });

    test('an unknown role throws FormatException', () {
      expect(
        () => ChatMessage.fromJson(
          <String, Object?>{'role': 'robot', 'content': 'hi'},
        ),
        throwsFormatException,
      );
    });
  });

  group('ToolProtocol', () {
    test('parse extracts the name of a well-formed block', () {
      final ToolCall? parsed = ToolProtocol.parse(
        'Sure. <tool name="device.battery">{}</tool>',
      );
      expect(parsed, isNotNull);
      final ToolCall call = parsed!;
      expect(call.name, 'device.battery');
      expect(call.arguments, isEmpty);
    });

    test('parse extracts JSON arguments', () {
      final ToolCall? parsed = ToolProtocol.parse(
        '<tool name="device.clipboard.write">{"text":"hi"}</tool>',
      );
      expect(parsed, isNotNull);
      final ToolCall call = parsed!;
      expect(call.name, 'device.clipboard.write');
      expect(call.arguments, <String, Object?>{'text': 'hi'});
    });

    test('parse returns null for prose', () {
      expect(
        ToolProtocol.parse('The sky looks blue because of scattering.'),
        isNull,
      );
    });

    test('stripCalls removes the block from the visible answer', () {
      final String stripped = ToolProtocol.stripCalls(
        'The answer follows.\n<tool name="device.battery">{}</tool>\nDone.',
      );
      expect(stripped, isNot(contains('<tool')));
      expect(stripped, contains('The answer follows.'));
      expect(stripped, contains('Done.'));
    });

    test('routeIntent ignores an explanation request', () {
      expect(ToolProtocol.routeIntent('why does the sky look blue'), isNull);
    });
  });
}

/// Builds an engine over [model] with deterministic greedy sampling.
ChatEngine _engine({
  required LanguageModelRuntime model,
  required Tokenizer tokenizer,
  ToolRegistry? tools,
  int maxNewTokens = 16,
  int minNewTokens = 4,
  int contextLength = 400,
  bool routeIntents = true,
  String systemPrompt = 'Be brief.',
}) {
  return ChatEngine(
    model: model,
    tokenizer: tokenizer,
    sampler: Sampler(temperature: 0),
    tools: tools,
    maxNewTokens: maxNewTokens,
    minNewTokens: minNewTokens,
    contextLength: contextLength,
    routeIntents: routeIntents,
    systemPrompt: systemPrompt,
  );
}

/// A tool whose description is long enough that the full catalogue overflows a
/// small window, and whose text carries the distinctive word `quasar`.
_RecordingTool _largeTool(String name) {
  return _RecordingTool(
    name: name,
    description: 'Reads the quasar calibration table and returns every '
        'annotated channel, provenance marker, confidence interval, drift '
        'estimate, and checksum the instrument recorded during the most '
        'recent survey pass over the same window.',
    parameters: <String, Object?>{
      'type': 'object',
      'properties': <String, Object?>{
        'channel': <String, Object?>{'type': 'string'},
        'window': <String, Object?>{'type': 'integer'},
        'precision': <String, Object?>{'type': 'number'},
        'includeDrift': <String, Object?>{'type': 'boolean'},
        'checksum': <String, Object?>{'type': 'string'},
      },
      'required': <String>['channel'],
    },
  );
}

/// A deterministic [LanguageModelRuntime] for driving the engine.
///
/// Three modes: favour a printable byte, favour EOS (with a printable byte as
/// runner-up so the engine's fallback writes visible text), or follow a fixed
/// list of token ids. [logitsForCalls] counts calls since the last
/// [resetCache], which pins down the incremental decode schedule.
class _StubModel implements LanguageModelRuntime {
  _StubModel({
    required this.vocabSize,
    bool alwaysEos = false,
    List<int>? tokens,
  })  : _alwaysEos = alwaysEos,
        _tokens = tokens == null ? null : List<int>.unmodifiable(tokens);

  @override
  final int vocabSize;

  final bool _alwaysEos;
  final List<int>? _tokens;
  int _step = 0;

  /// Number of `logitsFor` calls since the last [resetCache].
  int logitsForCalls = 0;

  /// Printable ASCII `A`, the fallback the engine should choose over a control id.
  static const int _printableByte = 0x41;

  @override
  int get contextLength => vocabSize;

  @override
  int get parameterCount => 0;

  @override
  List<double> logitsFor(List<int> prefix) {
    logitsForCalls++;
    final List<double> logits = List<double>.filled(vocabSize, 0.0);
    if (_alwaysEos) {
      if (_printableByte < vocabSize) {
        logits[_printableByte] = 5.0;
      }
      if (kEosId < vocabSize) {
        logits[kEosId] = 10.0;
      }
      return logits;
    }
    final List<int>? tokens = _tokens;
    if (tokens != null) {
      if (_step < tokens.length) {
        final int id = tokens[_step];
        _step++;
        if (id >= 0 && id < vocabSize) {
          logits[id] = 10.0;
        }
        return logits;
      }
      if (kEosId < vocabSize) {
        logits[kEosId] = 10.0;
      }
      return logits;
    }
    if (_printableByte < vocabSize) {
      logits[_printableByte] = 10.0;
    }
    return logits;
  }

  @override
  void resetCache() {
    _step = 0;
    logitsForCalls = 0;
  }
}

/// A tool that records whether it ran and echoes its payload.
class _RecordingTool extends FunctionTool {
  _RecordingTool({
    required String name,
    required String description,
    Map<String, Object?> parameters = const <String, Object?>{},
  })  : _name = name,
        _description = description,
        _parameters = parameters;

  final String _name;
  final String _description;
  final Map<String, Object?> _parameters;

  /// How many times [run] was entered.
  int runs = 0;

  /// The arguments of the most recent invocation.
  Map<String, Object?>? lastArgs;

  @override
  String get name => _name;

  @override
  String get description => _description;

  @override
  Map<String, Object?> get parameters => _parameters;

  @override
  bool get mutating => false;

  @override
  Future<Object?> run(Map<String, Object?> args) async {
    runs++;
    lastArgs = Map<String, Object?>.of(args);
    return <String, Object?>{'value': args['text'] ?? args['value'] ?? 'recorded'};
  }
}

/// A tool whose `invoke` throws, to prove the engine converts it to a result.
class _ThrowingTool extends Tool {
  _ThrowingTool(this.name);

  @override
  final String name;

  @override
  String get description => 'Always throws when invoked.';

  @override
  Map<String, Object?> get parameters => const <String, Object?>{
        'type': 'object',
      };

  @override
  bool get mutating => false;

  @override
  Future<ToolResult> invoke(Map<String, Object?> args) async {
    throw StateError('deliberate tool explosion');
  }
}