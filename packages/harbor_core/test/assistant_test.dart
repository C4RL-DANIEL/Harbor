// Tests for HarborAssistant: memory recall, reasoning and the
// capability toggles that drive the Settings switches.
//
// Uses a local deterministic stub model so every event type that
// the assistant can emit is exercised without a real tokenizer.

import 'dart:async';

import 'package:harbor_core/harbor_core.dart';
import 'package:test/test.dart';

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
  int logitsForCalls = 0;
  static const int _printableByte = 0x41;

  @override int get contextLength => vocabSize;
  @override int get parameterCount => 0;

  @override
  List<double> logitsFor(List<int> prefix) {
    logitsForCalls++;
    final List<double> logits = List<double>.filled(vocabSize, 0.0);
    if (_alwaysEos) {
      if (_printableByte < vocabSize) logits[_printableByte] = 5.0;
      if (kEosId < vocabSize) logits[kEosId] = 10.0;
      return logits;
    }
    final List<int>? tokens = _tokens;
    if (tokens != null) {
      if (_step < tokens.length) {
        final int id = tokens[_step++];
        if (id >= 0 && id < vocabSize) logits[id] = 10.0;
        return logits;
      }
      if (kEosId < vocabSize) logits[kEosId] = 10.0;
      return logits;
    }
    if (_printableByte < vocabSize) logits[_printableByte] = 10.0;
    return logits;
  }

  @override @override void resetCache() { _step = 0; logitsForCalls = 0; }
  int quickTokenCount(String text) => text.length;
  Future<void> initialize() async {}
  Future<void> dispose() async {}
}

void main() {
  group('HarborAssistant', () {
    test('initializes with full capabilities', () async {
      final HarborAssistant assistant = HarborAssistant(
        model: _StubModel(vocabSize: 260),
        tokenizer: ByteTokenizer(),
        tools: ToolRegistry(),
      );
      addTearDown(assistant.dispose);
      await assistant.initialize();
      expect(assistant.initialized, isTrue);
      expect(assistant.activeCapabilities.memory, isTrue);
      expect(assistant.activeCapabilities.learnFromConversation, isTrue);
      expect(assistant.activeCapabilities.thinking, isTrue);
    });

    test('remember / recall / forget / clear through the assistant', () async {
      final HarborAssistant assistant = HarborAssistant(
        model: _StubModel(vocabSize: 260),
        tokenizer: ByteTokenizer(),
      );
      addTearDown(assistant.dispose);
      await assistant.initialize();
      final MemoryEntry entry = assistant.remember('My staging server is at 10.0.0.5');
      expect(assistant.memory.length, 1);
      final List<MemoryMatch> hits = assistant.recall('where is my staging server');
      expect(hits.isNotEmpty, isTrue);
      expect(hits.first.entry.id, entry.id);
      final bool removed = await assistant.forget(entry.id);
      expect(removed, isTrue);
      expect(assistant.memory.length, 0);
      await assistant.clearMemory();
      expect(assistant.memory.length, 0);
    });

    test('setMemoryEnabled toggles memory', () async {
      final HarborAssistant assistant = HarborAssistant(
        model: _StubModel(vocabSize: 260),
        tokenizer: ByteTokenizer(),
      );
      addTearDown(assistant.dispose);
      await assistant.initialize();
      expect(assistant.activeCapabilities.memory, isTrue);
      assistant.setMemoryEnabled(false);
      expect(assistant.activeCapabilities.memory, isFalse);
      assistant.setMemoryEnabled(true);
      expect(assistant.activeCapabilities.memory, isTrue);
    });

    test('setLearningEnabled toggles learning', () async {
      final HarborAssistant assistant = HarborAssistant(
        model: _StubModel(vocabSize: 260),
        tokenizer: ByteTokenizer(),
      );
      addTearDown(assistant.dispose);
      await assistant.initialize();
      assistant.setLearningEnabled(false);
      expect(assistant.activeCapabilities.learnFromConversation, isFalse);
      assistant.setLearningEnabled(true);
      expect(assistant.activeCapabilities.learnFromConversation, isTrue);
    });

    test('setThinking pins depth and stops auto choice', () async {
      final HarborAssistant assistant = HarborAssistant(
        model: _StubModel(vocabSize: 260),
        tokenizer: ByteTokenizer(),
      );
      addTearDown(assistant.dispose);
      await assistant.initialize();
      expect(assistant.automaticThinking, isTrue);
      expect(assistant.thinkingStrategy, ThinkingStrategy.thorough);
      assistant.setThinking(ThinkingStrategy.concise);
      expect(assistant.automaticThinking, isFalse);
      expect(assistant.thinkingStrategy, ThinkingStrategy.concise);
      expect(assistant.activeCapabilities.thinking, isTrue);
      assistant.setThinking(ThinkingStrategy.none);
      expect(assistant.activeCapabilities.thinking, isFalse);
    });

    test('setAutomaticThinking restores per-turn depth selection', () async {
      final HarborAssistant assistant = HarborAssistant(
        model: _StubModel(vocabSize: 260),
        tokenizer: ByteTokenizer(),
      );
      addTearDown(assistant.dispose);
      await assistant.initialize();
      assistant.setThinking(ThinkingStrategy.none);
      assistant.setAutomaticThinking(enabled: true);
      expect(assistant.automaticThinking, isTrue);
      expect(assistant.thinkingStrategy, ThinkingStrategy.thorough);
    });

    test('respond emits thinking frames then tokens then finished', () async {
      final HarborAssistant assistant = HarborAssistant(
        model: _StubModel(vocabSize: 260),
        tokenizer: ByteTokenizer(),
      );
      addTearDown(assistant.dispose);
      await assistant.initialize();
      final List<ChatEvent> events = await assistant
          .respond(<ChatMessage>[ChatMessage.user('What is the capital of France?')])
          .toList();
      expect(events.whereType<ChatThinking>(), isNotEmpty);
      expect(events.whereType<ChatToken>(), isNotEmpty);
      expect(events.whereType<ChatFinished>(), isNotEmpty);
    });

    test('per-turn overrides work', () async {
      final HarborAssistant assistant = HarborAssistant(
        model: _StubModel(vocabSize: 260),
        tokenizer: ByteTokenizer(),
      );
      addTearDown(assistant.dispose);
      await assistant.initialize();
      final List<ChatEvent> noMemory = await assistant
          .respond(<ChatMessage>[ChatMessage.user('hi')], useMemory: false, thinking: ThinkingStrategy.none)
          .toList();
      expect(noMemory.whereType<ChatThinking>(), isEmpty);
      expect(noMemory.whereType<ChatFinished>(), isNotEmpty);
    });

    test('status reports memory count and thinking state', () async {
      final HarborAssistant assistant = HarborAssistant(
        model: _StubModel(vocabSize: 260),
        tokenizer: ByteTokenizer(),
      );
      addTearDown(assistant.dispose);
      await assistant.initialize();
      final AssistantStatus status = assistant.status();
      expect(status.memoryEntries, 0);
      expect(status.memoryLoaded, isTrue);
      expect(status.hasModel, isTrue);
      expect(status.capabilities.thinking, isTrue);
    });
  });
}
