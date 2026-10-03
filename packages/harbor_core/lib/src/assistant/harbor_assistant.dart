// HarborAssistant: the one object an application has to build.
//
// Before this file, wiring the assistant meant constructing a tokenizer, a
// model, a tool registry, a `MemoryStore`, a `MemoryExtractor` and a
// `ChatEngine` in the right order, then keeping five of them in sync when the
// user flipped a switch. That is the kind of wiring that is easy to get subtly
// wrong — and a stale engine is exactly how "memory is on" ends up not being on.
//
// The assistant owns that graph and rebuilds the engine whenever a setting
// changes, so a caller talks to one object and toggles named capabilities:
//
//     final assistant = HarborAssistant(
//       model: model,
//       tokenizer: tokenizer,
//       tools: tools,
//       memoryStorage: storage,
//     );
//     await assistant.initialize();          // loads memory, applies defaults
//     await for (final event in assistant.respond(history)) { … }
//     await assistant.setThinking(ThinkingStrategy.thorough);
//     await assistant.flush();               // persists memory
//
// Everything here is `dart:io`-free and Flutter-free, so the same assistant runs
// on the phone, on the server and in the browser tab.

import 'dart:async';

import '../chat/chat_engine.dart';
import '../chat/chat_message.dart';
import '../memory/memory_entry.dart';
import '../memory/memory_extractor.dart';
import '../memory/memory_store.dart';
import '../model/interfaces.dart';
import '../reasoning/thinking.dart';
import '../tools/tool_registry.dart';

/// The capabilities the assistant can run with, all on by default.
///
/// "Automatic and easy to operate" is a default, not a mode: an app that
/// constructs an assistant and does nothing else already gets memory and
/// reasoning, and every switch below exists so a user can turn something *off*.
class AssistantCapabilities {
  /// Creates a capability set.
  const AssistantCapabilities({
    this.memory = true,
    this.learnFromConversation = true,
    this.thinking = true,
    this.tools = true,
  });

  /// Whether stored memories are recalled into each prompt.
  final bool memory;

  /// Whether user turns are mined for new memories.
  final bool learnFromConversation;

  /// Whether answers are planned and checked before they are shown.
  final bool thinking;

  /// Whether the tool catalogue is consulted.
  final bool tools;

  /// A copy with selected fields replaced.
  AssistantCapabilities copyWith({
    bool? memory,
    bool? learnFromConversation,
    bool? thinking,
    bool? tools,
  }) {
    return AssistantCapabilities(
      memory: memory ?? this.memory,
      learnFromConversation:
          learnFromConversation ?? this.learnFromConversation,
      thinking: thinking ?? this.thinking,
      tools: tools ?? this.tools,
    );
  }

  /// Diagnostics form.
  Map<String, Object?> toJson() => <String, Object?>{
        'memory': memory,
        'learn_from_conversation': learnFromConversation,
        'thinking': thinking,
        'tools': tools,
      };
}

/// What the assistant is currently doing, for a status card.
class AssistantStatus {
  /// Creates a status snapshot.
  const AssistantStatus({
    required this.capabilities,
    required this.memoryEntries,
    required this.memoryLoaded,
    required this.hasModel,
    required this.hasTools,
    required this.toolsAvailable,
    required this.lastThinkingTrace,
  });

  /// Which capabilities are on.
  final AssistantCapabilities capabilities;

  /// How many memories are stored.
  final int memoryEntries;

  /// Whether memory finished loading from storage.
  final bool memoryLoaded;

  /// Whether a model is attached.
  final bool hasModel;

  /// Whether a tool registry was supplied.
  final bool hasTools;

  /// How many tools the registry exposes.
  final int toolsAvailable;

  /// The trace of the most recent reasoned turn, if any.
  final ThinkingTrace? lastThinkingTrace;

  /// Diagnostics form.
  Map<String, Object?> toJson() => <String, Object?>{
        'capabilities': capabilities.toJson(),
        'memory_entries': memoryEntries,
        'memory_loaded': memoryLoaded,
        'has_model': hasModel,
        'tools_available': toolsAvailable,
        'last_thinking': lastThinkingTrace?.toJson(),
      };
}

/// Owns the model, memory, reasoning and tool graph behind one API.
class HarborAssistant {
  /// Creates an assistant.
  ///
  /// Nothing expensive happens until [initialize]: the memory store is not read
  /// and no engine is built, so a widget or a request handler may construct one
  /// without blocking. [memoryStorage] is optional; without it the assistant
  /// still remembers within the session, it just does not survive a restart.
  HarborAssistant({
    required this.model,
    required this.tokenizer,
    this.tools,
    MemoryStorage? memoryStorage,
    this.capabilities = const AssistantCapabilities(),
    this.systemPrompt = kDefaultSystemPrompt,
    this.maxNewTokens = 160,
    this.minNewTokens = 4,
    this.contextLength = 256,
    this.routeIntents = true,
    this.autoThink = true,
    this.thinking,
    this.memoryCapacity = 256,
    this.memoryRecallLimit = 6,
    this.sampler,
  })  : _memory = MemoryStore(
          storage: memoryStorage ?? InMemoryMemoryStorage(),
          capacity: memoryCapacity,
        ) {
    // Passing [thinking] pins the depth and therefore stops the per-turn
    // chooser; leaving it null keeps automatic mode with `thorough` as the
    // ceiling the chooser may spend up to.
    _autoThinkActive = autoThink && thinking == null;
    _thinkingCeiling = thinking ?? ThinkingStrategy.thorough;
  }

  /// The model every turn is generated with.
  final LanguageModelRuntime model;

  /// The tokenizer shared with training and prompt rendering.
  final Tokenizer tokenizer;

  /// Tools the assistant may call, if any.
  final ToolRegistry? tools;

  /// The base capability set.
  final AssistantCapabilities capabilities;

  /// The system instruction placed at the head of every prompt.
  final String systemPrompt;

  /// Cap on tokens generated per turn.
  final int maxNewTokens;

  /// Floor on tokens generated before end-of-sequence is accepted.
  final int minNewTokens;

  /// The model window, in tokens.
  final int contextLength;

  /// Whether the deterministic intent router runs before generation.
  final bool routeIntents;

  /// Whether the reasoning depth is chosen per turn.
  ///
  /// True by default: the point of the automatic mode is that a greeting costs
  /// nothing while a hard question still gets a verification pass, and the user
  /// never has to make that call. Ignored when [thinking] pins a depth.
  final bool autoThink;

  /// A reasoning depth pinned for every turn, disabling [autoThink].
  ///
  /// Null means "let the assistant choose per turn". Pass
  /// [ThinkingStrategy.none] to turn reasoning off from the start, which is what
  /// a caller that wants the pre-thinking behaviour must do — because
  /// reasoning is ON by default, and an opt-out has to be expressible.
  final ThinkingStrategy? thinking;

  /// Maximum memories kept before the least useful is pruned.
  final int memoryCapacity;

  /// How many memories may enter one prompt.
  final int memoryRecallLimit;

  /// The sampler, when the caller wants a specific one.
  final Sampler? sampler;

  /// Whether the per-turn depth chooser is currently active.
  ///
  /// Starts as the constructor's [autoThink]; pinning a depth with
  /// [setThinking] is an instruction to stop choosing, so it turns this off.
  late bool _autoThinkActive;

  /// The ceiling the engine reasons at.
  ///
  /// In automatic mode this is [ThinkingStrategy.thorough]: the chooser may
  /// spend up to a full plan on a hard question and nothing on a greeting.
  /// Pinned mode uses exactly what the caller chose.
  late ThinkingStrategy _thinkingCeiling;

  final MemoryStore _memory;
  final MemoryExtractor _extractor = MemoryExtractor();
  final ReasoningPlanner _planner = const ReasoningPlanner();

  ChatEngine? _engine;
  Future<void>? _initializing;
  bool _initialized = false;
  ThinkingTrace? _lastTrace;

  /// Initialises [capabilities] into [\_active] so it is non-null from the start.
  late AssistantCapabilities _mutableCapabilities = capabilities;

  /// The long-term memory.
  MemoryStore get memory => _memory;

  /// The capabilities currently in force.
  AssistantCapabilities get activeCapabilities => _mutableCapabilities;

  /// Whether [initialize] has completed.
  bool get initialized => _initialized;

  /// The reasoning ceiling currently in force.
  ThinkingStrategy get thinkingStrategy => _thinkingCeiling;

  /// Whether the depth is chosen per turn.
  bool get automaticThinking => _autoThinkActive;

  /// The trace of the most recent reasoned turn, if any.
  ThinkingTrace? get lastThinkingTrace => _lastTrace;

  /// Loads memory and builds the first engine.
  ///
  /// Idempotent and safe to call from several places: the future itself is the
  /// guard, so a widget's `initState` and a request handler can both await it.
  Future<void> initialize() => _initializing ??= _initialize();

  Future<void> _initialize() async {
    await _memory.load();
    _rebuildEngine();
    _initialized = true;
  }

  /// Streams one turn.
  ///
  /// The engine is rebuilt on demand if a settings change has not been applied
  /// yet, so a caller can toggle a capability and immediately `respond` without
  /// an explicit rebuild step in between.
  ///
  /// [useMemory] and [thinking] override the standing configuration for this
  /// turn only. Overriding is a parameter rather than a rebuild because a
  /// per-request opt-out — an API call that must not read the user's memory,
  /// say — must not mutate a shared, long-lived assistant that other turns may
  /// be using at the same moment.
  Stream<ChatEvent> respond(
    List<ChatMessage> history, {
    bool? useMemory,
    ThinkingStrategy? thinking,
    int? maxNewTokens,
  }) {
    final ChatEngine engine = _engine ?? _buildEngine();
    return _tapTrace(
      engine.respond(
        history,
        useMemory: useMemory,
        thinking: thinking,
        maxNewTokens: maxNewTokens,
      ),
    );
  }

  /// Remembers [text] because the user or the caller asked for it.
  ///
  /// Importance is maximal by default: an explicit "remember this" must survive
  /// pruning far longer than an inferred topic.
  MemoryEntry remember(
    String text, {
    MemoryKind kind = MemoryKind.fact,
    String? subject,
    double importance = 1.0,
    String? source,
  }) {
    return _memory.remember(
      text,
      kind: kind,
      subject: subject,
      importance: importance,
      source: source ?? 'explicit',
    );
  }

  /// Forgets one memory, in the session and on disk.
  Future<bool> forget(String id) async {
    final bool removed = _memory.forget(id);
    if (removed) {
      await _memory.flush();
    }
    return removed;
  }

  /// Recalls memories relevant to [query].
  List<MemoryMatch> recall(String query, {int limit = 8}) =>
      _memory.recall(query, limit: limit);

  /// The memories worth injecting at the start of a session.
  MemoryRecall bootstrapBlock() => _memory.bootstrapBlock();

  /// Deletes every memory.
  Future<void> clearMemory() => _memory.clear();

  /// Persists memory now.
  Future<void> flush() => _memory.flush();

  /// Turns memory recall on or off.
  void setMemoryEnabled(bool enabled) {
    _mutableCapabilities = _mutableCapabilities.copyWith(memory: enabled);
    _rebuildEngine();
  }

  /// Turns learning from conversation on or off.
  void setLearningEnabled(bool enabled) {
    _extractor.enabled = enabled;
    _mutableCapabilities =
        _mutableCapabilities.copyWith(learnFromConversation: enabled);
    _rebuildEngine();
  }

  /// Pins the reasoning depth, disabling the per-turn choice.
  ///
  /// Passing [ThinkingStrategy.none] is how a caller turns thinking off; any
  /// other value pins that depth but leaves [autoThink]'s *choice* disabled, so
  /// the value passed is exactly what runs.
  /// Pins the depth, which also stops the per-turn chooser overriding it.
  ///
  /// [ThinkingStrategy.none] is how a caller turns reasoning off.
  void setThinking(ThinkingStrategy strategy) {
    _autoThinkActive = false;
    _thinkingCeiling = strategy;
    _mutableCapabilities = _mutableCapabilities.copyWith(
      thinking: strategy.isEnabled,
    );
    _rebuildEngine();
  }

  /// Turns the automatic reasoning depth back on.
  ///
  /// [autoThink] is a constructor field because it is a build-time policy, so
  /// "re-enable automatic" means: thinking on, depth chosen per turn.
  void setAutomaticThinking({bool enabled = true}) {
    _autoThinkActive = enabled;
    _thinkingCeiling = ThinkingStrategy.thorough;
    _mutableCapabilities = _mutableCapabilities.copyWith(thinking: enabled);
    _rebuildEngine();
  }

  /// Turns the tool catalogue on or off.
  void setToolsEnabled(bool enabled) {
    _mutableCapabilities = _mutableCapabilities.copyWith(tools: enabled);
    _rebuildEngine();
  }

  /// A snapshot for a status card or a health endpoint.
  AssistantStatus status() {
    return AssistantStatus(
      capabilities: _mutableCapabilities,
      memoryEntries: _memory.length,
      memoryLoaded: _memory.loaded,
      hasModel: true,
      hasTools: tools != null,
      toolsAvailable: tools?.length ?? 0,
      lastThinkingTrace: _lastTrace ?? _engine?.lastTrace,
    );
  }

  /// Releases the memory store's write-behind timer by flushing it.
  ///
  /// There is no `dispose` on a portable object with no owned streams; flushing
  /// is the only thing that must not be lost when a host shuts down.
  Future<void> dispose() => _memory.flush();

  /// Rebuilds the memoised engine from the current capabilities.
  void _rebuildEngine() {
    _engine = _buildEngine();
  }

  /// Builds an engine for the current capabilities.
  ChatEngine _buildEngine() {
    final AssistantCapabilities caps = _mutableCapabilities;
    return ChatEngine(
      model: model,
      tokenizer: tokenizer,
      sampler: sampler,
      tools: caps.tools ? tools : null,
      memory: caps.memory ? _memory : null,
      extractor: _extractor,
      learnFromUser: caps.learnFromConversation,
      systemPrompt: systemPrompt,
      maxNewTokens: maxNewTokens,
      minNewTokens: minNewTokens,
      contextLength: contextLength,
      routeIntents: routeIntents,
      // `thinking: true` means "reason"; how deeply is either the pinned
      // ceiling or the per-turn choice, which is what makes the switch a
      // one-tap decision rather than a second menu. The pinned strategy stays
      // the gate: `none` means off even in automatic mode, which is what makes
      // "off" mean off.
      thinkingStrategy: caps.thinking ? _thinkingCeiling : ThinkingStrategy.none,
      autoThink: caps.thinking && _autoThinkActive,
      planner: _planner,
      memoryRecallLimit: memoryRecallLimit,
    );
  }

  /// Records the engine's trace on the assistant as turns complete.
  ///
  /// The engine already exposes `lastTrace`, but a caller that holds the
  /// assistant rather than the engine should not have to reach through it.
  Stream<ChatEvent> _tapTrace(Stream<ChatEvent> events) async* {
    await for (final ChatEvent event in events) {
      if (event is ChatThinking && event.trace != null) {
        _lastTrace = event.trace;
      }
      yield event;
    }
  }
}