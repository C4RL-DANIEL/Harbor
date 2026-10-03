// The chat engine: one turn of conversation, as a stream.
//
// The engine is a stream rather than a `Future<String>` because a turn has
// visible internal structure the UI must show: tokens arriving one at a time,
// and a tool call happening in the middle (which on a phone can take a second
// and must not look like a freeze). Modelling the turn as a stream means the
// waiting state is a *rendered* state instead of an invisible one.
//
// It is also the only place that knows the priority order between the two tool
// mechanisms: the deterministic intent router runs first, because it is the one
// that is always right when it fires.
//
// Since this engine owns the generation loop, it is also where memory and
// reasoning live. Remembering a stated preference and checking a draft against
// the question are things a large model is trained to do implicitly; a model
// this small is not, so the engine does them explicitly and reports each step as
// an event the UI can render.

import 'dart:async';

import '../memory/memory_entry.dart';
import '../memory/memory_extractor.dart';
import '../memory/memory_store.dart';
import '../model/interfaces.dart';
import '../reasoning/thinking.dart';
import '../tools/tool.dart';
import '../tools/tool_registry.dart';
import 'chat_message.dart';
import 'tool_protocol.dart';

/// Something that happened during a turn.
sealed class ChatEvent {
  /// Const constructor for subclasses.
  const ChatEvent();
}

/// A piece of generated text.
class ChatToken extends ChatEvent {
  /// Creates a token event.
  const ChatToken(this.text);

  /// The text fragment, already decoded to a string.
  final String text;
}

/// A tool call is about to run.
class ChatToolStarted extends ChatEvent {
  /// Creates a start event.
  const ChatToolStarted(this.call);

  /// The call being run.
  final ToolCall call;
}

/// A tool call finished.
class ChatToolFinished extends ChatEvent {
  /// Creates a finish event.
  const ChatToolFinished(this.call, this.result);

  /// The call that ran.
  final ToolCall call;

  /// What it returned.
  final ToolResult result;
}

/// The assistant planned, or completed, a reasoning pass.
///
/// Emitted once with the plan as soon as it is formed (so the trace appears
/// while the answer is still generating) and once more with the finished
/// [ThinkingTrace] once the draft has been checked. A consumer that only wants
/// the final state can ignore every event whose `trace` is null.
class ChatThinking extends ChatEvent {
  /// Creates a thinking event.
  const ChatThinking({required this.plan, this.trace});

  /// The plan for this turn.
  final ThinkingPlan plan;

  /// The completed trace, once the turn has been verified.
  final ThinkingTrace? trace;
}

/// One or more new memories were written during this turn.
class ChatMemoryUpdated extends ChatEvent {
  /// Creates a memory event.
  const ChatMemoryUpdated(this.entries, {required this.explicit});

  /// The memories stored or refreshed.
  final List<MemoryEntry> entries;

  /// Whether the user explicitly asked for them to be remembered.
  final bool explicit;
}

/// The turn ended normally.
class ChatFinished extends ChatEvent {
  /// Creates a finish event.
  const ChatFinished({
    required this.text,
    required this.generatedTokens,
    required this.elapsed,
    this.stopReason,
  });

  /// The assistant's answer, with any tool markup removed.
  final String text;

  /// Number of tokens the model produced.
  final int generatedTokens;

  /// Wall-clock duration of the turn.
  final Duration elapsed;

  /// Why generation stopped: `eos`, `max_tokens`, or `context_limit`.
  final String? stopReason;
}

/// The turn ended because something went wrong.
class ChatFailed extends ChatEvent {
  /// Creates a failure event.
  const ChatFailed(this.message);

  /// Human-readable reason.
  final String message;
}

/// Turns a conversation into a stream of [ChatEvent]s.
class ChatEngine {
  /// Creates an engine over [model] and [tokenizer].
  ///
  /// [tools] is optional so the dashboard can run a model with no host at all;
  /// when it is null a routed intent still produces a visible "not available"
  /// tool result, because hiding the attempt would make the UI look broken.
  ///
  /// [memory] is optional for the same reason: without it the engine behaves
  /// exactly as it did before memory existed. [thinkingStrategy] defaults to
  /// [ThinkingStrategy.none] and [autoThink] defaults to false, so a caller that
  /// knows nothing about reasoning keeps the original single-pass behaviour.
  ChatEngine({
    required this.model,
    required this.tokenizer,
    Sampler? sampler,
    this.tools,
    this.memory,
    this.extractor,
    this.maxNewTokens = 160,
    this.minNewTokens = 4,
    this.contextLength = 256,
    this.routeIntents = true,
    this.systemPrompt = kDefaultSystemPrompt,
    this.thinkingStrategy = ThinkingStrategy.none,
    this.autoThink = false,
    this.planner = const ReasoningPlanner(),
    this.memoryRecallLimit = 6,
    this.learnFromUser = true,
  }) : sampler = sampler ?? Sampler();

  /// The model being driven.
  final LanguageModelRuntime model;

  /// The tokenizer shared with training.
  final Tokenizer tokenizer;

  /// Sampling parameters.
  final Sampler sampler;

  /// Tools the assistant may reach, if any.
  final ToolRegistry? tools;

  /// The long-term memory consulted before every turn, if any.
  final MemoryStore? memory;

  /// The extractor that turns user text into memories.
  ///
  /// When null and [memory] is set, a default enabled extractor is used; when
  /// [learnFromUser] is false nothing is ever written.
  final MemoryExtractor? extractor;

  /// Cap on tokens generated per turn.
  final int maxNewTokens;

  /// The fewest tokens to generate before end-of-sequence is allowed.
  ///
  /// A small or freshly initialised model frequently puts its highest
  /// probability on the end-of-sequence token, which produces a completely empty
  /// reply that looks like a crash. Forcing at least a few real tokens turns that
  /// failure into visible, judgeable (if poor) output, so the user sees that the
  /// model has not learned much rather than seeing nothing at all.
  final int minNewTokens;

  /// Cap on the rendered prompt length, in tokens.
  final int contextLength;

  /// The number of prompt tokens that may be spent, after reserving room to
  /// answer.
  ///
  /// Without the reservation a prompt can fill the whole window, and the
  /// generation loop then stops on its first check with `context_limit` and an
  /// empty reply — which reads as a crash rather than as "the conversation is
  /// too long". Half the window is reserved, capped by [maxNewTokens], so a
  /// caller that sets `maxNewTokens: 8` keeps most of the window for context and
  /// a caller that asks for 160 on a 128-token model still gets 64 tokens of
  /// history.
  int get promptBudget => promptBudgetFor(maxNewTokens);

  /// The number of prompt tokens available when a turn generates [cap] tokens.
  ///
  /// A per-turn override must move the reservation with it: answering the same
  /// conversation with a 512-token budget on a 512-token window needs the
  /// history squeezed far harder than a 32-token answer does, and using the
  /// engine's own [promptBudget] for an overridden cap would let the prompt
  /// fill the window and stop the turn on `context_limit` with nothing written.
  int promptBudgetFor(int cap) {
    final int reserve = cap.clamp(1, contextLength ~/ 2);
    return (contextLength - reserve).clamp(1, contextLength);
  }

  /// Whether the deterministic intent router is consulted.
  final bool routeIntents;

  /// The system instruction placed at the head of every prompt.
  final String systemPrompt;

  /// The reasoning depth used when [autoThink] is false.
  ///
  /// `none` means no thinking events and no extra generation pass.
  final ThinkingStrategy thinkingStrategy;

  /// Whether the engine chooses the reasoning depth per turn.
  ///
  /// When set *and* [thinkingStrategy] is not [ThinkingStrategy.none],
  /// [ReasoningPlanner.suggest] inspects the question and picks between a single
  /// check and the pinned depth's plan. It cannot turn reasoning on: a caller
  /// that pinned `none` gets no reasoning pass even for a hard question, which
  /// is what makes "off" mean off.
  final bool autoThink;

  /// The planner and verifier used when thinking is enabled.
  final ReasoningPlanner planner;

  /// How many memories may be injected into one prompt.
  final int memoryRecallLimit;

  /// Whether user turns are mined for memories.
  final bool learnFromUser;

  /// Whether the most recent turn invoked a tool.
  bool _lastTurnUsedTool = false;

  /// Whether the most recent turn invoked a tool.
  bool get lastTurnUsedTool => _lastTurnUsedTool;

  /// The trace of the most recent reasoned turn, if any.
  ThinkingTrace? _lastTrace;

  /// The trace of the most recent reasoned turn, if any.
  ThinkingTrace? get lastTrace => _lastTrace;

  /// Runs one turn over [history] and streams what happens.
  ///
  /// Never throws: an internal failure arrives as [ChatFailed] so a caller has
  /// exactly one place to handle errors.
  ///
  /// The optional overrides exist so one engine can serve requests that want a
  /// different budget, a different reasoning depth, or no memory at all, without
  /// the caller rebuilding an engine per request. [maxNewTokens] also moves the
  /// prompt reservation, via [promptBudgetFor], because a longer answer needs a
  /// shorter prompt. A null override means "use this engine's configured value".
  Stream<ChatEvent> respond(
    List<ChatMessage> history, {
    int? maxNewTokens,
    ThinkingStrategy? thinking,
    bool? useMemory,
  }) async* {
    final int tokenCap = maxNewTokens ?? this.maxNewTokens;
    final bool recallMemory = useMemory ?? (memory != null);
    final Stopwatch stopwatch = Stopwatch()..start();
    _lastTurnUsedTool = false;
    _lastTrace = null;
    try {
      if (history.isEmpty) {
        yield const ChatFailed('there is nothing to answer');
        return;
      }
      final List<ChatMessage> messages = List<ChatMessage>.of(history);

      if (routeIntents) {
        final ChatMessage? lastUser = _lastUserMessage(messages);
        final ToolCall? routed =
            lastUser == null ? null : ToolProtocol.routeIntent(lastUser.content);
        if (routed != null) {
          _lastTurnUsedTool = true;
          yield ChatToolStarted(routed);
          final ToolResult result = await _invoke(routed);
          yield ChatToolFinished(routed, result);
          messages.add(ChatMessage.tool(routed.name, result.summary));
        }
      }

      final ChatMessage? lastUser = _lastUserMessage(messages);

      // ---- Memory recall -------------------------------------------------
      // Inserted immediately before the newest user turn so the trimming in
      // `_render` — which never drops that turn — keeps the memory with it.
      final MemoryRecall recall =
          !recallMemory || memory == null || lastUser == null
          ? MemoryRecall.empty
          : memory!.recallBlock(
              lastUser.content,
              limit: memoryRecallLimit,
            );
      if (recall.block.isNotEmpty) {
        final int insertAt = _indexBeforeLastUser(messages);
        messages.insert(
          insertAt,
          ChatMessage.system('What you remember about this user:\n${recall.block}'),
        );
      }

      // ---- Reasoning plan ------------------------------------------------
      final ThinkingStrategy strategy =
          thinking ?? _resolveStrategy(lastUser?.content ?? '');
      final ThinkingPlan? plan = strategy.isEnabled && lastUser != null
          ? planner.plan(
              lastUser.content,
              strategy: strategy,
              hasTools: (tools?.length ?? 0) > 0,
              hasMemory: recall.block.isNotEmpty,
            )
          : null;
      final Stopwatch thinkingClock = Stopwatch()..start();
      if (plan != null) {
        yield ChatThinking(plan: plan);
      }

      // ---- Draft ---------------------------------------------------------
      _Generation? draft;
      await for (final _GenerationEvent event in _runGeneration(
        messages,
        maxNewTokens: tokenCap,
      )) {
        switch (event) {
          case _TokenPiece(:final String text):
            yield ChatToken(text);
          case _GenerationDone(:final _Generation generation):
            draft = generation;
        }
      }
      if (draft == null) {
        yield const ChatFailed('the model produced no result');
        return;
      }

      final String draftRaw = draft.text;
      final ToolCall? emitted =
          tools == null ? null : ToolProtocol.parse(draftRaw);
      final String draftText = ToolProtocol.stripCalls(draftRaw);

      if (emitted != null) {
        _lastTurnUsedTool = true;
        yield ChatToolStarted(emitted);
        final ToolResult result = await _invoke(emitted);
        yield ChatToolFinished(emitted, result);
        final String answer =
            draftText.isEmpty ? result.summary : draftText;

        if (plan != null) {
          thinkingClock.stop();
          final ThinkingTrace trace = _traceFor(
            plan: plan,
            question: lastUser?.content ?? '',
            draft: answer,
            finalText: answer,
            revisions: 0,
            elapsed: thinkingClock.elapsed,
          );
          _lastTrace = trace;
          yield ChatThinking(plan: plan, trace: trace);
        }
        final MemoryExtraction? learned = await _learn(lastUser, answer);
        if (learned != null) {
          yield ChatMemoryUpdated(learned.entries, explicit: learned.explicit);
        }
        stopwatch.stop();
        yield ChatFinished(
          // A model that emits nothing but a call has, in effect, produced the
          // tool's answer as its answer; showing an empty bubble instead would
          // look like a bug.
          text: answer,
          generatedTokens: draft.tokens,
          elapsed: stopwatch.elapsed,
          stopReason: '${draft.stopReason} (tool)',
        );
        return;
      }

      // ---- Verify and, at most once, repair ------------------------------
      String finalText = draftText;
      int revisions = 0;
      VerificationResult? verification;
      if (plan != null) {
        verification = planner.verify(
          question: lastUser?.content ?? '',
          draft: draftText,
          usedTool: _lastTurnUsedTool,
          toolNames: <String>{
            for (final Tool tool in tools?.tools ?? const <Tool>[]) tool.name,
          },
        );
        if (verification.needsRevision) {
          final String repairInstruction = planner.repairPrompt(
            question: lastUser?.content ?? '',
            draft: draftText,
            findings: verification.findings,
          );
          final List<ChatMessage> repairHistory = <ChatMessage>[
            ...messages,
            ChatMessage.assistant(draftText),
            ChatMessage.user(repairInstruction),
          ];
          final _Generation repaired =
              await _collect(repairHistory, maxNewTokens: tokenCap);
          final String repairedText = ToolProtocol.stripCalls(repaired.text);
          // Only accept the repair when it is actually better. A model this
          // small can easily answer the repair prompt worse than the original,
          // and replacing a flawed answer with a worse one helps nobody.
          if (repairedText.trim().isNotEmpty &&
              !_looksWorse(repairedText, draftText)) {
            finalText = repairedText;
            revisions = 1;
          }
        }
        thinkingClock.stop();
        final ThinkingTrace trace = _traceFor(
          plan: plan,
          question: lastUser?.content ?? '',
          draft: draftText,
          finalText: finalText,
          revisions: revisions,
          elapsed: thinkingClock.elapsed,
          findings: verification.findings,
          verdict: verification.verdict,
        );
        _lastTrace = trace;
        yield ChatThinking(plan: plan, trace: trace);
      }

      final MemoryExtraction? learned = await _learn(lastUser, finalText);
      if (learned != null) {
        yield ChatMemoryUpdated(learned.entries, explicit: learned.explicit);
      }

      stopwatch.stop();
      yield ChatFinished(
        text: finalText,
        generatedTokens: draft.tokens,
        elapsed: stopwatch.elapsed,
        stopReason: draft.stopReason,
      );
    } on Object catch (error) {
      yield ChatFailed('$error');
    }
  }

    /// The reasoning depth for a question.
  ///
  /// [autoThink] chooses *how deeply* to think; it never turns thinking on by
  /// itself. The pinned [thinkingStrategy] stays authoritative as the gate: when
  /// it is `none`, the turn does not reason no matter how complex the question
  /// is. Reading `autoThink` first (as an earlier revision did) made the two
  /// controls entangle — a caller could not ask for "off by default, automatic
  /// when enabled", which is the only configuration the settings UI can express.
  ThinkingStrategy _resolveStrategy(String question) {
    if (thinkingStrategy == ThinkingStrategy.none) {
      return ThinkingStrategy.none;
    }
    if (!autoThink) {
      return thinkingStrategy;
    }
    final ThinkingStrategy suggested = ReasoningPlanner.suggest(question);
    // Never exceed the pinned ceiling: `concise` means the caller bounded the
    // cost, and silently escalating to `thorough` would break that promise.
    if (suggested.index > thinkingStrategy.index) {
      return thinkingStrategy;
    }
    // A trivial greeting still earns no reasoning pass.
    return suggested;
  }

  /// Mines [user] (and, only when asked, the answer) for memories.
  ///
  /// Runs at the end of a turn so a memory is stored against text the user
  /// actually sent, never against a half-generated draft. Returns the extraction
  /// (or null) so the caller can report it as a [ChatMemoryUpdated] event.
  Future<MemoryExtraction?> _learn(ChatMessage? user, String answer) async {
    final MemoryStore? store = memory;
    if (store == null || !learnFromUser || user == null) {
      return null;
    }
    final MemoryExtractor miner = extractor ?? MemoryExtractor();
    final MemoryExtraction extraction = miner.extract(user.content);
    if (extraction.isEmpty) {
      return null;
    }
    for (final MemoryEntry entry in extraction.entries) {
      store.add(entry);
    }
    return extraction;
  }

  /// Whether [candidate] is a worse answer than [original].
  ///
  /// The two failure modes worth rejecting are a candidate that is much shorter
  /// (the model answered the repair instruction with a fragment) and one that is
  /// degenerate. A candidate that is merely different is accepted, because the
  /// check that asked for it found a real fault.
  static bool _looksWorse(String candidate, String original) {
    final String trimmed = candidate.trim();
    if (trimmed.isEmpty) {
      return true;
    }
    if (original.trim().length >= 24 &&
        trimmed.length < original.trim().length ~/ 3) {
      return true;
    }
    return ReasoningPlanner.isDegenerate(trimmed);
  }

  /// Builds the finished trace from a plan and a verification outcome.
  ThinkingTrace _traceFor({
    required ThinkingPlan plan,
    required String question,
    required String draft,
    required String finalText,
    required int revisions,
    required Duration elapsed,
    List<ReasoningFinding> findings = const <ReasoningFinding>[],
    VerificationVerdict verdict = VerificationVerdict.ok,
  }) {
    // Advance every step to done and record the outcome on the steps that have
    // one, so the rendered trace reflects what actually happened rather than
    // what was merely planned.
    final List<ThinkStep> steps = <ThinkStep>[
      for (final ThinkStep step in plan.steps)
        step.copyWith(
          status: ThinkStepStatus.done,
          result: _resultFor(step, findings),
        ),
    ];
    return ThinkingTrace(
      plan: plan.withSteps(steps),
      verdict: verdict,
      findings: findings,
      checkedDraft: _prefix(draft),
      finalText: _prefix(finalText),
      revisions: revisions,
      elapsed: elapsed,
    );
  }

  /// The one-line result shown for [step] in the trace.
  String? _resultFor(ThinkStep step, List<ReasoningFinding> findings) {
    final String title = step.title.toLowerCase();
    if (title.contains('check') || title.contains('sanity')) {
      return findings.isEmpty
          ? 'No problems found.'
          : findings.map((ReasoningFinding f) => f.message).join(' ');
    }
    if (title.contains('revise')) {
      return findings.isEmpty ? 'No revision needed.' : 'Revised the draft.';
    }
    return null;
  }

  static String _prefix(String text, [int limit = 160]) {
    final String cleaned = text.trim().replaceAll(RegExp(r'\s+'), ' ');
    return cleaned.length <= limit ? cleaned : '${cleaned.substring(0, limit)}…';
  }

  /// The highest-scoring token id that is not a control token.
  ///
  /// Control ids (begin, end, pad) carry no text, so emitting one would either
  /// stop the turn or write nothing. Every other id decodes to at least one
  /// byte, which is what makes this a safe fallback.
  int _bestNonControl(List<double> logits) {
    int best = -1;
    double bestScore = double.negativeInfinity;
    for (int id = 0; id < logits.length; id++) {
      if (id == tokenizer.bosId ||
          id == tokenizer.eosId ||
          id == tokenizer.padId) {
        continue;
      }
      if (logits[id] > bestScore) {
        bestScore = logits[id];
        best = id;
      }
    }
    // A vocabulary of nothing but control ids is impossible for a byte-level
    // tokenizer, but falling back to zero keeps this total rather than throwing
    // from the middle of a stream.
    return best < 0 ? 0 : best;
  }

  /// Runs the generation loop, streaming each piece and finally the result.
  ///
  /// Yielding the finished [_Generation] as the last event (rather than
  /// returning it) is what lets one implementation serve both the streamed draft
  /// — whose tokens the caller forwards to the UI — and the silent repair pass,
  /// which only wants the text.
  Stream<_GenerationEvent> _runGeneration(
    List<ChatMessage> messages, {
    int? maxNewTokens,
  }) async* {
    final int cap = maxNewTokens ?? this.maxNewTokens;
    final List<int> encoded =
        tokenizer.encode(renderPrompt(messages, maxNewTokens: cap));
    final int budget = promptBudgetFor(cap);
    final List<int> prefix = encoded.length > budget
        ? encoded.sublist(encoded.length - budget)
        : List<int>.of(encoded);
    if (prefix.isEmpty) {
      yield const _GenerationDone(
        _Generation(text: '', tokens: 0, stopReason: 'empty_prompt'),
      );
      return;
    }

    model.resetCache();
    final List<int> generated = <int>[];
    final StringBuffer text = StringBuffer();
    String? stopReason;
    while (generated.length < cap) {
      if (prefix.length >= contextLength) {
        stopReason = 'context_limit';
        break;
      }
      final List<double> logits = model.logitsFor(prefix);
      int next = sampler.pick(logits, generated: generated);
      if (next == tokenizer.eosId) {
        if (generated.length >= minNewTokens) {
          stopReason = 'eos';
          break;
        }
        // The sampler insisted on stopping far too early. Take the best token
        // that is not a control token instead, so the turn still produces
        // something the user can judge.
        next = _bestNonControl(logits);
      }
      generated.add(next);
      prefix.add(next);
      final String piece = tokenizer.decode(<int>[next]);
      text.write(piece);
      yield _TokenPiece(piece);
    }
    stopReason ??= 'max_tokens';
    yield _GenerationDone(
      _Generation(
        text: text.toString(),
        tokens: generated.length,
        stopReason: stopReason,
      ),
    );
  }

  /// Runs a generation pass and returns only its text.
  Future<_Generation> _collect(
    List<ChatMessage> messages, {
    int? maxNewTokens,
  }) async {
    _Generation? result;
    await for (final _GenerationEvent event in _runGeneration(
      messages,
      maxNewTokens: maxNewTokens,
    )) {
      if (event is _GenerationDone) {
        result = event.generation;
      }
    }
    return result ??
        const _Generation(text: '', tokens: 0, stopReason: 'empty_prompt');
  }

  /// Renders [history] into the flat transcript the model reads.
  ///
  /// Deterministic by construction: two calls with the same history produce byte
  /// identical prompts, which is what makes a generation reproducible and a bug
  /// report meaningful.
  String renderPrompt(List<ChatMessage> history, {int? maxNewTokens}) {
    final int cap = maxNewTokens ?? this.maxNewTokens;
    // Fewer tool details are better than a truncated conversation: the model
    // reads the last tokens of the prompt, so an oversized catalogue would push
    // the actual messages out of the window. The ladder degrades from the full
    // schema to names only and finally to no catalogue at all, and the first
    // variant that fits is the one the model (and the prompt preview) sees.
    String cheapest = '';
    for (final String catalogue in _catalogueLadder()) {
      final String candidate = _render(history, catalogue, budget: promptBudgetFor(cap));
      cheapest = candidate;
      if (tokenizer.encode(candidate).length <= promptBudgetFor(cap)) {
        return candidate;
      }
    }
    return cheapest;
  }

  /// Tool descriptions from most informative to least, ending with none.
  Iterable<String> _catalogueLadder() sync* {
    final ToolRegistry? registry = tools;
    if (registry == null || registry.length == 0) {
      yield '';
      return;
    }
    yield ToolProtocol.describeTools(registry.tools);
    yield registry.tools.map((Tool tool) => '- ${tool.name}').join('\n');
    yield '';
  }

  /// Renders [history] with a specific [toolCatalogue].
  String _render(
    List<ChatMessage> history,
    String toolCatalogue, {
    required int budget,
  }) {
    final StringBuffer buffer = StringBuffer()
      ..writeln('System: $systemPrompt');
    if (toolCatalogue.isNotEmpty) {
      buffer
        ..writeln('Tools:')
        ..writeln(toolCatalogue)
        ..writeln();
    }

    final List<String> lines = <String>[];
    for (final ChatMessage message in history) {
      switch (message.role) {
        case ChatRole.system:
          lines.add('System: ${message.content}');
        case ChatRole.user:
          lines.add('User: ${message.content}');
        case ChatRole.assistant:
          lines.add('Assistant: ${message.content}');
        case ChatRole.tool:
          lines.add('Tool(${message.name ?? 'unknown'}): ${message.content}');
      }
    }

    // Drop whole messages from the left until the transcript fits the budget.
    // Trimming by message rather than by token keeps the result readable, and the
    // system line is never dropped because losing the instructions
    // mid-conversation changes the model's behaviour completely.
    //
    // The newest user turn is never dropped either. An earlier revision trimmed
    // purely by length, so on a small window it removed the question and kept the
    // older answers — leaving the model to continue a conversation whose subject
    // it could no longer see, which is why every reply looked like fluent
    // nonsense.
    final int lastUser =
        history.lastIndexWhere((ChatMessage m) => m.role == ChatRole.user);
    int keepFrom = 0;
    String candidate = _assemble(buffer, lines);
    while (tokenizer.encode(candidate).length > budget) {
      if (lastUser < 0 || keepFrom >= lastUser) {
        break;
      }
      keepFrom++;
      candidate = _assemble(buffer, lines.sublist(keepFrom));
    }
    return candidate;
  }

  /// Joins the system preamble and the transcript into the final prompt.
  static String _assemble(StringBuffer preamble, List<String> lines) {
    return '$preamble${lines.join('\n')}\nAssistant:';
  }

  Future<ToolResult> _invoke(ToolCall call) async {
    final ToolRegistry? registry = tools;
    if (registry == null) {
      return ToolResult.unsupported(
        '${call.name} is not available on this platform',
      );
    }
    try {
      return await registry.invoke(call);
    } on Object catch (error) {
      return ToolResult.failure('${call.name} failed: $error');
    }
  }

  static ChatMessage? _lastUserMessage(List<ChatMessage> messages) {
    for (int i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role == ChatRole.user) {
        return messages[i];
      }
    }
    return null;
  }

  /// The index immediately before the newest user message.
  ///
  /// A memory block inserted here sits as close as possible to the question
  /// while still rendering before it, which matters because the model reads the
  /// end of the prompt most strongly.
  static int _indexBeforeLastUser(List<ChatMessage> messages) {
    for (int i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role == ChatRole.user) {
        return i;
      }
    }
    return messages.length;
  }
}

/// One generated pass: its text, token count and stop reason.
class _Generation {
  /// Creates a generation result.
  const _Generation({
    required this.text,
    required this.tokens,
    required this.stopReason,
  });

  /// The raw text produced, before tool markup is stripped.
  final String text;

  /// Number of tokens produced.
  final int tokens;

  /// Why the pass stopped.
  final String stopReason;
}

/// An event from the generation loop, private to this file.
sealed class _GenerationEvent {
  const _GenerationEvent();
}

/// A token produced by the generation loop.
class _TokenPiece extends _GenerationEvent {
  const _TokenPiece(this.text);

  final String text;
}

/// The terminal event of a generation pass.
class _GenerationDone extends _GenerationEvent {
  const _GenerationDone(this.generation);

  final _Generation generation;
}