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

import 'dart:async';

import '../model/interfaces.dart';
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
  ChatEngine({
    required this.model,
    required this.tokenizer,
    Sampler? sampler,
    this.tools,
    this.maxNewTokens = 160,
    this.minNewTokens = 4,
    this.contextLength = 256,
    this.routeIntents = true,
    this.systemPrompt = kDefaultSystemPrompt,
  }) : sampler = sampler ?? Sampler();

  /// The model being driven.
  final LanguageModelRuntime model;

  /// The tokenizer shared with training.
  final Tokenizer tokenizer;

  /// Sampling parameters.
  final Sampler sampler;

  /// Tools the assistant may reach, if any.
  final ToolRegistry? tools;

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
  int get promptBudget {
    final int reserve = maxNewTokens.clamp(1, contextLength ~/ 2);
    return (contextLength - reserve).clamp(1, contextLength);
  }

  /// Whether the deterministic intent router is consulted.
  final bool routeIntents;

  /// The system instruction placed at the head of every prompt.
  final String systemPrompt;

  /// Whether the most recent turn invoked a tool.
  bool _lastTurnUsedTool = false;

  /// Whether the most recent turn invoked a tool.
  bool get lastTurnUsedTool => _lastTurnUsedTool;

  /// Runs one turn over [history] and streams what happens.
  ///
  /// Never throws: an internal failure arrives as [ChatFailed] so a caller has
  /// exactly one place to handle errors.
  Stream<ChatEvent> respond(List<ChatMessage> history) async* {
    final Stopwatch stopwatch = Stopwatch()..start();
    _lastTurnUsedTool = false;
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

      final List<int> encoded = tokenizer.encode(renderPrompt(messages));
      final int budget = promptBudget;
      final List<int> prefix = encoded.length > budget
          ? encoded.sublist(encoded.length - budget)
          : List<int>.of(encoded);
      if (prefix.isEmpty) {
        yield const ChatFailed('the tokenizer produced an empty prompt');
        return;
      }

      model.resetCache();
      final List<int> generated = <int>[];
      final StringBuffer text = StringBuffer();
      String? stopReason;
      while (generated.length < maxNewTokens) {
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
        yield ChatToken(piece);
      }
      stopReason ??= 'max_tokens';

      final String raw = text.toString();
      final ToolCall? emitted = ToolProtocol.parse(raw);
      if (emitted != null && tools != null) {
        _lastTurnUsedTool = true;
        yield ChatToolStarted(emitted);
        final ToolResult result = await _invoke(emitted);
        yield ChatToolFinished(emitted, result);
        final String stripped = ToolProtocol.stripCalls(raw);
        stopwatch.stop();
        yield ChatFinished(
          // A model that emits nothing but a call has, in effect, produced the
          // tool's answer as its answer; showing an empty bubble instead would
          // look like a bug.
          text: stripped.isEmpty ? result.summary : stripped,
          generatedTokens: generated.length,
          elapsed: stopwatch.elapsed,
          stopReason: '$stopReason (tool)',
        );
        return;
      }

      stopwatch.stop();
      yield ChatFinished(
        text: ToolProtocol.stripCalls(raw),
        generatedTokens: generated.length,
        elapsed: stopwatch.elapsed,
        stopReason: stopReason,
      );
    } on Object catch (error) {
      yield ChatFailed('$error');
    }
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

  /// Renders [history] into the flat transcript the model reads.
  ///
  /// Deterministic by construction: two calls with the same history produce byte
  /// identical prompts, which is what makes a generation reproducible and a bug
  /// report meaningful.
  String renderPrompt(List<ChatMessage> history) {
    // Fewer tool details are better than a truncated conversation: the model
    // reads the last tokens of the prompt, so an oversized catalogue would push
    // the actual messages out of the window. The ladder degrades from the full
    // schema to names only and finally to no catalogue at all, and the first
    // variant that fits is the one the model (and the prompt preview) sees.
    String cheapest = '';
    for (final String catalogue in _catalogueLadder()) {
      final String candidate = _render(history, catalogue);
      cheapest = candidate;
      if (tokenizer.encode(candidate).length <= promptBudget) {
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
  String _render(List<ChatMessage> history, String toolCatalogue) {
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

    // Drop whole messages from the left until the transcript fits. Trimming by
    // message rather than by token keeps the result readable, and the system
    // line is never dropped because losing the instructions mid-conversation
    // changes the model's behaviour completely.
    String candidate = '$buffer${lines.join('\n')}\nAssistant:';
    while (lines.length > 1 &&
        tokenizer.encode(candidate).length > contextLength) {
      lines.removeAt(0);
      candidate = '$buffer${lines.join('\n')}\nAssistant:';
    }
    return candidate;
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
}