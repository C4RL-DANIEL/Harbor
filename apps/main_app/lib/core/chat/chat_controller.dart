// Chat state for the UI.
//
// A turn is consumed event by event so the screen can show three distinct states
// that a `Future<String>` would collapse into one: text arriving, a tool
// running, and the answer settling. The controller therefore exposes the
// in-progress text separately from the committed messages, because a
// half-generated sentence is not yet something the user may reply to.

import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:harbor_core/harbor_core.dart';

import '../memory/harbor_memory_storage.dart';

/// Drives a [ChatEngine] and holds the transcript.
///
/// The controller also owns the two things a turn now produces beyond text: the
/// reasoning trace, which the screen renders as a collapsible card, and the
/// memories the turn taught the assistant, which are surfaced so learning is
/// never silent.
class ChatController extends ChangeNotifier {
  /// Creates a controller.
  ChatController({
    required ChatEngine engine,
    required this.tokenizer,
    this.memory,
    this.transcript,
    this.maxMessages = 60,
  }) : _engine = engine;

  /// The engine that produces answers.
  ///
  /// Replaced wholesale by [replaceEngine] when a capability switch changes.
  /// [ChatEngine]'s configuration is immutable on purpose — a turn must not
  /// change shape halfway through — so "turn memory off" means "build the next
  /// engine without a memory store", and the controller owns that swap so no
  /// screen has to know how an engine is assembled.
  ChatEngine get engine => _engine;

  /// Swaps in a newly configured engine.
  ///
  /// The transcript is untouched: changing a capability must not look to the
  /// user like the conversation was thrown away. A swap requested mid-turn is
  /// deferred, because applying a new prompt shape between two tokens of the
  /// same answer would make the turn impossible to reproduce.
  void replaceEngine(ChatEngine engine) {
    if (_busy) {
      _pendingEngine = engine;
      return;
    }
    _engine = engine;
    _pendingEngine = null;
    notifyListeners();
  }

  /// The tokenizer, for the prompt preview and token counts.
  final Tokenizer tokenizer;

  /// The assistant's long-term memory, when one is attached.
  ///
  /// The engine already reads and writes this store during a turn; the
  /// controller holds the same reference so the UI can list, forget and flush
  /// without reaching through the engine.
  final MemoryStore? memory;

  /// Where the transcript is persisted between launches, when configured.
  final ChatTranscriptStore? transcript;

  /// How many messages to keep in the visible transcript.
  ///
  /// Bounded because the whole transcript is re-rendered into the prompt every
  /// turn: an unbounded history would make each reply slower than the last and
  /// eventually exceed the context window on its own.
  final int maxMessages;

  final List<ChatMessage> _messages = <ChatMessage>[];
  final StringBuffer _streaming = StringBuffer();
  final List<MemoryEntry> _remembered = <MemoryEntry>[];
  ChatEngine _engine;
  ChatEngine? _pendingEngine;
  StreamSubscription<ChatEvent>? _subscription;
  ToolCall? _activeTool;
  bool _busy = false;
  String? _error;
  String? _stopReason;
  int _generatedTokens = 0;
  Duration _lastTurnDuration = Duration.zero;
  ThinkingPlan? _thinkingPlan;
  ThinkingTrace? _thinkingTrace;
  bool _loadedTranscript = false;

  /// The committed transcript.
  List<ChatMessage> get messages => List<ChatMessage>.unmodifiable(_messages);

  /// The plan for the current or most recent reasoned turn.
  ThinkingPlan? get thinkingPlan => _thinkingPlan;

  /// The completed trace of the most recent reasoned turn.
  ThinkingTrace? get thinkingTrace => _thinkingTrace;

  /// Memories learned since the transcript was last cleared.
  List<MemoryEntry> get remembered => List<MemoryEntry>.unmodifiable(_remembered);

  /// Whether a reasoning trace is available for the current or last turn.
  bool get hasThinking => _thinkingPlan != null;

  /// Text produced so far in the current turn.
  String get streamingText => _streaming.toString();

  /// Whether a turn is in flight.
  bool get busy => _busy;

  /// The last failure, if any.
  String? get error => _error;

  /// Why the last turn stopped (`eos`, `max_tokens`, `context_limit`).
  String? get stopReason => _stopReason;

  /// Tokens produced by the last turn.
  int get generatedTokens => _generatedTokens;

  /// Duration of the last turn.
  Duration get lastTurnDuration => _lastTurnDuration;

  /// The tool currently running, if any.
  ToolCall? get activeTool => _activeTool;

  /// What the model would read for the current transcript.
  String get renderedPrompt => engine.renderPrompt(_messages);

  /// Restores the persisted transcript once, so a relaunch continues the
  /// conversation instead of starting from an empty bubble.
  ///
  /// Idempotent and safe to call from a widget's `initState`: the guard exists
  /// because a rebuild can re-enter it while the first read is still in flight.
  Future<void> restore() async {
    final ChatTranscriptStore? store = transcript;
    if (store == null || _loadedTranscript) {
      return;
    }
    _loadedTranscript = true;
    final List<ChatMessage> saved = await store.load();
    if (saved.isEmpty) {
      return;
    }
    _messages
      ..clear()
      ..addAll(saved);
    _trim();
    notifyListeners();
  }

  /// Adds a greeting so the screen is never empty on first open.
  void seedGreeting(String text) {
    if (_messages.isEmpty) {
      _messages.add(ChatMessage.assistant(text));
      notifyListeners();
    }
  }

  /// Sends [text] and streams the answer.
  Future<void> send(String text) async {
    final String trimmed = text.trim();
    if (trimmed.isEmpty || _busy) {
      return;
    }
    _messages.add(ChatMessage.user(trimmed));
    _trim();
    _streaming.clear();
    _error = null;
    _stopReason = null;
    _generatedTokens = 0;
    _activeTool = null;
    _thinkingPlan = null;
    _thinkingTrace = null;
    _busy = true;
    notifyListeners();

    final Completer<void> done = Completer<void>();
    _subscription = engine.respond(_messages).listen(
      _onEvent,
      onError: (Object error) {
        _error = '$error';
        if (!done.isCompleted) {
          done.complete();
        }
      },
      onDone: () {
        if (!done.isCompleted) {
          done.complete();
        }
      },
    );
    await done.future;
    await _subscription?.cancel();
    _subscription = null;
    _busy = false;
    _activeTool = null;
    final ChatEngine? pending = _pendingEngine;
    if (pending != null) {
      _engine = pending;
      _pendingEngine = null;
    }
    _persist();
    notifyListeners();
  }

  /// Cancels an in-flight turn.
  void stop() {
    _subscription?.cancel();
    _subscription = null;
    if (_streaming.isNotEmpty) {
      _messages.add(ChatMessage.assistant(_streaming.toString()));
      _streaming.clear();
    }
    _busy = false;
    _activeTool = null;
    _stopReason = 'cancelled';
    _persist();
    notifyListeners();
  }

  /// Clears the visible transcript, deliberately keeping the long-term memory.
  ///
  /// "Clear chat" is a request to forget *this conversation*, not to forget who
  /// the user is; conflating the two is how assistants become annoying to reset.
  /// [clearMemory] is the separate, explicit action.
  void clear() {
    _messages.clear();
    _streaming.clear();
    _error = null;
    _stopReason = null;
    _generatedTokens = 0;
    _thinkingPlan = null;
    _thinkingTrace = null;
    _remembered.clear();
    _persist();
    notifyListeners();
  }

  /// Forgets every stored memory and clears what was learned this session.
  Future<void> clearMemory() async {
    await memory?.clear();
    _remembered.clear();
    notifyListeners();
  }

  /// Forgets one memory by id.
  Future<bool> forgetMemory(String id) async {
    final bool removed = memory?.forget(id) ?? false;
    if (removed) {
      _remembered.removeWhere((MemoryEntry e) => e.id == id);
      unawaited(memory?.flush() ?? Future<void>.value());
      notifyListeners();
    }
    return removed;
  }

  /// Remembers [text] on the user's behalf.
  void remember(String text, {MemoryKind kind = MemoryKind.fact}) {
    final MemoryStore? store = memory;
    if (store == null) {
      return;
    }
    _remembered.add(store.remember(text, kind: kind, source: 'user command'));
    unawaited(store.flush());
    notifyListeners();
  }

  /// Writes the transcript and memory to storage now, e.g. on app pause.
  Future<void> flush() async {
    await transcript?.save(_messages);
    await memory?.flush();
  }

  /// Replaces the transcript, e.g. after loading a stored conversation.
  void replaceAll(List<ChatMessage> messages) {
    _messages
      ..clear()
      ..addAll(messages);
    _loadedTranscript = true;
    _trim();
    _persist();
    notifyListeners();
  }

  void _onEvent(ChatEvent event) {
    switch (event) {
      case ChatToken(:final String text):
        _streaming.write(text);
      case ChatToolStarted(:final ToolCall call):
        _activeTool = call;
      case ChatToolFinished(:final ToolCall call, :final ToolResult result):
        _activeTool = null;
        _messages.add(ChatMessage.tool(call.name, result.summary));
      case ChatThinking(:final ThinkingPlan plan, :final ThinkingTrace? trace):
        _thinkingPlan = plan;
        if (trace != null) {
          _thinkingTrace = trace;
        }
      case ChatMemoryUpdated(:final List<MemoryEntry> entries):
        for (final MemoryEntry entry in entries) {
          if (!_remembered.any((MemoryEntry e) => e.id == entry.id)) {
            _remembered.add(entry);
          }
        }
      case ChatFinished(
          :final String text,
          :final int generatedTokens,
          :final Duration elapsed,
          :final String? stopReason,
        ):
        _generatedTokens = generatedTokens;
        _lastTurnDuration = elapsed;
        _stopReason = stopReason;
        if (text.isNotEmpty) {
          _messages.add(ChatMessage.assistant(text));
        }
        _streaming.clear();
      case ChatFailed(:final String message):
        _error = message;
        if (_streaming.isNotEmpty) {
          _messages.add(ChatMessage.assistant(_streaming.toString()));
          _streaming.clear();
        }
    }
    _trim();
    notifyListeners();
  }

  void _trim() {
    while (_messages.length > maxMessages) {
      _messages.removeAt(0);
    }
  }

  /// Kick-off a debounced transcript save.
  ///
  /// Saves are throttled rather than done per message because a long chat
  /// session would otherwise write the whole transcript to storage on every
  /// bubble, which is the kind of I/O that shows up as a stutter on a phone.
  void _persist() {
    final ChatTranscriptStore? store = transcript;
    if (store == null) {
      return;
    }
    store.scheduleSave(_messages);
  }

  @override
  void dispose() {
    unawaited(_subscription?.cancel());
    super.dispose();
  }
}
