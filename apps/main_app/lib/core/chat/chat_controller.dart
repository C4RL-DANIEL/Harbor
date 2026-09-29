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

/// Drives a [ChatEngine] and holds the transcript.
class ChatController extends ChangeNotifier {
  /// Creates a controller.
  ChatController({
    required this.engine,
    required this.tokenizer,
    this.maxMessages = 60,
  });

  /// The engine that produces answers.
  final ChatEngine engine;

  /// The tokenizer, for the prompt preview and token counts.
  final Tokenizer tokenizer;

  /// How many messages to keep in the visible transcript.
  ///
  /// Bounded because the whole transcript is re-rendered into the prompt every
  /// turn: an unbounded history would make each reply slower than the last and
  /// eventually exceed the context window on its own.
  final int maxMessages;

  final List<ChatMessage> _messages = <ChatMessage>[];
  final StringBuffer _streaming = StringBuffer();
  StreamSubscription<ChatEvent>? _subscription;
  ToolCall? _activeTool;
  bool _busy = false;
  String? _error;
  String? _stopReason;
  int _generatedTokens = 0;
  Duration _lastTurnDuration = Duration.zero;

  /// The committed transcript.
  List<ChatMessage> get messages => List<ChatMessage>.unmodifiable(_messages);

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
    notifyListeners();
  }

  /// Clears the transcript.
  void clear() {
    _messages.clear();
    _streaming.clear();
    _error = null;
    _stopReason = null;
    _generatedTokens = 0;
    notifyListeners();
  }

  /// Replaces the transcript, e.g. after loading a stored conversation.
  void replaceAll(List<ChatMessage> messages) {
    _messages
      ..clear()
      ..addAll(messages);
    _trim();
    notifyListeners();
  }

  void _onEvent(ChatEvent event) {
    switch (event) {
      case ChatToken(:final String text):
        _streaming.write(text);
      case ChatToolStarted(:final ToolCall call):
        _activeTool = call;
        _messages.add(
          ChatMessage.assistant('Running ${call.name}…'),
        );
      case ChatToolFinished(:final ToolCall call, :final ToolResult result):
        _activeTool = null;
        _messages.add(ChatMessage.tool(call.name, result.summary));
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

  @override
  void dispose() {
    unawaited(_subscription?.cancel());
    super.dispose();
  }
}