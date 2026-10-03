// File-backed persistence for memory and the chat transcript.
//
// `harbor_core` defines the storage seams (`MemoryStorage`) and deliberately
// knows nothing about the file system, because the same code also runs in a
// browser tab. Everything that touches `dart:io` for memory lives here, so the
// portable package stays portable by construction rather than by discipline —
// the same split `corpus_persistence.dart` uses for the corpus.

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:harbor_core/harbor_core.dart';

/// A [MemoryStorage] backed by a single UTF-8 JSON file.
///
/// Writes go to a sibling temp file and are renamed over the target, so a crash
/// mid-write leaves the previous memory intact rather than a truncated document
/// that a tolerant loader would quietly accept as a smaller one.
class FileMemoryStorage implements MemoryStorage {
  /// Creates storage over [file].
  FileMemoryStorage(this.file);

  /// The file the memory is written to.
  final File file;

  @override
  Future<String?> read() async {
    try {
      if (!await file.exists()) {
        return null;
      }
      return file.readAsString();
    } on Object catch (error) {
      // Unreadable storage is recoverable: the assistant starts with an empty
      // memory and the next successful write replaces the broken file. Throwing
      // here would take the whole app down for one corrupt document.
      debugPrint('[harbor] memory read failed: $error');
      return null;
    }
  }

  @override
  Future<void> write(String payload) async {
    await file.parent.create(recursive: true);
    final File temporary = File('${file.path}.tmp');
    await temporary.writeAsString(payload, flush: true);
    await temporary.rename(file.path);
  }

  @override
  Future<void> delete() async {
    try {
      if (await file.exists()) {
        await file.delete();
      }
    } on Object catch (error) {
      debugPrint('[harbor] memory delete failed: $error');
    }
  }
}

/// Persists the visible transcript so a relaunch continues the conversation.
///
/// Separate from [FileMemoryStorage] on purpose: the transcript is ephemeral
/// chat history the user can wipe with "Clear chat", while memory is what the
/// assistant keeps. One file each means one can be deleted without silently
/// destroying the other.
class ChatTranscriptStore {
  /// Creates a store over [file].
  ChatTranscriptStore(this.file, {this.keepMessages = 60});

  /// The JSON-Lines file the transcript is written to.
  final File file;

  /// How many messages survive a restart.
  ///
  /// Bounded to what the model can actually attend over: storing two thousand
  /// messages that are never rendered again is storage with no purpose, and the
  /// prompt rebuild would pay for them on every turn.
  final int keepMessages;

  Timer? _debounce;
  bool _writing = false;
  bool _againWhenDone = false;

  /// Reads the saved transcript, oldest first.
  Future<List<ChatMessage>> load() async {
    try {
      if (!await file.exists()) {
        return const <ChatMessage>[];
      }
      final List<String> lines = await file.readAsLines();
      final List<ChatMessage> messages = <ChatMessage>[];
      for (final String line in lines) {
        if (line.trim().isEmpty) {
          continue;
        }
        final Object? decoded = jsonDecode(line);
        if (decoded is! Map<String, Object?>) {
          continue;
        }
        try {
          messages.add(ChatMessage.fromJson(decoded));
        } on FormatException {
          // A single unparseable message must not cost the whole history.
          continue;
        }
      }
      if (messages.length <= keepMessages) {
        return messages;
      }
      return messages.sublist(messages.length - keepMessages);
    } on Object catch (error) {
      debugPrint('[harbor] transcript read failed: $error');
      return const <ChatMessage>[];
    }
  }

  /// Schedules a debounced save.
  ///
  /// Coalesced because a chat session commits a message every few seconds while
  /// the transcript itself is re-written whole; the debounce turns a burst of
  /// updates into one write and the in-flight flag guarantees the *last* state
  /// is the one that ends up on disk.
  void scheduleSave(List<ChatMessage> messages,
      {Duration delay = const Duration(milliseconds: 600)}) {
    _debounce?.cancel();
    _debounce = Timer(delay, () => unawaited(save(messages)));
  }

  /// Writes [messages] now, as JSON Lines.
  Future<void> save(List<ChatMessage> messages) async {
    if (_writing) {
      _againWhenDone = true;
      return;
    }
    _writing = true;
    List<ChatMessage> current = messages;
    try {
      do {
        _againWhenDone = false;
        await _writeOnce(current);
      } while (_againWhenDone);
    } on Object catch (error) {
      debugPrint('[harbor] transcript write failed: $error');
    } finally {
      _writing = false;
    }
  }

  Future<void> _writeOnce(List<ChatMessage> messages) async {
    await file.parent.create(recursive: true);
    final List<ChatMessage> tail = messages.length <= keepMessages
        ? messages
        : messages.sublist(messages.length - keepMessages);
    final StringBuffer buffer = StringBuffer();
    for (final ChatMessage message in tail) {
      buffer.writeln(jsonEncode(message.toJson()));
    }
    final File temporary = File('${file.path}.tmp');
    await temporary.writeAsString(buffer.toString(), flush: true);
    await temporary.rename(file.path);
  }

  /// Drops the saved transcript.
  Future<void> clear() async {
    _debounce?.cancel();
    try {
      if (await file.exists()) {
        await file.delete();
      }
    } on Object catch (error) {
      debugPrint('[harbor] transcript clear failed: $error');
    }
  }

  /// Releases the pending debounce. Call from a controller's `dispose`.
  void dispose() => _debounce?.cancel();
}