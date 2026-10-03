// The assistant's switches: memory, learning and reasoning.
//
// These three flags are the whole difference between a tool that needs a
// manual and one that just works. They default to *on*: an assistant that
// forgets everything until you find a settings page is not an assistant with a
// feature, it is an assistant with a chore. The page exists so a person can
// turn things off — and the state survives a restart because the choices are
// theirs, not the build's.
//
// The class deliberately knows nothing about how an engine is assembled: it
// persists four values and fires [onChange], and the composition root decides
// what rebuilding means. That keeps this testable without a model.

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:harbor_core/harbor_core.dart';

/// Persistent assistant capability switches.
class AssistantSettings extends ChangeNotifier {
  /// Creates settings persisted to [file].
  AssistantSettings({required this.file});

  /// The JSON file holding the choices.
  final File file;

  /// Whether stored memories are recalled into every prompt.
  bool memoryEnabled = true;

  /// Whether user messages are mined for new memories.
  bool learningEnabled = true;

  /// Whether the reasoning depth is chosen per turn.
  ///
  /// True means automatic: a greeting gets no overhead and a hard question gets
  /// a verification pass, with nothing for the user to configure.
  bool autoThinking = true;

  /// The pinned depth used when [autoThinking] is off.
  ThinkingStrategy thinkingMode = ThinkingStrategy.thorough;

  /// Fired after any value changes, before the write lands.
  ///
  /// The composition root sets this to rebuild the engine, so a toggle takes
  /// effect on the very next turn instead of the next app launch.
  void Function()? onChange;

  Timer? _saveDebounce;

  /// Whether the assistant should currently be reasoning at all.
  bool get thinkingEnabled => autoThinking || thinkingMode.isEnabled;

  /// Reads the stored choices. Never throws: a corrupt file means defaults,
  /// because losing an assistant setting to one bad byte is not a trade anyone
  /// would choose.
  Future<void> load() async {
    try {
      if (!await file.exists()) {
        return;
      }
      final Object? decoded = jsonDecode(await file.readAsString());
      if (decoded is! Map<String, Object?>) {
        return;
      }
      memoryEnabled = _boolOf(decoded['memory'], memoryEnabled);
      learningEnabled = _boolOf(decoded['learning'], learningEnabled);
      autoThinking = _boolOf(decoded['auto_thinking'], autoThinking);
      final Object? mode = decoded['thinking_mode'];
      if (mode is String) {
        thinkingMode = ThinkingStrategy.fromWire(mode);
      }
    } on Object catch (error) {
      debugPrint('[harbor] assistant settings read failed: $error');
    }
  }

  /// Writes the current choices, debounced so a burst of toggles is one write.
  void save() {
    _saveDebounce?.cancel();
    _saveDebounce = Timer(const Duration(milliseconds: 300), _write);
  }

  Future<void> _write() async {
    try {
      await file.parent.create(recursive: true);
      final File temporary = File('${file.path}.tmp');
      await temporary.writeAsString(
        const JsonEncoder.withIndent('  ').convert(toJson()),
        flush: true,
      );
      await temporary.rename(file.path);
    } on Object catch (error) {
      debugPrint('[harbor] assistant settings write failed: $error');
    }
  }

  /// JSON form of the four values.
  Map<String, Object?> toJson() => <String, Object?>{
        'memory': memoryEnabled,
        'learning': learningEnabled,
        'auto_thinking': autoThinking,
        'thinking_mode': thinkingMode.wire,
      };

  /// Changes memory recall.
  void setMemoryEnabled(bool value) {
    if (memoryEnabled == value) {
      return;
    }
    memoryEnabled = value;
    _changed();
  }

  /// Changes automatic learning from conversation.
  void setLearningEnabled(bool value) {
    if (learningEnabled == value) {
      return;
    }
    learningEnabled = value;
    _changed();
  }

  /// Changes per-turn depth selection.
  void setAutoThinking(bool value) {
    if (autoThinking == value) {
      return;
    }
    autoThinking = value;
    _changed();
  }

  /// Pins a reasoning depth and turns the automatic choice off.
  ///
  /// Passing a real mode always disables `autoThinking`: setting the depth by
  /// hand *is* the request to stop choosing automatically. [ThinkingStrategy.none]
  /// disables reasoning entirely, and the automatic flag is cleared with it so
  /// the engine cannot silently re-enable it on the next complex question.
  void setThinkingMode(ThinkingStrategy mode) {
    if (thinkingMode == mode && !autoThinking) {
      return;
    }
    thinkingMode = mode;
    autoThinking = false;
    _changed();
  }

  /// Restores full-automatic reasoning.
  void setFullyAutomatic() {
    autoThinking = true;
    thinkingMode = ThinkingStrategy.none;
    _changed();
  }

  void _changed() {
    save();
    onChange?.call();
    notifyListeners();
  }

  static bool _boolOf(Object? value, bool fallback) =>
      value is bool ? value : fallback;

  @override
  void dispose() {
    _saveDebounce?.cancel();
    super.dispose();
  }
}