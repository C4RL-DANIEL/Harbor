// Building a chat engine from the user's capability switches.
//
// One function, because the same nine arguments must agree every time an engine
// is constructed: at launch, and again whenever the user flips memory, learning
// or reasoning in settings. Duplicating that call site is how a rebuilt engine
// ends up with a stale tool registry or a context length that no longer matches
// the model, and those bugs show up as nonsense answers rather than errors.

import 'package:harbor_core/harbor_core.dart';

import 'assistant_settings.dart';

/// Constructs the engine described by [settings].
///
/// [memory] is passed in every time and filtered here, so "memory off" is one
/// decision made in one place instead of a flag read at three call sites.
ChatEngine buildChatEngine({
  required AssistantSettings settings,
  required LanguageModelRuntime model,
  required Tokenizer tokenizer,
  required ToolRegistry? tools,
  required MemoryStore? memory,
  required int contextLength,
  int maxNewTokens = 96,
}) {
  return ChatEngine(
    model: model,
    tokenizer: tokenizer,
    tools: tools,
    memory: settings.memoryEnabled ? memory : null,
    extractor: MemoryExtractor(enabled: settings.learningEnabled),
    learnFromUser: settings.learningEnabled,
    maxNewTokens: maxNewTokens,
    contextLength: contextLength,
    // `autoThink` picks the depth per turn from the question itself, so a
    // greeting costs one pass and a multi-part request still gets checked;
    // pinning a mode instead makes every answer pay the same price. The pinned
    // strategy is only consulted when the automatic choice is off, so it is
    // passed as `none` while automatic mode is on rather than as a value the
    // engine would silently ignore.
    autoThink: settings.autoThinking,
    thinkingStrategy:
        settings.autoThinking ? ThinkingStrategy.none : settings.thinkingMode,
  );
}
