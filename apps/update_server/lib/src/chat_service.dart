// Chat service for the Harbor update server.
//
// Harbor Core already runs in three places: the Android AOT image, the browser
// bundle and this VM. The update server is the one that is always on and has a
// CPU to spare, so hosting the same runtime here gives the admin dashboard a
// chat endpoint without a second model implementation.
//
// The model is deliberately, honestly *untrained*. No pretrained checkpoint
// ships with the repository, so the service learns a byte-level tokenizer from
// an optional seed corpus and reports `stage: untrained` rather than dressing a
// randomly initialised transformer up as an assistant.

import 'dart:math' as math;

import 'package:harbor_core/harbor_core.dart';
import 'package:http/http.dart' as http;

/// Number of messages one `/api/v1/chat` request may carry.
///
/// The prompt is re-encoded on every turn and the model window is tiny, so a
/// long transcript buys nothing and only makes the endpoint a cheap way to spend
/// CPU. Twenty-four turns is far more history than any useful prompt survives.
const int kChatMessageLimit = 24;

/// Tokens generated per turn when the request omits `max_tokens`.
///
/// Matches [ChatEngine]'s own default so the streaming and buffered paths agree
/// about a request that says nothing.
const int kChatDefaultMaxTokens = 160;

/// Hard ceiling on a requested `max_tokens`.
///
/// A client cannot ask the server to generate an unbounded reply; the cap is why
/// a single request cannot occupy the model indefinitely.
const int kChatMaxTokensCap = 512;

/// Learned-tokenizer floor imposed by the byte vocabulary and its controls.
///
/// [BpeTrainer] appends merges above the 260 reserved ids, so the resolved
/// vocabulary of a tokenizer is `260 + merges.length` and the model must be
/// built with exactly that number.
const int _tokenizerFloor = 260;

/// Hosts the Harbor Core chat runtime for the update server.
///
/// Construction is cheap and allocation-free; the (comparatively expensive)
/// tokenizer training and model initialisation happen in [initialize], which is
/// idempotent and safe to call from several requests at once.
class HarborChatService {
  /// Creates a service.
  ///
  /// [config] selects the model shape. [seedCorpus] is the optional prose the
  /// tokenizer learns its merges from; when it is null or empty the plain byte
  /// vocabulary is used. [maxNewTokens] is the default cap used when a request
  /// does not override it (and must stay within [kChatMaxTokensCap]).
  HarborChatService({
    this.config = TinyLmConfig.browserPreset,
    String? seedCorpus,
    int maxNewTokens = kChatDefaultMaxTokens,
    MemoryStorage? memoryStorage,
    this.memoryEnabled = true,
    this.thinkingEnabled = true,
  })  : _seedCorpus = seedCorpus ?? '',
        memory = MemoryStore(
          storage: memoryStorage ?? InMemoryMemoryStorage(),
        ),
        maxNewTokens =
            maxNewTokens < 1 ? 1 : math.min(maxNewTokens, kChatMaxTokensCap);

  /// The requested model shape.
  ///
  /// The vocabulary is replaced by the tokenizer's resolved vocabulary at build
  /// time, because `browserPreset.vocabSize` need not match the seeded corpus.
  final TinyLmConfig config;

  /// The cap on tokens generated per turn, before a per-request override.
  final int maxNewTokens;

  final String _seedCorpus;

  /// Whether recalled memories are injected into a prompt.
  final bool memoryEnabled;

  /// Whether turns are planned and checked before the answer is shown.
  final bool thinkingEnabled;

  /// The assistant's long-term memory.
  ///
  /// Owned here rather than per-engine: every request shares one store, so a
  /// fact learned through the mobile client is visible to the dashboard on the
  /// next turn. Storage is the caller's choice; without it memory is
  /// process-scoped, which is the correct default for a test.
  final MemoryStore memory;

  TinyLm? _model;
  ByteTokenizer? _tokenizer;
  ChatEngine? _engine;
  ToolRegistry? _tools;
  http.Client? _client;
  Future<void>? _initialization;

  /// Builds the tokenizer, model and engine once, and only once.
  ///
  /// The future itself is the guard, so concurrent callers share one build and a
  /// second call after success returns the already-completed future. That is
  /// what makes it safe for a route handler to `await` on every request without
  /// re-training the tokenizer each time.
  Future<void> initialize() => _initialization ??= Future<void>.sync(() async {
        _buildOnce();
        // Memory is read, not built, so a corrupt file must not stop the model
        // from coming up. `load()` already swallows it; awaiting keeps the
        // first turn honest about what the store holds.
        await memory.load();
      });

  /// The tool catalogue the chat engine may advertise.
  ///
  /// Exposed (rather than only reachable through the engine) so the router can
  /// list it without knowing about `http` or how the registry is assembled.
  ToolRegistry get toolRegistry => _tools ??= toolCatalogue(client: _httpClient);

  /// Runs one turn over [history] and streams what happened.
  ///
  /// [maxNewTokens] overrides the service default for this turn only; the shared
  /// engine is reused when the value matches, and a short-lived engine is built
  /// otherwise because [ChatEngine.maxNewTokens] is immutable in Harbor Core.
  Stream<ChatEvent> respond(
    List<ChatMessage> history, {
    int? maxNewTokens,
    bool? useMemory,
    String? thinking,
  }) async* {
    await initialize();
    final ChatEngine engine = _engineFor(maxNewTokens);
    // `auto` means "let the engine choose per turn"; any other value is a
    // request that overrides the standing depth for this turn only.
    final ThinkingStrategy? override = thinking == null || thinking == 'auto'
        ? null
        : ThinkingStrategy.fromWire(thinking);
    yield* engine.respond(
      history,
      useMemory: useMemory,
      thinking: override,
    );
  }

  /// A JSON description of the model for `/api/v1/chat/status`.
  ///
  /// Reported without triggering [initialize]: `ready` has to mean "built",
  /// which would be a lie if asking for the status was itself what built it.
  Map<String, Object?> status() {
    final TinyLm? model = _model;
    final ByteTokenizer? tokenizer = _tokenizer;
    return <String, Object?>{
      'ready': model != null && tokenizer != null,
      // There is no pretrained checkpoint to load, so this stays honest no
      // matter how many turns have run.
      'stage': 'untrained',
      'vocabulary': tokenizer?.vocabSize ?? 0,
      'parameters': model?.totalParameterCount ?? 0,
      'layers': model?.config.nLayers ?? config.nLayers,
      'context_length': model?.contextLength ?? config.contextLength,
      // The seed corpus is a single caller-supplied string, so it is one
      // document; a caller that wants finer accounting can count its own lines.
      'seed_documents': _seedCorpus.trim().isEmpty ? 0 : 1,
      'seed_chars': _seedCorpus.length,
      'max_new_tokens': maxNewTokens,
      'memory': <String, Object?>{
        ...memory.describe(),
        'enabled': memoryEnabled,
      },
      'thinking': <String, Object?>{
        'enabled': thinkingEnabled,
        'auto': thinkingEnabled,
      },
    };
  }

  /// Assembles the tokenizer, model and engine exactly once.
  void _buildOnce() {
    final ByteTokenizer tokenizer = _seedCorpus.isEmpty
        ? ByteTokenizer()
        : ByteTokenizer.train(
            _seedCorpus,
            trainer: BpeTrainer(
              // Never learn more merges than the configured vocabulary can hold,
              // and never more than the trainer's practical budget.
              targetMerges: math.min(
                math.max(config.vocabSize - _tokenizerFloor, 0),
                320,
              ),
              maxSampleChars: 24000,
            ),
          );
    // Take the resolved vocabulary from the tokenizer: a preset whose
    // `vocabSize` is below the 260 floor would fail validation, and the
    // tokenizer is the authority on how many ids actually exist.
    final TinyLm model =
        TinyLm(config.copyWith(vocabSize: tokenizer.vocabSize));
    _tokenizer = tokenizer;
    _model = model;
    _engine = ChatEngine(
      model: model,
      tokenizer: tokenizer,
      tools: toolRegistry,
      memory: memoryEnabled ? memory : null,
      maxNewTokens: maxNewTokens,
      // The engine reserves room to answer inside this window and shortens the
      // tool catalogue before it would truncate the conversation, so the model's
      // real window is the right value here. An earlier revision passed a
      // pretend 4096-token budget to work around the catalogue overflowing the
      // window; that hid the problem instead of fixing it, and it promised the
      // caller more context than the model could attend.
      contextLength: model.contextLength,
      // Reasoning is on by default and the per-turn chooser picks the depth, so
      // a greeting is not charged for a plan while a hard question still gets
      // checked before its answer is shown.
      thinkingStrategy:
          thinkingEnabled ? ThinkingStrategy.thorough : ThinkingStrategy.none,
      autoThink: thinkingEnabled,
    );
  }

  ChatEngine _engineFor(int? requestedMaxNewTokens) {
    final TinyLm model = _model!;
    final ByteTokenizer tokenizer = _tokenizer!;
    final ChatEngine engine = _engine!;
    if (requestedMaxNewTokens == null ||
        requestedMaxNewTokens == engine.maxNewTokens) {
      return engine;
    }
    return ChatEngine(
      model: model,
      tokenizer: tokenizer,
      tools: toolRegistry,
      memory: memoryEnabled ? memory : null,
      maxNewTokens: requestedMaxNewTokens,
      contextLength: engine.contextLength,
      thinkingStrategy: engine.thinkingStrategy,
      autoThink: engine.autoThink,
    );
  }

  http.Client get _httpClient => _client ??= http.Client();

  /// Builds the catalogue offered by `GET /api/v1/tools`.
  ///
  /// Device tools are advertised but cannot run here: the server has no Android
  /// host, so every call resolves through the documented "unsupported" path
  /// (the caller returns null). Web tools are real, because the server is the
  /// one Harbor surface that legitimately owns an outbound HTTP client.
  static ToolRegistry toolCatalogue({required http.Client client}) {
    return ToolRegistry(<Tool>[
      ...deviceTools(_unsupportedHost),
      ...webTools(client: client, allow: _allowAnyHost),
    ]);
  }

  /// The host callable for a platform that has none.
  ///
  /// Returning null is the contract that makes a device tool report itself as
  /// unsupported instead of failing, which is exactly what a server should say.
  static Future<Object?> _unsupportedHost(
    String method,
    Map<String, Object?> args,
  ) async {
    return null;
  }

  /// Allows every host the URL validator already restricts to http(s).
  ///
  /// The catalogue is declarative today — no route invokes a tool — so the
  /// predicate only has to keep the tools constructible with a real client.
  static bool _allowAnyHost(Uri uri) => uri.hasScheme;
}