// The in-browser chat model for the Flutter Web dashboard.
//
// This file is deliberately Flutter-free and `dart:io`-free so the same source
// also compiles on the Dart VM: it only touches `harbor_core` and
// `package:http`, both of which dart2js understands. That is what lets the
// GitHub Pages build run a real model with no server process behind it.

import 'dart:math' as math;

import 'package:harbor_core/harbor_core.dart';
import 'package:http/http.dart' as http;

/// The only text the in-browser model ever sees.
///
/// The dashboard has no pretrained checkpoint to ship — a real one would be
/// megabytes, which is exactly what a static Pages bundle cannot carry — so the
/// model is trained from scratch on this bundled prose every time the page
/// loads. Keeping it inside the bundle rather than fetching it means the chat
/// pane works with no network at all, and keeping it short means the honesty
/// card can show the user every word the model has read.
const String kBrowserSeedCorpus = '''
Harbor is a private assistant that runs on the user's own device. It does not
send the user's words to a data centre, and it does not need an account. The
whole point of Harbor is that the model, the memory and the tools all live in
the same place the person does: on the phone, in the browser tab, or on a small
server the user controls.

The architecture has four layers. The first layer is the corpus pipeline. It
collects text from the device, from files the user chooses, and from public web
pages, and turns all of it into clean training documents. The web collector
strips scripts, styles and navigation, keeps the readable prose, and records the
source URL so a fact can be traced back. The corpus store keeps the documents
on disk as JSON, which means a run can be paused, inspected and resumed.

The second layer is the tokenizer. Harbor uses a byte level byte pair encoding
tokenizer. The base vocabulary is the 256 byte values, so any input is
encodable, including text in languages the model has never been trained on. The
tokenizer then learns merges from the corpus: the most frequent adjacent pairs
of bytes are fused into a single token, over and over, until the vocabulary is
full. A byte level vocabulary never has to drop a character it has not seen.

The third layer is the model itself. Harbor ships a small decoder only
transformer called TinyLm. Each layer applies two blocks: an attention block and
a mixture of experts block, both wrapped in residual connections with layer
normalisation in front. The attention block uses multi head latent attention,
often shortened to MLA. Instead of storing a full key and value vector for every
head, MLA compresses keys and values into a single shared latent of much lower
rank, and decompresses it on the fly. The compressed cache is far smaller than
the usual one, so a long conversation fits in the memory a phone actually has.
The trade is a small reconstruction error, which the model learns to tolerate
because the whole stack is trained end to end.

The mixture of experts block is the other half of the trick. Instead of running
one large feed forward network for every token, the block keeps several smaller
experts and a router. The router scores each expert and sends the token to the
top few, usually the top two. Only the chosen experts run, so the model can hold
many parameters while spending the compute of a much smaller one. Over a long
run, different experts specialise in different kinds of text, such as code,
prose, or tool calls.

The fourth layer is the tool layer. A tool is a small declaration with a name, a
description and a JSON schema of its arguments, plus an implementation. The
assistant can call a tool by emitting a short tag in its reply. The engine
extracts the call, validates the arguments, runs the tool and feeds the result
back. Device tools reach the phone through a platform channel: battery level,
storage, clipboard, notifications, location and the file system. Web tools run
in process and can fetch a public page, check whether a URL is reachable, or
inspect a JSON document. The browser cannot call Android APIs, so in the
dashboard only the web tools are available.

Training happens on the device. Harbor does not fine tune every weight. It keeps
the base model frozen and trains a small low rank adapter, usually called LoRA,
which adds a pair of narrow matrices to each layer it touches. The adapter has
orders of magnitude fewer parameters than the base model, so a round of training
finishes on battery and the result is a file small enough to share. The adapter
can be exported, imported and merged into the base weights when the user wants a
single file.

Updates travel over the air. The update server publishes release metadata, a
version number for each platform, the download URL and its SHA-256 digest, and
the minimum supported version. The client asks whether an update exists, checks
whether it is mandatory, downloads the artifact, verifies the digest and only
then installs it. A channel can be moved forward or held back, and a bad release
can be pinned by raising the minimum supported version. Because the client
verifies the digest before installing, a corrupted or tampered download is
rejected rather than executed.

Harbor keeps the pieces small and separate on purpose. The model runtime, the
tokenizer and the tool layer are plain Dart with no Flutter and no file system
dependency, so the same source runs on the Android image, on the update server
and in the browser. The update server is a shelf service that stores releases
and the feature flag matrix. The admin dashboard is a Flutter Web application
that edits that state. The browser build of the dashboard can also run a small
model from scratch, with no server at all, using the text in this seed corpus as
its only training data.
''';

/// The minimum transformer window the browser chat pane runs with.
///
/// [TinyLmConfig.browserPreset] declares a 96-token window, but the Harbor
/// system prompt plus the web-tool catalogue already tokenises to roughly 380
/// byte-level tokens. At 96 tokens the engine would drop the instructions from
/// the prompt and then report `context_limit` before emitting a single token.
/// Widening the window keeps the whole prompt intact and lets the model's
/// incremental decoder advance one step per generated token, which a larger
/// engine-only budget cannot do. 512 leaves room for the catalogue plus a full
/// 96-token reply and a short back-and-forth; the update server's chat service
/// documents the same trap.
const int _browserContextLength = 512;

/// Runs a tiny language model entirely inside the browser tab.
///
/// This class owns the whole in-browser lifecycle: it learns a tokenizer from
/// the bundled seed corpus, builds a [TinyLm], and wraps both in a
/// [ChatEngine]. Keeping that lifecycle here rather than in the Flutter widget
/// means the model half can also be exercised on the Dart VM, and it is what
/// lets the static GitHub Pages deployment answer chat with no server process.
class BrowserModel {
  /// Creates a model over [seedCorpus] with the shape [config].
  ///
  /// Nothing heavy happens here: no tokenizer is learned and no weights are
  /// allocated until [load] is called, so a widget can construct the object in
  /// a field initialiser without blocking its first frame.
  BrowserModel({
    String seedCorpus = kBrowserSeedCorpus,
    TinyLmConfig config = TinyLmConfig.browserPreset,
  })  : _seedCorpus = seedCorpus,
        _config = config {
    _tools = webTools(
      client: _client,
      allow: (Uri uri) => uri.scheme == 'https',
    );
    _registry = ToolRegistry(_tools);
  }

  final String _seedCorpus;
  final TinyLmConfig _config;

  /// The client shared by every web tool, owned and closed by this instance.
  ///
  /// Creating one client per tool would leak a connection pool for each tool in
  /// the catalogue, so all three web tools share this one.
  final http.Client _client = http.Client();

  late final List<Tool> _tools;
  late final ToolRegistry _registry;

  ByteTokenizer? _tokenizer;
  TinyLm? _model;
  ChatEngine? _engine;

  bool _loading = false;
  bool _disposed = false;
  double _progress = 0;
  String _stage = 'idle';

  /// Whether [load] has completed so [respond] can run a turn.
  ///
  /// The pane uses this to decide between "loading" and "ready" affordances
  /// without inspecting private fields.
  bool get ready => _engine != null;

  /// The tools this browser build can actually run.
  ///
  /// Only the in-process web tools are exposed: a browser tab cannot reach
  /// Android APIs, so the device tools are deliberately absent rather than
  /// advertised and then failed. The dashboard states that the device tools run
  /// in the Android app through its platform channel instead.
  List<Tool> get tools => List<Tool>.unmodifiable(_tools);

  /// The name-indexed view of [tools] that [ChatEngine] consumes.
  ///
  /// Exposed so the pane can invoke a tool directly and so a future caller can
  /// hand the same catalogue to another engine without rebuilding it.
  ToolRegistry get registry => _registry;

  /// Learns the tokenizer, builds the model, and warms the chat engine.
  ///
  /// Split across three awaited stages so the pane can drive a determinate
  /// progress bar; a single synchronous call would freeze the tab with nothing
  /// to show. It is idempotent because a widget lifecycle can trigger it more
  /// than once and a second build would otherwise redo all the work.
  Future<void> load() async {
    if (_engine != null || _loading || _disposed) {
      return;
    }
    _loading = true;
    try {
      _stage = 'Learning a vocabulary from the seed corpus';
      _progress = 0;
      await _yieldToUi();

      // The validator rejects a vocabulary below 260 (the 256 byte values plus
      // four control ids), and `browserPreset.vocabSize` is not guaranteed to be
      // above that floor for every caller. The resolved vocabulary therefore
      // always comes from the tokenizer, never from the raw preset.
      final ByteTokenizer tokenizer = ByteTokenizer.train(
        _seedCorpus,
        trainer: BpeTrainer(
          targetMerges: math.min(_config.vocabSize - kFirstMergeId, 320),
          maxSampleChars: 24000,
        ),
      );
      _tokenizer = tokenizer;
      _stage = 'Building the transformer';
      _progress = 1 / 3;
      await _yieldToUi();

      // The window is widened past the preset's 96 tokens for the reason given
      // on [_browserContextLength]; the resolved vocabulary still comes from the
      // tokenizer, never from the raw preset.
      final TinyLm model = TinyLm(
        _config.copyWith(
          vocabSize: tokenizer.vocabSize,
          contextLength: math.max(_config.contextLength, _browserContextLength),
        ),
      );
      _model = model;
      _stage = 'Warming the chat engine';
      _progress = 2 / 3;
      await _yieldToUi();

      _engine = ChatEngine(
        model: model,
        tokenizer: tokenizer,
        tools: _registry,
        maxNewTokens: 96,
        contextLength: model.contextLength,
      );
      _stage = 'untrained';
      _progress = 1;
    } finally {
      // Cleared in a `finally` so a failure at any stage cannot leave the model
      // stuck "loading" and permanently unavailable.
      _loading = false;
    }
  }

  /// Renders [history] exactly as the engine will feed it to the model.
  ///
  /// The pane offers this on demand so a user can see what the model was really
  /// asked; without it a truncated or mangled prompt is indistinguishable from
  /// a bad answer. Before [load] it returns an explanatory line rather than
  /// throwing, because the button may be pressed while the model is loading.
  String renderPrompt(List<ChatMessage> history) {
    final ChatEngine? engine = _engine;
    if (engine == null) {
      return 'The in-browser model is still loading; no prompt has been '
          'rendered yet.';
    }
    return engine.renderPrompt(history);
  }

  /// Streams one assistant turn over [history].
  ///
  /// Delegates straight to the engine so the pane observes token, tool and
  /// finish events as they happen. Before [load] it yields a single failure
  /// instead of throwing, because the widget may render a turn while loading.
  Stream<ChatEvent> respond(List<ChatMessage> history) {
    final ChatEngine? engine = _engine;
    if (engine == null) {
      return Stream<ChatEvent>.value(
        const ChatFailed('the in-browser model is not loaded yet'),
      );
    }
    return engine.respond(history);
  }

  /// A snapshot of the model's shape and load progress for the pane.
  ///
  /// `stage` settles on `untrained` once loading finishes: this model has no
  /// pretrained checkpoint, and the UI must not imply otherwise. The numeric
  /// fields are read from the resolved tokenizer and model so they describe
  /// what is actually running rather than the raw preset.
  Map<String, Object?> status() {
    final TinyLm? model = _model;
    final TinyLmConfig shape = model?.config ?? _config;
    return <String, Object?>{
      'ready': ready,
      'stage': _loading ? _stage : 'untrained',
      'vocabulary': _tokenizer?.vocabSize ?? _config.vocabSize,
      'parameters': model?.parameterCount ?? _config.estimatedParameters,
      'layers': shape.nLayers,
      'context_length': shape.contextLength,
      'seed_chars': _seedCorpus.length,
      'progress': _progress,
    };
  }

  /// Runs one of [tools] by [name] with [arguments].
  ///
  /// Kept separate from chat so the tools card can exercise a tool directly,
  /// which makes a failing tool debuggable without spending a model turn.
  Future<ToolResult> runTool(
    String name, [
    Map<String, Object?> arguments = const <String, Object?>{},
  ]) {
    if (_disposed) {
      return Future<ToolResult>.value(
        const ToolResult.failure('the browser model has been disposed'),
      );
    }
    return _registry.invokeByName(name, arguments);
  }

  /// Releases the HTTP connection pool shared by the web tools.
  ///
  /// Idempotent so a widget's `dispose` can call it without tracking whether
  /// the model ever finished loading or whether it was already released.
  void dispose() {
    if (_disposed) {
      return;
    }
    _disposed = true;
    _client.close();
  }

  /// Yields one event-loop turn so a progress bar can repaint between stages.
  ///
  /// Without it the three stages would complete in a single microtask chain and
  /// the pane would only ever observe the finished value.
  Future<void> _yieldToUi() =>
      Future<void>.delayed(const Duration(milliseconds: 16));
}