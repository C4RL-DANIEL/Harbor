// The model the app actually owns: tokenizer plus weights, on disk.
//
// Bootstrapping order matters and is the reason this class exists rather than a
// few lines in `main`. The tokenizer is learned *first*, from whatever corpus
// the device has gathered, and the model's vocabulary is then sized to whatever
// the trainer produced. Doing it the other way round — fixing a vocabulary and
// hoping the tokenizer fills it — either wastes embedding rows or forces a
// padding token into the vocabulary, and a padded vocabulary makes every
// checkpoint depend on a training detail that the checkpoint itself does not
// record.

import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:harbor_core/harbor_core.dart';

import '../corpus/corpus_persistence.dart';

/// Owns the on-device tokenizer and model, and their files.
class HarborModelRuntime extends ChangeNotifier {
  /// Creates a runtime persisting into [layout].
  ///
  /// [lowResource] shrinks the default config for older phones:
  /// fewer layers, shorter context, smaller KV cache. The runtime
  /// adapts generation to stay under the budget without blocking the UI.
  HarborModelRuntime({
    required this.layout,
    this.baseConfig = TinyLmConfig.onDevicePreset,
    this.lowResource = false,
  });

  /// Where the tokenizer and checkpoint live.
  final HarborStorageLayout layout;

  /// The architecture to build. Its `vocabSize` is a *request*: the tokenizer's
  /// actual vocabulary wins, because the embedding table must match the merges
  /// that exist rather than the merges that were hoped for.
  final TinyLmConfig baseConfig;

  /// Shrink the defaults for low-resource devices.
  final bool lowResource;

  ByteTokenizer? _tokenizer;
  TinyLm? _model;
  String _status = 'not initialised';
  String? _error;
  DateTime? _checkpointAt;

  /// The learned tokenizer, or null before [bootstrap].
  ByteTokenizer? get tokenizer => _tokenizer;

  /// The model, or null before [bootstrap].
  TinyLm? get model => _model;

  /// Whether both parts are ready.
  bool get ready => _tokenizer != null && _model != null;

  /// A short human status, shown in the UI while bootstrapping.
  String get status => _status;

  /// The last error, if bootstrapping or saving failed.
  String? get error => _error;

  /// When the loaded checkpoint was written.
  DateTime? get checkpointAt => _checkpointAt;

  /// Whether a checkpoint exists on disk.
  bool get hasCheckpoint => layout.modelFile.existsSync();

  /// The active configuration, including the resolved vocabulary.
  TinyLmConfig get config => _model?.config ?? baseConfig;

  /// Learns or loads a tokenizer, then loads or creates a model.
  ///
  /// [corpusText] is only used when no tokenizer exists yet. Never throws: a
  /// failure is recorded in [error] and reported through [status], because a
  /// model that cannot be built must not prevent the rest of the app from
  /// starting.
  Future<void> bootstrap(String corpusText) async {
    _error = null;
    try {
      _setStatus('Loading tokenizer');
      _tokenizer = await _loadOrTrainTokenizer(corpusText);
      final ByteTokenizer tokenizer = _tokenizer!;

      _setStatus('Building model');
      TinyLmConfig resolved =
          baseConfig.copyWith(vocabSize: tokenizer.vocabSize);
      // Low-resource devices get a smaller active config: fewer layers,
      // shorter context, tighter KV compression. The embedding table
      // still matches the tokenizer vocabulary.
      if (lowResource) {
        resolved = resolved.copyWith(
          nLayers: 1,
          contextLength: 128,
          kvRank: (resolved.kvRank / 2).clamp(4, 16).toInt(),
        );
      }
      _model = await _loadOrCreateModel(resolved);
      _setStatus(
        'Ready · ${_model!.config.summary} · '
        '${(_model!.totalParameterCount / 1000).toStringAsFixed(0)}k params',
      );
      notifyListeners();
    } on Object catch (error) {
      _error = '$error';
      _setStatus('Model unavailable: $error');
      notifyListeners();
    }
  }

  /// Persists the current tokenizer and model.
  Future<void> save() async {
    final ByteTokenizer? tokenizer = _tokenizer;
    final TinyLm? model = _model;
    if (tokenizer == null || model == null) {
      return;
    }
    try {
      await layout.tokenizerFile.parent.create(recursive: true);
      await layout.tokenizerFile
          .writeAsString(jsonEncode(tokenizer.toJson()), flush: true);
      final File temporary = File('${layout.modelFile.path}.tmp');
      await temporary.writeAsString(model.encode(), flush: true);
      await temporary.rename(layout.modelFile.path);
      _checkpointAt = DateTime.now();
      _error = null;
      notifyListeners();
    } on Object catch (error) {
      _error = 'Saving the model failed: $error';
      notifyListeners();
    }
  }

  /// Saves the current model under a caller-supplied checkpoint string.
  ///
  /// Used by the training session's checkpoint callback, which hands over a
  /// serialised model mid-run: writing it byte-for-byte avoids re-serialising
  /// the model while the training loop is between steps.
  Future<void> saveCheckpoint(String checkpointJson) async {
    try {
      await layout.trainingCheckpointFile.parent.create(recursive: true);
      final File temporary = File('${layout.trainingCheckpointFile.path}.tmp');
      await temporary.writeAsString(checkpointJson, flush: true);
      await temporary.rename(layout.trainingCheckpointFile.path);
      _checkpointAt = DateTime.now();
    } on Object catch (error) {
      _error = 'Checkpointing failed: $error';
      notifyListeners();
    }
  }

  /// Loads the most recent training checkpoint into the live model.
  Future<bool> restoreTrainingCheckpoint() async {
    try {
      if (!layout.trainingCheckpointFile.existsSync()) {
        return false;
      }
      final String payload = await layout.trainingCheckpointFile.readAsString();
      _model?.loadJson(_decodeJson(payload));
      notifyListeners();
      return true;
    } on Object catch (error) {
      _error = 'Could not restore the training checkpoint: $error';
      notifyListeners();
      return false;
    }
  }

  /// Deletes every artefact and returns to the un-initialised state.
  Future<void> reset() async {
    await layout.clear();
    _tokenizer = null;
    _model = null;
    _checkpointAt = null;
    _error = null;
    _setStatus('Reset · no model');
    notifyListeners();
  }

  Future<ByteTokenizer> _loadOrTrainTokenizer(String corpusText) async {
    final File file = layout.tokenizerFile;
    if (file.existsSync()) {
      try {
        return ByteTokenizer.fromJson(_decodeJson(await file.readAsString()));
      } on Object catch (error) {
        // A corrupt tokenizer is recoverable by retraining, and retraining is
        // cheap next to shipping a build that cannot start.
        _error = 'The stored tokenizer was unreadable ($error); retraining.';
      }
    }
    final int targetMerges =
        (baseConfig.vocabSize - 260).clamp(0, 320);
    _setStatus('Learning tokenizer from the corpus');
    final BpeTrainer trainer = BpeTrainer(
      targetMerges: targetMerges,
      maxSampleChars: 24000,
    );
    final List<BpeMerge> merges = trainer.train(corpusText);
    final ByteTokenizer tokenizer = ByteTokenizer(merges: merges);
    await file.parent.create(recursive: true);
    await file.writeAsString(jsonEncode(tokenizer.toJson()), flush: true);
    return tokenizer;
  }

  Future<TinyLm> _loadOrCreateModel(TinyLmConfig resolved) async {
    final File file = layout.modelFile;
    if (file.existsSync()) {
      try {
        final TinyLm loaded = TinyLm.decode(await file.readAsString());
        if (_compatible(loaded.config, resolved)) {
          return loaded;
        }
        // The vocabulary changed because the corpus grew enough to learn more
        // merges. The old weights are meaningless against the new ids, so a
        // fresh model is the only correct answer; the fit is kept on disk.
        _error = 'The stored checkpoint used ${loaded.config.summary}; '
            'starting a new model for ${resolved.summary}.';
      } on Object catch (error) {
        _error = 'The stored checkpoint was unreadable ($error); '
            'starting a new model.';
      }
    }
    if (!file.existsSync()) {
      _checkpointAt = null;
    }
    return TinyLm(resolved);
  }

  /// Whether a stored checkpoint can be reused for [resolved].
  ///
  /// Every shape field must match; a mismatch in any of them changes the meaning
  /// of the parameter vector and the checkpoint must not be loaded.
  static bool _compatible(TinyLmConfig stored, TinyLmConfig resolved) {
    return stored.vocabSize == resolved.vocabSize &&
        stored.dModel == resolved.dModel &&
        stored.nLayers == resolved.nLayers &&
        stored.nHeads == resolved.nHeads &&
        stored.headDim == resolved.headDim &&
        stored.kvRank == resolved.kvRank &&
        stored.contextLength == resolved.contextLength &&
        stored.numExperts == resolved.numExperts &&
        stored.topK == resolved.topK &&
        stored.expertHidden == resolved.expertHidden;
  }

  static Map<String, Object?> _decodeJson(String payload) {
    final Object? decoded = jsonDecode(payload);
    if (decoded is! Map<String, Object?>) {
      throw const FormatException('expected a JSON object');
    }
    return decoded;
  }

  void _setStatus(String value) {
    _status = value;
    if (kDebugMode) {
      debugPrint('[harbor] $value');
    }
  }
}