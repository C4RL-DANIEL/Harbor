// File-backed persistence for the corpus and the model checkpoint.
//
// `harbor_core` defines the storage seams (`CorpusStorage`) and knows nothing
// about the file system, because the same code also runs in a browser tab. This
// file is the Android/desktop implementation of those seams: everything that
// touches `dart:io` for the learning pipeline lives here, so the portable
// package stays portable by construction rather than by discipline.

import 'dart:io';

import 'package:harbor_core/harbor_core.dart';

/// A [CorpusStorage] backed by a single UTF-8 JSON file.
class FileCorpusStorage implements CorpusStorage {
  /// Creates storage over [file].
  FileCorpusStorage(this.file);

  /// The file the corpus is written to.
  final File file;

  @override
  Future<String?> read() async {
    if (!file.existsSync()) {
      return null;
    }
    return file.readAsString();
  }

  @override
  Future<void> write(String payload) async {
    await file.parent.create(recursive: true);
    // Write to a sibling and rename, so a crash mid-write cannot leave a
    // truncated corpus that the tolerant loader would silently accept as a
    // smaller one.
    final File temporary = File('${file.path}.tmp');
    await temporary.writeAsString(payload, flush: true);
    await temporary.rename(file.path);
  }

  @override
  Future<void> delete() async {
    if (file.existsSync()) {
      await file.delete();
    }
  }
}

/// The on-disk layout of everything the learning pipeline persists.
///
/// One place decides the file names so the settings screen, the reset action and
/// the diagnostics panel cannot disagree about what is stored where.
class HarborStorageLayout {
  /// Creates a layout rooted at [root].
  HarborStorageLayout(this.root);

  /// The directory every Harbor file lives in.
  final Directory root;

  /// The serialised corpus.
  File get corpusFile => File('${root.path}/corpus.json');

  /// The serialised tokenizer (learned merges).
  File get tokenizerFile => File('${root.path}/tokenizer.json');

  /// The most recent model checkpoint.
  File get modelFile => File('${root.path}/model.json');

  /// The rolling training checkpoint, written during a run.
  File get trainingCheckpointFile => File('${root.path}/training_checkpoint.json');

  /// A human-readable log of completed training rounds.
  File get trainingLogFile => File('${root.path}/training_log.jsonl');

  /// The assistant's long-term memory.
  File get memoryFile => File('${root.path}/memory.json');

  /// The visible chat transcript, restored on the next launch.
  File get transcriptFile => File('${root.path}/transcript.jsonl');

  /// Total bytes currently used by this layout, for the diagnostics panel.
  int get usedBytes {
    int total = 0;
    for (final File file in <File>[
      corpusFile,
      tokenizerFile,
      modelFile,
      trainingCheckpointFile,
      trainingLogFile,
      memoryFile,
      transcriptFile,
    ]) {
      if (file.existsSync()) {
        total += file.lengthSync();
      }
    }
    return total;
  }

  /// Deletes every persisted artefact.
  Future<void> clear() async {
    for (final File file in <File>[
      corpusFile,
      tokenizerFile,
      modelFile,
      trainingCheckpointFile,
      trainingLogFile,
      memoryFile,
      transcriptFile,
    ]) {
      if (file.existsSync()) {
        await file.delete();
      }
    }
  }
}

/// Appends one JSON object per line to the training log.
///
/// JSON Lines rather than a JSON array because a training run that is killed
/// mid-write leaves the previous rounds intact instead of producing a document
/// that no longer parses.
class TrainingLog {
  /// Creates a log over [file].
  TrainingLog(this.file);

  /// The append target.
  final File file;

  /// Appends [entry] as one line.
  Future<void> append(Map<String, Object?> entry) async {
    await file.parent.create(recursive: true);
    await file.writeAsString(
      '${_encode(entry)}\n',
      mode: FileMode.append,
      flush: true,
    );
  }

  /// The most recent [limit] entries, oldest first.
  ///
  /// Unparseable lines are skipped rather than throwing, so a half-written last
  /// line cannot make the history unreadable.
  Future<List<Map<String, Object?>>> recent({int limit = 20}) async {
    if (!file.existsSync()) {
      return const <Map<String, Object?>>[];
    }
    final List<String> lines = await file.readAsLines();
    final List<Map<String, Object?>> entries = <Map<String, Object?>>[];
    for (final String line in lines) {
      if (line.trim().isEmpty) {
        continue;
      }
      try {
        final Object? decoded = _decode(line);
        if (decoded is Map<String, Object?>) {
          entries.add(decoded);
        }
      } on FormatException {
        continue;
      }
    }
    if (entries.length <= limit) {
      return entries;
    }
    return entries.sublist(entries.length - limit);
  }

  static String _encode(Map<String, Object?> entry) {
    final StringBuffer buffer = StringBuffer('{');
    bool first = true;
    for (final MapEntry<String, Object?> pair in entry.entries) {
      if (!first) {
        buffer.write(',');
      }
      first = false;
      buffer
        ..write('"')
        ..write(pair.key.replaceAll('"', r'\"'))
        ..write('":')
        ..write(_literal(pair.value));
    }
    buffer.write('}');
    return buffer.toString();
  }

  static String _literal(Object? value) {
    if (value == null) {
      return 'null';
    }
    if (value is num || value is bool) {
      return '$value';
    }
    return '"${value.toString().replaceAll(r'\', r'\\').replaceAll('"', r'\"')}"';
  }

  static Object? _decode(String line) {
    // A tiny schema-free reader: the log only ever carries flat scalars, so a
    // full JSON parser would be a heavier dependency than the format needs.
    final String trimmed = line.trim();
    if (!trimmed.startsWith('{') || !trimmed.endsWith('}')) {
      throw const FormatException('not a JSON object');
    }
    final Map<String, Object?> out = <String, Object?>{};
    final String body = trimmed.substring(1, trimmed.length - 1);
    final RegExp entry = RegExp(r'"((?:[^"\\]|\\.)*)"\s*:\s*("(?:[^"\\]|\\.)*"|true|false|null|-?[0-9.]+)');
    for (final RegExpMatch match in entry.allMatches(body)) {
      final String key = match.group(1)!.replaceAll(r'\"', '"').replaceAll(r'\\', r'\');
      final String raw = match.group(2)!;
      if (raw.startsWith('"')) {
        out[key] = raw
            .substring(1, raw.length - 1)
            .replaceAll(r'\"', '"')
            .replaceAll(r'\\', r'\');
      } else if (raw == 'true') {
        out[key] = true;
      } else if (raw == 'false') {
        out[key] = false;
      } else if (raw == 'null') {
        out[key] = null;
      } else {
        out[key] = num.tryParse(raw);
      }
    }
    if (out.isEmpty) {
      throw const FormatException('no fields');
    }
    return out;
  }
}