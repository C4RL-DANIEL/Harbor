// Corpus persistence and quota management.
//
// This is the one place that decides what actually becomes training text. It
// sits behind a tiny storage seam (`CorpusStorage`) so the Android application
// can back it with a file in the app sandbox, the Flutter Web dashboard with
// `window.localStorage`, and tests with an in-memory map — without any of those
// environments leaking into the core package.
//
// The invariants the store guarantees:
//   * every stored document has been redacted, then hashed, then admitted;
//   * `documents` is in insertion order and never exposes a mutable list;
//   * a duplicate never enters, and every rejection is counted;
//   * a corrupt payload never throws — it loads as an empty store and is counted
//     in `loadFailures`, because losing a corpus is bad but crashing the app
//     that owns it is worse; and
//   * `trainingText` is reproducible for a given seed.

import 'dart:convert';
import 'dart:math' as math;

import 'corpus_document.dart';
import 'device_text.dart';

/// The persistence seam for a corpus payload.
///
/// The interface is intentionally three strings and no paths: the platform layer
/// owns *where* bytes live, and the core owns *what* they mean. Implementations
/// must round-trip [write] through [read] exactly and must not interpret the
/// payload.
abstract class CorpusStorage {
  /// Returns the stored payload, or `null` when nothing has been saved yet.
  Future<String?> read();

  /// Persists [payload], replacing any previous value.
  Future<void> write(String payload);

  /// Removes the stored payload; reading afterwards returns `null`.
  Future<void> delete();
}

/// An immutable snapshot of a store's contents and its rejection counters.
///
/// Counters are cumulative for the lifetime of the store (or since the last
/// [CorpusStore.load] / [CorpusStore.clear]) and exist so the training UI can
/// explain *why* a corpus is smaller than the user expected.
class CorpusStats {
  /// Creates a snapshot.
  const CorpusStats({
    required this.documentCount,
    required this.totalBytes,
    required this.totalChars,
    required this.bySource,
    required this.rejectedDuplicates,
    required this.rejectedTooLarge,
    required this.rejectedQuota,
  });

  /// Number of documents currently stored.
  final int documentCount;

  /// Sum of every stored document's UTF-8 byte length.
  final int totalBytes;

  /// Sum of every stored document's UTF-16 code-unit length.
  final int totalChars;

  /// Document count per [CorpusSource]; every enum value is present, defaulting
  /// to zero, so a dashboard can index it without null checks.
  final Map<CorpusSource, int> bySource;

  /// Documents refused because their content hash was already present.
  final int rejectedDuplicates;

  /// Documents refused because they exceeded the per-document byte limit.
  final int rejectedTooLarge;

  /// Documents refused because they would have exceeded the count or total-byte
  /// quota.
  final int rejectedQuota;
}

/// An in-memory corpus with redaction, deduplication, quotas and persistence.
///
/// Typical use: construct with a storage implementation, call [load] once at
/// startup, [add] or [addText] as text arrives, and [save] when the caller
/// decides the work is worth persisting. [add] deliberately does *not* save —
/// extraction and crawling produce many documents in a burst, and writing the
/// whole payload after each one would be wasteful; the owner batches saves.
class CorpusStore {
  /// Creates an empty store.
  ///
  /// [maxDocuments] and [maxTotalBytes] bound the whole corpus; a document that
  /// would cross either limit is rejected. [maxDocumentBytes] bounds a single
  /// document so one huge file cannot dominate training. Call [load] to populate
  /// the store from [storage].
  CorpusStore({
    required CorpusStorage storage,
    this.maxDocuments = 512,
    this.maxTotalBytes = 4 * 1024 * 1024,
    this.maxDocumentBytes = 512 * 1024,
  }) : _storage = storage;

  /// Maximum number of documents the store will hold.
  final int maxDocuments;

  /// Maximum total UTF-8 bytes the store will hold.
  final int maxTotalBytes;

  /// Maximum UTF-8 bytes allowed for a single document.
  final int maxDocumentBytes;

  final CorpusStorage _storage;
  final List<CorpusDocument> _documents = <CorpusDocument>[];
  final Set<String> _ids = <String>{};

  // Running totals are kept alongside the list so `stats` and quota checks stay
  // O(1) even when the corpus is at its limit.
  int _totalBytes = 0;
  int _totalChars = 0;
  int _rejectedDuplicates = 0;
  int _rejectedTooLarge = 0;
  int _rejectedQuota = 0;
  int _loadFailures = 0;

  /// Number of records skipped while parsing a stored payload.
  ///
  /// Covers a payload that is not JSON, a payload with no document list, and
  /// individual entries that are malformed. A non-zero value tells the UI that
  /// some previously stored text was lost, which is worth surfacing.
  int get loadFailures => _loadFailures;

  /// The stored documents in insertion order.
  ///
  /// Returns an unmodifiable view: callers must go through [add], [remove] or
  /// [clear] so the id set and the running totals cannot drift out of sync.
  List<CorpusDocument> get documents => List<CorpusDocument>.unmodifiable(_documents);

  /// A fresh snapshot of the contents and rejection counters.
  CorpusStats get stats {
    final Map<CorpusSource, int> bySource = <CorpusSource, int>{
      for (final CorpusSource source in CorpusSource.values) source: 0,
    };
    for (final CorpusDocument document in _documents) {
      bySource[document.source] = (bySource[document.source] ?? 0) + 1;
    }
    return CorpusStats(
      documentCount: _documents.length,
      totalBytes: _totalBytes,
      totalChars: _totalChars,
      bySource: bySource,
      rejectedDuplicates: _rejectedDuplicates,
      rejectedTooLarge: _rejectedTooLarge,
      rejectedQuota: _rejectedQuota,
    );
  }

  /// Loads the store from [CorpusStorage], replacing anything currently held.
  ///
  /// A missing payload yields an empty store. A corrupt payload is tolerated:
  /// unreadable records are counted in [loadFailures] and the rest are kept, so
  /// one damaged entry does not discard the whole corpus.
  Future<void> load() async {
    _reset();
    final String? payload = await _storage.read();
    if (payload == null || payload.trim().isEmpty) {
      return;
    }
    _absorb(payload);
  }

  /// Validates, redacts, hashes and inserts [document].
  ///
  /// Returns `false` without throwing when the redacted text is empty, when the
  /// document exceeds [maxDocumentBytes], when its hash is already stored, or
  /// when it would exceed [maxDocuments] or [maxTotalBytes]. Redaction runs
  /// first so a secret cannot influence the deduplication key, and the stored
  /// id is recomputed from the redacted text so two documents that differ only
  /// in a redacted secret still collapse to one.
  Future<bool> add(CorpusDocument document) async {
    final String text = TextRedactor.redact(document.text);
    if (text.trim().isEmpty) {
      return false;
    }
    final CorpusDocument normalised = CorpusDocument(
      id: CorpusDocument.hashOf(text),
      text: text,
      source: document.source,
      uri: document.uri,
      collectedAt: document.collectedAt,
      meta: document.meta,
    );
    return _insert(normalised);
  }

  /// Convenience wrapper around [add] for callers that have raw text only.
  ///
  /// The id is computed from the raw text here and recomputed after redaction
  /// inside [add]; the redundant hash is cheap next to the network or file read
  /// that produced the text.
  Future<bool> addText(
    String text, {
    required CorpusSource source,
    required String uri,
    Map<String, String> meta = const <String, String>{},
  }) {
    return add(
      CorpusDocument(
        id: CorpusDocument.hashOf(text),
        text: text,
        source: source,
        uri: uri,
        meta: meta,
      ),
    );
  }

  /// The training string: document texts joined by a blank line, shuffled
  /// deterministically and truncated to [maxChars].
  ///
  /// The shuffle exists so a corpus gathered in file-system order does not teach
  /// the model that all `README` files precede all `*.dart` files. It is seeded
  /// by [shuffleSeed] (a fixed constant by default) so the exact same training
  /// string can be regenerated after a crash or across the Android and Web
  /// builds, which is what makes a training run reproducible.
  ///
  /// [maxChars] is measured in the same UTF-16 code units as
  /// [CorpusDocument.charCount] and never splits a surrogate pair; `null` means
  /// the whole corpus.
  String trainingText({int? maxChars, int shuffleSeed = 0x48415242}) {
    if (_documents.isEmpty) {
      return '';
    }
    final List<CorpusDocument> shuffled = List<CorpusDocument>.of(_documents)
      ..shuffle(math.Random(shuffleSeed));
    final StringBuffer buffer = StringBuffer();
    for (int i = 0; i < shuffled.length; i++) {
      if (i > 0) {
        buffer.write('\n\n');
      }
      buffer.write(shuffled[i].text);
    }
    final String joined = buffer.toString();
    return maxChars == null ? joined : _truncate(joined, maxChars);
  }

  /// Removes the document with [id], if present.
  ///
  /// Removing an unknown id is a no-op rather than an error so a UI can act on a
  /// stale list without racing the store.
  Future<void> remove(String id) async {
    final int index = _documents.indexWhere((CorpusDocument document) => document.id == id);
    if (index < 0) {
      return;
    }
    final CorpusDocument removed = _documents.removeAt(index);
    _ids.remove(id);
    _totalBytes -= removed.bytes;
    _totalChars -= removed.charCount;
  }

  /// Empties the store in memory and deletes the persisted payload.
  ///
  /// Counters are reset too, so `stats` after a clear describes a genuinely
  /// empty corpus. This is the user-facing "forget everything" action, which is
  /// why it also removes the payload rather than writing an empty one.
  Future<void> clear() async {
    _reset();
    await _storage.delete();
  }

  /// Serialises the store to JSON through [CorpusStorage.write].
  ///
  /// The payload is versioned so a future format change can be detected instead
  /// of misread; [fromJson] currently ignores the version but a reader can rely
  /// on it being present.
  Future<void> save() async {
    await _storage.write(jsonEncode(_toJson()));
  }

  /// Rebuilds a store from [payload] without touching [storage].
  ///
  /// Tolerant in the same way as [load]: a payload that is not JSON, or has no
  /// document list, yields an empty store rather than an exception. The
  /// per-document and total quotas are the defaults; a caller that needs custom
  /// quotas should construct a store and call [load] instead.
  static CorpusStore fromJson(String payload, {required CorpusStorage storage}) {
    final CorpusStore store = CorpusStore(storage: storage);
    store._absorb(payload);
    return store;
  }

  /// Empties documents, ids, totals and all counters.
  void _reset() {
    _documents.clear();
    _ids.clear();
    _totalBytes = 0;
    _totalChars = 0;
    _rejectedDuplicates = 0;
    _rejectedTooLarge = 0;
    _rejectedQuota = 0;
    _loadFailures = 0;
  }

  /// Admits an already-redacted, already-hashed document, counting any refusal.
  bool _insert(CorpusDocument document) {
    final int bytes = document.bytes;
    if (bytes > maxDocumentBytes) {
      _rejectedTooLarge++;
      return false;
    }
    if (_ids.contains(document.id)) {
      _rejectedDuplicates++;
      return false;
    }
    if (_documents.length >= maxDocuments || _totalBytes + bytes > maxTotalBytes) {
      _rejectedQuota++;
      return false;
    }
    _documents.add(document);
    _ids.add(document.id);
    _totalBytes += bytes;
    _totalChars += document.charCount;
    return true;
  }

  /// Parses [payload] into an empty store, counting every skipped record.
  ///
  /// [FormatException] is the only thing `jsonDecode` throws for malformed JSON;
  /// type checks cover the rest, so this method cannot throw for any string
  /// input.
  void _absorb(String payload) {
    final Object? decoded;
    try {
      decoded = jsonDecode(payload);
    } on FormatException {
      _loadFailures++;
      return;
    }
    if (decoded is! Map<String, Object?>) {
      _loadFailures++;
      return;
    }
    final Object? rawDocuments = decoded['documents'];
    if (rawDocuments is! List<Object?>) {
      _loadFailures++;
      return;
    }
    for (final Object? entry in rawDocuments) {
      if (entry is! Map<String, Object?>) {
        _loadFailures++;
        continue;
      }
      try {
        _insert(CorpusDocument.fromJson(entry));
      } on FormatException {
        _loadFailures++;
      }
    }
  }

  /// The wire representation: a version marker plus every document's JSON.
  Map<String, Object?> _toJson() {
    return <String, Object?>{
      'version': 1,
      'documents': <Object?>[
        for (final CorpusDocument document in _documents) document.toJson(),
      ],
    };
  }
}

/// Truncates [text] to at most [maxChars] UTF-16 code units without splitting a
/// surrogate pair.
///
/// Kept as a file-private helper rather than a public utility: `maxChars` has
/// exactly this meaning everywhere in the corpus package, and exporting it would
/// invite callers to depend on a detail attached to no type.
String _truncate(String text, int maxChars) {
  if (maxChars <= 0) {
    return '';
  }
  if (text.length <= maxChars) {
    return text;
  }
  int end = maxChars;
  final int last = text.codeUnitAt(end - 1);
  if (last >= 0xD800 && last <= 0xDBFF) {
    end--;
  }
  return text.substring(0, end);
}