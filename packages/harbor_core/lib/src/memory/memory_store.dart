// The memory store: a bounded, deduplicating, lexically ranked memory.
//
// Storage is injected through [MemoryStorage] so this file stays `dart:io`-free
// and the identical store runs on the phone, on the server and in a browser tab
// (where storage is the browser's local storage). Persistence is write-behind:
// a caller may record memories in a hot loop without awaiting a disk write per
// memory, and [flush] is the one place that guarantees the file is current.

import 'dart:async';
import 'dart:math' as math;

import 'memory_entry.dart';

/// The persistence seam the store writes through.
///
/// Deliberately tiny — read, write, delete — because that is the whole contract
/// the three platforms can honour: a file on Android, a file on the server, and
/// a key in the browser. Anything richer would drag `dart:io` into the portable
/// package.
abstract class MemoryStorage {
  /// Returns the stored payload, or null when nothing has been written.
  Future<String?> read();

  /// Replaces the stored payload.
  Future<void> write(String payload);

  /// Removes the stored payload.
  Future<void> delete();
}

/// A memory that keeps nothing, for tests and for a platform without storage.
class InMemoryMemoryStorage implements MemoryStorage {
  /// Creates empty storage, optionally pre-filled with [payload].
  InMemoryMemoryStorage([this.payload]);

  /// The current payload, or null when nothing is stored.
  String? payload;

  @override
  Future<String?> read() async => payload;

  @override
  Future<void> write(String value) async => payload = value;

  @override
  Future<void> delete() async => payload = null;
}

/// How a memory was recalled, reported alongside the score.
class MemoryMatch {
  /// Creates a match.
  const MemoryMatch(this.entry, this.score);

  /// The memory that was recalled.
  final MemoryEntry entry;

  /// Relevance in `[0, 1]`, higher is closer to the query.
  final double score;
}

/// A ranked list of recalled memories plus the block rendered from it.
class MemoryRecall {
  /// Creates a recall result.
  const MemoryRecall({required this.matches, required this.block});

  /// An empty recall, used when nothing matched.
  static const MemoryRecall empty =
      MemoryRecall(matches: <MemoryMatch>[], block: '');

  /// The recalled memories, best first.
  final List<MemoryMatch> matches;

  /// The text injected into a prompt, or an empty string when nothing matched.
  final String block;

  /// Whether anything was recalled.
  bool get isEmpty => matches.isEmpty;

  /// The recalled entries, best first.
  List<MemoryEntry> get entries =>
      <MemoryEntry>[for (final MemoryMatch match in matches) match.entry];
}

/// A bounded store of [MemoryEntry] with lexical recall.
///
/// Capacity is enforced by pruning the *least useful* memory rather than the
/// oldest: value combines importance, how recently it was used, and how often,
/// so a decade-old identity fact outlives a chatty inferred topic.
class MemoryStore {
  /// Creates a store persisting through [storage].
  MemoryStore({
    required this.storage,
    this.capacity = 256,
    this.halfLife = const Duration(days: 30),
  });

  /// The persistence seam.
  final MemoryStorage storage;

  /// Maximum entries kept before the least useful is pruned.
  final int capacity;

  /// Age at which a memory's recency weight halves.
  final Duration halfLife;

  final Map<String, MemoryEntry> _entries = <String, MemoryEntry>{};
  bool _loaded = false;
  Future<void>? _pendingWrite;
  bool _dirty = false;

  /// Number of stored memories.
  int get length => _entries.length;

  /// Whether [load] has read storage at least once.
  bool get loaded => _loaded;

  /// Every entry, newest first.
  List<MemoryEntry> get entries {
    final List<MemoryEntry> sorted = _entries.values.toList()
      ..sort((MemoryEntry a, MemoryEntry b) =>
          b.createdAt.compareTo(a.createdAt),);
    return List<MemoryEntry>.unmodifiable(sorted);
  }

  /// Entries of one [kind], newest first.
  List<MemoryEntry> ofKind(MemoryKind kind) =>
      <MemoryEntry>[for (final MemoryEntry e in entries) if (e.kind == kind) e];

  /// Reads storage once. Never throws: an unreadable file leaves an empty store
  /// rather than preventing the assistant from starting.
  Future<void> load() async {
    if (_loaded) {
      return;
    }
    _loaded = true;
    try {
      final String? payload = await storage.read();
      if (payload == null || payload.trim().isEmpty) {
        return;
      }
      for (final MemoryEntry entry in MemoryEntry.decodeAll(payload)) {
        _entries[entry.id] = entry;
      }
    } on Object {
      _entries.clear();
    }
  }

  /// Inserts or updates [entry].
  ///
  /// A repeat of an existing memory merges into it: the newer text wins, the
  /// importance takes the maximum (an explicit "remember this" must not be
  /// diluted by a later vague mention) and the creation time is preserved so
  /// "when did I learn this" still means something.
  void add(MemoryEntry entry) {
    final MemoryEntry? existing = _entries[entry.id];
    if (existing == null) {
      _entries[entry.id] = entry;
    } else {
      _entries[entry.id] = entry.copyWith(
        createdAt: existing.createdAt,
        importance: math.max(existing.importance, entry.importance),
        uses: existing.uses,
        lastUsedAt: existing.lastUsedAt,
      );
    }
    _pruneIfNeeded();
    _markDirty();
  }

  /// Records [text] as [kind] and returns the stored entry.
  MemoryEntry remember(
    String text, {
    MemoryKind kind = MemoryKind.fact,
    String? subject,
    double importance = 0.5,
    String? source,
    DateTime? at,
  }) {
    final MemoryEntry entry = MemoryEntry.create(
      kind: kind,
      text: text,
      subject: subject,
      importance: importance,
      source: source,
      at: at,
    );
    add(entry);
    return entry;
  }

  /// Forgets one memory by id. Returns whether it existed.
  bool forget(String id) {
    final bool removed = _entries.remove(id) != null;
    if (removed) {
      _markDirty();
    }
    return removed;
  }

  /// Whether a memory with [id] exists.
  bool contains(String id) => _entries.containsKey(id);

  /// The memories most relevant to [query].
  ///
  /// Scoring is `idf-weighted keyword overlap × recency × importance`. Every
  /// factor is bounded and the result is normalised to `[0, 1]`, so a caller can
  /// apply one threshold across very different queries.
  List<MemoryMatch> recall(
    String query, {
    int limit = 6,
    double minScore = 0.05,
  }) {
    if (_entries.isEmpty || limit <= 0) {
      return const <MemoryMatch>[];
    }
    final List<String> queryTokens = MemoryEntry.keywordsOf(query);
    if (queryTokens.isEmpty) {
      return const <MemoryMatch>[];
    }
    // Document frequency per query token, so a token that appears in every
    // memory contributes almost nothing and a rare one dominates.
    final Map<String, int> documentFrequency = <String, int>{};
    for (final MemoryEntry entry in _entries.values) {
      for (final String token in entry.keywords.toSet()) {
        documentFrequency[token] = (documentFrequency[token] ?? 0) + 1;
      }
    }
    final DateTime now = DateTime.now().toUtc();
    final List<MemoryMatch> matches = <MemoryMatch>[];
    for (final MemoryEntry entry in _entries.values) {
      final double score = _score(
        entry,
        queryTokens,
        documentFrequency,
        now,
      );
      if (score >= minScore) {
        matches.add(MemoryMatch(entry, score));
      }
    }
    matches.sort((MemoryMatch a, MemoryMatch b) => b.score.compareTo(a.score));
    return matches.length <= limit
        ? matches
        : matches.sublist(0, limit);
  }

  /// Records that [entries] were just used, so usefulness decays correctly.
  void markUsed(Iterable<MemoryEntry> used, {DateTime? at}) {
    final DateTime moment = at ?? DateTime.now().toUtc();
    bool touched = false;
    for (final MemoryEntry entry in used) {
      final MemoryEntry? current = _entries[entry.id];
      if (current == null) {
        continue;
      }
      _entries[current.id] = current.copyWith(
        lastUsedAt: moment,
        uses: current.uses + 1,
      );
      touched = true;
    }
    if (touched) {
      _markDirty();
    }
  }

  /// Recalls and renders [query] into one prompt block.
  ///
  /// The block is labelled by kind so the model can tell a stated preference
  /// from an inferred topic, and it is capped in both entries and characters so
  /// a chatty store cannot crowd the actual conversation out of the window.
  MemoryRecall recallBlock(
    String query, {
    int limit = 6,
    int maxChars = 600,
  }) {
    final List<MemoryMatch> matches = recall(query, limit: limit);
    if (matches.isEmpty) {
      return MemoryRecall.empty;
    }
    final List<MemoryMatch> kept = <MemoryMatch>[];
    final StringBuffer buffer = StringBuffer();
    for (final MemoryMatch match in matches) {
      final String line =
          '- [${match.entry.label}] ${match.entry.text}';
      if (buffer.length + line.length + 1 > maxChars && kept.isNotEmpty) {
        break;
      }
      buffer.writeln(line);
      kept.add(match);
    }
    markUsed(<MemoryEntry>[for (final MemoryMatch m in kept) m.entry]);
    return MemoryRecall(
      matches: List<MemoryMatch>.unmodifiable(kept),
      block: buffer.toString().trimRight(),
    );
  }

  /// The `n` most useful memories, ignoring any query.
  ///
  /// Used by the memory panel and by the automatic bootstrap, where there is no
  /// query yet but the assistant still benefits from its standing context.
  List<MemoryEntry> mostUseful({int limit = 8}) {
    final DateTime now = DateTime.now().toUtc();
    final List<MemoryEntry> sorted = _entries.values.toList()
      ..sort((MemoryEntry a, MemoryEntry b) =>
          _usefulness(b, now).compareTo(_usefulness(a, now)),);
    return sorted.length <= limit ? sorted : sorted.sublist(0, limit);
  }

  /// Everything worth injecting before the first message of a session.
  ///
  /// Identity and preference outrank everything else here: knowing the user's
  /// name and how they want to be answered is worth more at turn zero than any
  /// number of remembered topics.
  MemoryRecall bootstrapBlock({int maxChars = 400}) {
    final List<MemoryEntry> ranked = <MemoryEntry>[
      ...ofKind(MemoryKind.identity),
      ...ofKind(MemoryKind.preference),
      ...ofKind(MemoryKind.goal),
      ...ofKind(MemoryKind.fact),
    ];
    if (ranked.isEmpty) {
      return MemoryRecall.empty;
    }
    final DateTime now = DateTime.now().toUtc();
    ranked.sort((MemoryEntry a, MemoryEntry b) =>
        _usefulness(b, now).compareTo(_usefulness(a, now)),);
    final StringBuffer buffer = StringBuffer();
    final List<MemoryMatch> matches = <MemoryMatch>[];
    for (final MemoryEntry entry in ranked) {
      final String line = '- [${entry.label}] ${entry.text}';
      if (buffer.length + line.length + 1 > maxChars && matches.isNotEmpty) {
        break;
      }
      buffer.writeln(line);
      matches.add(MemoryMatch(entry, _usefulness(entry, now)));
      if (matches.length >= 8) {
        break;
      }
    }
    if (matches.isEmpty) {
      return MemoryRecall.empty;
    }
    return MemoryRecall(
      matches: List<MemoryMatch>.unmodifiable(matches),
      block: buffer.toString().trimRight(),
    );
  }

  /// Writes the store through [storage], coalescing concurrent callers.
  ///
  /// A caller that records several memories in a row produces one write, not
  /// one per memory, because every call while a write is in flight joins the
  /// same future.
  Future<void> flush() {
    if (!_dirty) {
      return _pendingWrite ?? Future<void>.value();
    }
    _dirty = false;
    final String payload = MemoryEntry.encodeAll(entries);
    final Future<void> write = () async {
      try {
        await storage.write(payload);
      } on Object {
        // A failed write must not surface as an unhandled error from a
        // fire-and-forget save; the store stays in memory and the next flush
        // tries again.
        _dirty = true;
      } finally {
        _pendingWrite = null;
      }
    }();
    _pendingWrite = write;
    return write;
  }

  /// Deletes every memory, in memory and on disk.
  Future<void> clear() async {
    _entries.clear();
    _dirty = false;
    _pendingWrite = null;
    try {
      await storage.delete();
    } on Object {
      // Clearing is best-effort for the same reason loading is.
    }
  }

  /// Replaces the whole store, used by an import or a restore.
  Future<void> replaceAll(List<MemoryEntry> entries) async {
    _entries
      ..clear()
      ..addEntries(<MapEntry<String, MemoryEntry>>[
        for (final MemoryEntry entry in entries)
          MapEntry<String, MemoryEntry>(entry.id, entry),
      ]);
    _pruneIfNeeded();
    _dirty = true;
    await flush();
  }

  /// A JSON snapshot for diagnostics and for the memory panel.
  Map<String, Object?> describe() => <String, Object?>{
        'entries': length,
        'capacity': capacity,
        'kinds': <String, Object?>{
          for (final MemoryKind kind in MemoryKind.values)
            kind.wire: ofKind(kind).length,
        },
        'loaded': _loaded,
      };

  void _markDirty() {
    _dirty = true;
    // Fire-and-forget write-behind. `flush` is the awaitable form and is what a
    // lifecycle callback or a test should use.
    unawaited(flush());
  }

  void _pruneIfNeeded() {
    if (_entries.length <= capacity) {
      return;
    }
    final DateTime now = DateTime.now().toUtc();
    final List<MemoryEntry> byValue = _entries.values.toList()
      ..sort((MemoryEntry a, MemoryEntry b) =>
          _usefulness(a, now).compareTo(_usefulness(b, now)),);
    final int excess = _entries.length - capacity;
    for (int i = 0; i < excess; i++) {
      _entries.remove(byValue[i].id);
    }
  }

  double _score(
    MemoryEntry entry,
    List<String> queryTokens,
    Map<String, int> documentFrequency,
    DateTime now,
  ) {
    final Set<String> entryTokens = entry.keywords.toSet();
    double overlap = 0;
    double weightSum = 0;
    final int total = _entries.length;
    for (final String token in queryTokens) {
      final int frequency = documentFrequency[token] ?? 0;
      // Smoothed inverse document frequency, always positive.
      final double idf =
          math.log((total + 1) / (frequency + 1)).abs() + 0.5;
      weightSum += idf;
      if (entryTokens.contains(token)) {
        overlap += idf;
      }
    }
    if (weightSum <= 0) {
      return 0;
    }
    final double lexical = overlap / weightSum;
    if (lexical <= 0) {
      return 0;
    }
    return (lexical * _recency(entry, now) * (0.5 + 0.5 * entry.importance))
        .clamp(0.0, 1.0);
  }

  /// Exponential decay with the store's half-life.
  double _recency(MemoryEntry entry, DateTime now) {
    final double days =
        now.difference(entry.lastUsedAt).inMinutes / (60 * 24);
    final double halfLifeDays = math.max(halfLife.inMinutes / (60 * 24), 0.001);
    return math.pow(0.5, days / halfLifeDays).toDouble().clamp(0.0, 1.0);
  }

  /// A memory's standing value: usefulness when no query is available.
  double _usefulness(MemoryEntry entry, DateTime now) {
    final double useBoost = math.log(entry.uses + 1) / 2;
    return ((0.5 + 0.5 * entry.importance) *
            _recency(entry, now) *
            (1 + useBoost.clamp(0.0, 1.0)))
        .clamp(0.0, 1.0);
  }
}