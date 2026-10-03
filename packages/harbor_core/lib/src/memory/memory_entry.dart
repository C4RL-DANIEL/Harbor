// Persistent memory: one thing the assistant remembers.
//
// Harbor's memory is deliberately *not* a vector database. A vector store would
// need an embedding model, and an embedding model would need a pretrained
// checkpoint that this repository does not ship. What it uses instead is the
// oldest retrieval method that still works: normalised keyword overlap, scored
// with inverse document frequency, plus a recency half-life and an explicit
// importance. That is deterministic, costs no parameters, runs in a browser tab
// and can be unit-tested exactly — all of which are properties a phone-sized
// assistant actually needs.
//
// A memory is a value object. The store owns identity and pruning; an entry only
// knows how to score itself and how to round-trip through JSON.

import 'dart:convert';

/// What kind of thing a memory records.
///
/// The kind is not decoration: it decides how confidently the extractor may
/// create one, how the retrieval block labels it, and which memories pruning
/// sacrifices first when the store is full.
enum MemoryKind {
  /// Who the user is: their name, their language, how they want to be addressed.
  identity('identity'),

  /// A stable like, dislike or style preference ("prefers short answers").
  preference('preference'),

  /// A durable fact about the user or their world ("lives in Lisbon").
  fact('fact'),

  /// Something the user is trying to achieve across turns.
  goal('goal'),

  /// A subject the user keeps returning to, inferred when no rule fires.
  topic('topic'),

  /// A compact record of an exchange worth remembering verbatim.
  episode('episode');

  const MemoryKind(this.wire);

  /// The stable name written to storage.
  ///
  /// Spelled out rather than using `Enum.name` so a rename of the Dart symbol
  /// cannot silently invalidate every stored memory.
  final String wire;

  /// Parses [wire], returning null for a value this build does not know.
  ///
  /// Returning null rather than throwing is what lets a newer build write a
  /// memory kind that an older one skips instead of discarding the whole file.
  static MemoryKind? tryParse(String value) {
    for (final MemoryKind kind in MemoryKind.values) {
      if (kind.wire == value) {
        return kind;
      }
    }
    return null;
  }
}

/// One remembered statement, with the statistics retrieval needs.
class MemoryEntry {
  /// Creates an entry. Prefer [MemoryEntry.create], which derives the id and the
  /// keyword set so two callers cannot disagree about them.
  const MemoryEntry({
    required this.id,
    required this.kind,
    required this.text,
    required this.keywords,
    required this.createdAt,
    required this.lastUsedAt,
    this.subject,
    this.source,
    this.importance = 0.5,
    this.uses = 0,
  });

  /// Builds an entry, deriving a stable [id] and the keyword set from [text].
  ///
  /// The id is a hash of kind and normalised text, so saying the same thing
  /// twice updates one memory instead of growing the store without bound.
  factory MemoryEntry.create({
    required MemoryKind kind,
    required String text,
    String? subject,
    String? source,
    double importance = 0.5,
    DateTime? at,
  }) {
    final String cleaned = clean(text);
    final DateTime moment = at ?? DateTime.now().toUtc();
    final String subjectKey = subject == null ? '' : normalize(subject);
    return MemoryEntry(
      id: stableId(kind, subjectKey.isEmpty ? cleaned : '$subjectKey:$cleaned'),
      kind: kind,
      text: cleaned,
      keywords: keywordsOf('$subjectKey $cleaned'),
      createdAt: moment,
      lastUsedAt: moment,
      subject: subjectKey.isEmpty ? null : subjectKey,
      source: source,
      importance: importance.clamp(0.0, 1.0),
    );
  }

  /// Stable identity, used to deduplicate repeated statements.
  final String id;

  /// What kind of thing this is.
  final MemoryKind kind;

  /// The remembered content, already cleaned.
  final String text;

  /// Normalised tokens used for lexical scoring.
  final List<String> keywords;

  /// When the memory was first recorded, in UTC.
  final DateTime createdAt;

  /// When retrieval last returned it, in UTC.
  final DateTime lastUsedAt;

  /// Normalised subject key, e.g. `name` or `location`, when one is known.
  final String? subject;

  /// Where it came from, e.g. `conversation`.
  final String? source;

  /// How important the memory is, in `[0, 1]`. The user's explicit "remember
  /// this" is 1.0; an inferred topic is much lower.
  final double importance;

  /// How many times retrieval has surfaced it.
  final int uses;

  /// A copy with selected fields replaced.
  MemoryEntry copyWith({
    MemoryKind? kind,
    String? text,
    List<String>? keywords,
    DateTime? createdAt,
    DateTime? lastUsedAt,
    String? subject,
    String? source,
    double? importance,
    int? uses,
  }) {
    return MemoryEntry(
      id: id,
      kind: kind ?? this.kind,
      text: text ?? this.text,
      keywords: keywords ?? this.keywords,
      createdAt: createdAt ?? this.createdAt,
      lastUsedAt: lastUsedAt ?? this.lastUsedAt,
      subject: subject ?? this.subject,
      source: source ?? this.source,
      importance: importance ?? this.importance,
      uses: uses ?? this.uses,
    );
  }

  /// A one-line label for the retrieval block and the memory list UI.
  String get label {
    switch (kind) {
      case MemoryKind.identity:
        return 'Identity';
      case MemoryKind.preference:
        return 'Preference';
      case MemoryKind.fact:
        return 'Fact';
      case MemoryKind.goal:
        return 'Goal';
      case MemoryKind.topic:
        return 'Topic';
      case MemoryKind.episode:
        return 'Episode';
    }
  }

  /// JSON form written to storage.
  Map<String, Object?> toJson() => <String, Object?>{
        'id': id,
        'kind': kind.wire,
        'text': text,
        'keywords': keywords,
        'created_at': createdAt.toUtc().toIso8601String(),
        'last_used_at': lastUsedAt.toUtc().toIso8601String(),
        if (subject != null) 'subject': subject,
        if (source != null) 'source': source,
        'importance': importance,
        'uses': uses,
      };

  /// Parses [toJson], returning null for an entry this build cannot use.
  ///
  /// Null rather than an exception: one unreadable line in a memory file must
  /// not cost the user every other memory they ever saved.
  static MemoryEntry? tryFromJson(Map<String, Object?> json) {
    final Object? kindRaw = json['kind'];
    final Object? textRaw = json['text'];
    if (kindRaw is! String || textRaw is! String || textRaw.trim().isEmpty) {
      return null;
    }
    final MemoryKind? kind = MemoryKind.tryParse(kindRaw);
    if (kind == null) {
      return null;
    }
    final String cleaned = clean(textRaw);
    if (cleaned.isEmpty) {
      return null;
    }
    final DateTime created =
        _parseTime(json['created_at']) ?? DateTime.now().toUtc();
    final DateTime used = _parseTime(json['last_used_at']) ?? created;
    final Object? keywordRaw = json['keywords'];
    final List<String> keywords = keywordRaw is List<Object?>
        ? <String>[
            for (final Object? keyword in keywordRaw)
              if (keyword is String && keyword.trim().isNotEmpty)
                keyword.trim().toLowerCase(),
          ]
        : keywordsOf(cleaned);
    final Object? importanceRaw = json['importance'];
    return MemoryEntry(
      id: json['id'] is String
          ? json['id']! as String
          : stableId(kind, cleaned),
      kind: kind,
      text: cleaned,
      keywords: keywords.isEmpty ? keywordsOf(cleaned) : keywords,
      createdAt: created,
      lastUsedAt: used,
      subject: json['subject'] is String ? json['subject']! as String : null,
      source: json['source'] is String ? json['source']! as String : null,
      importance: importanceRaw is num
          ? importanceRaw.toDouble().clamp(0.0, 1.0)
          : 0.5,
      uses: json['uses'] is int ? json['uses']! as int : 0,
    );
  }

  @override
  String toString() => '$label($id): $text';

  /// Lower-cases, strips surrounding punctuation and collapses whitespace.
  static String clean(String value) {
    String out = value.trim();
    out = out.replaceAll(RegExp(r'\s+'), ' ');
    out = out.replaceAll(RegExp(r'^[\s\-–—:;,."`\x27]+'), '');
    out = out.replaceAll(RegExp(r'[\s"`\x27]+$'), '');
    out = out.replaceAll(RegExp(r'[.,;:!?]+$'), '');
    return out.trim();
  }

  /// Normalises [value] to lowercase alphanumerics joined by single spaces.
  static String normalize(String value) {
    return value
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z0-9]+'), ' ')
        .trim();
  }

  /// The meaningful tokens of [value].
  ///
  /// Stop words are dropped because they appear in nearly every entry and so
  /// carry no retrieval signal; the idf term downstream would down-weight them
  /// anyway, and removing them keeps the stored file smaller.
  static List<String> keywordsOf(String value) {
    final List<String> out = <String>[];
    final Set<String> seen = <String>{};
    for (final String token in normalize(value).split(' ')) {
      if (token.length < 2 || _stopWords.contains(token) || !seen.add(token)) {
        continue;
      }
      out.add(token);
      if (out.length >= 32) {
        break;
      }
    }
    return out;
  }

  /// A stable, dependency-free FNV-1a hash rendered as a short hex id.
  ///
  /// The same input always yields the same id on every platform, which is what
  /// makes "saying it twice" an update rather than a duplicate.
  static String stableId(MemoryKind kind, String normalizedText) {
    final String payload = '${kind.wire}|${normalize(normalizedText)}';
    int hash = 0x811c9dc5;
    for (final int unit in payload.codeUnits) {
      hash ^= unit;
      hash = (hash * 0x01000193) & 0xFFFFFFFF;
    }
    return 'mem_${hash.toRadixString(16).padLeft(8, '0')}';
  }

  static DateTime? _parseTime(Object? value) {
    if (value is! String) {
      return null;
    }
    return DateTime.tryParse(value)?.toUtc();
  }

  /// Encodes [entries] as a JSON array, oldest first.
  static String encodeAll(List<MemoryEntry> entries) {
    return jsonEncode(<Object?>[
      for (final MemoryEntry entry in entries) entry.toJson(),
    ]);
  }

  /// Decodes the array written by [encodeAll], skipping unusable entries.
  static List<MemoryEntry> decodeAll(String payload) {
    final Object? decoded = jsonDecode(payload);
    if (decoded is! List<Object?>) {
      throw const FormatException('a memory file must be a JSON array');
    }
    final List<MemoryEntry> entries = <MemoryEntry>[];
    for (final Object? item in decoded) {
      if (item is! Map<String, Object?>) {
        continue;
      }
      final MemoryEntry? entry = MemoryEntry.tryFromJson(item);
      if (entry != null) {
        entries.add(entry);
      }
    }
    return entries;
  }

  /// Words that carry no retrieval signal on their own.
  static const Set<String> _stopWords = <String>{
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'your', 'with', 'that',
    'this', 'have', 'has', 'had', 'was', 'were', 'will', 'would', 'can',
    'could', 'should', 'about', 'into', 'from', 'they', 'them', 'their',
    'what', 'when', 'where', 'which', 'who', 'how', 'why', 'there', 'here',
    'then', 'than', 'also', 'just', 'some', 'any', 'all', 'out', 'own',
    'too', 'very', 'much', 'more', 'most', 'other', 'only', 'over', 'under',
    'again', 'been', 'being', 'does', 'did', 'doing', 'its', 'it', 'is',
    'am', 'be', 'as', 'at', 'by', 'in', 'of', 'on', 'or', 'to', 'up', 'we',
    'us', 'our', 'i', 'me', 'my', 'mine', 'he', 'she', 'his', 'her',
  };
}