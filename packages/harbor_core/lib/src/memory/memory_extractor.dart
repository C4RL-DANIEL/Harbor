// Automatic memory extraction: turning what the user says into what Harbor
// remembers, with no model and no second network call.
//
// There is no pretrained classifier behind this. A small, ordered table of
// patterns runs over each finished sentence and, when one fires, records the
// sentence as a memory of the matching kind. The table is deliberately biased
// towards precision: a missed memory costs one re-statement, while a false one
// corrupts every future prompt with a claim the user never made. Patterns
// therefore require an explicit first-person construction ("I prefer …",
// "my name is …") or a direct instruction ("remember that …"), and never fire
// on a question or an imperative aimed at the assistant.
//
// Extraction is written to be fed a token stream. [MemoryExtractor.observe]
// accumulates streamed assistant tokens and returns memories only for completed
// sentences, which is what lets memory learning happen *during* a reply without
// waiting for it to finish or re-reading it afterwards.

import 'memory_entry.dart';

/// The outcome of scanning one piece of text.
class MemoryExtraction {
  /// Creates an extraction result.
  const MemoryExtraction({
    required this.entries,
    required this.explicit,
  });

  /// An empty result.
  static const MemoryExtraction none =
      MemoryExtraction(entries: <MemoryEntry>[], explicit: false);

  /// Memories worth storing, best first.
  final List<MemoryEntry> entries;

  /// Whether the user explicitly asked for something to be remembered.
  ///
  /// The assistant says so out loud when this is true; silently obeying a
  /// "remember this" and saying nothing is indistinguishable from ignoring it.
  final bool explicit;

  /// Whether anything was found.
  bool get isEmpty => entries.isEmpty;
}

/// One ordered extraction rule.
class _Rule {
  const _Rule({
    required this.pattern,
    required this.kind,
    required this.subject,
    required this.importance,
    this.requiresCue = false,
  });

  final RegExp pattern;
  final MemoryKind kind;
  final String subject;
  final double importance;

  /// Whether the captured group is itself a cue used to pick the kind, for the
  /// explicit "remember …" form whose kind depends on what follows.
  final bool requiresCue;
}

/// Extracts memories from a user's or assistant's text.
class MemoryExtractor {
  /// Creates an extractor.
  ///
  /// [enabled] lets a caller keep the object in place while turning learning
  /// off, which is what the automatic-mode switch does.
  MemoryExtractor({this.enabled = true, this.maxEntriesPerText = 4});

  /// Whether extraction runs at all.
  bool enabled;

  /// Ceiling on memories taken from one message.
  ///
  /// A pasted biography could otherwise fill the store in a single turn; the
  /// first few statements are also the ones a speaker led with, and so the ones
  /// most likely to be the point.
  final int maxEntriesPerText;

  final StringBuffer _pending = StringBuffer();

  /// Extracts every memory in [text].
  ///
  /// The result is deduplicated by id, so a message that repeats a preference
  /// yields one entry.
  MemoryExtraction extract(String text) {
    if (!enabled || text.trim().isEmpty) {
      return MemoryExtraction.none;
    }
    final List<MemoryEntry> found = <MemoryEntry>[];
    final Set<String> seen = <String>{};
    bool explicit = false;

    for (final String sentence in _sentences(text)) {
      for (final _Rule rule in _rules) {
        final RegExpMatch? match = rule.pattern.firstMatch(sentence);
        if (match == null) {
          continue;
        }
        final String captured = _payload(match);
        if (captured.length < 2) {
          continue;
        }
        final String statement = _statement(sentence, match);
        final MemoryKind kind = rule.requiresCue
            ? _kindFromCue(captured)
            : rule.kind;
        final double importance =
            rule.requiresCue ? 1.0 : rule.importance;
        if (rule.requiresCue) {
          explicit = true;
        }
        final MemoryEntry entry = MemoryEntry.create(
          kind: kind,
          text: statement,
          subject: rule.subject,
          source: 'conversation',
          importance: importance,
        );
        if (seen.add(entry.id)) {
          found.add(entry);
        }
        // One rule per sentence: the most specific rule is listed first, so the
        // first match is the best interpretation and a later broader one would
        // only add a weaker duplicate.
        break;
      }
      if (found.length >= maxEntriesPerText) {
        break;
      }
    }
    if (found.isEmpty) {
      return MemoryExtraction.none;
    }
    return MemoryExtraction(
      entries: List<MemoryEntry>.unmodifiable(found),
      explicit: explicit,
    );
  }

  /// Feeds streamed [piece]s and returns memories for completed sentences.
  ///
  /// Text is held until a sentence boundary or [flushChars] accumulate, so the
  /// extractor never sees a half-written "my name is Al" and stores an
  /// accidental name. [flush] drains whatever remains at the end of a turn.
  MemoryExtraction observe(String piece, {int flushChars = 240}) {
    if (!enabled) {
      return MemoryExtraction.none;
    }
    _pending.write(piece);
    final String buffered = _pending.toString();
    if (buffered.length < flushChars && !_hasBoundary(buffered)) {
      return MemoryExtraction.none;
    }
    return flush();
  }

  /// Extracts from whatever is buffered and clears the buffer.
  MemoryExtraction flush() {
    if (_pending.isEmpty) {
      return MemoryExtraction.none;
    }
    final String text = _pending.toString();
    _pending.clear();
    return extract(text);
  }

  /// Clears the streaming buffer, e.g. when a turn is abandoned.
  void reset() => _pending.clear();

  /// Whether anything is buffered for the next flush.
  bool get hasPending => _pending.isNotEmpty;

  /// Splits [text] into sentences, keeping the terminator.
  ///
  /// A plain `split('.')` would break on "3.5" and on an abbreviation; a
  /// boundary is only honoured when the punctuation is followed by whitespace
  /// or the end of the text.
  static List<String> _sentences(String text) {
    final List<String> out = <String>[];
    final StringBuffer current = StringBuffer();
    final List<int> units = text.trim().runes.toList(growable: false);
    for (int i = 0; i < units.length; i++) {
      final int unit = units[i];
      current.writeCharCode(unit);
      final bool terminator =
          unit == 0x2E || unit == 0x21 || unit == 0x3F || unit == 0x0A;
      final bool atEnd = i == units.length - 1;
      final bool followedBySpace =
          !atEnd && (units[i + 1] == 0x20 || units[i + 1] == 0x0A);
      if (terminator && (atEnd || followedBySpace)) {
        final String sentence = current.toString().trim();
        if (sentence.isNotEmpty) {
          out.add(sentence);
        }
        current.clear();
      }
    }
    final String tail = current.toString().trim();
    if (tail.isNotEmpty) {
      out.add(tail);
    }
    return out;
  }

  /// Whether [text] holds a sentence boundary.
  ///
  /// A terminator followed by a space is the obvious case, but in a token
  /// stream the period usually arrives as the *last* character of the final
  /// piece with nothing after it yet. Requiring a following space meant a
  /// completed sentence went unrecognised until the next token turned up, so the
  /// last thing the user said could only be learned at end-of-turn flush time.
  static bool _hasBoundary(String text) =>
      text.contains('. ') ||
      text.contains('! ') ||
      text.contains('? ') ||
      text.contains('\n') ||
      RegExp(r'[.!?]$').hasMatch(text);

  /// The meaningful capture of [match].
  ///
  /// Rules are written with the payload in group 1 and any optional connector in
  /// group 2, so this prefers the last non-captured-connector group.
  static String _payload(RegExpMatch match) {
    for (int group = 1; group < match.groupCount + 1; group++) {
      final String? value = match.group(group);
      if (value != null && value.trim().length >= 2) {
        return value.trim();
      }
    }
    return '';
  }

  /// The full sentence, so the stored text reads as the user wrote it.
  static String _statement(String sentence, RegExpMatch match) {
    // Keep the whole sentence when it is short; otherwise keep from the match,
    // so a paragraph-long sentence does not store its unrelated preamble.
    if (sentence.length <= 200) {
      return sentence;
    }
    final String fromMatch = sentence.substring(match.start).trim();
    return fromMatch.length <= 200 ? fromMatch : fromMatch.substring(0, 200);
  }

  /// Picks a memory kind for an explicit "remember …" payload.
  static MemoryKind _kindFromCue(String payload) {
    final String lower = payload.toLowerCase();
    if (RegExp(r'\b(prefer|like|love|hate|dislike|favourite|favorite|always|never|avoid)\b')
        .hasMatch(lower)) {
      return MemoryKind.preference;
    }
    if (RegExp(r'\b(want|need|plan|goal|trying|working on|learn|build)\b')
        .hasMatch(lower)) {
      return MemoryKind.goal;
    }
    if (RegExp(r'\b(my name|i am called|call me)\b').hasMatch(lower)) {
      return MemoryKind.identity;
    }
    return MemoryKind.fact;
  }

  /// Ordered most-specific first; the first match in a sentence wins.
  static final List<_Rule> _rules = <_Rule>[
    // ---- Explicit instruction -------------------------------------------
    _Rule(
      pattern: RegExp(
        r'\b(?:remember|note|keep in mind)\s+(?:that\s+)?(.+)$',
        caseSensitive: false,
      ),
      kind: MemoryKind.fact,
      subject: 'explicit',
      importance: 1.0,
      requiresCue: true,
    ),

    // ---- Identity --------------------------------------------------------
    _Rule(
      pattern: RegExp(
        r'\bmy name is\s+(.+)$|\bi am called\s+(.+)$|\bcall me\s+(.+)$',
        caseSensitive: false,
      ),
      kind: MemoryKind.identity,
      subject: 'name',
      importance: 0.95,
    ),
    _Rule(
      pattern: RegExp(
        r"\bi(?:'m| am)\s+(?:a|an)\s+([a-z][a-z0-9 +#-]{1,60}?)(?:[.!?,]|$)",
        caseSensitive: false,
      ),
      kind: MemoryKind.identity,
      subject: 'role',
      importance: 0.85,
    ),
    _Rule(
      pattern: RegExp(
        r'\bmy (?:preferred )?(?:language|locale) is\s+(.+)$',
        caseSensitive: false,
      ),
      kind: MemoryKind.identity,
      subject: 'language',
      importance: 0.9,
    ),
    // "Write in Portuguese", "answer in Spanish": about the conversation, not
    // about the user, but stored as a preference precisely because it is one.
    _Rule(
      pattern: RegExp(
        r'\b(?:write|answer|respond|reply)(?:\s+to me)?\s+in\s+([A-Za-z][A-Za-z -]{2,30})$',
        caseSensitive: false,
      ),
      kind: MemoryKind.preference,
      subject: 'language',
      importance: 0.9,
    ),

    // ---- Preferences -----------------------------------------------------
    _Rule(
      pattern: RegExp(
        r"\bi(?:'d| would) rather\s+(?:you\s+)?(.+)$",
        caseSensitive: false,
      ),
      kind: MemoryKind.preference,
      subject: 'preference',
      importance: 0.9,
    ),
    _Rule(
      pattern: RegExp(
        r"\bi\s+(?:do not|don't|dont)\s+(?:like|want|use|need)\s+(.+)$",
        caseSensitive: false,
      ),
      kind: MemoryKind.preference,
      subject: 'preference',
      importance: 0.85,
    ),
    _Rule(
      pattern: RegExp(
        r'\bi\s+(?:prefer|like|love|enjoy|hate|dislike|avoid|always|never|usually)\s+(.+)$',
        caseSensitive: false,
      ),
      kind: MemoryKind.preference,
      subject: 'preference',
      importance: 0.85,
    ),
    _Rule(
      pattern: RegExp(
        r'\bmy favou?rite\s+(?:\w+\s+)?is\s+(.+)$',
        caseSensitive: false,
      ),
      kind: MemoryKind.preference,
      subject: 'favourite',
      importance: 0.85,
    ),

    // ---- Goals -----------------------------------------------------------
    _Rule(
      pattern: RegExp(
        r'\bi\s+(?:want|need|plan|intend|hope)\s+to\s+(.+)$',
        caseSensitive: false,
      ),
      kind: MemoryKind.goal,
      subject: 'goal',
      importance: 0.8,
    ),
    _Rule(
      pattern: RegExp(
        r"\bi(?:'m| am)\s+(?:trying|working|planning)\s+(?:to|on)\s+(.+)$",
        caseSensitive: false,
      ),
      kind: MemoryKind.goal,
      subject: 'goal',
      importance: 0.8,
    ),
    _Rule(
      pattern: RegExp(r'\bmy goal is\s+(.+)$', caseSensitive: false),
      kind: MemoryKind.goal,
      subject: 'goal',
      importance: 0.85,
    ),

    // ---- Facts -----------------------------------------------------------
    _Rule(
      pattern: RegExp(
        r'\bi\s+(?:live|am based|reside)\s+in\s+(.+)$',
        caseSensitive: false,
      ),
      kind: MemoryKind.fact,
      subject: 'location',
      importance: 0.9,
    ),
    _Rule(
      pattern: RegExp(r"\bi(?:'m| am) from\s+(.+)$", caseSensitive: false),
      kind: MemoryKind.fact,
      subject: 'origin',
      importance: 0.85,
    ),
    _Rule(
      pattern: RegExp(
        r'\bi\s+(?:work|study|teach)\s+(?:at|as|in|for)\s+(.+)$',
        caseSensitive: false,
      ),
      kind: MemoryKind.fact,
      subject: 'occupation',
      importance: 0.85,
    ),
    // A possessive fact: "my birthday is …", "my sister is called …", but not
    // "my question is …", which is about the conversation and not the user.
    _Rule(
      pattern: RegExp(
        r'\bmy\s+('
        r'birthday|anniversary|timezone|time zone|email|phone|address|age|'
        r'blood type|allergies|medication|condition|sister|brother|mother|'
        r'father|wife|husband|partner|son|daughter|dog|cat|car|job|company|'
        r'team|school|university|degree|major|hometown'
        r')\s+is\s+(.+)$',
        caseSensitive: false,
      ),
      kind: MemoryKind.fact,
      subject: 'personal',
      importance: 0.85,
    ),
  ];
}