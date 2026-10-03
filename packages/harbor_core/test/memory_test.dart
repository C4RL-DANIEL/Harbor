// Tests for the memory subsystem: entry identity and JSON round-tripping, the
// bounded store's ranking / pruning / persistence, and the rule-based extractor.
//
// Store scoring decays against the real clock, so the timestamps that drive
// ranking are built as explicit offsets from a single captured instant rather
// than fixed calendar dates: the *difference* between two memories is then exact
// and independent of when the suite runs. Where a test asserts a precisely
// round-tripped value it uses a fixed `DateTime.utc` instead.

import 'package:harbor_core/harbor_core.dart';
import 'package:test/test.dart';

/// Runs every memory test.
void main() {
  /// The instant the suite started, used as the origin for relative timestamps.
  final DateTime now = DateTime.now().toUtc();

  /// A UTC moment the given amount before [now].
  DateTime ago({int days = 0, int minutes = 0, int hours = 0}) {
    return now.subtract(Duration(days: days, hours: hours, minutes: minutes));
  }

  /// A store over throwaway in-memory storage.
  MemoryStore newStore({int capacity = 256}) {
    return MemoryStore(storage: InMemoryMemoryStorage(), capacity: capacity);
  }

  group('MemoryKind', () {
    test('carries a stable wire name, a display label, and a tolerant parser', () {
      expect(MemoryKind.tryParse('identity'), MemoryKind.identity);
      expect(MemoryKind.tryParse('goal'), MemoryKind.goal);
      expect(MemoryKind.tryParse('episode'), MemoryKind.episode);
      // A kind written by a newer build must parse to null so the reader can
      // skip one entry instead of rejecting the whole file.
      expect(MemoryKind.tryParse('belief'), isNull);
      expect(MemoryKind.tryParse('Identity'), isNull);

      final Map<String, String> labelsByWire = <String, String>{
        for (final MemoryKind kind in MemoryKind.values)
          kind.wire: MemoryEntry.create(kind: kind, text: 'some statement').label,
      };
      expect(labelsByWire, <String, String>{
        'identity': 'Identity',
        'preference': 'Preference',
        'fact': 'Fact',
        'goal': 'Goal',
        'topic': 'Topic',
        'episode': 'Episode',
      });
    });
  });

  group('MemoryEntry.create identity', () {
    test('maps the same statement said twice to one id, and folds the subject in', () {
      final MemoryEntry first = MemoryEntry.create(
        kind: MemoryKind.identity,
        text: 'My name is Daniel.',
        at: ago(days: 4),
      );
      final MemoryEntry second = MemoryEntry.create(
        kind: MemoryKind.identity,
        text: 'My name is Daniel.',
        at: ago(days: 1),
      );
      expect(first.id, isNotEmpty);
      expect(first.id, startsWith('mem_'));
      expect(first.text, 'My name is Daniel');
      // Case, punctuation and spacing are normalised before hashing, so these
      // other spellings of the same statement collapse onto the same id.
      final MemoryEntry third = MemoryEntry.create(
        kind: MemoryKind.identity,
        text: '  my NAME is   daniel  ',
        at: now,
      );
      expect(second.id, first.id);
      expect(third.id, first.id);
      // `clean()` normalises whitespace and punctuation but deliberately keeps
      // the user's own capitalisation: a memory is a record of what was said,
      // not a normalised key. The id is what ignores case.
      expect(third.text, 'my NAME is daniel');
      // A stated subject takes part in the id, so a keyed statement is its own
      // memory rather than a duplicate of the bare sentence.
      final MemoryEntry keyed = MemoryEntry.create(
        kind: MemoryKind.identity,
        text: 'My name is Daniel.',
        subject: 'name',
        at: ago(days: 4),
      );
      expect(keyed.id, isNot(first.id));
    });

    test('derives a different id for the same text recorded under another kind', () {
      final MemoryEntry asFact = MemoryEntry.create(
        kind: MemoryKind.fact,
        text: 'I live in Lisbon.',
        at: ago(days: 2),
      );
      final MemoryEntry asGoal = MemoryEntry.create(
        kind: MemoryKind.goal,
        text: 'I live in Lisbon.',
        at: ago(days: 2),
      );
      expect(asFact.id, isNot(asGoal.id));
    });

    test('keeps subject and keyword set consistent with the cleaned text', () {
      final MemoryEntry bare = MemoryEntry.create(
        kind: MemoryKind.fact,
        text: 'My sister is called Ana.',
        at: ago(days: 1),
      );
      final MemoryEntry keyed = MemoryEntry.create(
        kind: MemoryKind.fact,
        text: 'My sister is called Ana.',
        subject: 'Sister',
        at: ago(days: 1),
      );
      expect(bare.subject, isNull);
      expect(keyed.subject, 'sister');
      // The subject participates in both the id and the keyword set.
      expect(keyed.id, isNot(bare.id));
      expect(keyed.keywords, contains('sister'));
      expect(bare.keywords, contains('sister'));
      expect(keyed.keywords, contains('ana'));
      expect(keyed.source, isNull);
    });

    test('clamps importance into [0, 1]', () {
      expect(
        MemoryEntry.create(kind: MemoryKind.fact, text: 'a thing', importance: 5).importance,
        1.0,
      );
      expect(
        MemoryEntry.create(kind: MemoryKind.fact, text: 'a thing', importance: -3).importance,
        0.0,
      );
      expect(
        MemoryEntry.create(kind: MemoryKind.fact, text: 'a thing').importance,
        0.5,
        reason: 'the default importance is middling, not maximal',
      );
    });
  });

  group('MemoryEntry keywords and cleaning', () {
    test('keywordsOf drops stop words and short tokens and dedupes', () {
      expect(
        MemoryEntry.keywordsOf('the quick and the brown fox a b c xy'),
        <String>['quick', 'brown', 'fox', 'xy'],
      );
      expect(MemoryEntry.keywordsOf('who are you and what is this'), isEmpty);
      expect(MemoryEntry.keywordsOf(''), isEmpty);
      // The same token appearing twice contributes one entry.
      expect(MemoryEntry.keywordsOf('kayaking kayak kayaking'), <String>['kayaking', 'kayak']);
    });

    test('keywordsOf caps the stored token list at 32', () {
      final String long = List<String>.generate(60, (int i) => 'word$i').join(' ');
      final List<String> keywords = MemoryEntry.keywordsOf(long);
      expect(keywords, hasLength(32));
      expect(keywords.first, 'word0');
      expect(keywords.last, 'word31');
    });

    test('clean strips surrounding punctuation and whitespace but keeps inner apostrophes', () {
      expect(MemoryEntry.clean('   "Don\'t stop!"   '), 'Don\'t stop');
      expect(MemoryEntry.clean('...well, ok...'), 'well, ok');
      expect(MemoryEntry.clean('Hello, world!'), 'Hello, world');
      expect(MemoryEntry.clean('— note: it\'s fine;  '), 'note: it\'s fine');
      expect(MemoryEntry.clean('a   b\n\tc'), 'a b c');
      expect(MemoryEntry.clean('!!!'), isEmpty);
    });
  });

  group('MemoryEntry JSON', () {
    test('encodeAll and decodeAll round-trip every field', () {
      final DateTime created = DateTime.utc(2024, 3, 4, 5, 6, 7);
      final DateTime used = DateTime.utc(2024, 9, 10, 11, 12, 13);
      final List<MemoryEntry> entries = <MemoryEntry>[
        MemoryEntry.create(
              kind: MemoryKind.identity,
              text: 'My name is Daniel.',
              subject: 'name',
              source: 'conversation',
              importance: 0.95,
              at: created,
            )
            .copyWith(lastUsedAt: used, uses: 3),
        MemoryEntry.create(kind: MemoryKind.topic, text: 'Dart generics.', at: created),
      ];
      final String payload = MemoryEntry.encodeAll(entries);
      final List<MemoryEntry> restored = MemoryEntry.decodeAll(payload);

      expect(restored, hasLength(2));
      final MemoryEntry daniel = restored.first;
      expect(daniel.id, entries.first.id);
      expect(daniel.kind, MemoryKind.identity);
      expect(daniel.text, 'My name is Daniel');
      expect(daniel.subject, 'name');
      expect(daniel.source, 'conversation');
      expect(daniel.importance, 0.95);
      expect(daniel.uses, 3);
      expect(daniel.createdAt, created);
      expect(daniel.lastUsedAt, used);
      expect(daniel.keywords, entries.first.keywords);
      // `label` is derived, not stored, so it survives as well.
      expect(daniel.label, 'Identity');
      expect(restored.last.kind, MemoryKind.topic);
    });

    test('decodeAll skips an unknown kind and a malformed row instead of throwing', () {
      const String payload = '[{"kind":"fact","text":"keep me"},'
          '{"kind":"future","text":"skip me"},'
          '{"kind":"fact"},'
          '{"text":"no kind"},'
          '{"kind":"fact","text":"   "},'
          '"not even an object",'
          '42]';
      final List<MemoryEntry> restored = MemoryEntry.decodeAll(payload);
      expect(restored, hasLength(1));
      expect(restored.single.text, 'keep me');
      expect(MemoryEntry.tryFromJson(<String, Object?>{'kind': 'fact', 'text': ''}), isNull);
      expect(MemoryEntry.tryFromJson(<String, Object?>{'text': 'no kind'}), isNull);
    });

    test('decodeAll rejects a payload that is not a JSON array', () {
      expect(() => MemoryEntry.decodeAll('{"kind":"fact"}'), throwsFormatException);
      expect(() => MemoryEntry.decodeAll('not json'), throwsFormatException);
    });

    test('a decoded row normalises its timestamps and regenerates missing keywords', () {
      final MemoryEntry restored = MemoryEntry.decodeAll(
        '[{"kind":"fact","text":"Lives in Lisbon",'
        '"created_at":"2024-03-04T05:06:07+02:00"}]',
      ).single;
      expect(restored.createdAt, DateTime.utc(2024, 3, 4, 3, 6, 7));
      expect(restored.createdAt.isUtc, isTrue);
      // No last_used_at: it falls back to the creation time.
      expect(restored.lastUsedAt, restored.createdAt);
      // No keywords: they are derived from the text.
      expect(restored.keywords, <String>['lives', 'lisbon']);
      // No importance / uses: the middling defaults apply.
      expect(restored.importance, 0.5);
      expect(restored.uses, 0);
    });
  });

  group('MemoryStore.remember and bookkeeping', () {
    test('remembering the same text twice leaves one merged entry', () {
      final MemoryStore store = newStore();
      final MemoryEntry first = store.remember(
        'I prefer short answers.',
        kind: MemoryKind.preference,
        importance: 0.2,
        at: ago(days: 10),
      );
      final MemoryEntry second = store.remember(
        'I prefer short answers',
        kind: MemoryKind.preference,
        importance: 0.9,
        at: now,
      );
      expect(second.id, first.id);
      expect(store.length, 1);
      final MemoryEntry stored = store.entries.single;
      // The maximum importance survives; a later vague mention must not dilute
      // an explicit one (or vice versa).
      expect(stored.importance, 0.9);
      expect(stored.createdAt, first.createdAt);
      expect(stored.lastUsedAt, first.lastUsedAt);
      expect(stored.uses, 0);
      expect(store.contains(first.id), isTrue);
      expect(store.forget('mem_does_not_exist'), isFalse);
    });

    test('reports entries newest first, per kind, and in a describe snapshot', () {
      final MemoryStore store = newStore();
      store.remember('Old memory about kayaking.', kind: MemoryKind.fact, at: ago(days: 9));
      store.remember('New memory about pottery.', kind: MemoryKind.goal, at: ago(days: 1));
      store.remember('Middle memory about bread.', kind: MemoryKind.fact, at: ago(days: 5));

      expect(store.entries.map((MemoryEntry e) => e.text).toList(), <String>[
        'New memory about pottery',
        'Middle memory about bread',
        'Old memory about kayaking',
      ]);
      expect(store.ofKind(MemoryKind.fact), hasLength(2));
      expect(store.ofKind(MemoryKind.goal).single.text, 'New memory about pottery');
      expect(store.ofKind(MemoryKind.topic), isEmpty);
      expect(store.loaded, isFalse);

      final Map<String, Object?> described = store.describe();
      expect(described['entries'], 3);
      expect(described['capacity'], 256);
      expect(described['loaded'], isFalse);
      final Map<String, Object?> kinds = described['kinds']! as Map<String, Object?>;
      expect(kinds['fact'], 2);
      expect(kinds['goal'], 1);
      expect(kinds['topic'], 0);
    });
  });

  group('MemoryStore.recall', () {
    test('ranks a memory sharing a rare query token above one sharing none', () {
      final MemoryStore store = newStore();
      store.remember('The user enjoys kayaking on weekends.', at: ago(hours: 1));
      store.remember('The assistant explains things simply.', at: ago(hours: 1));

      final List<MemoryMatch> matches = store.recall('kayaking', minScore: 0.0);
      expect(matches, isNotEmpty);
      expect(matches.first.entry.text, 'The user enjoys kayaking on weekends');
      for (final MemoryMatch match in matches) {
        expect(match.score, inInclusiveRange(0.0, 1.0));
      }
      // The memory that shares nothing is scored zero and is dropped by the
      // default threshold rather than merely ranked lower.
      expect(store.recall('kayaking').length, 1);
    });

    test('an unrelated query and a stop-word-only query recall nothing', () {
      final MemoryStore store = newStore();
      store.remember('The user enjoys kayaking on weekends.', at: now);
      expect(store.recall('quantum tunneling'), isEmpty);
      // Every query token is a stop word, so there is nothing to match on.
      expect(store.recall('who are you and what is this'), isEmpty);
      expect(newStore().recall('kayaking'), isEmpty);
    });

    test('recency breaks a tie between two memories with equal overlap', () {
      final MemoryStore store = newStore();
      store.remember('Fresh kayaking gear review.', at: ago(hours: 1));
      store.remember('Stale kayaking gear review.', at: ago(days: 120));

      final List<MemoryMatch> matches = store.recall('kayaking', minScore: 0.0);
      expect(matches, hasLength(2));
      // Stored text keeps its words but loses the sentence terminator.
      expect(matches.first.entry.text, 'Fresh kayaking gear review');
      expect(matches.first.score, greaterThan(matches.last.score));
      // The stale one has decayed past a third of its value.
      expect(matches.last.score, lessThan(matches.first.score / 3));
    });

    test('honours the limit and rejects a non-positive one', () {
      final MemoryStore store = newStore();
      for (final int i in <int>[0, 1, 2, 3]) {
        store.remember('Kayaking trip number $i with a paddle.', at: ago(minutes: i));
      }
      expect(store.recall('kayaking', limit: 2), hasLength(2));
      expect(store.recall('kayaking', limit: 0), isEmpty);
    });
  });

  group('MemoryStore capacity', () {
    test('prunes the least useful entries rather than the newest', () {
      final MemoryStore store = newStore(capacity: 3);
      final MemoryEntry chatter = store.remember(
        'Low value chatter one.',
        importance: 0.1,
        at: ago(days: 1),
      );
      store.remember(
        'Low value chatter two.',
        importance: 0.1,
        at: ago(days: 2),
      );
      final MemoryEntry identity = store.remember(
        'High value identity anchor.',
        importance: 0.95,
        at: ago(days: 3),
      );
      final MemoryEntry goal = store.remember(
        'High value goal statement.',
        importance: 0.9,
        at: ago(days: 4),
      );
      store.remember(
        'Medium value fact statement.',
        importance: 0.5,
        at: ago(days: 5),
      );

      expect(store.length, 3);
      expect(store.contains(chatter.id), isFalse);
      expect(store.contains(identity.id), isTrue);
      expect(store.contains(goal.id), isTrue);
      // The retained set is exactly the three highest-importance memories,
      // even though they are the older ones.
      // `entries` is newest-first, so this asserts the retained *set* rather than
      // a ranking the getter never promised.
      expect(
        store.entries.map((MemoryEntry e) => e.importance).toList(),
        orderedEquals(<double>[0.95, 0.9, 0.5]),
      );
    });
  });

  group('MemoryStore.recallBlock', () {
    test('renders one labelled line per memory and respects maxChars', () {
      final MemoryStore store = newStore();
      store.remember('The user prefers tabs over spaces.', kind: MemoryKind.preference, at: ago(hours: 1));
      store.remember('The user lives in Lisbon.', kind: MemoryKind.fact, at: ago(hours: 1));

      final MemoryRecall recall = store.recallBlock('tabs lisbon');
      expect(recall.isEmpty, isFalse);
      expect(recall.block, '- [Preference] The user prefers tabs over spaces\n- [Fact] The user lives in Lisbon');
      expect(recall.entries, hasLength(2));

      // One line always survives, so a tiny budget yields exactly one line and
      // never an empty-but-matched block.
      final MemoryRecall tiny = store.recallBlock('tabs lisbon', maxChars: 20);
      expect(tiny.matches, hasLength(1));
      expect(tiny.block, '- [Preference] The user prefers tabs over spaces');
    });

    test('returns MemoryRecall.empty when nothing matches', () {
      final MemoryStore store = newStore();
      store.remember('The user prefers tabs over spaces.', kind: MemoryKind.preference, at: now);
      final MemoryRecall recall = store.recallBlock('nothing about that at all');
      expect(recall.isEmpty, isTrue);
      expect(recall.block, isEmpty);
      expect(identical(recall, MemoryRecall.empty), isTrue);
      expect(MemoryRecall.empty.entries, isEmpty);
    });

    test('bumps uses and lastUsedAt only on the memories it returned', () {
      final MemoryStore store = newStore();
      final MemoryEntry hit = store.remember('The user prefers tabs over spaces.', at: ago(days: 3));
      final MemoryEntry miss = store.remember('The user enjoys kayaking.', at: ago(days: 3));

      store.recallBlock('tabs');
      final MemoryEntry storedHit = store.entries.firstWhere((MemoryEntry e) => e.id == hit.id);
      expect(storedHit.uses, 1);
      expect(storedHit.lastUsedAt.difference(storedHit.createdAt), greaterThan(Duration.zero));
      final MemoryEntry storedMiss = store.entries.firstWhere((MemoryEntry e) => e.id == miss.id);
      expect(storedMiss.uses, 0);
      expect(storedMiss.lastUsedAt, storedMiss.createdAt);

      store.recallBlock('tabs');
      expect(
        store.entries.firstWhere((MemoryEntry e) => e.id == hit.id).uses,
        2,
      );
      // Marking an entry the store does not hold is a no-op, not an insert.
      final MemoryEntry stranger = MemoryEntry.create(
        kind: MemoryKind.fact,
        text: 'Never stored at all.',
        at: now,
      );
      store.markUsed(<MemoryEntry>[stranger]);
      expect(store.length, 2);
    });
  });

  group('MemoryStore.bootstrapBlock', () {
    test('prioritises identity and preference over a higher-value topic', () {
      final MemoryStore store = newStore();
      store.remember('Chatty topic about dart generics.', kind: MemoryKind.topic, importance: 1.0, at: now);
      store.remember('My name is Daniel.', kind: MemoryKind.identity, importance: 0.95, at: now);
      store.remember('I prefer concise replies.', kind: MemoryKind.preference, importance: 0.85, at: now);
      store.remember('I want to ship my app.', kind: MemoryKind.goal, importance: 0.8, at: now);
      store.remember('I live in Lisbon.', kind: MemoryKind.fact, importance: 0.9, at: now);

      final MemoryRecall recall = store.bootstrapBlock();
      expect(recall.block, contains('My name is Daniel'));
      expect(recall.block, contains('I prefer concise replies'));
      expect(recall.block, contains('I live in Lisbon'));
      // An inferred topic never earns a place in the standing context, however
      // important its own score looks.
      expect(recall.block, isNot(contains('dart generics')));
      expect(recall.matches.first.entry.kind, MemoryKind.identity);
      expect(recall.block, startsWith('- [Identity] '));
      for (final MemoryMatch match in recall.matches) {
        expect(match.score, inInclusiveRange(0.0, 1.0));
      }

      expect(newStore().bootstrapBlock().block, isEmpty);
      expect(identical(newStore().bootstrapBlock(), MemoryRecall.empty), isTrue);
    });
  });

  group('MemoryStore persistence', () {
    test('flush writes storage and a new store over it loads the same entries', () async {
      final InMemoryMemoryStorage storage = InMemoryMemoryStorage();
      final MemoryStore writer = MemoryStore(storage: storage);
      final DateTime created = ago(days: 2);
      final DateTime used = ago(hours: 1);
      final MemoryEntry entry = writer.remember(
        'My name is Daniel.',
        kind: MemoryKind.identity,
        subject: 'name',
        importance: 0.95,
        at: created,
      );
      writer.markUsed(<MemoryEntry>[entry], at: used);
      await writer.flush();
      expect(storage.payload, isNotNull);

      final MemoryStore reader = MemoryStore(storage: storage);
      expect(reader.length, 0);
      await reader.load();
      expect(reader.loaded, isTrue);
      expect(reader.length, 1);
      final MemoryEntry restored = reader.entries.single;
      expect(restored.id, entry.id);
      expect(restored.kind, MemoryKind.identity);
      expect(restored.text, 'My name is Daniel');
      expect(restored.subject, 'name');
      expect(restored.source, isNull);
      expect(restored.importance, 0.95);
      expect(restored.uses, 1);
      expect(restored.createdAt, created);
      expect(restored.lastUsedAt, used);
      // The recalled memory reappears in the new store too.
      expect(reader.recall('what is my name').single.entry.text, 'My name is Daniel');
    });

    test('load never throws on unreadable storage and leaves the store empty', () async {
      final MemoryStore garbage = MemoryStore(storage: InMemoryMemoryStorage('not json'));
      await garbage.load();
      expect(garbage.loaded, isTrue);
      expect(garbage.length, 0);
      // Still usable afterwards.
      garbage.remember('A fresh memory about pottery.', at: now);
      expect(garbage.length, 1);

      final MemoryStore partial = MemoryStore(
        storage: InMemoryMemoryStorage('[{"kind":"fact","text":"good one"},{"kind":"bogus","text":"bad one"}]'),
      );
      await partial.load();
      expect(partial.length, 1);
      expect(partial.entries.single.text, 'good one');

      // Empty storage is not an error, and load is read-once.
      final InMemoryMemoryStorage storage = InMemoryMemoryStorage();
      final MemoryStore store = MemoryStore(storage: storage);
      await store.load();
      storage.payload = MemoryEntry.encodeAll(<MemoryEntry>[
        MemoryEntry.create(kind: MemoryKind.fact, text: 'Written after the load.', at: now),
      ]);
      await store.load();
      expect(store.length, 0);
    });

    test('forget plus flush removes it from storage, and clear deletes the file', () async {
      final InMemoryMemoryStorage storage = InMemoryMemoryStorage();
      final MemoryStore store = MemoryStore(storage: storage);
      final MemoryEntry keep = store.remember('Keeper about kayaking.', at: now);
      final MemoryEntry drop = store.remember('Dropper about pottery.', at: now);
      await store.flush();
      expect(MemoryEntry.decodeAll(storage.payload!), hasLength(2));

      expect(store.forget(drop.id), isTrue);
      await store.flush();
      final MemoryStore reloaded = MemoryStore(storage: storage);
      await reloaded.load();
      expect(reloaded.length, 1);
      expect(reloaded.entries.single.id, keep.id);

      await store.clear();
      expect(store.length, 0);
      expect(storage.payload, isNull);
    });

    test('a clean store flushes without writing, and a fresh store has no payload', () async {
      final InMemoryMemoryStorage storage = InMemoryMemoryStorage();
      final MemoryStore store = MemoryStore(storage: storage);
      await store.flush();
      expect(storage.payload, isNull);
      await store.load();
      expect(store.length, 0);
    });
  });

  group('MemoryExtractor positives', () {
    test('records an identity memory for a stated name', () {
      final MemoryExtraction result = MemoryExtractor().extract('My name is Daniel.');
      expect(result.explicit, isFalse);
      expect(result.entries, hasLength(1));
      final MemoryEntry entry = result.entries.single;
      expect(entry.kind, MemoryKind.identity);
      expect(entry.subject, 'name');
      expect(entry.text, 'My name is Daniel');
      expect(entry.source, 'conversation');
      expect(entry.importance, greaterThan(0.9));
    });

    test('records an identity memory for "I am called" and a stated role', () {
      final MemoryExtraction called = MemoryExtractor().extract('I am called Sofia.');
      expect(called.entries.single.kind, MemoryKind.identity);
      expect(called.entries.single.subject, 'name');

      final MemoryExtraction role = MemoryExtractor().extract('I am a nurse.');
      expect(role.entries.single.kind, MemoryKind.identity);
      expect(role.entries.single.subject, 'role');
      expect(role.entries.single.text, 'I am a nurse');
    });

    test('records a preference for "I prefer ..." and for a language directive', () {
      final MemoryExtraction prefer = MemoryExtractor().extract('I prefer short answers.');
      expect(prefer.explicit, isFalse);
      expect(prefer.entries.single.kind, MemoryKind.preference);
      expect(prefer.entries.single.subject, 'preference');
      expect(prefer.entries.single.text, 'I prefer short answers');

      final MemoryExtraction language = MemoryExtractor().extract('Write in Portuguese');
      expect(language.entries.single.kind, MemoryKind.preference);
      expect(language.entries.single.subject, 'language');
      expect(language.entries.single.text, 'Write in Portuguese');

      final MemoryExtraction dislike = MemoryExtractor().extract("I don't like long replies.");
      expect(dislike.entries.single.kind, MemoryKind.preference);
    });

    test('records a goal for a stated intention', () {
      final MemoryExtraction result = MemoryExtractor().extract('I want to ship my app this month');
      expect(result.entries.single.kind, MemoryKind.goal);
      expect(result.entries.single.subject, 'goal');
      expect(result.entries.single.text, 'I want to ship my app this month');
    });

    test('records a fact for a stated location', () {
      final MemoryExtraction result = MemoryExtractor().extract('I live in Lisbon.');
      expect(result.entries.single.kind, MemoryKind.fact);
      expect(result.entries.single.subject, 'location');
      expect(result.entries.single.text, 'I live in Lisbon');

      final MemoryExtraction work = MemoryExtractor().extract('I work at Acme.');
      expect(work.entries.single.kind, MemoryKind.fact);
      expect(work.entries.single.subject, 'occupation');
    });

    test('treats "Remember that ..." as explicit and re-classifies by cue', () {
      final MemoryExtraction sister = MemoryExtractor().extract('Remember that my sister is called Ana.');
      expect(sister.explicit, isTrue);
      expect(sister.entries.single.kind, MemoryKind.fact);
      expect(sister.entries.single.subject, 'explicit');
      expect(sister.entries.single.importance, 1.0);
      expect(sister.entries.single.text, 'Remember that my sister is called Ana');

      final MemoryExtraction tabs = MemoryExtractor().extract('Remember that I prefer tabs.');
      expect(tabs.explicit, isTrue);
      // The cue wins over the rule's default kind: a remembered preference is
      // still a preference.
      expect(tabs.entries.single.kind, MemoryKind.preference);
      expect(tabs.entries.single.importance, 1.0);

      final MemoryExtraction goal = MemoryExtractor().extract('Remember that I want to learn Portuguese.');
      expect(goal.entries.single.kind, MemoryKind.goal);
      expect(goal.explicit, isTrue);
    });

    test('stores what it extracts under the same id when the text repeats', () {
      final MemoryExtractor extractor = MemoryExtractor();
      final MemoryStore store = newStore();
      for (final String text in <String>['I prefer short answers.', 'I prefer short answers.']) {
        for (final MemoryEntry entry in extractor.extract(text).entries) {
          store.add(entry);
        }
      }
      expect(store.length, 1);
    });
  });

  group('MemoryExtractor precision', () {
    test('never fires on a question', () {
      for (final String text in <String>[
        'What is the weather?',
        'How do I reset my password?',
        'Where are my files?',
      ]) {
        final MemoryExtraction result = MemoryExtractor().extract(text);
        expect(result.entries, isEmpty, reason: text);
        expect(result.explicit, isFalse, reason: text);
        expect(result.isEmpty, isTrue, reason: text);
      }
    });

    test('never fires on an imperative aimed at the assistant', () {
      for (final String text in <String>[
        'Please fix the error in my code.',
        'List my files.',
        'Summarise this document for me.',
      ]) {
        expect(MemoryExtractor().extract(text).entries, isEmpty, reason: text);
      }
    });

    test('never fires on a third-person or possessive statement about the conversation', () {
      for (final String text in <String>[
        'You should prefer the other option.',
        'My question is about the tokenizer.',
      ]) {
        expect(MemoryExtractor().extract(text).entries, isEmpty, reason: text);
      }
    });

    test('fires on "call me" even when it is not a naming request', () {
      // Observed behaviour, not intended behaviour: the identity rule's
      // `call me` alternative has no anchor, so any sentence containing it is
      // stored as the user's name. See the accompanying bug report.
      final MemoryExtraction result = MemoryExtractor().extract('You can call me later.');
      expect(result.entries, hasLength(1));
      expect(result.entries.single.kind, MemoryKind.identity);
      expect(result.entries.single.subject, 'name');
      expect(result.explicit, isFalse);
    });

    test('never yields more than maxEntriesPerText memories from one message', () {
      const String message = 'My name is Ana. I live in Porto. I prefer tea. '
          'I want to build a app. My goal is world peace.';
      final MemoryExtraction capped = MemoryExtractor().extract(message);
      expect(capped.entries, hasLength(4));
      expect(MemoryExtractor(maxEntriesPerText: 2).extract(message).entries, hasLength(2));
      // A single sentence is still a single sentence.
      expect(MemoryExtractor(maxEntriesPerText: 1).extract(message).entries, hasLength(1));
    });

    test('dedupes a statement repeated inside one message', () {
      final MemoryExtraction result = MemoryExtractor().extract('I prefer tea. I prefer tea.');
      expect(result.entries, hasLength(1));
    });

    test('yields nothing when disabled or when the text is blank', () {
      expect(MemoryExtractor(enabled: false).extract('My name is Daniel.').entries, isEmpty);
      expect(MemoryExtractor().extract('   ').entries, isEmpty);
      expect(MemoryExtraction.none.isEmpty, isTrue);
      expect(MemoryExtraction.none.explicit, isFalse);
    });
  });

  group('MemoryExtractor streaming', () {
    test('withholds a partial sentence until the boundary arrives', () {
      final MemoryExtractor extractor = MemoryExtractor();
      final MemoryExtraction partial = extractor.observe('My name is ');
      expect(partial.entries, isEmpty);
      expect(extractor.hasPending, isTrue);
      expect(extractor.observe('Da'), isEmpty);
      expect(extractor.hasPending, isTrue, reason: 'a half-written name must never be stored');
    });

    test('emits the memory as soon as the sentence completes', () {
      final MemoryExtractor extractor = MemoryExtractor();
      expect(extractor.observe('My name is ').entries, isEmpty);
      final MemoryExtraction done = extractor.observe('Daniel.');
      expect(done.entries, hasLength(1));
      expect(done.entries.single.kind, MemoryKind.identity);
      expect(done.entries.single.text, 'My name is Daniel');
      expect(extractor.hasPending, isFalse);
      // The buffer is already drained, so the end-of-turn flush adds nothing.
      expect(extractor.flush().entries, isEmpty);
    });

    test('flush stores a sentence that never received a terminator', () {
      final MemoryExtractor extractor = MemoryExtractor();
      expect(extractor.observe('I live in Lisbon').entries, isEmpty);
      expect(extractor.hasPending, isTrue);
      final MemoryExtraction flushed = extractor.flush();
      expect(flushed.entries.single.kind, MemoryKind.fact);
      expect(flushed.entries.single.text, 'I live in Lisbon');
      expect(extractor.flush().entries, isEmpty);
    });

    test('emits every completed sentence held in the buffer', () {
      final MemoryExtractor extractor = MemoryExtractor();
      final MemoryExtraction result = extractor.observe('My name is Daniel. I live in Lisbon.');
      expect(result.entries, hasLength(2));
      expect(result.entries.map((MemoryEntry e) => e.kind).toList(), <MemoryKind>[
        MemoryKind.identity,
        MemoryKind.fact,
      ]);
    });

    test('reset drops the pending buffer and a disabled extractor buffers nothing', () {
      final MemoryExtractor extractor = MemoryExtractor();
      extractor.observe('My name is ');
      extractor.reset();
      expect(extractor.hasPending, isFalse);
      expect(extractor.flush().entries, isEmpty);

      final MemoryExtractor disabled = MemoryExtractor(enabled: false);
      expect(disabled.observe('My name is Daniel.').entries, isEmpty);
      expect(disabled.hasPending, isFalse);
    });
  });
}
