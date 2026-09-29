// A byte-level byte-pair-encoding tokenizer.
//
// Design notes, because the choices here are load-bearing for the rest of the
// pipeline:
//
//  * The base vocabulary is the 256 byte values, so *any* input is encodable.
//    That matters because the corpus collector feeds the tokenizer arbitrary
//    bytes read off a device; a word-level tokenizer would have to drop or
//    substitute everything outside its vocabulary, silently destroying training
//    signal.
//  * Learned merges are appended above the four reserved control ids, so a
//    checkpoint can be validated by `vocabSize - 260 == merges.length`.
//  * `decode(encode(text)) == text` holds for every well-formed string. A string
//    containing a lone surrogate is normalised to U+FFFD, because that is what
//    UTF-8 encoding does; a Dart String can hold a lone surrogate but no byte
//    sequence can represent one.
//  * Merge application follows the classic BPE rule: repeatedly apply the
//    lowest-rank merge present anywhere, merging *all* of its occurrences in one
//    pass, until no known pair remains. This is order-deterministic and does not
//    depend on hash iteration order.

import 'dart:convert';
import 'dart:typed_data';

import '../model/interfaces.dart';

/// Id of the beginning-of-sequence token.
const int kBosId = 256;

/// Id of the end-of-sequence token, which also stops generation.
const int kEosId = 257;

/// Id of the padding token used to fill incomplete windows.
const int kPadId = 258;

/// Id of the replacement token emitted for an unknown id.
const int kUnkId = 259;

/// Id of the first learned merge; merge with rank `r` has id `260 + r`.
const int kFirstMergeId = 260;

/// Hard ceiling on [ByteTokenizer.vocabSize].
///
/// `BpeMerge.pairKey` packs a pair as `(left << 12) | right`, which only
/// round-trips while every id fits in twelve bits. Ids start at 260, so this
/// caps the merge table at 3 836 entries — orders of magnitude above anything an
/// on-device model would carry, but it is checked rather than assumed.
const int _maxVocabSize = 4096;

/// One learned byte-pair merge.
///
/// [rank] is the position in the merge table and therefore the priority: a
/// lower rank is applied first.
class BpeMerge {
  /// Creates a merge of [left] followed by [right] with priority [rank].
  const BpeMerge(this.left, this.right, this.rank);

  /// Left member of the merged pair.
  final int left;

  /// Right member of the merged pair.
  final int right;

  /// Priority; lower is applied first.
  final int rank;

  /// Id assigned to the merged token.
  int get tokenId => kFirstMergeId + rank;

  /// Packs the pair into a single integer key.
  static int pairKey(int left, int right) => (left << 12) | right;

  /// JSON form, used by model checkpoints.
  Map<String, Object?> toJson() => <String, Object?>{
        'left': left,
        'right': right,
        'rank': rank,
      };

  /// Parses [BpeMerge.toJson].
  static BpeMerge fromJson(Map<String, Object?> json) {
    final Object? left = json['left'];
    final Object? right = json['right'];
    final Object? rank = json['rank'];
    if (left is! int || right is! int || rank is! int) {
      throw const FormatException('merge must carry int left/right/rank');
    }
    if (left < 0 || right < 0 || left >= 1 << 12 || right >= 1 << 12) {
      throw FormatException('merge pair ($left, $right) is out of range');
    }
    return BpeMerge(left, right, rank);
  }

  @override
  String toString() => 'BpeMerge($left, $right, rank $rank)';
}

/// Learns [BpeMerge]s from a bounded sample of text.
class BpeTrainer {
  /// Creates a trainer that stops after [targetMerges] merges.
  ///
  /// [maxSampleChars] bounds the training sample. BPE training is
  /// `O(sample * merges)`, so an unbounded sample would make the on-device and
  /// in-browser paths unresponsive; 24 000 characters is plenty to discover the
  /// common byte sequences of English prose and of source code.
  const BpeTrainer({
    this.targetMerges = 320,
    this.maxSampleChars = 24000,
  });

  /// Number of merges to learn.
  final int targetMerges;

  /// Maximum number of characters taken from the input.
  final int maxSampleChars;

  /// Learns merges from [text] and returns them in rank order.
  List<BpeMerge> train(String text) {
    final String sample = text.length > maxSampleChars
        ? text.substring(0, maxSampleChars)
        : text;
    if (sample.isEmpty || targetMerges <= 0) {
      return const <BpeMerge>[];
    }

    // Doubly linked list over the sample's bytes. `next[i]` of the final node is
    // -1. Merging rewires the links instead of rebuilding the list, which keeps
    // each merge pass linear in the number of surviving symbols.
    final Uint8List symbols = Uint8List.fromList(utf8.encode(sample));
    final int length = symbols.length;
    final Int32List ids = Int32List(length);
    final Int32List next = Int32List(length);
    for (int i = 0; i < length; i++) {
      ids[i] = symbols[i];
      next[i] = i + 1 < length ? i + 1 : -1;
    }

    final List<BpeMerge> merges = <BpeMerge>[];
    for (int rank = 0; rank < targetMerges; rank++) {
      final Map<int, int> counts = <int, int>{};
      int cursor = 0;
      while (cursor >= 0) {
        final int following = next[cursor];
        if (following < 0) {
          break;
        }
        final int key = BpeMerge.pairKey(ids[cursor], ids[following]);
        counts[key] = (counts[key] ?? 0) + 1;
        cursor = following;
      }
      if (counts.isEmpty) {
        break;
      }

      // Most frequent pair wins; ties are broken toward the lower packed key so
      // the result never depends on hash iteration order.
      int bestKey = -1;
      int bestCount = 1;
      for (final MapEntry<int, int> entry in counts.entries) {
        if (entry.value > bestCount ||
            (entry.value == bestCount && (bestKey < 0 || entry.key < bestKey))) {
          bestKey = entry.key;
          bestCount = entry.value;
        }
      }
      if (bestKey < 0 || bestCount < 2) {
        // A pair seen once is not a pattern; stop rather than memorise noise.
        break;
      }

      final int left = bestKey >> 12;
      final int right = bestKey & 0xFFF;
      final int newId = kFirstMergeId + rank;
      merges.add(BpeMerge(left, right, rank));

      // Merge every occurrence of the chosen pair in this pass.
      cursor = 0;
      while (cursor >= 0) {
        final int following = next[cursor];
        if (following < 0) {
          break;
        }
        if (ids[cursor] == left && ids[following] == right) {
          ids[cursor] = newId;
          next[cursor] = next[following];
          cursor = next[cursor];
        } else {
          cursor = following;
        }
      }
    }
    return merges;
  }
}

/// A byte-level BPE tokenizer, optionally with learned merges.
class ByteTokenizer implements Tokenizer {
  /// Builds a tokenizer over [merges], which must be in rank order.
  ///
  /// Throws [ArgumentError] when the rank sequence has a gap, because a gap
  /// would make `tokenId` collide with a later merge.
  ByteTokenizer({List<BpeMerge> merges = const <BpeMerge>[]})
      : merges = List<BpeMerge>.unmodifiable(merges) {
    for (int i = 0; i < merges.length; i++) {
      if (merges[i].rank != i) {
        throw ArgumentError(
          'merge at index $i has rank ${merges[i].rank}; ranks must be dense '
          'and start at 0',
        );
      }
    }
    if (kFirstMergeId + merges.length > _maxVocabSize) {
      throw ArgumentError(
        'at most ${_maxVocabSize - kFirstMergeId} merges are supported, got '
        '${merges.length}',
      );
    }
    _expandTokenBytes();
    for (final BpeMerge merge in merges) {
      _mergeByPair[BpeMerge.pairKey(merge.left, merge.right)] = merge;
    }
  }

  /// Trains a tokenizer on [text].
  ///
  /// Convenience for the common "learn a vocabulary, then use it" flow.
  factory ByteTokenizer.train(
    String text, {
    BpeTrainer trainer = const BpeTrainer(),
  }) {
    return ByteTokenizer(merges: trainer.train(text));
  }

  /// The merges this tokenizer applies, in rank order.
  final List<BpeMerge> merges;

  final Map<int, BpeMerge> _mergeByPair = <int, BpeMerge>{};
  final Map<int, Uint8List> _tokenBytes = <int, Uint8List>{};

  @override
  int get vocabSize => kFirstMergeId + merges.length;

  @override
  int get bosId => kBosId;

  @override
  int get eosId => kEosId;

  @override
  int get padId => kPadId;

  /// Id of the replacement token. Not part of the [Tokenizer] contract because
  /// a byte-level vocabulary never needs it; exposed for diagnostics.
  int get unkId => kUnkId;

  /// Number of learned merges.
  int get mergeCount => merges.length;

  /// The byte sequence a token expands to.
  ///
  /// Control ids have no bytes; a merged token expands to the concatenation of
  /// its two members' bytes, computed once at construction.
  Uint8List bytesOf(int id) {
    if (id < 0 || id >= vocabSize) {
      return Uint8List(0);
    }
    if (id < kFirstMergeId) {
      if (id < 256) {
        return Uint8List.fromList(<int>[id]);
      }
      return Uint8List(0);
    }
    return _tokenBytes[id] ?? Uint8List(0);
  }

  void _expandTokenBytes() {
    for (int id = 0; id < 256; id++) {
      _tokenBytes[id] = Uint8List.fromList(<int>[id]);
    }
    for (final BpeMerge merge in merges) {
      final Uint8List left = _tokenBytes[merge.left] ?? Uint8List(0);
      final Uint8List right = _tokenBytes[merge.right] ?? Uint8List(0);
      final Uint8List combined = Uint8List(left.length + right.length)
        ..setRange(0, left.length, left)
        ..setRange(left.length, left.length + right.length, right);
      _tokenBytes[merge.tokenId] = combined;
    }
  }

  @override
  List<int> encode(String text, {bool addBos = false, bool addEos = false}) {
    if (text.isEmpty && !addBos && !addEos) {
      return <int>[];
    }
    final Uint8List raw = Uint8List.fromList(utf8.encode(text));
    List<int> ids = List<int>.generate(raw.length, (int i) => raw[i]);

    // Apply the lowest-rank available merge, then rescan. Outer loop is bounded
    // by the merge count because each pass consumes at least one rank.
    while (ids.length > 1) {
      int bestRank = -1;
      for (int i = 0; i + 1 < ids.length; i++) {
        final BpeMerge? merge = _mergeByPair[BpeMerge.pairKey(ids[i], ids[i + 1])];
        if (merge != null && (bestRank < 0 || merge.rank < bestRank)) {
          bestRank = merge.rank;
        }
      }
      if (bestRank < 0) {
        break;
      }
      final List<int> merged = <int>[];
      for (int i = 0; i < ids.length;) {
        if (i + 1 < ids.length) {
          final BpeMerge? merge =
              _mergeByPair[BpeMerge.pairKey(ids[i], ids[i + 1])];
          if (merge != null && merge.rank == bestRank) {
            merged.add(merge.tokenId);
            i += 2;
            continue;
          }
        }
        merged.add(ids[i]);
        i++;
      }
      ids = merged;
    }

    final List<int> out = <int>[];
    if (addBos) {
      out.add(kBosId);
    }
    out.addAll(ids);
    if (addEos) {
      out.add(kEosId);
    }
    return out;
  }

  @override
  String decode(List<int> ids) {
    final BytesBuilder builder = BytesBuilder(copy: false);
    for (final int id in ids) {
      if (id == kBosId || id == kEosId || id == kPadId) {
        continue;
      }
      if (id == kUnkId || id < 0 || id >= vocabSize) {
        // Unknown ids carry no bytes; dropping them is safer than emitting a
        // placeholder that would corrupt the round trip for valid text.
        continue;
      }
      builder.add(bytesOf(id));
    }
    return utf8.decode(builder.takeBytes(), allowMalformed: true);
  }

  @override
  String describeToken(int id) {
    if (id == kBosId) {
      return '<bos>';
    }
    if (id == kEosId) {
      return '<eos>';
    }
    if (id == kPadId) {
      return '<pad>';
    }
    if (id == kUnkId) {
      return '<unk>';
    }
    if (id < 0 || id >= vocabSize) {
      return '<invalid:$id>';
    }
    final Uint8List bytes = bytesOf(id);
    if (id < 256) {
      if (id >= 0x20 && id < 0x7F) {
        return "'${String.fromCharCode(id)}'";
      }
      return r'\x' + id.toRadixString(16).padLeft(2, '0');
    }
    return "'${utf8.decode(bytes, allowMalformed: true).replaceAll('\n', r'\n')}'";
  }

  /// JSON form, small enough to embed in a model checkpoint.
  Map<String, Object?> toJson() => <String, Object?>{
        'merges': <Object?>[for (final BpeMerge merge in merges) merge.toJson()],
      };

  /// Parses [ByteTokenizer.toJson].
  static ByteTokenizer fromJson(Map<String, Object?> json) {
    final Object? raw = json['merges'];
    if (raw is! List<Object?>) {
      throw const FormatException('tokenizer payload must carry a "merges" list');
    }
    return ByteTokenizer(
      merges: <BpeMerge>[
        for (final Object? entry in raw)
          if (entry is Map<String, Object?>)
            BpeMerge.fromJson(entry)
          else
            throw const FormatException('merge entry must be a JSON object'),
      ],
    );
  }

  @override
  String toString() =>
      'ByteTokenizer(vocab: $vocabSize, merges: ${merges.length})';
}