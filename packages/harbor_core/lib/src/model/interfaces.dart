// The narrow contracts that let the model runtime, the chat engine, and the
// corpus pipeline be developed and tested independently.
//
// Deliberately dependency-free: nothing here imports Flutter, `dart:io`, or any
// other module in the package, so a consumer can depend on the shape of the
// runtime without pulling in the transformer implementation.

import 'dart:math' as math;

/// A reversible text <-> token-id mapping.
///
/// Implementations must satisfy the round-trip law
/// `decode(encode(text)) == text` for every well-formed string. A string that
/// contains a lone surrogate cannot round-trip through any byte-oriented
/// encoding, because UTF-8 has no representation for an unpaired surrogate; it
/// is normalised to U+FFFD, which is what `utf8.encode` itself does.
abstract class Tokenizer {
  /// Number of distinct token ids, i.e. ids run from 0 to `vocabSize - 1`.
  int get vocabSize;

  /// Id of the beginning-of-sequence token.
  int get bosId;

  /// Id of the end-of-sequence token. Generation stops when it is sampled.
  int get eosId;

  /// Id of the padding token, used to fill a batch slot.
  int get padId;

  /// Encodes [text] into token ids.
  ///
  /// When [addBos] is set the sequence starts with [bosId]; when [addEos] is
  /// set it ends with [eosId]. The two are independent so a training window can
  /// ask for a leading BOS without a trailing EOS.
  List<int> encode(String text, {bool addBos = false, bool addEos = false});

  /// Decodes token ids back into text, skipping [bosId], [eosId] and [padId].
  String decode(List<int> ids);

  /// Renders a single id for debugging output, e.g. `"the"` or `"<unk>"`.
  String describeToken(int id);
}

/// The read-only surface the chat engine and the web dashboard need from a
/// language model. The trainer depends on the wider `TrainableLanguageModel`
/// interface in `model/trainable_model.dart`, which extends this one.
abstract class LanguageModelRuntime {
  /// Must equal the [Tokenizer.vocabSize] the model was constructed with.
  int get vocabSize;

  /// Maximum number of prefix tokens the model will attend over.
  int get contextLength;

  /// Total number of trainable scalars (weights plus any active adapter).
  int get parameterCount;

  /// Next-token logits for the full prefix [prefix].
  ///
  /// The returned list always has [vocabSize] entries. Implementations may keep
  /// an internal incremental cache and must therefore invalidate it when the
  /// prefix is not an extension of the previous call; [resetCache] forces that.
  List<double> logitsFor(List<int> prefix);

  /// Discards any incremental decoding state. Must be called before the first
  /// `logitsFor` of an independent sequence.
  void resetCache();
}

/// A deterministic token sampler.
///
/// Held as a concrete class rather than an interface because its behaviour is
/// part of the distribution the chat surface reports: temperature, top-k and a
/// repetition penalty applied to the tokens already generated.
class Sampler {
  /// Creates a sampler.
  ///
  /// [temperature] scales the logits before the softmax; `<= 0` means greedy.
  /// [topK] keeps only the [topK] highest-scoring candidates (`<= 0` disables
  /// the filter). [repetitionPenalty] divides the logit of every token already
  /// present in the generated prefix by this factor when it is `> 1`.
  /// [seed] makes sampling reproducible, which is what the tests rely on.
  Sampler({
    this.temperature = 0.8,
    this.topK = 40,
    this.repetitionPenalty = 1.0,
    int seed = 0x5EED5EED,
  }) : _random = math.Random(seed);

  /// Logit scaling applied before the softmax.
  final double temperature;

  /// Number of candidates kept after filtering; `<= 0` keeps all of them.
  final int topK;

  /// Penalty applied to tokens already generated.
  final double repetitionPenalty;

  final math.Random _random;

  /// Picks the next token id from [logits].
  ///
  /// [generated] are the ids produced so far in this response; they are used by
  /// the repetition penalty only and never mutated.
  int pick(List<double> logits, {List<int> generated = const <int>[]}) {
    if (logits.isEmpty) {
      throw ArgumentError.value(logits, 'logits', 'must not be empty');
    }
    final List<double> adjusted = List<double>.of(logits);
    if (repetitionPenalty > 1.0 && generated.isNotEmpty) {
      for (final int id in generated) {
        if (id < 0 || id >= adjusted.length) {
          continue;
        }
        adjusted[id] =
            adjusted[id] > 0 ? adjusted[id] / repetitionPenalty : adjusted[id] * repetitionPenalty;
      }
    }
    if (temperature <= 0) {
      return _argmax(adjusted);
    }

    // Rank candidates by logit and keep the top-k, then sample from the
    // renormalised softmax over that subset.
    final List<int> order = List<int>.generate(adjusted.length, (int i) => i)
      ..sort((int a, int b) => adjusted[b].compareTo(adjusted[a]));
    final int keep = topK <= 0 ? order.length : math.min(topK, order.length);
    final List<int> candidates = order.sublist(0, keep);

    final double maxLogit = adjusted[candidates.first];
    double total = 0;
    final List<double> weights = <double>[];
    for (final int id in candidates) {
      final double weight =
          math.exp((adjusted[id] - maxLogit) / temperature);
      weights.add(weight);
      total += weight;
    }
    if (!total.isFinite || total <= 0) {
      return candidates.first;
    }
    double threshold = _random.nextDouble() * total;
    for (int i = 0; i < candidates.length; i++) {
      threshold -= weights[i];
      if (threshold <= 0) {
        return candidates[i];
      }
    }
    return candidates.last;
  }

  /// Index of the largest value, ties resolved toward the lower index.
  int _argmax(List<double> values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
  }
}