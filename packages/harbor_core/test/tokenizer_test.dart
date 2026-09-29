import 'dart:convert';
import 'dart:math';

import 'package:harbor_core/src/tokenizer/byte_tokenizer.dart';
import 'package:test/test.dart';

const BpeTrainer _trainer = BpeTrainer(targetMerges: 40, maxSampleChars: 4000);

final String _trainingText = List<String>.filled(
  40,
  'the quick brown fox jumps over the lazy dog. 日本語 🚀 ',
).join();

ByteTokenizer _mergeFree() => ByteTokenizer();

ByteTokenizer _trained() =>
    ByteTokenizer.train(_trainingText, trainer: _trainer);

String _printableAscii(int length, Random random) {
  final StringBuffer buffer = StringBuffer();
  for (int i = 0; i < length; i++) {
    buffer.writeCharCode(32 + random.nextInt(95));
  }
  return buffer.toString();
}

final List<String> _roundTripSamples = <String>[
  '',
  'hello world',
  'line one\n\tindented\r\nline two\ttabbed',
  'héllo wörld — 日本語 🚀',
  _printableAscii(4000, Random(20240917)),
  latin1.decode(List<int>.generate(256, (int i) => i)),
];

void main() {
  test('decode(encode(text)) round-trips for both tokenizers', () {
    final List<ByteTokenizer> tokenizers = <ByteTokenizer>[
      _mergeFree(),
      _trained(),
    ];
    for (final ByteTokenizer tokenizer in tokenizers) {
      for (final String sample in _roundTripSamples) {
        expect(
          tokenizer.decode(tokenizer.encode(sample)),
          sample,
          reason: 'round trip failed for a ${sample.length}-character sample',
        );
      }
    }
  });

  test('vocabulary layout', () {
    final ByteTokenizer mergeFree = _mergeFree();
    expect(mergeFree.vocabSize, 260);
    expect(mergeFree.vocabSize, 260 + mergeFree.merges.length);
    expect(mergeFree.bosId, 256);
    expect(mergeFree.eosId, 257);
    expect(mergeFree.padId, 258);
    expect(mergeFree.unkId, 259);

    final ByteTokenizer trained = _trained();
    expect(trained.vocabSize, 260 + trained.merges.length);
    expect(trained.mergeCount, trained.merges.length);
    expect(trained.merges, isNotEmpty);
  });

  test('bos and eos flags wrap the sequence and are independent', () {
    final ByteTokenizer tokenizer = _mergeFree();
    final List<int> plain = tokenizer.encode('ab');
    expect(tokenizer.encode('ab', addBos: true), <int>[256, ...plain]);
    expect(tokenizer.encode('ab', addEos: true), <int>[...plain, 257]);
    expect(
      tokenizer.encode('ab', addBos: true, addEos: true),
      <int>[256, ...plain, 257],
    );
    expect(tokenizer.encode('', addBos: true), <int>[256]);
    expect(tokenizer.encode('', addEos: true), <int>[257]);
    expect(tokenizer.encode('', addBos: true, addEos: true), <int>[256, 257]);
  });

  test('decode skips control ids and drops invalid ids', () {
    final ByteTokenizer tokenizer = _mergeFree();
    expect(
      tokenizer.decode(<int>[256, 257, 258, 259, 999999, -7, 104, 105]),
      'hi',
    );
    expect(tokenizer.decode(<int>[256, 257, 258]), '');
  });

  test('dense ranks and the merge ceiling are enforced', () {
    expect(
      () => ByteTokenizer(
        merges: <BpeMerge>[const BpeMerge(1, 2, 0), const BpeMerge(3, 4, 2)],
      ),
      throwsArgumentError,
    );
    final List<BpeMerge> tooMany = <BpeMerge>[
      for (int rank = 0; rank <= 3836; rank++) BpeMerge(0, 1, rank),
    ];
    expect(tooMany.length, 3837);
    expect(() => ByteTokenizer(merges: tooMany), throwsArgumentError);
    expect(
      () => ByteTokenizer(merges: tooMany.sublist(0, 3836)),
      returnsNormally,
    );
  });

  test('the trainer learns merges that actually compress text', () {
    final String sentence = 'the quick brown fox jumps over the lazy dog. ';
    final String corpus = List<String>.filled(40, sentence).join();
    final ByteTokenizer tokenizer =
        ByteTokenizer.train(corpus, trainer: _trainer);
    expect(tokenizer.merges, isNotEmpty);

    final List<int> encoded = tokenizer.encode(corpus);
    final int byteLength = utf8.encode(corpus).length;
    expect(encoded.length, lessThan(byteLength * 0.75));

    final List<int> first = tokenizer.encode(corpus);
    final List<int> second = tokenizer.encode(corpus);
    expect(second, first);
    for (final int id in encoded) {
      expect(id, lessThan(4096));
    }
  });

  test('rank ordering is respected', () {
    final ByteTokenizer tokenizer = _trained();
    expect(tokenizer.merges, isNotEmpty);
    final BpeMerge merge = tokenizer.merges.first;
    if (merge.left < 256 && merge.right < 256) {
      final String pair = latin1.decode(<int>[merge.left, merge.right]);
      expect(tokenizer.encode(pair), <int>[merge.tokenId]);
    } else {
      expect(merge, isNotNull);
    }
  });

  test('describeToken renders controls, characters and control bytes', () {
    final ByteTokenizer tokenizer = _mergeFree();
    expect(tokenizer.describeToken(256), '<bos>');
    expect(tokenizer.describeToken(257), '<eos>');
    expect(tokenizer.describeToken(258), '<pad>');
    expect(tokenizer.describeToken(65), "'A'");
    expect(tokenizer.describeToken(0), r'\x00');
    expect(tokenizer.describeToken(31), r'\x1f');
  });

  test('JSON round-trips a trained tokenizer', () {
    final ByteTokenizer tokenizer = _trained();
    final ByteTokenizer restored = ByteTokenizer.fromJson(tokenizer.toJson());
    expect(restored.vocabSize, tokenizer.vocabSize);
    const String sample = 'the quick brown fox jumps over the lazy dog. 日本語 🚀';
    expect(restored.encode(sample), tokenizer.encode(sample));

    expect(
      () => ByteTokenizer.fromJson(<String, Object?>{'merges': 'nope'}),
      throwsFormatException,
    );
    expect(
      () => ByteTokenizer.fromJson(<String, Object?>{
        'merges': <Object?>[42],
      }),
      throwsFormatException,
    );
  });

  test('a merge-free tokenizer emits raw UTF-8 bytes', () {
    final ByteTokenizer tokenizer = _mergeFree();
    for (final String sample in _roundTripSamples) {
      expect(tokenizer.encode(sample).length, utf8.encode(sample).length);
    }
  });
}
