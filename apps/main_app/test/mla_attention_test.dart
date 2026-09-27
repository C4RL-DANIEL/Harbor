// Unit tests for the pure-Dart Multi-Head Latent Attention (MLA) engine.
//
// The system under test is `lib/core/engine/mla_attention.dart`. These tests
// exercise only the public API: geometry, cache accounting, the decode-step
// forward pass, the sliding-window KV cache, and the memory claim that is the
// whole point of MLA.
//
// Run with:
//   flutter test test/mla_attention_test.dart

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:main_app/core/engine/mla_attention.dart';

/// A small but non-degenerate MLA geometry used across the forward tests.
MlaConfig _smallConfig({int maxSeqLen = 16}) => MlaConfig(
      hiddenSize: 16,
      numHeads: 2,
      headDim: 8,
      kvLoraRank: 6,
      ropeHeadDim: 4,
      maxSeqLen: maxSeqLen,
    );

/// Deterministic dense vector helper.
Float32List _vec(int n, [double offset = 0.0]) => Float32List.fromList(
      List<double>.generate(n, (int i) => (i + 1) * 0.125 + offset),
    );

void main() {
  group('MlaConfig geometry', () {
    test('cache and non-RoPE geometry are derived from the base fields', () {
      const MlaConfig config = MlaConfig(
        hiddenSize: 64,
        numHeads: 4,
        headDim: 16,
        kvLoraRank: 12,
        ropeHeadDim: 6,
      );
      expect(config.cacheFloatsPerToken, config.kvLoraRank + config.ropeHeadDim);
      expect(config.cacheFloatsPerToken, 18);
      expect(config.nopeHeadDim, config.headDim - config.ropeHeadDim);
      expect(config.nopeHeadDim, 10);
    });

    test('qLoraRank 0 falls back to the full hidden width', () {
      const MlaConfig config = MlaConfig(
        hiddenSize: 64,
        numHeads: 4,
        headDim: 16,
        kvLoraRank: 12,
        ropeHeadDim: 6,
        maxSeqLen: 8,
      );
      expect(
        MultiHeadLatentAttention(config, seed: 1).describe()['q_lora_rank'],
        64,
      );

      const MlaConfig compressed = MlaConfig(
        hiddenSize: 64,
        numHeads: 4,
        headDim: 16,
        kvLoraRank: 12,
        ropeHeadDim: 6,
        qLoraRank: 24,
        maxSeqLen: 8,
      );
      expect(
        MultiHeadLatentAttention(compressed, seed: 1).describe()['q_lora_rank'],
        24,
      );
    });

    test('rejects ropeHeadDim larger than headDim', () {
      expect(
        () => MlaConfig(
          hiddenSize: 64,
          numHeads: 4,
          headDim: 8,
          kvLoraRank: 4,
          ropeHeadDim: 12,
        ),
        throwsA(isA<AssertionError>()),
      );
    });

    test('rejects non-positive geometry', () {
      expect(
        () => MlaConfig(
          hiddenSize: 0,
          numHeads: 4,
          headDim: 8,
          kvLoraRank: 4,
          ropeHeadDim: 4,
        ),
        throwsA(isA<AssertionError>()),
      );
      expect(
        () => MlaConfig(
          hiddenSize: 64,
          numHeads: 0,
          headDim: 8,
          kvLoraRank: 4,
          ropeHeadDim: 4,
        ),
        throwsA(isA<AssertionError>()),
      );
      expect(
        () => MlaConfig(
          hiddenSize: 64,
          numHeads: 4,
          headDim: 0,
          kvLoraRank: 4,
          ropeHeadDim: 4,
        ),
        throwsA(isA<AssertionError>()),
      );
      expect(
        () => MlaConfig(
          hiddenSize: 64,
          numHeads: 4,
          headDim: 8,
          kvLoraRank: 0,
          ropeHeadDim: 4,
        ),
        throwsA(isA<AssertionError>()),
      );
      expect(
        () => MlaConfig(
          hiddenSize: 64,
          numHeads: 4,
          headDim: 8,
          kvLoraRank: 4,
          ropeHeadDim: 0,
        ),
        throwsA(isA<AssertionError>()),
      );
      expect(
        () => MlaConfig(
          hiddenSize: 64,
          numHeads: 4,
          headDim: 8,
          kvLoraRank: 4,
          ropeHeadDim: 4,
          maxSeqLen: 0,
        ),
        throwsA(isA<AssertionError>()),
      );
    });

    test('copyWith preserves geometry and replaces the window length', () {
      final MlaConfig base = _smallConfig();
      final MlaConfig longer = base.copyWith(maxSeqLen: 128);
      expect(longer.hiddenSize, base.hiddenSize);
      expect(longer.numHeads, base.numHeads);
      expect(longer.headDim, base.headDim);
      expect(longer.kvLoraRank, base.kvLoraRank);
      expect(longer.ropeHeadDim, base.ropeHeadDim);
      expect(longer.maxSeqLen, 128);
      expect(base.maxSeqLen, 16);
    });
  });

  group('MlaCacheStats accounting', () {
    // hiddenSize 64, numHeads 4, headDim 16, kvLoraRank 12, ropeHeadDim 6.
    const MlaConfig config = MlaConfig(
      hiddenSize: 64,
      numHeads: 4,
      headDim: 16,
      kvLoraRank: 12,
      ropeHeadDim: 6,
    );

    test('bytes per token match the float32 layout', () {
      const MlaCacheStats stats = MlaCacheStats(tokens: 100, config: config);
      // MLA keeps kvLoraRank + ropeHeadDim floats per token.
      expect(stats.mlaBytesPerToken, (12 + 6) * 4);
      expect(stats.mlaBytesPerToken, 72);
      // Vanilla MHA keeps 2 * numHeads * headDim floats per token.
      expect(stats.vanillaBytesPerToken, 2 * 4 * 16 * 4);
      expect(stats.vanillaBytesPerToken, 512);
      expect(
        stats.compressionRatio,
        closeTo(stats.vanillaBytesPerToken / stats.mlaBytesPerToken, 1e-12),
      );
    });

    test('100-token cache bytes and savings are exact', () {
      const MlaCacheStats stats = MlaCacheStats(tokens: 100, config: config);
      expect(stats.mlaBytes, 7200);
      expect(stats.vanillaBytes, 51200);
      expect(stats.bytesSaved, stats.vanillaBytes - stats.mlaBytes);
      expect(stats.bytesSaved, 44000);
      expect(stats.bytesSaved, greaterThan(0));
    });

    test('toJson exposes the documented contract', () {
      const MlaCacheStats stats = MlaCacheStats(tokens: 100, config: config);
      final Map<String, Object?> json = stats.toJson();
      expect(
        json.keys.toSet(),
        <String>{
          'tokens',
          'mla_bytes_per_token',
          'vanilla_bytes_per_token',
          'mla_bytes',
          'vanilla_bytes',
          'bytes_saved',
          'compression_ratio',
          'kv_lora_rank',
          'rope_head_dim',
        },
      );
      expect(json['tokens'], 100);
      expect(json['mla_bytes_per_token'], 72);
      expect(json['vanilla_bytes_per_token'], 512);
      expect(json['mla_bytes'], 7200);
      expect(json['vanilla_bytes'], 51200);
      expect(json['bytes_saved'], 44000);
      expect(json['compression_ratio'], closeTo(512.0 / 72.0, 1e-3));
      expect(json['kv_lora_rank'], 12);
      expect(json['rope_head_dim'], 6);
    });
  });

  group('KV-cache memory claim', () {
    test('a realistic MLA layer is >10x smaller than vanilla MHA', () {
      const MlaConfig config = MlaConfig(
        hiddenSize: 512,
        numHeads: 16,
        headDim: 64,
        kvLoraRank: 128,
        ropeHeadDim: 32,
      );
      const int tokens = 4096;
      const MlaCacheStats stats = MlaCacheStats(tokens: tokens, config: config);

      expect(stats.mlaBytesPerToken, (128 + 32) * 4);
      expect(stats.mlaBytesPerToken, 640);
      expect(stats.vanillaBytesPerToken, 2 * 16 * 64 * 4);
      expect(stats.vanillaBytesPerToken, 8192);
      expect(stats.compressionRatio, greaterThan(10.0));
      expect(stats.compressionRatio, closeTo(12.8, 1e-9));
      expect(stats.mlaBytes, 640 * tokens);
      expect(stats.vanillaBytes, 8192 * tokens);
      expect(stats.bytesSaved, (8192 - 640) * tokens);
      expect(stats.bytesSaved, greaterThan(0));
    });
  });

  group('MultiHeadLatentAttention.forward', () {
    test('output is hiddenSize wide and position is echoed back', () {
      final MlaConfig config = _smallConfig();
      final MultiHeadLatentAttention attn = MultiHeadLatentAttention(config, seed: 7);
      final Float32List x = _vec(config.hiddenSize);

      final MlaForwardResult r0 = attn.forward(x, 0);
      expect(r0.output.length, config.hiddenSize);
      expect(r0.position, 0);

      final MlaForwardResult r1 = attn.forward(_vec(config.hiddenSize, 0.5), 1);
      expect(r1.output.length, config.hiddenSize);
      expect(r1.position, 1);

      final MlaForwardResult r2 = attn.forward(_vec(config.hiddenSize, 1.0), 2);
      expect(r2.output.length, config.hiddenSize);
      expect(r2.position, 2);
    });

    test('attendedTokens grows 1, 2, 3 across successive decode steps', () {
      final MlaConfig config = _smallConfig();
      final MultiHeadLatentAttention attn = MultiHeadLatentAttention(config, seed: 3);
      final int attended0 = attn.forward(_vec(config.hiddenSize), 0).attendedTokens;
      final int attended1 = attn.forward(_vec(config.hiddenSize, 0.3), 1).attendedTokens;
      final int attended2 = attn.forward(_vec(config.hiddenSize, 0.6), 2).attendedTokens;
      expect(attended0, 1);
      expect(attended1, 2);
      expect(attended2, 3);
      expect(attn.cache.length, 3);
      expect(attn.cacheStats.tokens, 3);
    });

    test('same seed and same input produce bit-identical outputs', () {
      final MlaConfig config = _smallConfig();
      final MultiHeadLatentAttention a = MultiHeadLatentAttention(config, seed: 1234);
      final MultiHeadLatentAttention b = MultiHeadLatentAttention(config, seed: 1234);
      final Float32List x = _vec(config.hiddenSize, 0.75);

      final MlaForwardResult ra = a.forward(x, 0);
      final MlaForwardResult rb = b.forward(x, 0);
      expect(ra.output, orderedEquals(rb.output));
      expect(
        ra.output.buffer.asUint8List(),
        rb.output.buffer.asUint8List(),
      );
    });

    test('different seeds produce different outputs', () {
      final MlaConfig config = _smallConfig();
      final MultiHeadLatentAttention a = MultiHeadLatentAttention(config, seed: 1);
      final MultiHeadLatentAttention b = MultiHeadLatentAttention(config, seed: 2);
      final Float32List x = _vec(config.hiddenSize);

      final MlaForwardResult ra = a.forward(x, 0);
      final MlaForwardResult rb = b.forward(x, 0);
      expect(ra.output, isNot(orderedEquals(rb.output)));
    });

    test('rejects an input whose width differs from hiddenSize', () {
      final MlaConfig config = _smallConfig();
      final MultiHeadLatentAttention attn = MultiHeadLatentAttention(config, seed: 9);
      expect(
        () => attn.forward(Float32List(config.hiddenSize + 1), 0),
        throwsA(isA<MlaShapeException>()),
      );
      expect(
        () => attn.forward(Float32List(config.hiddenSize - 1), 0),
        throwsA(isA<MlaShapeException>()),
      );
    });

    test('an odd ropeHeadDim is rejected by RoPE with an assertion', () {
      // MlaConfig itself allows odd ropeHeadDim; the RoPE table requires an
      // even dimension and asserts when the attention layer builds it.
      const MlaConfig odd = MlaConfig(
        hiddenSize: 16,
        numHeads: 2,
        headDim: 8,
        kvLoraRank: 6,
        ropeHeadDim: 5,
        maxSeqLen: 8,
      );
      expect(odd.ropeHeadDim, 5);
      expect(odd.nopeHeadDim, 3);
      expect(
        () => MultiHeadLatentAttention(odd, seed: 1),
        throwsA(isA<AssertionError>()),
      );
    });

    test('RoPE additionally guards position and vector width', () {
      final RotaryEmbedding rope =
          RotaryEmbedding(dim: 4, theta: 10000.0, maxSeqLen: 4);
      expect(
        () => rope.applyInPlace(Float32List(3), 0),
        throwsA(isA<MlaShapeException>()),
      );
      expect(
        () => rope.applyInPlace(Float32List(4), 4),
        throwsA(isA<MlaCacheException>()),
      );
      expect(
        () => rope.applyInPlace(Float32List(4), -1),
        throwsA(isA<MlaCacheException>()),
      );
      final Float32List rotated = rope.apply(Float32List.fromList(<double>[1, 0, 1, 0]), 0);
      expect(rotated[0], closeTo(1.0, 1e-6));
      expect(rotated[1], closeTo(0.0, 1e-6));
    });

    test('forwardSequence returns a hiddenSize output equal to stepwise decode', () {
      final MlaConfig config = _smallConfig();
      final List<Float32List> tokens = <Float32List>[
        _vec(config.hiddenSize, 0.0),
        _vec(config.hiddenSize, 0.4),
        _vec(config.hiddenSize, 0.8),
      ];

      final MultiHeadLatentAttention sequential =
          MultiHeadLatentAttention(config, seed: 55);
      final Float32List seqOut = sequential.forwardSequence(tokens);
      expect(seqOut.length, config.hiddenSize);

      final MultiHeadLatentAttention stepwise =
          MultiHeadLatentAttention(config, seed: 55);
      Float32List last = Float32List(config.hiddenSize);
      for (int i = 0; i < tokens.length; i++) {
        last = stepwise.forward(tokens[i], i).output;
      }
      expect(seqOut, orderedEquals(last));
      expect(sequential.cache.length, tokens.length);
      expect(stepwise.cache.length, tokens.length);
    });

    test('forwardSequence respects startPosition and rejects an empty prompt', () {
      final MlaConfig config = _smallConfig();
      final MultiHeadLatentAttention attn = MultiHeadLatentAttention(config, seed: 12);
      expect(
        () => attn.forwardSequence(<Float32List>[]),
        throwsA(isA<MlaShapeException>()),
      );
      attn.forwardSequence(<Float32List>[_vec(config.hiddenSize)], startPosition: 3);
      expect(attn.cache.length, 1);
    });

    test('resetCache zeroes both length and memory stats', () {
      final MlaConfig config = _smallConfig();
      final MultiHeadLatentAttention attn = MultiHeadLatentAttention(config, seed: 21);
      attn.forward(_vec(config.hiddenSize), 0);
      attn.forward(_vec(config.hiddenSize, 0.2), 1);
      expect(attn.cache.length, 2);
      expect(attn.cacheStats.tokens, 2);
      expect(attn.cacheStats.mlaBytes, 2 * config.cacheFloatsPerToken * 4);

      attn.resetCache();
      expect(attn.cache.length, 0);
      expect(attn.cacheStats.tokens, 0);
      expect(attn.cacheStats.mlaBytes, 0);
      expect(attn.cacheStats.bytesSaved, 0);
    });

    test('describe reports the geometry and cache contract', () {
      final MlaConfig config = _smallConfig();
      final MultiHeadLatentAttention attn = MultiHeadLatentAttention(config, seed: 4);
      attn.forward(_vec(config.hiddenSize), 0);
      final Map<String, Object?> info = attn.describe();
      expect(info['architecture'], 'multi_head_latent_attention');
      expect(info['hidden_size'], 16);
      expect(info['num_heads'], 2);
      expect(info['head_dim'], 8);
      expect(info['nope_head_dim'], 4);
      expect(info['rope_head_dim'], 4);
      expect(info['kv_lora_rank'], 6);
      expect(info['max_seq_len'], 16);
      expect(info['tokens'], 1);
      // mla = (6 + 4) * 4 = 40 B/token, vanilla = 2 * 2 * 8 * 4 = 128 B/token.
      expect(info['mla_bytes_per_token'], 40);
      expect(info['vanilla_bytes_per_token'], 128);
      expect(info['compression_ratio'], closeTo(128.0 / 40.0, 1e-3));
    });
  });

  group('MlaKvCache', () {
    const MlaConfig config = MlaConfig(
      hiddenSize: 8,
      numHeads: 1,
      headDim: 4,
      kvLoraRank: 3,
      ropeHeadDim: 2,
      maxSeqLen: 4,
    );

    test('append / slotFor / latentAt / ropeAt round-trips', () {
      final MlaKvCache cache = MlaKvCache(config);
      expect(cache.length, 0);
      expect(cache.isSaturated, isFalse);

      final Float32List latent = Float32List.fromList(<double>[1, 2, 3]);
      final Float32List rope = Float32List.fromList(<double>[0.25, 0.5]);
      cache.append(latent, rope);

      expect(cache.length, 1);
      expect(cache.stats.tokens, 1);
      expect(cache.slotFor(0), 0);
      expect(cache.latentAt(0), orderedEquals(latent));
      expect(cache.ropeAt(0), orderedEquals(rope));

      final Float32List latent2 = Float32List.fromList(<double>[4, 5, 6]);
      final Float32List rope2 = Float32List.fromList(<double>[0.75, 1.0]);
      cache.append(latent2, rope2);
      expect(cache.slotFor(1), 1);
      expect(cache.latentAt(1), orderedEquals(latent2));
      expect(cache.ropeAt(1), orderedEquals(rope2));
    });

    test('append rejects tensors of the wrong width', () {
      final MlaKvCache cache = MlaKvCache(config);
      expect(
        () => cache.append(Float32List(2), Float32List(2)),
        throwsA(isA<MlaShapeException>()),
      );
      expect(
        () => cache.append(Float32List(3), Float32List(3)),
        throwsA(isA<MlaShapeException>()),
      );
    });

    test('out-of-range logical indices throw MlaCacheException', () {
      final MlaKvCache cache = MlaKvCache(config);
      cache.append(Float32List.fromList(<double>[1, 1, 1]), Float32List(2));
      expect(() => cache.slotFor(-1), throwsA(isA<MlaCacheException>()));
      expect(() => cache.slotFor(1), throwsA(isA<MlaCacheException>()));
      expect(() => cache.latentAt(-1), throwsA(isA<MlaCacheException>()));
      expect(() => cache.ropeAt(1), throwsA(isA<MlaCacheException>()));
    });

    test('sliding window saturates at maxSeqLen and evicts the oldest tokens', () {
      final MlaKvCache cache = MlaKvCache(config);
      for (int t = 0; t < 6; t++) {
        cache.append(
          Float32List.fromList(<double>[t + 0.0, t + 0.5, t + 0.25]),
          Float32List.fromList(<double>[t + 0.0, t + 0.125]),
        );
        if (t < 3) {
          expect(cache.length, t + 1);
          expect(cache.isSaturated, isFalse);
        }
      }

      // maxSeqLen is 4: after the 4th append the window is saturated and the
      // oldest token (t=0) has been overwritten by t=4.
      expect(cache.length, 4);
      expect(cache.isSaturated, isTrue);
      expect(cache.length, config.maxSeqLen);
      // Physical slot order after two wraps: oldest (t=2) lives in slot 2.
      expect(cache.slotFor(0), 2);
      expect(cache.slotFor(3), 1);
      expect(cache.latentAt(0), orderedEquals(<double>[2, 2.5, 2.25]));
      expect(cache.latentAt(1), orderedEquals(<double>[3, 3.5, 3.25]));
      expect(cache.latentAt(2), orderedEquals(<double>[4, 4.5, 4.25]));
      expect(cache.latentAt(3), orderedEquals(<double>[5, 5.5, 5.25]));
      expect(cache.ropeAt(0), orderedEquals(<double>[2, 2.125]));
      expect(cache.ropeAt(3), orderedEquals(<double>[5, 5.125]));
      expect(() => cache.slotFor(4), throwsA(isA<MlaCacheException>()));
    });

    test('reset rewinds the window so the next append starts at slot zero', () {
      final MlaKvCache cache = MlaKvCache(config);
      for (int t = 0; t < 6; t++) {
        cache.append(
          Float32List.fromList(<double>[t + 0.0, t + 0.0, t + 0.0]),
          Float32List.fromList(<double>[t + 0.0, t + 0.0]),
        );
      }
      expect(cache.isSaturated, isTrue);
      cache.reset();
      expect(cache.length, 0);
      expect(cache.isSaturated, isFalse);
      expect(cache.stats.tokens, 0);
      cache.append(
        Float32List.fromList(<double>[9, 9, 9]),
        Float32List.fromList(<double>[9, 9]),
      );
      expect(cache.slotFor(0), 0);
      expect(cache.latentAt(0), orderedEquals(<double>[9, 9, 9]));
    });
  });
}