// Compact, allocation-friendly serialisation helpers for model checkpoints.
//
// A checkpoint is dominated by parameter buffers. Encoding them as JSON arrays
// of decimal doubles costs roughly 20 bytes per scalar and forces the decoder
// through thousands of `double.parse` calls; a 300 000-parameter model would
// produce a ~6 MB document and a visible stall on a mid-range phone. Encoding
// the raw little-endian bytes as base64 costs ~5.3 bytes per scalar, decodes
// with a single `Float64List.view`, and is exactly lossless because IEEE-754
// bits — unlike decimal text — round-trip without rounding.

import 'dart:convert';
import 'dart:typed_data';

/// Encodes [values] as base64 over its raw little-endian IEEE-754 bytes.
String encodeFloats(Float64List values) {
  final Uint8List bytes = values.buffer
      .asUint8List(values.offsetInBytes, values.lengthInBytes);
  return base64Encode(bytes);
}

/// The inverse of [encodeFloats].
///
/// Throws [FormatException] when the payload is not valid base64 or does not
/// describe a whole number of doubles, so a truncated checkpoint fails loudly
/// instead of silently loading a half-initialised model.
Float64List decodeFloats(String payload) {
  final Uint8List bytes;
  try {
    bytes = base64Decode(payload);
  } on FormatException catch (error) {
    throw FormatException('parameter payload is not valid base64: ${error.message}');
  }
  if (bytes.lengthInBytes % 8 != 0) {
    throw FormatException(
      'parameter payload has ${bytes.lengthInBytes} bytes, which is not a '
      'whole number of float64 values',
    );
  }
  return Float64List.view(
    bytes.buffer,
    bytes.offsetInBytes,
    bytes.lengthInBytes ~/ 8,
  );
}

/// Reads a `List<Object?>` of finite numbers into a [Float64List].
///
/// Used by the human-readable parts of a checkpoint (tokenizer merges, config
/// values) where base64 would be unhelpful.
Float64List floatsFromJson(Object? raw, {required int expectedLength, required String field}) {
  if (raw is! List<Object?>) {
    throw FormatException('"$field" must be a JSON array');
  }
  if (raw.length != expectedLength) {
    throw FormatException(
      '"$field" has ${raw.length} entries, expected $expectedLength',
    );
  }
  final Float64List out = Float64List(expectedLength);
  for (int i = 0; i < expectedLength; i++) {
    final Object? value = raw[i];
    if (value is! num) {
      throw FormatException('"$field"[$i] is not a number');
    }
    out[i] = value.toDouble();
  }
  return out;
}

/// A stable, low-cost checksum of a checkpoint payload.
///
/// This is not a cryptographic digest: it exists so the loader can tell a
/// corrupted file from a valid one without importing `package:crypto` on the
/// hot path. The model integrity that actually matters (the APK SHA-256) is
/// verified by the update engine, not here.
String payloadChecksum(String payload) {
  int hash = 0x811c9dc5;
  for (int i = 0; i < payload.length; i++) {
    hash ^= payload.codeUnitAt(i);
    hash = (hash * 0x01000193) & 0xFFFFFFFF;
  }
  return hash.toRadixString(16).padLeft(8, '0');
}