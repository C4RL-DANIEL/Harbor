// The unit of text the corpus pipeline moves around, plus the enum that records
// where it came from.
//
// A `CorpusDocument` is deliberately dumb: it holds text and provenance and
// knows how to serialise itself. Deduplication, quota accounting and persistence
// live in `CorpusStore`, and extraction/redaction live in the device and web
// helpers, so this file stays dependency-free apart from a hash function and the
// UTF-8 codec. That keeps it importable from every platform Harbor targets,
// including Flutter Web where `dart:io` does not exist.

import 'dart:convert';

import 'package:crypto/crypto.dart';

/// Where a corpus document came from.
///
/// The distinction matters to the trainer and to the privacy story: text read
/// from the user's own device (`device`) may be shown back to them and removed
/// on request, text pulled off the web (`web`) carries a URL that must be
/// re-checked against the crawl policy, text shipped inside the application
/// (`bundled`) is trusted boilerplate, and text the user typed or pasted
/// (`user`) is treated as intentional and high-value.
enum CorpusSource {
  /// A file read from the user's device by the Android application.
  device,

  /// A page fetched by the web corpus collector.
  web,

  /// A text resource compiled into the application.
  bundled,

  /// Text the user typed, pasted or explicitly imported.
  user,
}

/// One immutable piece of corpus text together with its provenance.
///
/// The [id] is the SHA-256 hex digest of [text], which makes it the natural
/// deduplication key: two documents with identical text collide by construction
/// and the store can reject the second one without comparing whole strings.
/// Because the id is content derived, callers that mutate text must build a new
/// document (or go through `CorpusStore.add`, which re-hashes after redaction).
class CorpusDocument {
  /// Creates a document.
  ///
  /// [id] must be the content hash produced by [hashOf] for [text]; [uri] is the
  /// provenance string (an absolute file path for device text, an absolute URL
  /// for web text). [collectedAt] is stored in memory in whatever zone the
  /// caller supplies and serialised as UTC; [meta] carries small, string-only
  /// annotations such as a page title or the original file name.
  const CorpusDocument({
    required this.id,
    required this.text,
    required this.source,
    required this.uri,
    this.collectedAt,
    this.meta = const <String, String>{},
  });

  /// Content hash of [text] as a lowercase SHA-256 hex string.
  ///
  /// This is the deduplication key. It is also what makes the store safe to
  /// rebuild from disk: the same text always hashes to the same id on every
  /// platform, including in JavaScript, because the hash is computed over the
  /// UTF-8 bytes rather than over the platform's native string representation.
  final String id;

  /// The extracted, redacted text itself.
  final String text;

  /// Which half of the pipeline produced this document.
  final CorpusSource source;

  /// Provenance for the user: an absolute file path or an absolute URL.
  final String uri;

  /// When the text was collected, or `null` for bundled text that has no
  /// meaningful collection time.
  final DateTime? collectedAt;

  /// Small string-only annotations (page title, original file name, ...).
  ///
  /// Kept string-valued on purpose so the JSON representation is stable and the
  /// web dashboard can render it without a schema.
  final Map<String, String> meta;

  /// Length of [text] in UTF-8 bytes.
  ///
  /// Computed on demand rather than cached so the class stays a cheap `const`
  /// value; callers that need it in a tight loop should hold the result.
  int get bytes => utf8.encode(text).length;

  /// Length of [text] in UTF-16 code units, matching `String.length`.
  ///
  /// This is the same unit the `maxChars` limits use, so truncation and
  /// accounting agree with each other.
  int get charCount => text.length;

  /// Serialises this document to a JSON-compatible map.
  ///
  /// [collectedAt] is written as a UTC ISO-8601 string so a document collected
  /// on a device in one time zone round-trips to an identical instant in a
  /// browser in another; [meta] is written as a plain string map. The result is
  /// exactly what [fromJson] accepts, so `fromJson(toJson())` is the identity.
  Map<String, Object?> toJson() {
    final Map<String, Object?> json = <String, Object?>{
      'id': id,
      'text': text,
      'source': source.name,
      'uri': uri,
      'meta': meta,
    };
    final DateTime? collected = collectedAt;
    if (collected != null) {
      json['collectedAt'] = collected.toUtc().toIso8601String();
    }
    return json;
  }

  /// Rebuilds a document from [json], the shape produced by [toJson].
  ///
  /// Throws [FormatException] when a required field is missing or has the wrong
  /// type, when [collectedAt] is not a parseable ISO-8601 string, or when the
  /// source name is not one of [CorpusSource]'s values. The store catches that
  /// exception so a single damaged record cannot poison a whole payload.
  static CorpusDocument fromJson(Map<String, Object?> json) {
    final Object? id = json['id'];
    final Object? text = json['text'];
    final Object? source = json['source'];
    final Object? uri = json['uri'];
    if (id is! String || id.isEmpty) {
      throw const FormatException('CorpusDocument.id must be a non-empty string');
    }
    if (text is! String) {
      throw const FormatException('CorpusDocument.text must be a string');
    }
    if (source is! String) {
      throw const FormatException('CorpusDocument.source must be a string');
    }
    if (uri is! String) {
      throw const FormatException('CorpusDocument.uri must be a string');
    }

    CorpusSource? parsedSource;
    for (final CorpusSource candidate in CorpusSource.values) {
      if (candidate.name == source) {
        parsedSource = candidate;
        break;
      }
    }
    if (parsedSource == null) {
      throw FormatException('unknown CorpusDocument.source "$source"');
    }

    final Object? rawCollectedAt = json['collectedAt'];
    DateTime? collectedAt;
    if (rawCollectedAt != null) {
      if (rawCollectedAt is! String) {
        throw const FormatException('CorpusDocument.collectedAt must be a string');
      }
      final DateTime? parsed = DateTime.tryParse(rawCollectedAt);
      if (parsed == null) {
        throw FormatException('unparseable CorpusDocument.collectedAt "$rawCollectedAt"');
      }
      collectedAt = parsed.toUtc();
    }

    final Map<String, String> meta = <String, String>{};
    final Object? rawMeta = json['meta'];
    if (rawMeta != null) {
      if (rawMeta is! Map<String, Object?>) {
        throw const FormatException('CorpusDocument.meta must be an object');
      }
      for (final MapEntry<String, Object?> entry in rawMeta.entries) {
        final Object? value = entry.value;
        meta[entry.key] = value is String ? value : '${value ?? ''}';
      }
    }

    return CorpusDocument(
      id: id,
      text: text,
      source: parsedSource,
      uri: uri,
      collectedAt: collectedAt,
      meta: meta,
    );
  }

  /// The lowercase SHA-256 hex digest of [text]'s UTF-8 bytes.
  ///
  /// Used by the store to derive the deduplication key and by tests as a stable
  /// fingerprint. Hashing the encoded bytes rather than the string keeps the
  /// result identical on the VM and in JavaScript.
  static String hashOf(String text) {
    final List<int> bytes = utf8.encode(text);
    return sha256.convert(bytes).toString();
  }

  /// Renders a short, human-readable summary for logs and assertions.
  ///
  /// Intentionally omits [text] so a debug dump never leaks corpus content.
  @override
  String toString() =>
      'CorpusDocument(${source.name}, ${bytes}B, $uri, $id)';
}