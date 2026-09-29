// Pure text helpers for the device half of the corpus.
//
// Nothing in this file touches the file system or any platform channel: the
// Android application reads a file (or picks one through the storage access
// framework), passes the *name* and the raw *bytes* in, and gets text back. That
// split is what lets the exact same logic run under `dart test` on a workstation
// and inside a Flutter Web build, and it is why the extractor can be tested
// without creating temporary files.
//
// Two jobs live here:
//   * `DeviceTextExtractor` decides whether a file is text at all and decodes it
//     defensively; and
//   * `TextRedactor` scrubs anything that must never enter a training corpus or
//     leave the device.

import 'dart:convert';
// `dart:typed_data` is imported for `Uint8List`, which is the concrete type the
// platform hands us and which avoids copying when the byte list is already one.
import 'dart:typed_data';

/// Turns a file's name and bytes into corpus text, or refuses it.
///
/// The extension check is intentionally narrow: a model trained on the bytes of
/// arbitrary binaries learns noise, and an APK or a JPEG can be megabytes of it.
/// Anything whose extension is not on the list is skipped before a single byte
/// is inspected.
class DeviceTextExtractor {
  /// File extensions (lowercase, without the dot) that the extractor accepts.
  ///
  /// The set spans prose, structured data, markup and source code because all of
  /// them are useful training text for a byte-level tokenizer; it deliberately
  /// excludes archives, images, audio and compiled artifacts.
  static const Set<String> textExtensions = <String>{
    'txt',
    'md',
    'markdown',
    'rst',
    'log',
    'csv',
    'tsv',
    'json',
    'jsonl',
    'yaml',
    'yml',
    'xml',
    'html',
    'htm',
    'dart',
    'js',
    'ts',
    'py',
    'java',
    'kt',
    'swift',
    'go',
    'rs',
    'c',
    'h',
    'cpp',
    'cs',
    'rb',
    'php',
    'sh',
    'bash',
    'ini',
    'toml',
    'cfg',
    'conf',
    'properties',
    'gradle',
    'sql',
    'tex',
    'org',
    'adoc',
  };

  /// Whether [fileName]'s extension is in [textExtensions].
  ///
  /// Accepts both `/` and `\` as directory separators so a Windows-style path
  /// pasted into an Android picker still resolves, and ignores a trailing
  /// directory separator by looking only at the final segment. A name with no
  /// dot at all (for example `README`) is rejected rather than guessed at,
  /// because guessing wrong wastes a decode of potentially large bytes.
  static bool looksLikeText(String fileName) {
    final String base = fileName.replaceAll('\\', '/').split('/').last;
    final int dot = base.lastIndexOf('.');
    if (dot < 0 || dot == base.length - 1) {
      return false;
    }
    final String extension = base.substring(dot + 1).toLowerCase();
    return textExtensions.contains(extension);
  }

  /// Whether [bytes] look like binary rather than text.
  ///
  /// The heuristic inspects a bounded prefix and counts bytes that cannot appear
  /// in text: NUL and the other C0 control bytes (apart from tab, newline,
  /// carriage return and form feed) plus DEL. NUL is weighted extra heavily
  /// because it is the single strongest signal of a binary container and because
  /// a legitimately encoded UTF-16 file is detected by its BOM before this check
  /// is ever reached. The threshold is deliberately generous (more than 30% of
  /// the sample) so stray control bytes in a mostly-text log do not disqualify
  /// the whole file.
  static bool isProbablyBinary(List<int> bytes) {
    if (bytes.isEmpty) {
      return false;
    }
    final int sample = bytes.length < 8192 ? bytes.length : 8192;
    int suspicious = 0;
    for (int i = 0; i < sample; i++) {
      final int byte = bytes[i];
      if (byte == 0x00) {
        suspicious += 4;
      } else if (byte == 0x7F) {
        suspicious++;
      } else if (byte < 0x20 &&
          byte != 0x09 &&
          byte != 0x0A &&
          byte != 0x0D &&
          byte != 0x0C) {
        suspicious++;
      }
    }
    return suspicious / sample > 0.30;
  }

  /// Decodes [bytes] for [fileName], or returns `null` when the file is skipped.
  ///
  /// A file is skipped when its extension is unsupported, when it looks binary,
  /// when its decoded text is more than roughly 30% non-printable characters, or
  /// when decoding is impossible. The decoding path is forgiving by design:
  ///
  ///   * a UTF-8 BOM is removed, and a UTF-16 BOM switches to the matching
  ///     UTF-16 decoder because some Windows editors still write that;
  ///   * NUL bytes are stripped, since they break Dart strings and tokenizers;
  ///   * an invalid UTF-8 sequence becomes U+FFFD instead of throwing, because
  ///     one bad byte in a long log should not throw away the whole file; and
  ///   * the result is truncated to [maxChars] on a code-unit boundary that never
  ///     splits a surrogate pair.
  static String? extract(String fileName, List<int> bytes, {int maxChars = 200000}) {
    if (!looksLikeText(fileName)) {
      return null;
    }
    if (bytes.isEmpty) {
      return '';
    }

    // A UTF-16 BOM means the raw bytes are full of NULs, so the binary
    // heuristic must be bypassed and the pairs reassembled directly.
    if (bytes.length >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE) {
      return _finalise(_decodeUtf16(bytes.sublist(2), littleEndian: true), maxChars);
    }
    if (bytes.length >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF) {
      return _finalise(_decodeUtf16(bytes.sublist(2), littleEndian: false), maxChars);
    }

    if (isProbablyBinary(bytes)) {
      return null;
    }

    // Strip NULs before decoding: `utf8.decode` would turn each one into a U+FFFD
    // and inflate the non-printable ratio for files that are otherwise fine.
    final Uint8List cleaned = Uint8List.fromList(
      bytes.where((int byte) => byte != 0x00).toList(growable: false),
    );
    final String decoded = utf8.decode(cleaned, allowMalformed: true);
    return _finalise(decoded, maxChars);
  }

  /// Applies the printable-character and length limits shared by every path.
  static String? _finalise(String text, int maxChars) {
    final String withoutBom = text.startsWith('\uFEFF') ? text.substring(1) : text;
    if (_nonPrintableRatio(withoutBom) > 0.30) {
      return null;
    }
    return _truncate(withoutBom, maxChars);
  }

  /// Fraction of the sampled prefix that cannot be printed, including the
  /// replacement character that malformed UTF-8 decodes to.
  static double _nonPrintableRatio(String text) {
    if (text.isEmpty) {
      return 0;
    }
    final int sample = text.length < 65536 ? text.length : 65536;
    int suspicious = 0;
    for (int i = 0; i < sample; i++) {
      final int unit = text.codeUnitAt(i);
      if (unit == 0xFFFD) {
        suspicious++;
      } else if (unit < 0x20 &&
          unit != 0x09 &&
          unit != 0x0A &&
          unit != 0x0D &&
          unit != 0x0C) {
        suspicious++;
      } else if (unit == 0x7F) {
        suspicious++;
      }
    }
    return suspicious / sample;
  }

  /// Truncates [text] to at most [maxChars] code units without splitting a
  /// surrogate pair, which would corrupt the final character.
  static String _truncate(String text, int maxChars) {
    if (maxChars <= 0) {
      return '';
    }
    if (text.length <= maxChars) {
      return text;
    }
    int end = maxChars;
    final int last = text.codeUnitAt(end - 1);
    if (last >= 0xD800 && last <= 0xDBFF) {
      end--;
    }
    return text.substring(0, end);
  }

  /// Decodes UTF-16 code units from [bytes] according to the BOM that preceded
  /// them; an odd trailing byte is ignored because it cannot form a code unit.
  static String _decodeUtf16(List<int> bytes, {required bool littleEndian}) {
    final List<int> units = <int>[];
    for (int i = 0; i + 1 < bytes.length; i += 2) {
      final int first = bytes[i];
      final int second = bytes[i + 1];
      units.add(littleEndian ? first | (second << 8) : (first << 8) | second);
    }
    return String.fromCharCodes(units);
  }
}

/// Removes material that must never leave the device or enter a corpus.
///
/// Redaction runs before hashing and insertion in `CorpusStore`, so a secret
/// that is stripped here also does not influence the deduplication key. The
/// replacements are stable marker strings instead of empty strings so the
/// surrounding prose keeps its shape and the fact that something was removed is
/// visible to the user reviewing their corpus.
class TextRedactor {
  /// A PEM private-key block, including an explicitly labelled variant such as
  /// `-----BEGIN RSA PRIVATE KEY-----`.
  ///
  /// The end marker is optional so a truncated paste that contains only the
  /// header is still recognised and removed; without it, a lone
  /// `-----BEGIN PRIVATE KEY-----` would be reported as clean and the key body
  /// that follows it would be treated as ordinary text.
  static final RegExp _privateKey = RegExp(
    r'-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?'
    r'(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|$)',
  );

  /// An `Authorization: Bearer ...` header; the scheme name and header are kept
  /// so the redacted text still reads as a header.
  static final RegExp _authorizationHeader = RegExp(
    r'(authorization\s*:\s*bearer\s+)([^\s,;]+)',
    caseSensitive: false,
  );

  /// A bare `Bearer <token>` occurrence, for tokens pasted outside a header.
  static final RegExp _bearerToken = RegExp(
    r'\bbearer\s+([A-Za-z0-9._~+/\-]{8,})',
    caseSensitive: false,
  );

  /// `scheme://user:password@` credentials embedded in a URL.
  static final RegExp _urlCredentials = RegExp(
    r'([a-zA-Z][a-zA-Z0-9+.\-]*://)([^/\s:@]+):([^/\s@]+)@',
  );

  /// A conventional e-mail address.
  static final RegExp _email = RegExp(
    r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}',
  );

  /// A long hexadecimal run: 32 characters or more, the shortest common digest.
  static final RegExp _hexSecret = RegExp(
    r'(?<![A-Za-z0-9])[0-9a-fA-F]{32,}(?![A-Za-z0-9])',
  );

  /// A long base64 run, which covers API keys, JWTs and encoded key material.
  static final RegExp _base64Secret = RegExp(
    r'(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/=])',
  );

  /// A candidate phone number: optional country code, then digits with the usual
  /// separators. A match is only treated as a secret when it carries at least
  /// seven digits, which keeps short numbers such as years or ports intact.
  static final RegExp _phone = RegExp(
    r'(?<![A-Za-z0-9])\+?\d[\d\s().\-]{4,}\d(?![A-Za-z0-9])',
  );

  /// Replaces every recognised secret in [text] with a bracketed marker.
  ///
  /// Order is significant: private keys and authorization headers are removed
  /// before URL credentials, URL credentials before e-mail addresses (an address
  /// can look like the host part of a credentialed URL), and long opaque tokens
  /// last, once shorter structured secrets no longer overlap with them.
  static String redact(String text) {
    String out = text;
    out = out.replaceAll(_privateKey, '[REDACTED_PRIVATE_KEY]');
    out = out.replaceAllMapped(
      _authorizationHeader,
      (Match match) => '${match.group(1)}[REDACTED]',
    );
    out = out.replaceAllMapped(
      _bearerToken,
      (Match match) => 'Bearer [REDACTED]',
    );
    out = out.replaceAllMapped(
      _urlCredentials,
      (Match match) => '${match.group(1)}[REDACTED_CREDENTIALS]@',
    );
    out = out.replaceAll(_email, '[REDACTED_EMAIL]');
    out = out.replaceAll(_hexSecret, '[REDACTED_SECRET]');
    out = out.replaceAll(_base64Secret, '[REDACTED_SECRET]');
    out = out.replaceAllMapped(_phone, (Match match) {
      final String candidate = match.group(0) ?? '';
      return _digitCount(candidate) >= 7 ? '[REDACTED_PHONE]' : candidate;
    });
    return out;
  }

  /// Whether [text] contains anything [redact] would remove.
  ///
  /// The store uses this as a cheap second opinion when surfacing a corpus to
  /// the user, and the collector uses it to flag pages that look like they
  /// leaked credentials. It is intentionally conservative: a false positive
  /// hides a little prose, a false negative leaks a key.
  static bool containsLikelySecret(String text) {
    if (_privateKey.hasMatch(text)) {
      return true;
    }
    if (_authorizationHeader.hasMatch(text) || _bearerToken.hasMatch(text)) {
      return true;
    }
    if (_urlCredentials.hasMatch(text)) {
      return true;
    }
    if (_email.hasMatch(text)) {
      return true;
    }
    if (_hexSecret.hasMatch(text) || _base64Secret.hasMatch(text)) {
      return true;
    }
    for (final RegExpMatch match in _phone.allMatches(text)) {
      final String candidate = match.group(0) ?? '';
      if (_digitCount(candidate) >= 7) {
        return true;
      }
    }
    return false;
  }

  /// Counts ASCII digits in [candidate]; used to reject short numeric runs.
  static int _digitCount(String candidate) {
    int digits = 0;
    for (int i = 0; i < candidate.length; i++) {
      final int unit = candidate.codeUnitAt(i);
      if (unit >= 0x30 && unit <= 0x39) {
        digits++;
      }
    }
    return digits;
  }
}