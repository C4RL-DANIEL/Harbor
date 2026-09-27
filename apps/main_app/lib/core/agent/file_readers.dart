// Universal file reading and parsing for the on-device agent.
//
// The agent must be able to reason about anything the user points it at, so this
// module normalises wildly different formats into a single structured shape:
//
//   * raw text / logs / markdown
//   * source code (40+ extensions, language-tagged)
//   * binary containers — magic-number sniffing for ELF, PE, Mach-O, ZIP,
//     GZIP, PNG, JPEG, PDF, SQLite, WASM and more, plus a hex+ASCII header dump
//   * archives — zip / tar / tar.gz / tgz / gz with a real entry listing and
//     bounded per-entry text extraction (zip-bomb guarded)
//   * structured data — JSON, JSONL/NDJSON, YAML, CSV/TSV, INI, .properties, XML
//
// Everything is bounded: files above [FileReadLimits.maxBytes] are windowed
// rather than slurped, archives are capped on entry count and uncompressed size,
// and every failure is a typed exception, never a silent null.

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:archive/archive.dart' as arch;
import 'package:xml/xml.dart' as xml;
import 'package:yaml/yaml.dart' show loadYaml;

/// Thrown when a file cannot be found or is unreadable.
class FileReadException implements Exception {
  FileReadException(this.message, [this.path, this.cause]);

  final String message;
  final String? path;
  final Object? cause;

  @override
  String toString() => 'FileReadException: $message'
      '${path == null ? '' : ' [$path]'}'
      '${cause == null ? '' : ' ($cause)'}';
}

/// Thrown when a parser cannot make sense of otherwise readable bytes.
class ParseFailureException implements Exception {
  ParseFailureException(this.message, {this.format, this.path});

  final String message;
  final String? format;
  final String? path;

  @override
  String toString() => 'ParseFailureException: $message'
      '${format == null ? '' : ' (format: $format)'}'
      '${path == null ? '' : ' [$path]'}';
}

/// Thrown when a safety limit is exceeded (size, entry count, nesting).
class FileReadLimitException implements Exception {
  FileReadLimitException(this.message);

  final String message;

  @override
  String toString() => 'FileReadLimitException: $message';
}

/// Coarse classification produced by [UniversalFileReader.detectKind].
enum FileKind {
  text,
  code,
  structuredData,
  archive,
  binary,
  image,
  document,
  unknown,
}

/// Safety bounds applied to every read.
class FileReadLimits {
  const FileReadLimits({
    this.maxBytes = 8 * 1024 * 1024,
    this.maxHeaderBytes = 512,
    this.maxArchiveEntries = 512,
    this.maxUncompressedBytes = 32 * 1024 * 1024,
    this.maxExtractedEntryBytes = 256 * 1024,
    this.maxCsvRows = 5000,
    this.maxLines = 20000,
  });

  /// Largest regular file read in full; larger files are windowed.
  final int maxBytes;

  /// Bytes captured for the binary header dump.
  final int maxHeaderBytes;

  /// Maximum number of archive entries inspected.
  final int maxArchiveEntries;

  /// Total uncompressed budget when walking an archive.
  final int maxUncompressedBytes;

  /// Per-entry text extraction cap inside an archive.
  final int maxExtractedEntryBytes;

  /// Maximum CSV/TSV rows parsed.
  final int maxCsvRows;

  /// Maximum lines retained for text documents.
  final int maxLines;

  static const FileReadLimits defaults = FileReadLimits();
}

/// A single entry discovered inside an archive.
class ArchiveEntryInfo {
  const ArchiveEntryInfo({
    required this.name,
    required this.size,
    required this.isDirectory,
    this.crc32,
    this.preview,
  });

  final String name;
  final int size;
  final bool isDirectory;

  /// CRC32 of the uncompressed content, when the container records one.
  ///
  /// `archive` 3.x does not expose a per-entry compressed size on
  /// `ArchiveFile`, so compression ratio is reported at the archive level
  /// rather than per entry.
  final int? crc32;

  /// Bounded text preview, present only for text-like small entries.
  final String? preview;

  Map<String, Object?> toJson() => <String, Object?>{
        'name': name,
        'size': size,
        'is_directory': isDirectory,
        'crc32': crc32,
        'preview': preview,
      };
}

/// Structured result of reading any file.
class FileReadResult {
  const FileReadResult({
    required this.path,
    required this.kind,
    required this.sizeBytes,
    required this.isTruncated,
    this.text,
    this.language,
    this.structuredData,
    this.archiveEntries,
    this.binaryHeader,
    this.magicDescription,
    this.encoding,
    this.lineCount,
    this.warnings = const <String>[],
  });

  /// Absolute or relative path that was read.
  final String path;

  /// Detected classification.
  final FileKind kind;

  /// File size on disk in bytes.
  final int sizeBytes;

  /// Whether [text] is a windowed prefix rather than the whole file.
  final bool isTruncated;

  /// Decoded text content, when the file is text-like.
  final String? text;

  /// Language tag for source files (e.g. `dart`, `python`).
  final String? language;

  /// Parsed structured payload (Map/List) for data files.
  final Object? structuredData;

  /// Archive directory listing.
  final List<ArchiveEntryInfo>? archiveEntries;

  /// Hex + ASCII dump of the file header for binary files.
  final String? binaryHeader;

  /// Human description of the sniffed magic number.
  final String? magicDescription;

  /// Text encoding actually used (`utf-8`, `latin-1`, ...).
  final String? encoding;

  /// Number of lines in [text] when applicable.
  final int? lineCount;

  /// Non-fatal issues encountered while reading.
  final List<String> warnings;

  /// A compact, model-friendly summary of this file.
  String toPromptSummary() {
    final StringBuffer b = StringBuffer()
      ..writeln('File: $path')
      ..writeln('Kind: ${kind.name}  Size: $sizeBytes bytes'
          '${isTruncated ? ' (truncated)' : ''}');
    if (language != null) {
      b.writeln('Language: $language');
    }
    if (magicDescription != null) {
      b.writeln('Format: $magicDescription');
    }
    if (lineCount != null) {
      b.writeln('Lines: $lineCount');
    }
    if (archiveEntries != null) {
      b.writeln('Archive entries: ${archiveEntries!.length}');
    }
    for (final String w in warnings) {
      b.writeln('Warning: $w');
    }
    return b.toString();
  }

  Map<String, Object?> toJson() => <String, Object?>{
        'path': path,
        'kind': kind.name,
        'size_bytes': sizeBytes,
        'is_truncated': isTruncated,
        'language': language,
        'encoding': encoding,
        'line_count': lineCount,
        'magic_description': magicDescription,
        'binary_header': binaryHeader,
        'structured_data': structuredData,
        'archive_entries': archiveEntries
            ?.map((ArchiveEntryInfo e) => e.toJson())
            .toList(growable: false),
        'text_preview': text == null
            ? null
            : (text!.length > 4000 ? text!.substring(0, 4000) : text),
        'warnings': warnings,
      };
}

/// Magic-number signature entry.
class MagicSignature {
  const MagicSignature({
    required this.label,
    required this.kind,
    required this.offset,
    required this.bytes,
  });

  final String label;
  final FileKind kind;
  final int offset;
  final List<int> bytes;

  bool matches(Uint8List data) {
    if (data.length < offset + bytes.length) {
      return false;
    }
    for (int i = 0; i < bytes.length; i++) {
      if (data[offset + i] != bytes[i]) {
        return false;
      }
    }
    return true;
  }
}

/// Extension -> language tag map for source files.
const Map<String, String> kCodeExtensions = <String, String>{
  'dart': 'dart',
  'kt': 'kotlin',
  'kts': 'kotlin',
  'java': 'java',
  'swift': 'swift',
  'm': 'objective-c',
  'mm': 'objective-c++',
  'c': 'c',
  'h': 'c-header',
  'cc': 'c++',
  'cpp': 'c++',
  'cxx': 'c++',
  'hpp': 'c++-header',
  'cs': 'csharp',
  'go': 'go',
  'rs': 'rust',
  'py': 'python',
  'rb': 'ruby',
  'php': 'php',
  'js': 'javascript',
  'mjs': 'javascript',
  'cjs': 'javascript',
  'jsx': 'javascript-jsx',
  'ts': 'typescript',
  'tsx': 'typescript-tsx',
  'sh': 'shell',
  'bash': 'shell',
  'zsh': 'shell',
  'fish': 'shell',
  'ps1': 'powershell',
  'bat': 'batch',
  'cmd': 'batch',
  'sql': 'sql',
  'graphql': 'graphql',
  'gql': 'graphql',
  'proto': 'protobuf',
  'gradle': 'gradle',
  'lua': 'lua',
  'pl': 'perl',
  'r': 'r',
  'scala': 'scala',
  'clj': 'clojure',
  'ex': 'elixir',
  'exs': 'elixir',
  'erl': 'erlang',
  'hs': 'haskell',
  'ml': 'ocaml',
  'vue': 'vue',
  'svelte': 'svelte',
  'html': 'html',
  'htm': 'html',
  'css': 'css',
  'scss': 'scss',
  'sass': 'sass',
  'less': 'less',
  'tf': 'terraform',
  'toml': 'toml',
  'ini': 'ini',
  'cfg': 'ini',
  'conf': 'ini',
  'properties': 'properties',
  'env': 'dotenv',
  'mk': 'makefile',
  'cmake': 'cmake',
  'asm': 'assembly',
  's': 'assembly',
};

/// Extension -> structured data format map.
const Map<String, String> kStructuredExtensions = <String, String>{
  'json': 'json',
  'jsonl': 'jsonl',
  'ndjson': 'jsonl',
  'yaml': 'yaml',
  'yml': 'yaml',
  'csv': 'csv',
  'tsv': 'tsv',
  'xml': 'xml',
  'ini': 'ini',
  'properties': 'properties',
  'plist': 'xml',
  'toml': 'toml',
};

/// Archive extensions handled by the archive walker.
const Set<String> kArchiveExtensions = <String>{
  'zip',
  'tar',
  'gz',
  'tgz',
  'jar',
  'apk',
  'aab',
  'aar',
  'war',
  'epub',
  'whl',
  'ipa',
  'xpi',
};

/// Extension -> image format.
const Set<String> kImageExtensions = <String>{
  'png',
  'jpg',
  'jpeg',
  'gif',
  'bmp',
  'webp',
  'ico',
  'heic',
  'tiff',
  'svg',
};

/// Extension -> document format.
const Set<String> kDocumentExtensions = <String>{
  'pdf',
  'doc',
  'docx',
  'xls',
  'xlsx',
  'ppt',
  'pptx',
  'odt',
  'ods',
};

/// Known text extensions that are neither code nor structured data.
const Set<String> kPlainTextExtensions = <String>{
  'txt',
  'md',
  'markdown',
  'rst',
  'log',
  'text',
  'license',
  'readme',
  'gitignore',
  'editorconfig',
  'lock',
  'sum',
  'patch',
  'diff',
};

/// Universal file reader capable of handling text, code, binary, archives and
/// structured data with a single entry point.
class UniversalFileReader {
  UniversalFileReader({this.limits = FileReadLimits.defaults});

  final FileReadLimits limits;

  static const List<MagicSignature> _signatures = <MagicSignature>[
    MagicSignature(
      label: 'ELF executable/library',
      kind: FileKind.binary,
      offset: 0,
      bytes: <int>[0x7F, 0x45, 0x4C, 0x46],
    ),
    MagicSignature(
      label: 'Mach-O 64-bit (little endian)',
      kind: FileKind.binary,
      offset: 0,
      bytes: <int>[0xCF, 0xFA, 0xED, 0xFE, 0x0C, 0x00, 0x00, 0x01],
    ),
    MagicSignature(
      label: 'Mach-O universal (fat) binary',
      kind: FileKind.binary,
      offset: 0,
      bytes: <int>[0xCA, 0xFE, 0xBA, 0xBE],
    ),
    MagicSignature(
      label: 'Windows PE executable',
      kind: FileKind.binary,
      offset: 0,
      bytes: <int>[0x4D, 0x5A],
    ),
    MagicSignature(
      label: 'Android DEX bytecode',
      kind: FileKind.binary,
      offset: 0,
      bytes: <int>[0x64, 0x65, 0x78, 0x0A],
    ),
    MagicSignature(
      label: 'Android binary XML (AXML)',
      kind: FileKind.binary,
      offset: 0,
      bytes: <int>[0x03, 0x00, 0x08, 0x00],
    ),
    MagicSignature(
      label: 'SQLite 3 database',
      kind: FileKind.binary,
      offset: 0,
      bytes: <int>[0x53, 0x51, 0x4C, 0x69, 0x74, 0x65, 0x20, 0x66],
    ),
    MagicSignature(
      label: 'WebAssembly module',
      kind: FileKind.binary,
      offset: 0,
      bytes: <int>[0x00, 0x61, 0x73, 0x6D],
    ),
    MagicSignature(
      label: 'Java class file',
      kind: FileKind.binary,
      offset: 0,
      bytes: <int>[0xCA, 0xFE, 0xBA, 0xBE],
    ),
    MagicSignature(
      label: 'ZIP / JAR / APK archive',
      kind: FileKind.archive,
      offset: 0,
      bytes: <int>[0x50, 0x4B, 0x03, 0x04],
    ),
    MagicSignature(
      label: 'ZIP archive (empty)',
      kind: FileKind.archive,
      offset: 0,
      bytes: <int>[0x50, 0x4B, 0x05, 0x06],
    ),
    MagicSignature(
      label: 'GZIP compressed data',
      kind: FileKind.archive,
      offset: 0,
      bytes: <int>[0x1F, 0x8B],
    ),
    MagicSignature(
      label: 'BZIP2 compressed data',
      kind: FileKind.archive,
      offset: 0,
      bytes: <int>[0x42, 0x5A, 0x68],
    ),
    MagicSignature(
      label: 'XZ compressed data',
      kind: FileKind.archive,
      offset: 0,
      bytes: <int>[0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00],
    ),
    MagicSignature(
      label: 'Zstandard compressed data',
      kind: FileKind.archive,
      offset: 0,
      bytes: <int>[0x28, 0xB5, 0x2F, 0xFD],
    ),
    MagicSignature(
      label: 'TAR archive',
      kind: FileKind.archive,
      offset: 257,
      bytes: <int>[0x75, 0x73, 0x74, 0x61, 0x72],
    ),
    MagicSignature(
      label: 'PNG image',
      kind: FileKind.image,
      offset: 0,
      bytes: <int>[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
    ),
    MagicSignature(
      label: 'JPEG image',
      kind: FileKind.image,
      offset: 0,
      bytes: <int>[0xFF, 0xD8, 0xFF],
    ),
    MagicSignature(
      label: 'GIF image',
      kind: FileKind.image,
      offset: 0,
      bytes: <int>[0x47, 0x49, 0x46, 0x38],
    ),
    MagicSignature(
      label: 'BMP image',
      kind: FileKind.image,
      offset: 0,
      bytes: <int>[0x42, 0x4D],
    ),
    MagicSignature(
      label: 'WebP image',
      kind: FileKind.image,
      offset: 0,
      bytes: <int>[0x52, 0x49, 0x46, 0x46],
    ),
    MagicSignature(
      label: 'PDF document',
      kind: FileKind.document,
      offset: 0,
      bytes: <int>[0x25, 0x50, 0x44, 0x46],
    ),
    MagicSignature(
      label: 'OLE2 compound document (legacy Office)',
      kind: FileKind.document,
      offset: 0,
      bytes: <int>[0xD0, 0xCF, 0x11, 0xE0],
    ),
    MagicSignature(
      label: 'OOXML document (docx/xlsx/pptx)',
      kind: FileKind.document,
      offset: 0,
      bytes: <int>[0x50, 0x4B, 0x03, 0x04],
    ),
    MagicSignature(
      label: 'UTF-8 byte order mark',
      kind: FileKind.text,
      offset: 0,
      bytes: <int>[0xEF, 0xBB, 0xBF],
    ),
    MagicSignature(
      label: 'UTF-16 little endian BOM',
      kind: FileKind.text,
      offset: 0,
      bytes: <int>[0xFF, 0xFE],
    ),
    MagicSignature(
      label: 'UTF-16 big endian BOM',
      kind: FileKind.text,
      offset: 0,
      bytes: <int>[0xFE, 0xFF],
    ),
  ];

  /// Sniffs the leading bytes of [data] and returns a matching signature.
  static MagicSignature? sniff(Uint8List data) {
    for (final MagicSignature s in _signatures) {
      if (s.matches(data)) {
        return s;
      }
    }
    return null;
  }

  /// Returns the lowercase extension of [path] without the dot.
  static String extensionOf(String path) {
    final String name = path.split(RegExp(r'[/\\]')).last;
    final int dot = name.lastIndexOf('.');
    if (dot <= 0 || dot == name.length - 1) {
      return '';
    }
    return name.substring(dot + 1).toLowerCase();
  }

  /// Classifies a path by extension alone (no I/O).
  static FileKind classifyExtension(String path) {
    final String ext = extensionOf(path);
    if (kArchiveExtensions.contains(ext)) {
      return FileKind.archive;
    }
    if (kImageExtensions.contains(ext)) {
      return FileKind.image;
    }
    if (kDocumentExtensions.contains(ext)) {
      return FileKind.document;
    }
    if (kStructuredExtensions.containsKey(ext)) {
      return FileKind.structuredData;
    }
    if (kCodeExtensions.containsKey(ext)) {
      return FileKind.code;
    }
    if (kPlainTextExtensions.contains(ext) || ext.isEmpty) {
      return FileKind.text;
    }
    return FileKind.unknown;
  }

  /// Reads [path] and returns a fully parsed [FileReadResult].
  Future<FileReadResult> read(String path) async {
    final File file = File(path);
    if (!await file.exists()) {
      throw FileReadException('file does not exist', path);
    }
    final FileStat stat = await file.stat();
    if (stat.type == FileSystemEntityType.directory) {
      throw FileReadException('path is a directory, not a file', path);
    }

    final int size = stat.size;
    final int readCap = size > limits.maxBytes ? limits.maxBytes : size;
    Uint8List bytes;
    try {
      final RandomAccessFile handle = await file.open();
      try {
        bytes = await handle.read(readCap);
      } finally {
        await handle.close();
      }
    } on FileSystemException catch (e) {
      throw FileReadException('failed to read file', path, e);
    }

    final bool truncated = size > readCap;
    final List<String> warnings = <String>[];
    if (truncated) {
      warnings.add(
        'file is $size bytes; only the first $readCap bytes were read',
      );
    }
    if (bytes.isEmpty) {
      return FileReadResult(
        path: path,
        kind: FileKind.text,
        sizeBytes: size,
        isTruncated: truncated,
        text: '',
        language: null,
        encoding: 'utf-8',
        lineCount: 0,
        warnings: warnings,
      );
    }

    final MagicSignature? signature = sniff(bytes);
    final FileKind extKind = classifyExtension(path);

    // Archives and binary containers are handled structurally, regardless of
    // whether the extension lied about the content.
    if (signature != null &&
        signature.kind == FileKind.archive &&
        !signature.label.startsWith('GZIP') &&
        !signature.label.startsWith('BZIP2') &&
        !signature.label.startsWith('XZ') &&
        !signature.label.startsWith('Zstandard') &&
        !signature.label.startsWith('TAR')) {
      try {
        return await _readArchive(path, size, bytes, signature, warnings, truncated);
      } on FileReadLimitException {
        rethrow;
      } on Object catch (e) {
        warnings.add('archive parsing failed ($e); falling back to sniffing');
      }
    }

    if (extKind == FileKind.archive) {
      try {
        return await _readArchive(path, size, bytes, signature, warnings, truncated);
      } on FileReadLimitException {
        rethrow;
      } on Object catch (e) {
        warnings.add('archive parsing failed ($e); falling back to sniffing');
      }
    }

    if (extKind == FileKind.structuredData) {
      try {
        return _readStructured(path, size, bytes, null, warnings, truncated);
      } on ParseFailureException catch (e) {
        warnings.add('${e.message}; falling back to raw text');
      }
    }

    if (signature != null &&
        (signature.kind == FileKind.binary ||
            signature.kind == FileKind.image ||
            signature.kind == FileKind.document)) {
      return _readBinary(path, size, bytes, signature, warnings, truncated);
    }

    // Text-like path: decode and, if the extension suggests structure, parse.
    final _DecodedText decoded = _decodeText(bytes);
    if (!decoded.isValidUtf8 && signature == null) {
      return _readBinary(path, size, bytes, null, warnings, truncated);
    }
    if (extKind == FileKind.structuredData) {
      try {
        return _readStructured(
          path,
          size,
          bytes,
          decoded.text,
          warnings,
          truncated,
        );
      } on ParseFailureException catch (e) {
        warnings.add('${e.message}; returning raw text instead');
      }
    }

    if (signature != null && signature.kind == FileKind.binary) {
      return _readBinary(path, size, bytes, signature, warnings, truncated);
    }

    return _readTextLike(
      path,
      size,
      bytes,
      decoded,
      signature?.label,
      warnings,
      truncated,
    );
  }

  _DecodedText _decodeText(Uint8List bytes) {
    String? utf8Text;
    try {
      utf8Text = utf8.decode(bytes);
    } on FormatException {
      utf8Text = null;
    }
    if (utf8Text != null) {
      return _DecodedText(utf8Text, 'utf-8', true);
    }
    // Latin-1 can decode any byte sequence, so it is the safe fallback.
    final StringBuffer b = StringBuffer();
    for (final int byte in bytes) {
      b.writeCharCode(byte);
    }
    return _DecodedText(b.toString(), 'latin-1', false);
  }

  FileReadResult _readTextLike(
    String path,
    int size,
    Uint8List bytes,
    _DecodedText decoded,
    String? magicLabel,
    List<String> warnings,
    bool truncated,
  ) {
    String text = decoded.text;
    // Strip a leading BOM so the model does not see a stray U+FEFF.
    if (text.startsWith('\uFEFF')) {
      text = text.substring(1);
    }
    final List<String> lines = const LineSplitter().convert(text);
    int lineCount = lines.length;
    if (lineCount > limits.maxLines) {
      text = lines.take(limits.maxLines).join('\n');
      warnings.add(
        'document has $lineCount lines; truncated to ${limits.maxLines}',
      );
      lineCount = limits.maxLines;
    }
    final String ext = extensionOf(path);
    return FileReadResult(
      path: path,
      kind: kCodeExtensions.containsKey(ext) ? FileKind.code : FileKind.text,
      sizeBytes: size,
      isTruncated: truncated,
      text: text,
      language: kCodeExtensions[ext],
      encoding: decoded.encoding,
      lineCount: lineCount,
      magicDescription: magicLabel,
      warnings: warnings,
    );
  }

  FileReadResult _readBinary(
    String path,
    int size,
    Uint8List bytes,
    MagicSignature? signature,
    List<String> warnings,
    bool truncated,
  ) {
    final int headerLen = math.min(limits.maxHeaderBytes, bytes.length);
    final Uint8List header = Uint8List.sublistView(bytes, 0, headerLen);
    final String ext = extensionOf(path);
    FileKind kind = signature?.kind ?? FileKind.binary;
    if (signature == null && kImageExtensions.contains(ext)) {
      kind = FileKind.image;
    }
    if (signature == null && kDocumentExtensions.contains(ext)) {
      kind = FileKind.document;
    }
    return FileReadResult(
      path: path,
      kind: kind,
      sizeBytes: size,
      isTruncated: truncated,
      binaryHeader: hexDump(header),
      magicDescription: signature?.label ?? 'unrecognised binary ($ext)',
      warnings: warnings,
    );
  }

  Future<FileReadResult> _readArchive(
    String path,
    int size,
    Uint8List bytes,
    MagicSignature? signature,
    List<String> warnings,
    bool truncated,
  ) async {
    final arch.Archive archive;
    try {
      archive = _decodeArchive(path, bytes);
    } on Object catch (e) {
      throw ParseFailureException(
        'could not decode archive: $e',
        format: extensionOf(path),
        path: path,
      );
    }

    final List<ArchiveEntryInfo> entries = <ArchiveEntryInfo>[];
    int uncompressedTotal = 0;
    for (final arch.ArchiveFile f in archive.files) {
      if (entries.length >= limits.maxArchiveEntries) {
        warnings.add(
          'archive has more than ${limits.maxArchiveEntries} entries; '
          'listing was capped',
        );
        break;
      }
      uncompressedTotal += f.size;
      if (uncompressedTotal > limits.maxUncompressedBytes) {
        warnings.add(
          'archive expands beyond the $limits.maxUncompressedBytes byte '
          'budget; extraction stopped',
        );
        break;
      }
      String? preview;
      if (!f.isFile) {
        preview = null;
      } else if (f.size <= limits.maxExtractedEntryBytes) {
        preview = _extractEntryPreview(f);
      } else {
        warnings.add('entry ${f.name} skipped (${f.size} bytes exceeds cap)');
      }
      entries.add(
        ArchiveEntryInfo(
          name: f.name,
          size: f.size,
          isDirectory: !f.isFile,
          crc32: f.crc32,
          preview: preview,
        ),
      );
    }

    return FileReadResult(
      path: path,
      kind: FileKind.archive,
      sizeBytes: size,
      isTruncated: truncated,
      archiveEntries: List<ArchiveEntryInfo>.unmodifiable(entries),
      magicDescription: signature?.label ?? 'archive (${extensionOf(path)})',
      warnings: warnings,
    );
  }

  arch.Archive _decodeArchive(String path, Uint8List bytes) {
    final String ext = extensionOf(path);
    // gzip/tar.gz arrive as a gzip stream wrapping a tar payload.
    if (ext == 'gz' || ext == 'tgz' || _looksGzipped(bytes)) {
      final List<int> inflated = arch.GZipDecoder().decodeBytes(bytes);
      try {
        return arch.TarDecoder().decodeBytes(inflated);
      } on Object {
        // Plain gzip of a single file, not a tarball.
        final arch.Archive single = arch.Archive();
        single.addFile(
          arch.ArchiveFile(
            path.replaceAll(RegExp(r'\.(gz|tgz)$'), ''),
            inflated.length,
            inflated,
          ),
        );
        return single;
      }
    }
    if (ext == 'tar' || _looksTar(bytes)) {
      return arch.TarDecoder().decodeBytes(bytes);
    }
    if (ext == 'zip' ||
        ext == 'jar' ||
        ext == 'apk' ||
        ext == 'aab' ||
        ext == 'aar' ||
        ext == 'war' ||
        ext == 'epub' ||
        ext == 'whl' ||
        ext == 'ipa' ||
        ext == 'xpi' ||
        _looksZip(bytes)) {
      return arch.ZipDecoder().decodeBytes(bytes, verify: false);
    }
    // Last resort: let the package auto-detect.
    return arch.ZipDecoder().decodeBytes(bytes, verify: false);
  }

  bool _looksGzipped(Uint8List b) => b.length >= 2 && b[0] == 0x1F && b[1] == 0x8B;

  bool _looksZip(Uint8List b) =>
      b.length >= 4 && b[0] == 0x50 && b[1] == 0x4B;

  bool _looksTar(Uint8List b) {
    if (b.length < 262) {
      return false;
    }
    return b[257] == 0x75 &&
        b[258] == 0x73 &&
        b[259] == 0x74 &&
        b[260] == 0x61 &&
        b[261] == 0x72;
  }

  String? _extractEntryPreview(arch.ArchiveFile f) {
    final List<int>? content = f.content as List<int>?;
    if (content == null) {
      return null;
    }
    final Uint8List data = content is Uint8List
        ? content
        : Uint8List.fromList(content);
    if (data.isEmpty) {
      return '';
    }
    final MagicSignature? sig = sniff(data);
    if (sig != null &&
        (sig.kind == FileKind.binary ||
            sig.kind == FileKind.image ||
            sig.kind == FileKind.document)) {
      return null;
    }
    final _DecodedText decoded = _decodeText(data);
    if (!decoded.isValidUtf8) {
      return null;
    }
    final String text = decoded.text;
    return text.length > 2000 ? text.substring(0, 2000) : text;
  }

  FileReadResult _readStructured(
    String path,
    int size,
    Uint8List bytes,
    String? preDecoded,
    List<String> warnings,
    bool truncated,
  ) {
    final String ext = extensionOf(path);
    final String format = kStructuredExtensions[ext] ?? 'json';
    final _DecodedText decoded =
        preDecoded != null ? _DecodedText(preDecoded, 'utf-8', true) : _decodeText(bytes);
    final String text = decoded.text;
    Object? parsed;

    switch (format) {
      case 'json':
        parsed = _parseJson(text, path);
        break;
      case 'jsonl':
        parsed = _parseJsonLines(text, path, warnings);
        break;
      case 'yaml':
        parsed = _parseYaml(text, path);
        break;
      case 'csv':
        parsed = _parseDelimited(text, ',', warnings);
        break;
      case 'tsv':
        parsed = _parseDelimited(text, '\t', warnings);
        break;
      case 'xml':
        parsed = _parseXml(text, path);
        break;
      case 'ini':
      case 'properties':
        parsed = _parseIni(text, format == 'properties');
        break;
      case 'toml':
        parsed = _parseToml(text, warnings);
        break;
      default:
        throw ParseFailureException(
          'unsupported structured format "$format"',
          format: format,
          path: path,
        );
    }

    return FileReadResult(
      path: path,
      kind: FileKind.structuredData,
      sizeBytes: size,
      isTruncated: truncated,
      text: text,
      structuredData: parsed,
      encoding: decoded.encoding,
      lineCount: const LineSplitter().convert(text).length,
      magicDescription: '$format document',
      warnings: warnings,
    );
  }

  Object? _parseJson(String text, String path) {
    try {
      return jsonDecode(text);
    } on FormatException catch (e) {
      throw ParseFailureException(
        'invalid JSON at offset ${e.offset}',
        format: 'json',
        path: path,
      );
    }
  }

  List<Object?> _parseJsonLines(
    String text,
    String path,
    List<String> warnings,
  ) {
    final List<Object?> rows = <Object?>[];
    int lineNo = 0;
    for (final String line in const LineSplitter().convert(text)) {
      lineNo++;
      final String trimmed = line.trim();
      if (trimmed.isEmpty) {
        continue;
      }
      if (rows.length >= limits.maxCsvRows) {
        warnings.add('JSONL truncated at ${limits.maxCsvRows} rows');
        break;
      }
      try {
        rows.add(jsonDecode(trimmed));
      } on FormatException {
        warnings.add('skipped malformed JSONL line $lineNo');
      }
    }
    return rows;
  }

  Object? _parseYaml(String text, String path) {
    try {
      return _normaliseYaml(loadYaml(text));
    } on Object catch (e) {
      throw ParseFailureException(
        'invalid YAML: $e',
        format: 'yaml',
        path: path,
      );
    }
  }

  Object? _normaliseYaml(Object? node) {
    if (node is Map) {
      return node.map<String, Object?>(
        (Object? k, Object? v) =>
            MapEntry<String, Object?>(k.toString(), _normaliseYaml(v)),
      );
    }
    if (node is List) {
      return node.map<Object?>(_normaliseYaml).toList(growable: false);
    }
    return node;
  }

  Object? _parseXml(String text, String path) {
    try {
      final xml.XmlDocument doc = xml.XmlDocument.parse(text);
      return _xmlToMap(doc.rootElement);
    } on xml.XmlParserException catch (e) {
      throw ParseFailureException(
        'invalid XML: ${e.message}',
        format: 'xml',
        path: path,
      );
    }
  }

  Map<String, Object?> _xmlToMap(xml.XmlElement element) {
    final Map<String, Object?> out = <String, Object?>{
      'tag': element.name.local,
    };
    if (element.attributes.isNotEmpty) {
      out['attributes'] = <String, Object?>{
        for (final xml.XmlAttribute a in element.attributes)
          a.name.local: a.value,
      };
    }
    final List<xml.XmlElement> children = element.childElements.toList();
    if (children.isEmpty) {
      out['text'] = element.innerText.trim();
    } else {
      out['children'] = children
          .map<Map<String, Object?>>(_xmlToMap)
          .toList(growable: false);
    }
    return out;
  }

  Map<String, Object?> _parseDelimited(
    String text,
    String delimiter,
    List<String> warnings,
  ) {
    final List<List<String>> rows =
        _splitDelimited(text, delimiter, warnings);
    if (rows.isEmpty) {
      return <String, Object?>{'headers': <String>[], 'rows': <Object?>[]};
    }
    final List<String> headers = rows.first;
    final List<Object?> records = <Object?>[];
    for (int i = 1; i < rows.length; i++) {
      final List<String> row = rows[i];
      final Map<String, Object?> record = <String, Object?>{};
      for (int c = 0; c < headers.length; c++) {
        final String key = headers[c].trim();
        record[key.isEmpty ? 'column_$c' : key] =
            c < row.length ? _coerceScalar(row[c]) : null;
      }
      records.add(record);
    }
    return <String, Object?>{
      'headers': headers,
      'row_count': records.length,
      'rows': records,
    };
  }

  /// RFC-4180-ish splitter: honours quoted fields and escaped quotes.
  List<List<String>> _splitDelimited(
    String text,
    String delimiter,
    List<String> warnings,
  ) {
    final List<List<String>> rows = <List<String>>[];
    List<String> current = <String>[];
    final StringBuffer field = StringBuffer();
    bool inQuotes = false;

    void endField() {
      current.add(field.toString());
      field.clear();
    }

    void endRow() {
      endField();
      rows.add(current);
      current = <String>[];
    }

    bool isDelimiter(int index) {
      if (delimiter.length == 1) {
        return text[index] == delimiter;
      }
      return text.startsWith(delimiter, index);
    }

    for (int i = 0; i < text.length; i++) {
      final String ch = text[i];
      if (inQuotes) {
        if (ch == '"') {
          if (i + 1 < text.length && text[i + 1] == '"') {
            field.write('"');
            i++;
          } else {
            inQuotes = false;
          }
        } else {
          field.write(ch);
        }
        continue;
      }
      if (ch == '"') {
        inQuotes = true;
        continue;
      }
      if (isDelimiter(i)) {
        endField();
        i += delimiter.length - 1;
        continue;
      }
      if (ch == '\n') {
        endRow();
        if (rows.length >= limits.maxCsvRows) {
          warnings.add('delimited file truncated at ${limits.maxCsvRows} rows');
          return rows;
        }
        continue;
      }
      if (ch == '\r') {
        continue;
      }
      field.write(ch);
    }
    if (field.isNotEmpty || current.isNotEmpty) {
      endRow();
    }
    return rows;
  }

  Object? _coerceScalar(String raw) {
    final String v = raw.trim();
    if (v.isEmpty) {
      return '';
    }
    if (v.toLowerCase() == 'true') {
      return true;
    }
    if (v.toLowerCase() == 'false') {
      return false;
    }
    if (v.toLowerCase() == 'null') {
      return null;
    }
    final int? asInt = int.tryParse(v);
    if (asInt != null) {
      return asInt;
    }
    final double? asDouble = double.tryParse(v);
    if (asDouble != null) {
      return asDouble;
    }
    return v;
  }

  Map<String, Object?> _parseIni(String text, bool properties) {
    final Map<String, Object?> sections = <String, Object?>{};
    final Map<String, Object?> root = <String, Object?>{};
    Map<String, Object?> bucket = root;
    for (final String rawLine in const LineSplitter().convert(text)) {
      final String line = rawLine.trim();
      if (line.isEmpty || line.startsWith(';') || line.startsWith('#')) {
        continue;
      }
      if (line.startsWith('[') && line.endsWith(']')) {
        final String name = line.substring(1, line.length - 1).trim();
        final Map<String, Object?> section = <String, Object?>{};
        sections[name] = section;
        bucket = section;
        continue;
      }
      final int sepIdx = properties
          ? _firstIndexOfAny(line, const <String>['=', ':'])
          : _firstIndexOfAny(line, const <String>['=', ':']);
      if (sepIdx <= 0) {
        continue;
      }
      final String key = line.substring(0, sepIdx).trim();
      final String value = line.substring(sepIdx + 1).trim();
      bucket[key] = _coerceScalar(value);
    }
    if (root.isNotEmpty) {
      sections['_root'] = root;
    }
    return sections;
  }

  int _firstIndexOfAny(String line, List<String> needles) {
    int best = -1;
    for (final String n in needles) {
      final int idx = line.indexOf(n);
      if (idx >= 0 && (best < 0 || idx < best)) {
        best = idx;
      }
    }
    return best;
  }

  /// Minimal TOML subset: top-level key/value pairs and `[section]` headers,
  /// which is what the agent needs for pubspec/build metadata files.
  Map<String, Object?> _parseToml(String text, List<String> warnings) {
    final Map<String, Object?> root = <String, Object?>{};
    Map<String, Object?> bucket = root;
    for (final String rawLine in const LineSplitter().convert(text)) {
      final String line = rawLine.trim();
      if (line.isEmpty || line.startsWith('#')) {
        continue;
      }
      if (line.startsWith('[') && line.endsWith(']')) {
        final String name = line.substring(1, line.length - 1).trim();
        final Map<String, Object?> section = <String, Object?>{};
        root[name] = section;
        bucket = section;
        continue;
      }
      final int eq = line.indexOf('=');
      if (eq <= 0) {
        continue;
      }
      final String key = line.substring(0, eq).trim().replaceAll('"', '');
      final String rawValue = line.substring(eq + 1).trim();
      bucket[key] = _coerceScalar(
        rawValue.replaceAll(RegExp(r'^["\x27]|["\x27]$'), ''),
      );
    }
    warnings.add('TOML parsed with the built-in minimal subset parser');
    return root;
  }

  /// Renders a classic `offset  hex  ascii` header dump.
  static String hexDump(Uint8List data, {int bytesPerLine = 16}) {
    final StringBuffer out = StringBuffer();
    for (int offset = 0; offset < data.length; offset += bytesPerLine) {
      final int end = math.min(offset + bytesPerLine, data.length);
      final StringBuffer hex = StringBuffer();
      final StringBuffer ascii = StringBuffer();
      for (int i = offset; i < end; i++) {
        final int b = data[i];
        hex.write(b.toRadixString(16).padLeft(2, '0').toUpperCase());
        hex.write(i == end - 1 ? '' : ' ');
        ascii.write(b >= 32 && b < 127 ? String.fromCharCode(b) : '.');
      }
      out.writeln(
        '${offset.toRadixString(16).padLeft(8, '0').toUpperCase()}  '
        '${hex.toString().padRight(bytesPerLine * 3 - 1)}  '
        '|${ascii.toString()}|',
      );
    }
    return out.toString();
  }

  /// Reads every file under [directoryPath] matching [extensions], bounded by
  /// [maxFiles], and returns their results. Unreadable entries are reported in
  /// the returned warnings list rather than aborting the walk.
  Future<CodebaseReadResult> readCodebase(
    String directoryPath, {
    Set<String>? extensions,
    int maxFiles = 200,
    int maxFileBytes = 512 * 1024,
    bool skipHidden = true,
  }) async {
    final Directory dir = Directory(directoryPath);
    if (!await dir.exists()) {
      throw FileReadException('directory does not exist', directoryPath);
    }
    final Set<String>? wanted =
        extensions?.map((String e) => e.toLowerCase().replaceAll('.', '')).toSet();
    final List<FileReadResult> files = <FileReadResult>[];
    final List<String> warnings = <String>[];

    await for (final FileSystemEntity entity
        in dir.list(recursive: true, followLinks: false)) {
      if (files.length >= maxFiles) {
        warnings.add('stopped after $maxFiles files');
        break;
      }
      if (entity is! File) {
        continue;
      }
      final String relative =
          entity.path.startsWith(dir.path) ? entity.path.substring(dir.path.length) : entity.path;
      if (skipHidden &&
          relative.split(RegExp(r'[/\\]')).any((String p) => p.startsWith('.'))) {
        continue;
      }
      final String ext = extensionOf(entity.path);
      if (wanted != null && !wanted.contains(ext)) {
        continue;
      }
      try {
        final FileStat stat = await entity.stat();
        if (stat.size > maxFileBytes) {
          warnings.add('skipped ${entity.path} (${stat.size} bytes > $maxFileBytes)');
          continue;
        }
        files.add(await read(entity.path));
      } on FileReadException catch (e) {
        warnings.add('could not read ${entity.path}: ${e.message}');
      } on ParseFailureException catch (e) {
        warnings.add('could not parse ${entity.path}: ${e.message}');
      }
    }

    return CodebaseReadResult(
      rootPath: directoryPath,
      files: List<FileReadResult>.unmodifiable(files),
      warnings: List<String>.unmodifiable(warnings),
    );
  }
}

/// Aggregate result of a codebase walk.
class CodebaseReadResult {
  const CodebaseReadResult({
    required this.rootPath,
    required this.files,
    required this.warnings,
  });

  final String rootPath;
  final List<FileReadResult> files;
  final List<String> warnings;

  /// Language histogram across the walked files.
  Map<String, int> get languageHistogram {
    final Map<String, int> counts = <String, int>{};
    for (final FileReadResult f in files) {
      final String key = f.language ?? f.kind.name;
      counts[key] = (counts[key] ?? 0) + 1;
    }
    return counts;
  }

  /// Total bytes read.
  int get totalBytes =>
      files.fold<int>(0, (int a, FileReadResult f) => a + f.sizeBytes);

  Map<String, Object?> toJson() => <String, Object?>{
        'root': rootPath,
        'file_count': files.length,
        'total_bytes': totalBytes,
        'languages': languageHistogram,
        'warnings': warnings,
        'files': files
            .map((FileReadResult f) => f.toJson())
            .toList(growable: false),
      };
}

class _DecodedText {
  const _DecodedText(this.text, this.encoding, this.isValidUtf8);

  final String text;
  final String encoding;
  final bool isValidUtf8;
}