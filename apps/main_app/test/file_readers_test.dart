// Unit tests for the universal file reader in core/agent/file_readers.dart.
//
// Every fixture is built under a per-test Directory.systemTemp root and removed
// again in tearDown; nothing here touches the network or the host's real files.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:main_app/core/agent/file_readers.dart';

/// One magic-number case: a file name, its leading bytes, the kind the reader
/// must report and a fragment the human description must mention.
class _MagicCase {
  const _MagicCase(this.name, this.magic, this.kind, this.mention);

  final String name;
  final List<int> magic;
  final FileKind kind;
  final String mention;
}

/// A three-entry archive used by the zip / tar / tar.gz tests: one text file,
/// one directory and one binary entry.
Archive buildSampleArchive() {
  final Archive archive = Archive();
  archive.addFile(ArchiveFile.string('hello.txt', 'hello archive\n'));
  final ArchiveFile directory = ArchiveFile('sub/', 0, <int>[]);
  directory.isFile = false;
  archive.addFile(directory);
  archive.addFile(
    ArchiveFile('logo.png', 8, <int>[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
  );
  return archive;
}

void main() {
  late Directory root;

  setUp(() async {
    root = await Directory.systemTemp.createTemp('harbor_file_readers_');
  });

  tearDown(() async {
    if (await root.exists()) {
      await root.delete(recursive: true);
    }
  });

  File writeText(String relative, String content) {
    final File file = File('${root.path}/$relative');
    file.parent.createSync(recursive: true);
    file.writeAsStringSync(content);
    return file;
  }

  File writeBytes(String relative, List<int> bytes) {
    final File file = File('${root.path}/$relative');
    file.parent.createSync(recursive: true);
    file.writeAsBytesSync(bytes);
    return file;
  }

  group('extensionOf / classifyExtension', () {
    test('extensionOf strips the leading dot and lowercases', () {
      expect(UniversalFileReader.extensionOf('a/b.dart'), 'dart');
      expect(UniversalFileReader.extensionOf('x.JSON'), 'json');
      expect(UniversalFileReader.extensionOf('dir\\file.TXT'), 'txt');
      // No dot, a dotfile and a trailing dot all have no extension.
      expect(UniversalFileReader.extensionOf('noext'), '');
      expect(UniversalFileReader.extensionOf('.gitignore'), '');
      expect(UniversalFileReader.extensionOf('trailing.'), '');
    });

    test('classifyExtension maps representative paths to FileKind', () {
      expect(UniversalFileReader.classifyExtension('a/b.dart'), FileKind.code);
      expect(UniversalFileReader.classifyExtension('x.json'), FileKind.structuredData);
      expect(UniversalFileReader.classifyExtension('y.zip'), FileKind.archive);
      expect(UniversalFileReader.classifyExtension('z.png'), FileKind.image);
      expect(UniversalFileReader.classifyExtension('q.pdf'), FileKind.document);
      expect(UniversalFileReader.classifyExtension('n.txt'), FileKind.text);
      expect(UniversalFileReader.classifyExtension('noext'), FileKind.text);
      expect(UniversalFileReader.classifyExtension('weird.qqq'), FileKind.unknown);
      expect(UniversalFileReader.classifyExtension('.gitignore'), FileKind.text);
      expect(UniversalFileReader.classifyExtension('trailing.'), FileKind.text);
    });
  });

  group('text and code', () {
    test('reads a text file with exact content, line count and encoding', () async {
      final File file = writeText('notes/plain.txt', 'hello\nworld\n');
      final FileReadResult result = await UniversalFileReader().read(file.path);

      expect(result.kind, FileKind.text);
      expect(result.text, 'hello\nworld\n');
      expect(result.lineCount, 2);
      expect(result.encoding, 'utf-8');
      expect(result.isTruncated, isFalse);
      expect(result.language, isNull);
      expect(result.binaryHeader, isNull);
      expect(result.warnings, isEmpty);
      expect(result.sizeBytes, file.lengthSync());
    });

    test('tags a .dart file as code with language dart', () async {
      final File file = writeText('lib/main.dart', 'void main() {}\n');
      final FileReadResult result = await UniversalFileReader().read(file.path);

      expect(result.kind, FileKind.code);
      expect(result.language, 'dart');
      expect(result.text, 'void main() {}\n');
      expect(result.structuredData, isNull);
    });

    test('strips a leading UTF-8 BOM from decoded text', () async {
      final File file = writeBytes(
        'bom.txt',
        <int>[0xEF, 0xBB, 0xBF, ...utf8.encode('hello bom\n')],
      );
      final FileReadResult result = await UniversalFileReader().read(file.path);

      expect(result.text, 'hello bom\n');
      expect(result.text!.contains('\uFEFF'), isFalse);
      expect(result.encoding, 'utf-8');
      expect(result.magicDescription, 'UTF-8 byte order mark');
    });

    test('falls back to latin-1 for invalid UTF-8 bytes', () async {
      final File file = writeBytes('latin.txt', <int>[0xFF, 0xFE, 0x41]);
      final FileReadResult result = await UniversalFileReader().read(file.path);

      expect(result.encoding, 'latin-1');
      expect(result.text, 'ÿþA');
      expect(result.kind, FileKind.text);
      expect(result.text, isNotNull);
    });

    test('reads an empty file as empty text', () async {
      final File file = writeText('empty.txt', '');
      final FileReadResult result = await UniversalFileReader().read(file.path);

      expect(result.kind, FileKind.text);
      expect(result.text, '');
      expect(result.lineCount, 0);
      expect(result.encoding, 'utf-8');
      expect(result.isTruncated, isFalse);
    });
  });

  group('structured data', () {
    test('parses a JSON object into a deep-equal Dart structure', () async {
      final File file = writeText('data/object.json', '{"name":"harbor","count":2,"tags":["a","b"]}');
      final FileReadResult result = await UniversalFileReader().read(file.path);

      expect(result.kind, FileKind.structuredData);
      expect(result.magicDescription, 'json document');
      expect(
        result.structuredData,
        <String, Object?>{
          'name': 'harbor',
          'count': 2,
          'tags': <Object?>['a', 'b'],
        },
      );
      final Map<String, Object?> data = result.structuredData! as Map<String, Object?>;
      expect(data['count'], 2);
      expect(data['tags'], <Object?>['a', 'b']);
    });

    test('parses a JSON array', () async {
      final File file = writeText('data/array.json', '[1,2,3]');
      final FileReadResult result = await UniversalFileReader().read(file.path);

      expect(result.kind, FileKind.structuredData);
      expect(result.structuredData, <Object?>[1, 2, 3]);
    });

    test('invalid JSON degrades to raw text and records warnings', () async {
      final File file = writeText('data/broken.json', '{bad');
      final FileReadResult result = await UniversalFileReader().read(file.path);

      // The reader never throws for a readable file: it falls back to text.
      expect(result.kind, FileKind.text);
      expect(result.text, '{bad');
      expect(result.structuredData, isNull);
      expect(result.warnings, isNotEmpty);
      expect(result.warnings.first, contains('invalid JSON'));
      expect(result.warnings.last, contains('raw text'));
    });

    test('JSONL keeps valid rows and warns about the malformed one', () async {
      final File file = writeText(
        'data/rows.jsonl',
        '{"a":1}\n{"b":2}\n{"c":3}\nnot json\n',
      );
      final FileReadResult result = await UniversalFileReader().read(file.path);

      expect(result.kind, FileKind.structuredData);
      final List<Object?> rows = result.structuredData! as List<Object?>;
      expect(rows.length, 3);
      expect(rows.first, <String, Object?>{'a': 1});
      expect(rows.last, <String, Object?>{'c': 3});
      expect(result.warnings, hasLength(1));
      expect(result.warnings.single, contains('line 4'));
    });

    test('YAML is normalised into Map<String, Object?> and List<Object?>', () async {
      final File file = writeText(
        'data/config.yaml',
        'service:\n'
        '  port: 8080\n'
        '  enabled: true\n'
        '  hosts:\n'
        '    - a\n'
        '    - b\n'
        '1: one\n',
      );
      final FileReadResult result = await UniversalFileReader().read(file.path);

      expect(result.kind, FileKind.structuredData);
      final Map<String, Object?> data = result.structuredData! as Map<String, Object?>;
      // A numeric YAML key is coerced to a String key.
      expect(data.keys, containsAll(<String>['service', '1']));
      expect(data['1'], 'one');
      final Map<String, Object?> service = data['service']! as Map<String, Object?>;
      expect(service['port'], 8080);
      expect(service['enabled'], isTrue);
      expect(service['hosts'], isA<List<Object?>>());
      expect(service['hosts'], <Object?>['a', 'b']);
    });

    test('CSV honours quoting, escaped quotes and scalar coercion', () async {
      final File file = writeText(
        'data/table.csv',
        'name,count,amount,flag,note,empty\n'
        '"a,b",42,3.5,true,"say ""hi"", please",\n',
      );
      final FileReadResult result = await UniversalFileReader().read(file.path);

      expect(result.kind, FileKind.structuredData);
      final Map<String, Object?> data = result.structuredData! as Map<String, Object?>;
      expect(
        data['headers'],
        <String>['name', 'count', 'amount', 'flag', 'note', 'empty'],
      );
      expect(data['row_count'], 1);
      final List<Object?> rows = data['rows']! as List<Object?>;
      final Map<String, Object?> row = rows.single as Map<String, Object?>;
      expect(row['name'], 'a,b');
      expect(row['count'], isA<int>());
      expect(row['count'], 42);
      expect(row['amount'], isA<double>());
      expect(row['amount'], 3.5);
      expect(row['flag'], isA<bool>());
      expect(row['flag'], isTrue);
      expect(row['note'], 'say "hi", please');
      expect(row['empty'], '');
    });

    test('INI parses sections, both separators, comments and the _root bucket', () async {
      final File file = writeText(
        'config/app.ini',
        '; a comment\n'
        '# another comment\n'
        'root_key=root_value\n'
        '[server]\n'
        'a = 1\n'
        'b: true\n'
        'c = hello\n',
      );
      final FileReadResult result = await UniversalFileReader().read(file.path);

      expect(result.kind, FileKind.structuredData);
      expect(result.magicDescription, 'ini document');
      final Map<String, Object?> data = result.structuredData! as Map<String, Object?>;
      expect(data.keys, containsAll(<String>['_root', 'server']));
      final Map<String, Object?> rootBucket = data['_root']! as Map<String, Object?>;
      expect(rootBucket['root_key'], 'root_value');
      final Map<String, Object?> server = data['server']! as Map<String, Object?>;
      expect(server['a'], 1);
      expect(server['b'], isTrue);
      expect(server['c'], 'hello');
    });

    test('XML becomes nested tag/attributes/text/children maps', () async {
      final File file = writeText(
        'data/doc.xml',
        '<?xml version="1.0"?>'
        '<root id="1"><child>hi</child><child>bye</child></root>',
      );
      final FileReadResult result = await UniversalFileReader().read(file.path);

      expect(result.kind, FileKind.structuredData);
      expect(result.magicDescription, 'xml document');
      final Map<String, Object?> data = result.structuredData! as Map<String, Object?>;
      expect(data['tag'], 'root');
      expect(data['attributes'], <String, Object?>{'id': '1'});
      final List<Object?> children = data['children']! as List<Object?>;
      expect(children, hasLength(2));
      expect((children.first as Map<String, Object?>)['tag'], 'child');
      expect((children.first as Map<String, Object?>)['text'], 'hi');
      expect((children.last as Map<String, Object?>)['text'], 'bye');
    });

    test('a plain element exposes its trimmed text', () async {
      final File file = writeText('data/simple.xml', '<note>  hi there  </note>');
      final FileReadResult result = await UniversalFileReader().read(file.path);
      final Map<String, Object?> data = result.structuredData! as Map<String, Object?>;
      expect(data['tag'], 'note');
      expect(data['text'], 'hi there');
    });
  });

  group('magic-number sniffing', () {
    test('classifies binary containers by their leading bytes', () async {
      const List<_MagicCase> cases = <_MagicCase>[
        _MagicCase(
          'elf.bin',
          <int>[0x7F, 0x45, 0x4C, 0x46, 0x02, 0x01, 0x01, 0x00],
          FileKind.binary,
          'ELF',
        ),
        _MagicCase(
          'png.bin',
          <int>[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
          FileKind.image,
          'PNG',
        ),
        _MagicCase(
          'doc.bin',
          <int>[0x25, 0x50, 0x44, 0x46, 0x2D, 0x31, 0x2E, 0x37],
          FileKind.document,
          'PDF',
        ),
        _MagicCase(
          'db.bin',
          <int>[0x53, 0x51, 0x4C, 0x69, 0x74, 0x65, 0x20, 0x66, 0x6F, 0x72],
          FileKind.binary,
          'SQLite',
        ),
        _MagicCase(
          'mod.bin',
          <int>[0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00],
          FileKind.binary,
          'WebAssembly',
        ),
      ];

      final UniversalFileReader reader = UniversalFileReader();
      for (final _MagicCase testCase in cases) {
        final File file = writeBytes(testCase.name, testCase.magic);
        final FileReadResult result = await reader.read(file.path);
        expect(result.kind, testCase.kind, reason: testCase.name);
        expect(result.magicDescription, contains(testCase.mention), reason: testCase.name);
        expect(result.binaryHeader, isNotNull, reason: testCase.name);
      }
    });

    test('sniffs ZIP and GZIP containers as archives', () async {
      final UniversalFileReader reader = UniversalFileReader();

      // A real, valid zip is recognised from an opaque .bin name.
      final File zipFile = writeBytes('sniffed.bin', ZipEncoder().encode(buildSampleArchive())!);
      final FileReadResult zipResult = await reader.read(zipFile.path);
      expect(zipResult.kind, FileKind.archive);
      expect(zipResult.magicDescription!.toLowerCase(), contains('zip'));

      final File gzFile = writeBytes(
        'sniffed.gz',
        GZipEncoder().encode(utf8.encode('plain gzip payload\n'))!,
      );
      final FileReadResult gzResult = await reader.read(gzFile.path);
      expect(gzResult.kind, FileKind.archive);
      expect(gzResult.magicDescription!.toLowerCase(), contains('gzip'));
    });

    test('hexDump renders offset, hex pairs and an ASCII gutter', () {
      final String dump = UniversalFileReader.hexDump(
        Uint8List.fromList(<int>[0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x00, 0xFF]),
      );
      final List<String> lines = dump.trim().split('\n');
      expect(lines, hasLength(1));
      expect(lines.first.startsWith('00000000  '), isTrue);
      expect(lines.first, contains('48 65 6C 6C 6F 00 FF'));
      expect(lines.first, endsWith('|Hello..|'));
    });

    test('hexDump advances the offset on subsequent lines', () {
      final Uint8List data = Uint8List(18);
      data[0] = 0x41;
      data[1] = 0x42;
      data[16] = 0x5A;
      final List<String> lines = UniversalFileReader.hexDump(data).trim().split('\n');
      expect(lines, hasLength(2));
      expect(lines[0].startsWith('00000000'), isTrue);
      expect(lines[0], contains('|AB'));
      expect(lines[1].startsWith('00000010'), isTrue);
      expect(lines[1], endsWith('|Z.|'));
    });
  });

  group('archives', () {
    test('a real zip lists entries with sizes, directories and text previews', () async {
      final File file = writeBytes('bundle.zip', ZipEncoder().encode(buildSampleArchive())!);
      final FileReadResult result = await UniversalFileReader().read(file.path);

      expect(result.kind, FileKind.archive);
      expect(result.magicDescription!.toLowerCase(), contains('zip'));
      final List<ArchiveEntryInfo> entries = result.archiveEntries!;
      expect(entries, hasLength(3));
      expect(
        entries.map((ArchiveEntryInfo e) => e.name).toSet(),
        <String>{'hello.txt', 'sub/', 'logo.png'},
      );

      final ArchiveEntryInfo textEntry =
          entries.firstWhere((ArchiveEntryInfo e) => e.name == 'hello.txt');
      expect(textEntry.isDirectory, isFalse);
      expect(textEntry.size, 'hello archive\n'.length);
      expect(textEntry.preview, 'hello archive\n');

      final ArchiveEntryInfo dirEntry =
          entries.firstWhere((ArchiveEntryInfo e) => e.name == 'sub/');
      expect(dirEntry.isDirectory, isTrue);
      expect(dirEntry.preview, isNull);

      final ArchiveEntryInfo binaryEntry =
          entries.firstWhere((ArchiveEntryInfo e) => e.name == 'logo.png');
      expect(binaryEntry.preview, isNull);
    });

    test('a real tar archive is walked', () async {
      final File file = writeBytes('bundle.tar', TarEncoder().encode(buildSampleArchive()));
      final FileReadResult result = await UniversalFileReader().read(file.path);

      expect(result.kind, FileKind.archive);
      expect(result.magicDescription!.toLowerCase(), contains('tar'));
      final List<ArchiveEntryInfo> entries = result.archiveEntries!;
      expect(entries, hasLength(3));
      final ArchiveEntryInfo textEntry =
          entries.firstWhere((ArchiveEntryInfo e) => e.name == 'hello.txt');
      expect(textEntry.preview, 'hello archive\n');
      expect(
        entries.firstWhere((ArchiveEntryInfo e) => e.name == 'sub/').isDirectory,
        isTrue,
      );
    });

    test('a gzip-compressed tar archive is inflated and walked', () async {
      final List<int> tarBytes = TarEncoder().encode(buildSampleArchive());
      final File file = writeBytes('bundle.tar.gz', GZipEncoder().encode(tarBytes)!);
      final FileReadResult result = await UniversalFileReader().read(file.path);

      expect(result.kind, FileKind.archive);
      expect(result.magicDescription!.toLowerCase(), contains('gzip'));
      expect(result.archiveEntries, hasLength(3));
      expect(
        result.archiveEntries!
            .firstWhere((ArchiveEntryInfo e) => e.name == 'hello.txt')
            .preview,
        'hello archive\n',
      );
    });
  });

  group('failures and limits', () {
    test('a missing file throws FileReadException', () async {
      final UniversalFileReader reader = UniversalFileReader();
      await expectLater(
        reader.read('${root.path}/does_not_exist.txt'),
        throwsA(isA<FileReadException>()),
      );
    });

    test('a directory path passed to read throws FileReadException', () async {
      final Directory directory = Directory('${root.path}/a_directory')..createSync();
      final UniversalFileReader reader = UniversalFileReader();
      await expectLater(
        reader.read(directory.path),
        throwsA(isA<FileReadException>()),
      );
    });

    test('maxBytes windows a large file and warns', () async {
      final File file = writeText('big.txt', 'x' * 100);
      final UniversalFileReader reader =
          UniversalFileReader(limits: const FileReadLimits(maxBytes: 32));
      final FileReadResult result = await reader.read(file.path);

      expect(result.isTruncated, isTrue);
      expect(result.sizeBytes, 100);
      expect(result.text, 'x' * 32);
      expect(result.warnings.single, contains('only the first 32 bytes'));
    });

    test('maxLines truncates a long document and warns', () async {
      final File file = writeText('lines.txt', 'l1\nl2\nl3\nl4\n');
      final UniversalFileReader reader =
          UniversalFileReader(limits: const FileReadLimits(maxLines: 2));
      final FileReadResult result = await reader.read(file.path);

      expect(result.isTruncated, isFalse);
      expect(result.lineCount, 2);
      expect(result.text, 'l1\nl2');
      expect(result.warnings.single, contains('truncated to 2'));
    });

    test('maxCsvRows truncates a table and warns', () async {
      final File file = writeText('rows.csv', 'a,b\n1,2\n3,4\n5,6\n7,8\n');
      final UniversalFileReader reader =
          UniversalFileReader(limits: const FileReadLimits(maxCsvRows: 2));
      final FileReadResult result = await reader.read(file.path);

      final Map<String, Object?> data = result.structuredData! as Map<String, Object?>;
      expect(data['row_count'], 1);
      expect(result.warnings.single, contains('truncated at 2 rows'));
    });
  });

  group('readCodebase', () {
    test('walks nested directories with an extension filter and skips hidden paths', () async {
      final File a = writeText('a.dart', 'void a() {}\n');
      final File b = writeText('sub/b.dart', 'void b() {}\n');
      writeText('sub/notes.txt', 'not dart\n');
      writeText('.hidden/h.dart', 'void h() {}\n');
      writeText('.ghost.dart', 'void g() {}\n');

      final CodebaseReadResult result = await UniversalFileReader().readCodebase(
        root.path,
        extensions: <String>{'dart'},
      );

      expect(result.rootPath, root.path);
      expect(result.files, hasLength(2));
      expect(result.languageHistogram, <String, int>{'dart': 2});
      expect(result.totalBytes, a.lengthSync() + b.lengthSync());
      expect(
        result.files.map((FileReadResult f) => f.path).toSet(),
        <String>{a.path, b.path},
      );
    });

    test('includes hidden paths when skipHidden is false', () async {
      writeText('a.dart', 'void a() {}\n');
      writeText('.hidden/h.dart', 'void h() {}\n');
      writeText('.ghost.dart', 'void g() {}\n');

      final CodebaseReadResult result = await UniversalFileReader().readCodebase(
        root.path,
        extensions: <String>{'dart'},
        skipHidden: false,
      );

      expect(result.files, hasLength(3));
      expect(result.languageHistogram['dart'], 3);
    });

    test('a non-existent directory throws FileReadException', () async {
      await expectLater(
        UniversalFileReader().readCodebase('${root.path}/nope'),
        throwsA(isA<FileReadException>()),
      );
    });
  });

  group('result rendering', () {
    test('toJson exposes the documented keys', () async {
      final File file = writeText('render.txt', 'render me\n');
      final FileReadResult result = await UniversalFileReader().read(file.path);
      final Map<String, Object?> json = result.toJson();

      expect(
        json.keys,
        containsAll(<String>[
          'path',
          'kind',
          'size_bytes',
          'is_truncated',
          'language',
          'encoding',
          'line_count',
          'magic_description',
          'binary_header',
          'structured_data',
          'archive_entries',
          'text_preview',
          'warnings',
        ]),
      );
      expect(json['path'], file.path);
      expect(json['kind'], 'text');
      expect(json['text_preview'], 'render me\n');
    });

    test('toPromptSummary names the path and the kind', () async {
      final File file = writeText('summary.txt', 'summary\n');
      final FileReadResult result = await UniversalFileReader().read(file.path);
      final String summary = result.toPromptSummary();

      expect(summary, contains('File: ${file.path}'));
      expect(summary, contains('Kind: text'));
      expect(summary, contains('Lines: 1'));
    });
  });
}