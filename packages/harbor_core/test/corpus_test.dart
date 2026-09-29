// Tests for the corpus half of the pipeline: the document model, the device
// extractor and redactor, the HTML converter, the web collector's policy and
// robots handling, and the store with its quotas.
//
// Everything here is hermetic. The web collector is driven through
// `package:http/testing.dart`'s `MockClient`, so no test touches the network,
// and the store is backed by an in-memory `CorpusStorage`, so no test touches
// the file system.

import 'dart:convert';

import 'package:harbor_core/src/corpus/corpus_document.dart';
import 'package:harbor_core/src/corpus/corpus_store.dart';
import 'package:harbor_core/src/corpus/device_text.dart';
import 'package:harbor_core/src/corpus/html_text.dart';
import 'package:harbor_core/src/corpus/web_collector.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:test/test.dart';

/// A `CorpusStorage` that keeps the payload in memory.
///
/// The store is deliberately storage-agnostic; this fake exercises the same
/// contract the Android file backend and the Web `localStorage` backend
/// implement, without any platform dependency.
class _MemoryStorage implements CorpusStorage {
  String? payload;
  int writeCount = 0;
  int deleteCount = 0;

  @override
  Future<String?> read() async => payload;

  @override
  Future<void> write(String value) async {
    payload = value;
    writeCount++;
  }

  @override
  Future<void> delete() async {
    payload = null;
    deleteCount++;
  }
}

void main() {
  group('CorpusDocument', () {
    test('toJson/fromJson round-trips text, meta and UTC collectedAt', () {
      final CorpusDocument original = CorpusDocument(
        id: CorpusDocument.hashOf('round trip text'),
        text: 'round trip text',
        source: CorpusSource.web,
        uri: 'https://example.com/page',
        collectedAt: DateTime.utc(2024, 5, 17, 12, 30, 45),
        meta: <String, String>{'title': 'Example', 'contentType': 'text/html'},
      );

      final Map<String, Object?> json = original.toJson();
      final CorpusDocument restored = CorpusDocument.fromJson(json);

      expect(restored.id, original.id);
      expect(restored.text, original.text);
      expect(restored.source, original.source);
      expect(restored.uri, original.uri);
      expect(restored.collectedAt, original.collectedAt);
      expect(restored.collectedAt!.isUtc, isTrue);
      expect(restored.meta, original.meta);
      expect(json['collectedAt'], isA<String>());
      expect((json['collectedAt']! as String).endsWith('Z'), isTrue);
    });

    test('round-trips a document with no collectedAt and no meta', () {
      final CorpusDocument original = CorpusDocument(
        id: CorpusDocument.hashOf('bundled'),
        text: 'bundled',
        source: CorpusSource.bundled,
        uri: 'bundled://welcome.txt',
      );
      final CorpusDocument restored = CorpusDocument.fromJson(original.toJson());
      expect(restored.collectedAt, isNull);
      expect(restored.meta, isEmpty);
    });

    test('bytes and charCount report UTF-8 bytes and code units', () {
      final CorpusDocument document = CorpusDocument(
        id: CorpusDocument.hashOf('Café'),
        text: 'Café',
        source: CorpusSource.device,
        uri: 'file:///notes.txt',
      );
      expect(document.charCount, 4);
      expect(document.bytes, 5);
    });

    test('hashOf is stable and differs for different text', () {
      expect(CorpusDocument.hashOf('hello'), CorpusDocument.hashOf('hello'));
      expect(CorpusDocument.hashOf('hello').length, 64);
      expect(
        CorpusDocument.hashOf('hello'),
        isNot(CorpusDocument.hashOf('hello!')),
      );
    });

    test('fromJson throws FormatException on bad input', () {
      expect(
        () => CorpusDocument.fromJson(<String, Object?>{'id': 'x'}),
        throwsFormatException,
      );
      expect(
        () => CorpusDocument.fromJson(<String, Object?>{
          'id': 'x',
          'text': 't',
          'source': 'not-a-source',
          'uri': 'u',
        }),
        throwsFormatException,
      );
      expect(
        () => CorpusDocument.fromJson(<String, Object?>{
          'id': 'x',
          'text': 't',
          'source': 'device',
          'uri': 'u',
          'collectedAt': 'not a date',
        }),
        throwsFormatException,
      );
      expect(
        () => CorpusDocument.fromJson(<String, Object?>{
          'id': 'x',
          'text': 't',
          'source': 'device',
          'uri': 'u',
          'meta': 'nope',
        }),
        throwsFormatException,
      );
    });
  });

  group('DeviceTextExtractor', () {
    test('looksLikeText accepts known extensions and rejects others', () {
      expect(DeviceTextExtractor.looksLikeText('notes.txt'), isTrue);
      expect(DeviceTextExtractor.looksLikeText('README.MD'), isTrue);
      expect(DeviceTextExtractor.looksLikeText('/tmp/dir/script.dart'), isTrue);
      expect(DeviceTextExtractor.looksLikeText(r'C:\Users\me\a.log'), isTrue);
      expect(DeviceTextExtractor.looksLikeText('photo.jpg'), isFalse);
      expect(DeviceTextExtractor.looksLikeText('archive.zip'), isFalse);
      expect(DeviceTextExtractor.looksLikeText('README'), isFalse);
      expect(DeviceTextExtractor.looksLikeText('trailing.'), isFalse);
    });

    test('extract decodes UTF-8 text', () {
      final String source = 'Hello, corpus! Café — naïve.';
      final String? text = DeviceTextExtractor.extract(
        'notes.txt',
        utf8.encode(source),
      );
      expect(text, source);
    });

    test('extract rejects a NUL-laden binary blob', () {
      final List<int> blob = <int>[
        0x00, 0x00, 0x01, 0x00, 0x00, 0x42, 0x00, 0x00,
        0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x7F, 0x00,
      ];
      expect(DeviceTextExtractor.isProbablyBinary(blob), isTrue);
      expect(DeviceTextExtractor.extract('blob.txt', blob), isNull);
    });

    test('extract does not mistake UTF-8 prose for binary', () {
      expect(
        DeviceTextExtractor.isProbablyBinary(utf8.encode('plenty of prose here')),
        isFalse,
      );
    });

    test('extract truncates at maxChars', () {
      final String? text = DeviceTextExtractor.extract(
        'notes.txt',
        utf8.encode('abcdefghij'),
        maxChars: 5,
      );
      expect(text, 'abcde');
      expect(text!.length, 5);
    });

    test('extract honours a UTF-8 BOM', () {
      final List<int> bytes = <int>[
        0xEF, 0xBB, 0xBF,
        ...utf8.encode('hello bom'),
      ];
      expect(DeviceTextExtractor.extract('notes.txt', bytes), 'hello bom');
    });

    test('extract honours a UTF-16 little-endian BOM', () {
      final List<int> body = <int>[];
      for (final int unit in 'hi there'.codeUnits) {
        body.add(unit & 0xFF);
        body.add((unit >> 8) & 0xFF);
      }
      final List<int> bytes = <int>[0xFF, 0xFE, ...body];
      expect(DeviceTextExtractor.extract('wide.txt', bytes), 'hi there');
    });

    test('extract replaces invalid UTF-8 instead of throwing', () {
      final List<int> bytes = <int>[
        ...utf8.encode('Hello world, this is a long enough valid prefix. '),
        0xFF,
        0xFE,
      ];
      final String? text = DeviceTextExtractor.extract('notes.txt', bytes);
      expect(text, isNotNull);
      expect(text, contains('\uFFFD'));
      expect(text, contains('valid prefix'));
    });

    test('extract skips unsupported extensions', () {
      expect(DeviceTextExtractor.extract('photo.jpg', utf8.encode('x')), isNull);
    });

    test('extract returns empty text for empty input', () {
      expect(DeviceTextExtractor.extract('empty.txt', <int>[]), '');
    });
  });

  group('TextRedactor', () {
    test('removes an Authorization bearer token', () {
      const String source = 'Authorization: Bearer abc123secretvalue';
      final String redacted = TextRedactor.redact(source);
      expect(redacted, isNot(contains('abc123secretvalue')));
      expect(redacted, contains('[REDACTED]'));
    });

    test('removes an e-mail address', () {
      const String source = 'Please contact alice@example.com about this.';
      final String redacted = TextRedactor.redact(source);
      expect(redacted, isNot(contains('alice@example.com')));
      expect(redacted, contains('[REDACTED_EMAIL]'));
    });

    test('removes a PEM private key block', () {
      const String source = 'before\n'
          '-----BEGIN PRIVATE KEY-----\n'
          'MIIBVwIBADANBgkqhkiG9w0BAQEFAASCAUEwggE9AgEAAoGBAL\n'
          '-----END PRIVATE KEY-----\n'
          'after';
      final String redacted = TextRedactor.redact(source);
      expect(redacted, isNot(contains('MIIBVwIBADANBgkqhkiG9w0BAQEFAASCAUEwggE9')));
      expect(redacted, contains('[REDACTED_PRIVATE_KEY]'));
      expect(redacted, contains('before'));
      expect(redacted, contains('after'));
    });

    test('removes credentials embedded in a URL', () {
      const String source = 'See https://alice:s3cr3t@example.com/private now';
      final String redacted = TextRedactor.redact(source);
      expect(redacted, isNot(contains('s3cr3t')));
      expect(redacted, isNot(contains('alice:')));
      expect(redacted, contains('example.com'));
      expect(redacted, contains('[REDACTED_CREDENTIALS]'));
    });

    test('removes a long hexadecimal secret', () {
      const String source = 'api key 0123456789abcdef0123456789abcdef end';
      final String redacted = TextRedactor.redact(source);
      expect(redacted, isNot(contains('0123456789abcdef0123456789abcdef')));
      expect(redacted, contains('[REDACTED_SECRET]'));
    });

    test('removes a phone-like digit run', () {
      const String source = 'Call +1 (555) 123-4567 tomorrow.';
      final String redacted = TextRedactor.redact(source);
      expect(redacted, isNot(contains('555')));
      expect(redacted, contains('[REDACTED_PHONE]'));
    });

    test('containsLikelySecret is true for secrets and false for prose', () {
      expect(
        TextRedactor.containsLikelySecret('Authorization: Bearer abc123secretvalue'),
        isTrue,
      );
      expect(
        TextRedactor.containsLikelySecret('mail alice@example.com'),
        isTrue,
      );
      expect(
        TextRedactor.containsLikelySecret('-----BEGIN PRIVATE KEY-----'),
        isTrue,
      );
      expect(
        TextRedactor.containsLikelySecret('https://alice:s3cr3t@example.com/x'),
        isTrue,
      );
      expect(
        TextRedactor.containsLikelySecret('0123456789abcdef0123456789abcdef'),
        isTrue,
      );
      expect(
        TextRedactor.containsLikelySecret(
          'The quick brown fox jumps over the lazy dog near the river bank.',
        ),
        isFalse,
      );
    });

    test('redaction is idempotent on already-clean prose', () {
      const String prose = 'A perfectly ordinary sentence about harbors and boats.';
      expect(TextRedactor.redact(prose), prose);
    });
  });

  group('html', () {
    test('htmlToText drops script/style/head and decodes entities', () {
      const String markup = '<html><head><title>Ignored Head</title>'
          '<style>body { color: red; }</style></head>'
          '<body><script>var secret = 1;</script>'
          '<h1>Harbor &amp; corpus</h1>'
          '<p>First&nbsp;paragraph</p>'
          '<p>Caf&#233; is open</p>'
          '</body></html>';
      final String text = htmlToText(markup);
      expect(text, contains('Harbor & corpus'));
      expect(text, contains('First paragraph'));
      expect(text, contains('Café'));
      expect(text, isNot(contains('secret')));
      expect(text, isNot(contains('color: red')));
      expect(text, isNot(contains('Ignored Head')));
    });

    test('htmlToText turns block tags into newlines', () {
      final String text = htmlToText('<p>one</p><p>two</p><br>three');
      final List<String> lines =
          text.split('\n').where((String line) => line.isNotEmpty).toList();
      expect(lines, <String>['one', 'two', 'three']);
      expect(htmlToText('alpha<br>beta'), 'alpha\nbeta');
    });

    test('decodeHtmlEntities handles named, decimal and hex references', () {
      expect(
        decodeHtmlEntities('&unknown; &amp; &#65; &#x41; &nbsp;end'),
        '&unknown; & A A  end',
      );
    });

    test('htmlTitle finds and cleans the title', () {
      expect(
        htmlTitle('<html><head><title>  Hello &amp; World  </title></head></html>'),
        'Hello & World',
      );
      expect(htmlTitle('<html><body>no title</body></html>'), isNull);
      expect(htmlTitle('<title></title>'), isNull);
    });

    test('htmlLinks resolves relative hrefs and de-duplicates', () {
      final Uri base = Uri.parse('https://example.com/dir/page.html');
      const String markup = '<a href="a.html">A</a>'
          '<a href="/b">B</a>'
          '<a href="a.html">A again</a>'
          "<a href='c.html'>C</a>"
          '<a href="https://other.example/x">D</a>'
          '<a href="#fragment">E</a>'
          '<a href="mailto:x@example.com">F</a>';
      final List<Uri> links = htmlLinks(markup, base);
      expect(
        links.map((Uri uri) => uri.toString()).toList(),
        <String>[
          'https://example.com/dir/a.html',
          'https://example.com/b',
          'https://example.com/dir/c.html',
          'https://other.example/x',
        ],
      );
    });

    test('htmlLinks honours the limit', () {
      final Uri base = Uri.parse('https://example.com/');
      const String markup = '<a href="1">1</a><a href="2">2</a><a href="3">3</a>';
      expect(htmlLinks(markup, base, limit: 2).length, 2);
      expect(htmlLinks(markup, base, limit: 0), isEmpty);
    });
  });

  group('WebCorpusPolicy', () {
    test('allows only permitted schemes and hosts', () {
      const WebCorpusPolicy policy = WebCorpusPolicy();
      expect(policy.allows(Uri.parse('https://example.com/x')), isTrue);
      expect(policy.allows(Uri.parse('http://example.com/x')), isFalse);

      const WebCorpusPolicy restricted = WebCorpusPolicy(
        allowedHosts: <String>{'example.com'},
      );
      expect(restricted.allows(Uri.parse('https://example.com/x')), isTrue);
      expect(restricted.allows(Uri.parse('https://evil.example/x')), isFalse);
    });

    test('copyWith replaces only the supplied fields', () {
      const WebCorpusPolicy base = WebCorpusPolicy(
        allowedHosts: <String>{'example.com'},
      );
      final WebCorpusPolicy copy = base.copyWith(
        maxBytes: 1024,
        allowedSchemes: <String>{'http', 'https'},
      );
      expect(copy.maxBytes, 1024);
      expect(copy.allowedSchemes, contains('http'));
      expect(copy.allowedHosts, <String>{'example.com'});
      expect(copy.obeyRobotsTxt, isTrue);
      expect(copy.userAgent, 'HarborCorpusBot/1.0');
    });
  });

  group('RobotsRules', () {
    test('parses a star group into disallow prefixes', () {
      final RobotsRules rules = RobotsRules.parse(
        '# comment\nUser-agent: *\nDisallow: /private\nDisallow: /tmp\n',
      );
      expect(rules.disallow, <String>['/private', '/tmp']);
      expect(rules.allowsPath('/private/secret'), isFalse);
      expect(rules.allowsPath('/tmp'), isFalse);
      expect(rules.allowsPath('/public/page'), isTrue);
    });

    test('ignores groups for other user agents', () {
      final RobotsRules rules = RobotsRules.parse(
        'User-agent: Googlebot\nDisallow: /\n\n'
        'User-agent: *\nDisallow: /tmp\n',
      );
      expect(rules.disallow, <String>['/tmp']);
      expect(rules.allowsPath('/'), isTrue);
      expect(rules.allowsPath('/tmp/x'), isFalse);
    });

    test('Disallow: / blocks everything', () {
      final RobotsRules rules = RobotsRules.parse('User-agent: *\nDisallow: /\n');
      expect(rules.allowsPath('/'), isFalse);
      expect(rules.allowsPath('/anything/at/all'), isFalse);
    });

    test('a file with no star group allows everything', () {
      final RobotsRules rules = RobotsRules.parse(
        'User-agent: Googlebot\nDisallow: /\n',
      );
      expect(rules.allowsPath('/anything'), isTrue);
    });

    test('strips comments and skips malformed lines', () {
      final RobotsRules rules = RobotsRules.parse(
        'garbage line\n\nUser-agent: *\nDisallow: /secret # trailing comment\n',
      );
      expect(rules.disallow, <String>['/secret']);
      expect(rules.allowsPath('/secret/x'), isFalse);
    });
  });

  group('WebCorpusCollector', () {
    test('collect returns a document for a textual page', () async {
      const String page = '<html><head><title>Page</title></head><body>'
          '<script>window.secret = 1;</script>'
          '<p>This is a sufficiently long paragraph of ordinary prose.</p>'
          '</body></html>';
      final http.Client client = MockClient((http.Request request) async {
        return http.Response(
          page,
          200,
          headers: <String, String>{'content-type': 'text/html; charset=utf-8'},
        );
      });
      final WebCorpusCollector collector = WebCorpusCollector(
        client: client,
        policy: const WebCorpusPolicy(obeyRobotsTxt: false),
      );

      final CorpusDocument? collected =
          await collector.collect(Uri.parse('https://example.com/page'));
      expect(collected, isNotNull);
      final CorpusDocument document = collected!;
      expect(document.text, contains('sufficiently long paragraph'));
      expect(document.text, isNot(contains('window.secret')));
      expect(document.source, CorpusSource.web);
      expect(document.uri, 'https://example.com/page');
      expect(document.meta['title'], 'Page');
      expect(document.id, CorpusDocument.hashOf(document.text));
      expect(document.collectedAt, isNotNull);
    });

    test('collect returns null for a 404', () async {
      final http.Client client = MockClient((http.Request request) async {
        return http.Response('missing', 404);
      });
      final WebCorpusCollector collector = WebCorpusCollector(
        client: client,
        policy: const WebCorpusPolicy(obeyRobotsTxt: false),
      );
      expect(
        await collector.collect(Uri.parse('https://example.com/missing')),
        isNull,
      );
    });

    test('collect refuses a disallowed host without any HTTP request', () async {
      int calls = 0;
      final http.Client client = MockClient((http.Request request) async {
        calls++;
        return http.Response('never', 200);
      });
      final WebCorpusCollector collector = WebCorpusCollector(
        client: client,
        policy: const WebCorpusPolicy(
          allowedHosts: <String>{'allowed.example'},
          obeyRobotsTxt: false,
        ),
      );
      expect(
        await collector.collect(Uri.parse('https://blocked.example/x')),
        isNull,
      );
      expect(calls, 0);
    });

    test('collect refuses a body over the byte cap', () async {
      final String body = '<p>${'x' * 500}</p>';
      final http.Client client = MockClient((http.Request request) async {
        return http.Response(
          body,
          200,
          headers: <String, String>{'content-type': 'text/html'},
        );
      });
      final WebCorpusCollector collector = WebCorpusCollector(
        client: client,
        policy: const WebCorpusPolicy(obeyRobotsTxt: false, maxBytes: 64),
      );
      expect(
        await collector.collect(Uri.parse('https://example.com/big')),
        isNull,
      );
    });

    test('collect refuses a non-textual content type', () async {
      final http.Client client = MockClient((http.Request request) async {
        return http.Response(
          'binaryish',
          200,
          headers: <String, String>{'content-type': 'image/png'},
        );
      });
      final WebCorpusCollector collector = WebCorpusCollector(
        client: client,
        policy: const WebCorpusPolicy(obeyRobotsTxt: false),
      );
      expect(
        await collector.collect(Uri.parse('https://example.com/image')),
        isNull,
      );
    });

    test('collect refuses a path robots.txt disallows', () async {
      bool pageRequested = false;
      final http.Client client = MockClient((http.Request request) async {
        if (request.url.path == '/robots.txt') {
          return http.Response(
            'User-agent: *\nDisallow: /\n',
            200,
            headers: <String, String>{'content-type': 'text/plain'},
          );
        }
        pageRequested = true;
        return http.Response(
          '<p>${'long enough page prose ' * 4}</p>',
          200,
          headers: <String, String>{'content-type': 'text/html'},
        );
      });
      final WebCorpusCollector collector = WebCorpusCollector(client: client);

      expect(
        await collector.collect(Uri.parse('https://example.com/page')),
        isNull,
      );
      expect(pageRequested, isFalse);
    });

    test('caches robots.txt per host', () async {
      int robotsRequests = 0;
      final http.Client client = MockClient((http.Request request) async {
        if (request.url.path == '/robots.txt') {
          robotsRequests++;
          return http.Response(
            'User-agent: *\nDisallow: /admin\n',
            200,
            headers: <String, String>{'content-type': 'text/plain'},
          );
        }
        return http.Response(
          '<p>${'long enough page prose ' * 4}</p>',
          200,
          headers: <String, String>{'content-type': 'text/html'},
        );
      });
      final WebCorpusCollector collector = WebCorpusCollector(client: client);

      expect(
        await collector.collect(Uri.parse('https://example.com/one')),
        isNotNull,
      );
      expect(
        await collector.collect(Uri.parse('https://example.com/two')),
        isNotNull,
      );
      expect(robotsRequests, 1);
    });

    test('re-checks the host allow-list on every redirect hop', () async {
      final List<String> hostedRequests = <String>[];
      final http.Client client = MockClient((http.Request request) async {
        hostedRequests.add(request.url.host);
        if (request.url.host == 'good.example') {
          return http.Response(
            '',
            302,
            headers: <String, String>{'location': 'https://evil.example/x'},
            isRedirect: true,
          );
        }
        return http.Response('should never be fetched', 200);
      });
      final WebCorpusCollector collector = WebCorpusCollector(
        client: client,
        policy: const WebCorpusPolicy(
          allowedHosts: <String>{'good.example'},
          obeyRobotsTxt: false,
        ),
      );

      expect(
        await collector.collect(Uri.parse('https://good.example/start')),
        isNull,
      );
      expect(hostedRequests, <String>['good.example']);
    });

    test('follows an allowed redirect', () async {
      final http.Client client = MockClient((http.Request request) async {
        if (request.url.path == '/start') {
          return http.Response(
            '',
            301,
            headers: <String, String>{'location': '/final'},
            isRedirect: true,
          );
        }
        return http.Response(
          '<p>${'final destination prose ' * 4}</p>',
          200,
          headers: <String, String>{'content-type': 'text/html'},
        );
      });
      final WebCorpusCollector collector = WebCorpusCollector(
        client: client,
        policy: const WebCorpusPolicy(obeyRobotsTxt: false),
      );

      final CorpusDocument? collected =
          await collector.collect(Uri.parse('https://example.com/start'));
      expect(collected, isNotNull);
      expect(collected!.uri, 'https://example.com/final');
    });

    test('throws WebCorpusException on a transport failure', () async {
      final http.Client client = MockClient((http.Request request) async {
        throw http.ClientException('connection reset', request.url);
      });
      final WebCorpusCollector collector = WebCorpusCollector(
        client: client,
        policy: const WebCorpusPolicy(obeyRobotsTxt: false),
      );
      await expectLater(
        collector.collect(Uri.parse('https://example.com/down')),
        throwsA(isA<WebCorpusException>()),
      );
    });

    test('robotsFor returns null when robots.txt is missing', () async {
      final http.Client client = MockClient((http.Request request) async {
        return http.Response('nope', 404);
      });
      final WebCorpusCollector collector = WebCorpusCollector(client: client);
      expect(
        await collector.robotsFor(Uri.parse('https://example.com/')),
        isNull,
      );
    });
  });

  group('CorpusStore', () {
    test('rejects duplicates and counts them', () async {
      final CorpusStore store = CorpusStore(storage: _MemoryStorage());
      expect(
        await store.addText(
          'a duplicated document body',
          source: CorpusSource.device,
          uri: 'file:///a.txt',
        ),
        isTrue,
      );
      expect(
        await store.addText(
          'a duplicated document body',
          source: CorpusSource.web,
          uri: 'https://example.com/a',
        ),
        isFalse,
      );
      expect(store.documents.length, 1);
      expect(store.stats.rejectedDuplicates, 1);
    });

    test('collapses documents that differ only by a redacted secret', () async {
      final CorpusStore store = CorpusStore(storage: _MemoryStorage());
      expect(
        await store.addText(
          'write to alice@example.com today',
          source: CorpusSource.device,
          uri: 'file:///a.txt',
        ),
        isTrue,
      );
      expect(
        await store.addText(
          'write to bob@example.com today',
          source: CorpusSource.device,
          uri: 'file:///b.txt',
        ),
        isFalse,
      );
      expect(store.documents.length, 1);
      expect(store.stats.rejectedDuplicates, 1);
    });

    test('redacts before storing', () async {
      final CorpusStore store = CorpusStore(storage: _MemoryStorage());
      await store.addText(
        'Authorization: Bearer abcdef1234567890',
        source: CorpusSource.device,
        uri: 'file:///secret.txt',
      );
      expect(store.documents.single.text, isNot(contains('abcdef1234567890')));
      expect(store.documents.single.text, contains('[REDACTED]'));
    });

    test('rejects an empty-after-redaction document', () async {
      final CorpusStore store = CorpusStore(storage: _MemoryStorage());
      expect(
        await store.addText(
          '   \n\t ',
          source: CorpusSource.device,
          uri: 'file:///blank.txt',
        ),
        isFalse,
      );
      expect(store.documents, isEmpty);
    });

    test('rejects a document larger than maxDocumentBytes', () async {
      final CorpusStore store = CorpusStore(
        storage: _MemoryStorage(),
        maxDocumentBytes: 10,
      );
      expect(
        await store.addText(
          'this body is definitely longer than ten bytes',
          source: CorpusSource.device,
          uri: 'file:///big.txt',
        ),
        isFalse,
      );
      expect(store.stats.rejectedTooLarge, 1);
    });

    test('rejects a document that would exceed the count quota', () async {
      final CorpusStore store = CorpusStore(
        storage: _MemoryStorage(),
        maxDocuments: 1,
      );
      expect(
        await store.addText(
          'the first document',
          source: CorpusSource.device,
          uri: 'file:///1.txt',
        ),
        isTrue,
      );
      expect(
        await store.addText(
          'the second document',
          source: CorpusSource.device,
          uri: 'file:///2.txt',
        ),
        isFalse,
      );
      expect(store.stats.rejectedQuota, 1);
    });

    test('rejects a document that would exceed the byte quota', () async {
      final CorpusStore store = CorpusStore(
        storage: _MemoryStorage(),
        maxTotalBytes: 20,
      );
      expect(
        await store.addText(
          '123456789012345',
          source: CorpusSource.device,
          uri: 'file:///1.txt',
        ),
        isTrue,
      );
      expect(
        await store.addText(
          'abcdefghijklmno',
          source: CorpusSource.device,
          uri: 'file:///2.txt',
        ),
        isFalse,
      );
      expect(store.stats.rejectedQuota, 1);
    });

    test('stats reports totals and every source', () async {
      final CorpusStore store = CorpusStore(storage: _MemoryStorage());
      await store.addText(
        'device text',
        source: CorpusSource.device,
        uri: 'file:///d.txt',
      );
      await store.addText(
        'web text',
        source: CorpusSource.web,
        uri: 'https://example.com/',
      );
      final CorpusStats stats = store.stats;
      expect(stats.documentCount, 2);
      expect(stats.totalChars, 'device text'.length + 'web text'.length);
      expect(stats.totalBytes, utf8.encode('device text').length + utf8.encode('web text').length);
      expect(stats.bySource.keys.toSet(), CorpusSource.values.toSet());
      expect(stats.bySource[CorpusSource.device], 1);
      expect(stats.bySource[CorpusSource.web], 1);
      expect(stats.bySource[CorpusSource.bundled], 0);
    });

    test('documents is unmodifiable and in insertion order', () async {
      final CorpusStore store = CorpusStore(storage: _MemoryStorage());
      await store.addText(
        'first in',
        source: CorpusSource.device,
        uri: 'file:///1.txt',
      );
      await store.addText(
        'second in',
        source: CorpusSource.device,
        uri: 'file:///2.txt',
      );
      expect(
        store.documents.map((CorpusDocument document) => document.text).toList(),
        <String>['first in', 'second in'],
      );
      expect(() => store.documents.clear(), throwsUnsupportedError);
    });

    test('trainingText is deterministic and contains every document', () async {
      final CorpusStore store = CorpusStore(storage: _MemoryStorage());
      await store.addText(
        'alpha document text',
        source: CorpusSource.device,
        uri: 'file:///a.txt',
      );
      await store.addText(
        'beta document text',
        source: CorpusSource.web,
        uri: 'https://b.example/',
      );
      await store.addText(
        'gamma document text',
        source: CorpusSource.bundled,
        uri: 'bundled://g.txt',
      );

      final String first = store.trainingText(shuffleSeed: 12345);
      final String second = store.trainingText(shuffleSeed: 12345);
      expect(first, second);
      expect(first, contains('alpha document text'));
      expect(first, contains('beta document text'));
      expect(first, contains('gamma document text'));

      final String capped = store.trainingText(maxChars: 12, shuffleSeed: 12345);
      expect(capped.length, 12);
      expect(store.trainingText(maxChars: 0), isEmpty);
    });

    test('trainingText of an empty store is empty', () {
      final CorpusStore store = CorpusStore(storage: _MemoryStorage());
      expect(store.trainingText(), isEmpty);
    });

    test('save then fromJson round-trips count and text', () async {
      final _MemoryStorage storage = _MemoryStorage();
      final CorpusStore store = CorpusStore(storage: storage);
      await store.addText(
        'round trip body one',
        source: CorpusSource.device,
        uri: 'file:///1.txt',
      );
      await store.addText(
        'round trip body two',
        source: CorpusSource.web,
        uri: 'https://example.com/two',
      );
      await store.save();
      expect(storage.writeCount, 1);
      expect(storage.payload, contains('"version":1'));

      final CorpusStore rebuilt = CorpusStore.fromJson(
        storage.payload!,
        storage: _MemoryStorage(),
      );
      expect(rebuilt.documents.length, 2);
      expect(
        rebuilt.documents.map((CorpusDocument document) => document.text).toSet(),
        <String>{'round trip body one', 'round trip body two'},
      );
      expect(rebuilt.loadFailures, 0);
    });

    test('load reads a previously saved payload', () async {
      final _MemoryStorage storage = _MemoryStorage();
      final CorpusStore source = CorpusStore(storage: storage);
      await source.addText(
        'persisted corpus text',
        source: CorpusSource.user,
        uri: 'user://pasted',
      );
      await source.save();

      final CorpusStore loaded = CorpusStore(storage: storage);
      await loaded.load();
      expect(loaded.documents.single.text, 'persisted corpus text');
      expect(loaded.documents.single.source, CorpusSource.user);
    });

    test('a corrupt payload yields an empty store with loadFailures', () {
      final CorpusStore store = CorpusStore.fromJson(
        'this is not json at all',
        storage: _MemoryStorage(),
      );
      expect(store.documents, isEmpty);
      expect(store.loadFailures, greaterThan(0));
    });

    test('a partially corrupt payload keeps good entries and counts bad ones', () async {
      final _MemoryStorage storage = _MemoryStorage();
      final CorpusStore seed = CorpusStore(storage: storage);
      await seed.addText(
        'the good entry',
        source: CorpusSource.device,
        uri: 'file:///good.txt',
      );
      await seed.save();
      final String damaged =
          storage.payload!.replaceFirst('"documents":[', '"documents":[7,{"id":"broken"},');

      final CorpusStore store = CorpusStore.fromJson(
        damaged,
        storage: _MemoryStorage(),
      );
      expect(store.documents.length, 1);
      expect(store.documents.single.text, 'the good entry');
      expect(store.loadFailures, 2);
    });

    test('load tolerates a payload that is not an object', () async {
      final _MemoryStorage storage = _MemoryStorage();
      storage.payload = '[]';
      final CorpusStore store = CorpusStore(storage: storage);
      await store.load();
      expect(store.documents, isEmpty);
      expect(store.loadFailures, 1);
    });

    test('remove deletes a document and frees its quota', () async {
      final CorpusStore store = CorpusStore(storage: _MemoryStorage());
      await store.addText(
        'to be removed',
        source: CorpusSource.device,
        uri: 'file:///gone.txt',
      );
      final String id = store.documents.single.id;
      await store.remove(id);
      expect(store.documents, isEmpty);
      expect(store.stats.totalBytes, 0);
      expect(store.stats.totalChars, 0);
      await store.remove(id);
      expect(store.documents, isEmpty);
    });

    test('clear empties the store and deletes the payload', () async {
      final _MemoryStorage storage = _MemoryStorage();
      final CorpusStore store = CorpusStore(storage: storage);
      await store.addText(
        'something to forget',
        source: CorpusSource.device,
        uri: 'file:///forget.txt',
      );
      await store.save();
      expect(storage.payload, isNotNull);

      await store.clear();
      expect(store.documents, isEmpty);
      expect(store.stats.documentCount, 0);
      expect(store.stats.totalBytes, 0);
      expect(store.stats.rejectedDuplicates, 0);
      expect(store.loadFailures, 0);
      expect(storage.payload, isNull);
      expect(storage.deleteCount, 1);
    });
  });
}