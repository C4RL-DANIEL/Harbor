// Minimal, dependency-free HTML-to-text conversion for the web corpus.
//
// Harbor does not want a full DOM parser in its core package: the package must
// compile to JavaScript and stay small, and the collector only needs readable
// prose plus a bounded set of links. The converter below is therefore
// deliberately heuristic. It removes the elements whose contents are never
// prose (`script`, `style`, `head`), treats block-level tags as paragraph
// breaks, decodes the entities that actually appear on ordinary pages, and
// collapses whitespace so the tokenizer sees clean text.
//
// The same functions are used to extract a page title and to enumerate links
// for the bounded crawl in `web_collector.dart`.

/// Matches an HTML comment, which is dropped entirely.
final RegExp _commentPattern = RegExp(r'<!--[\s\S]*?-->');

/// Matches an element whose entire content is non-prose. The back-reference to
/// the opening tag name keeps the pairing honest (`<script>` ... `</script>`).
final RegExp _droppedElementPattern = RegExp(
  r'<(script|style|head|noscript|template|svg|iframe)\b[^>]*>[\s\S]*?</\1\s*>',
  caseSensitive: false,
);

/// Matches an opening or closing block-level tag; each becomes a line break.
final RegExp _blockTagPattern = RegExp(
  r'</?(?:p|div|br|li|ul|ol|dl|dt|dd|tr|td|th|table|thead|tbody|tfoot|'
  r'h[1-6]|section|article|header|footer|aside|nav|main|blockquote|pre|'
  r'figure|figcaption|hr|form|fieldset|address|details|summary)\b[^>]*>',
  caseSensitive: false,
);

/// Matches any remaining tag, which is removed without leaving a break.
final RegExp _anyTagPattern = RegExp(r'<[^>]*>');

/// Matches a named, decimal or hexadecimal character reference.
final RegExp _entityPattern = RegExp(r'&(#[xX]?[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);');

/// The named entities common enough to be worth decoding by hand. Unknown names
/// are left untouched, which is the safe choice: a literal `&foo;` in prose is
/// preserved rather than silently deleted.
const Map<String, String> _namedEntities = <String, String>{
  'amp': '&',
  'lt': '<',
  'gt': '>',
  'quot': '"',
  'apos': "'",
  'nbsp': ' ',
  'copy': '\u00A9',
  'reg': '\u00AE',
  'trade': '\u2122',
  'deg': '\u00B0',
  'plusmn': '\u00B1',
  'times': '\u00D7',
  'divide': '\u00F7',
  'frac12': '\u00BD',
  'frac14': '\u00BC',
  'frac34': '\u00BE',
  'sup2': '\u00B2',
  'sup3': '\u00B3',
  'micro': '\u00B5',
  'middot': '\u00B7',
  'bull': '\u2022',
  'hellip': '\u2026',
  'mdash': '\u2014',
  'ndash': '\u2013',
  'lsquo': '\u2018',
  'rsquo': '\u2019',
  'ldquo': '\u201C',
  'rdquo': '\u201D',
  'laquo': '\u00AB',
  'raquo': '\u00BB',
  'sect': '\u00A7',
  'para': '\u00B6',
  'dagger': '\u2020',
  'permil': '\u2030',
  'euro': '\u20AC',
  'pound': '\u00A3',
  'yen': '\u00A5',
  'cent': '\u00A2',
};

/// Converts an HTML document into readable plain text.
///
/// The transformation runs in four passes: comments and non-prose elements are
/// removed, block-level tags become newlines, the remaining tags are deleted,
/// and entities are decoded. Finally whitespace is collapsed so indentation and
/// markup artifacts do not waste tokenizer vocabulary. The result is suitable
/// for training but is not a byte-exact rendering of the page.
String htmlToText(String html) {
  if (html.isEmpty) {
    return '';
  }
  String text = html.replaceAll(_commentPattern, ' ');
  text = text.replaceAll(_droppedElementPattern, ' ');
  text = text.replaceAll(_blockTagPattern, '\n');
  text = text.replaceAll(_anyTagPattern, '');
  text = decodeHtmlEntities(text);
  return collapseWhitespace(text);
}

/// Extracts the contents of the document's first `<title>`, or `null`.
///
/// Titles are useful provenance: the collector stores one in a document's meta
/// map so the dashboard can show *what* a URL contained without re-fetching it.
/// Entities are decoded and whitespace collapsed so the value is display ready.
String? htmlTitle(String html) {
  final RegExpMatch? match = RegExp(
    r'<title\b[^>]*>([\s\S]*?)</title>',
    caseSensitive: false,
  ).firstMatch(html);
  if (match == null) {
    return null;
  }
  final String title = collapseWhitespace(decodeHtmlEntities(match.group(1) ?? ''));
  return title.isEmpty ? null : title;
}

/// Collects same-document hyperlinks for a bounded crawl.
///
/// Every `href` on an anchor is resolved against [base], which turns the mixture
/// of relative, root-relative and absolute URLs found on real pages into
/// absolute ones. Fragment-only links, `mailto:`, `javascript:` and other
/// non-HTTP schemes are skipped because the collector cannot fetch them, and the
/// result is de-duplicated while preserving document order so a crawl is
/// reproducible. At most [limit] links are returned; a value of zero or less
/// yields an empty list.
List<Uri> htmlLinks(String html, Uri base, {int limit = 50}) {
  if (limit <= 0 || html.isEmpty) {
    return <Uri>[];
  }
  final RegExp anchorPattern = RegExp(
    r'''<a\b[^>]*?\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))''',
    caseSensitive: false,
  );
  final List<Uri> links = <Uri>[];
  final Set<String> seen = <String>{};
  for (final RegExpMatch match in anchorPattern.allMatches(html)) {
    final String href = (match.group(1) ?? match.group(2) ?? match.group(3) ?? '').trim();
    if (href.isEmpty || href.startsWith('#')) {
      continue;
    }
    Uri resolved;
    try {
      resolved = base.resolve(href);
    } on FormatException {
      continue;
    }
    if (resolved.scheme != 'http' && resolved.scheme != 'https') {
      continue;
    }
    final String key = resolved.toString();
    if (seen.add(key)) {
      links.add(resolved);
      if (links.length >= limit) {
        break;
      }
    }
  }
  return links;
}

/// Decodes named, decimal (`&#233;`) and hexadecimal (`&#xE9;`) character
/// references.
///
/// Exposed separately from [htmlToText] because the title and link extractors
/// need entity-free text without the tag stripping. Unknown named entities and
/// out-of-range numeric references are returned unchanged.
String decodeHtmlEntities(String text) {
  if (!text.contains('&')) {
    return text;
  }
  return text.replaceAllMapped(_entityPattern, (Match match) {
    final String body = match.group(1) ?? '';
    if (body.startsWith('#')) {
      final bool hex = body.length > 1 && (body[1] == 'x' || body[1] == 'X');
      final String digits = hex ? body.substring(2) : body.substring(1);
      final int? codePoint = int.tryParse(digits, radix: hex ? 16 : 10);
      if (codePoint == null ||
          codePoint <= 0 ||
          codePoint > 0x10FFFF ||
          (codePoint >= 0xD800 && codePoint <= 0xDFFF)) {
        return match.group(0) ?? '';
      }
      return String.fromCharCodes(<int>[codePoint]);
    }
    return _namedEntities[body.toLowerCase()] ?? match.group(0) ?? '';
  });
}

/// Collapses every run of spaces, tabs and other horizontal whitespace to a
/// single space and every run of blank lines to at most one.
///
/// Corpus text is fed to a byte-level tokenizer, so repeated indentation is pure
/// waste; at the same time paragraph breaks carry meaning and are preserved.
String collapseWhitespace(String text) {
  if (text.isEmpty) {
    return '';
  }
  final RegExp horizontal = RegExp(r'[ \t\r\f\v\u00A0\u2000-\u200A]+');
  final List<String> lines = text.split('\n');
  final List<String> out = <String>[];
  for (final String raw in lines) {
    final String line = raw.replaceAll(horizontal, ' ').trim();
    if (line.isEmpty) {
      if (out.isNotEmpty && out.last.isNotEmpty) {
        out.add('');
      }
    } else {
      out.add(line);
    }
  }
  while (out.isNotEmpty && out.last.isEmpty) {
    out.removeLast();
  }
  return out.join('\n');
}