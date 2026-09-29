// The web half of the corpus: a policy object, a robots.txt parser and a
// collector that fetches pages without ever stepping outside the policy.
//
// Design constraints that shaped this file:
//
//   * No real network access in tests. The `http.Client` is injected, so tests
//     hand in `package:http/testing.dart`'s `MockClient` and assert on requests
//     instead of hitting the internet.
//   * The host allow-list must survive redirects. `http`'s automatic redirect
//     following would hide intermediate hops, so redirects are followed by hand
//     and the policy is re-checked on every `Location`.
//   * A page that violates the policy is *skipped* (`null`), while a transport
//     failure is *thrown* (`WebCorpusException`), so a UI can distinguish "we
//     chose not to" from "the network broke".
//
// `dart:io` is deliberately absent: this must compile for Flutter Web, where
// `http` picks the browser transport itself.

import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'corpus_document.dart';
import 'html_text.dart';

/// The rules a crawl must obey: which URIs are reachable and how much may be
/// downloaded from each.
///
/// The defaults are the conservative ones Harbor ships with: HTTPS only, no host
/// allow-list (meaning "any host", which the application narrows before a real
/// crawl), a 512 KiB body cap, three redirects, robots.txt obeyed and an honest
/// user agent that names the bot.
class WebCorpusPolicy {
  /// Creates a policy; every field has a safe default, so an application only
  /// has to supply what it wants to differ from.
  const WebCorpusPolicy({
    this.allowedHosts = const <String>{},
    this.allowedSchemes = const <String>{'https'},
    this.maxBytes = 512 * 1024,
    this.maxRedirects = 3,
    this.obeyRobotsTxt = true,
    this.userAgent = 'HarborCorpusBot/1.0',
  });

  /// Hosts the crawl may visit; an empty set means any host is allowed.
  final Set<String> allowedHosts;

  /// URI schemes the crawl may use. Defaults to HTTPS only because plain HTTP
  /// can be tampered with by any intermediary.
  final Set<String> allowedSchemes;

  /// Maximum response body size in bytes; a larger body is refused rather than
  /// silently truncated, because a truncated page trains on a broken document.
  final int maxBytes;

  /// Maximum number of redirect hops to follow before giving up.
  final int maxRedirects;

  /// Whether `robots.txt` is fetched and honoured before a page is requested.
  final bool obeyRobotsTxt;

  /// The `User-Agent` header sent on every request, including robots.txt.
  final String userAgent;

  /// Whether [uri] is reachable under this policy.
  ///
  /// A URI is allowed when its scheme is in [allowedSchemes] and either
  /// [allowedHosts] is empty or it names [uri]'s host. Host comparison is exact
  /// (no subdomain wildcards) so an allow-list entry cannot accidentally widen
  /// into a whole domain.
  bool allows(Uri uri) {
    if (!allowedSchemes.contains(uri.scheme)) {
      return false;
    }
    if (allowedHosts.isEmpty) {
      return true;
    }
    return allowedHosts.contains(uri.host);
  }

  /// Returns a copy of this policy with the supplied fields replaced.
  ///
  /// Every parameter is optional and `null` means "keep the current value",
  /// which lets a caller narrow one dimension of a shared policy without
  /// restating the rest.
  WebCorpusPolicy copyWith({
    Set<String>? allowedHosts,
    Set<String>? allowedSchemes,
    int? maxBytes,
    int? maxRedirects,
    bool? obeyRobotsTxt,
    String? userAgent,
  }) {
    return WebCorpusPolicy(
      allowedHosts: allowedHosts ?? this.allowedHosts,
      allowedSchemes: allowedSchemes ?? this.allowedSchemes,
      maxBytes: maxBytes ?? this.maxBytes,
      maxRedirects: maxRedirects ?? this.maxRedirects,
      obeyRobotsTxt: obeyRobotsTxt ?? this.obeyRobotsTxt,
      userAgent: userAgent ?? this.userAgent,
    );
  }
}

/// The subset of `robots.txt` that this crawler acts on.
///
/// Only the `User-agent: *` groups are considered; rules written for a named
/// crawler are deliberately ignored, because pretending to be that crawler would
/// be dishonest and because Harbor's user agent is its own. `Allow` directives
/// are not modelled: the crawler follows the conservative union of every
/// `Disallow` rule that applies to it.
class RobotsRules {
  /// Creates a rule set from [disallow], a list of path prefixes.
  const RobotsRules({required this.disallow});

  /// A rule set that allows everything; useful for hosts whose robots.txt was
  /// fetched but contained no relevant group.
  static const RobotsRules empty = RobotsRules(disallow: <String>[]);

  /// Path prefixes that are off limits for `User-agent: *`, in file order.
  final List<String> disallow;

  /// Whether [path] may be fetched under these rules.
  ///
  /// A path is refused when it starts with any non-empty rule prefix. The `$`
  /// end-anchor and `*` wildcards some sites use are not interpreted; treating
  /// them as ordinary prefix characters is the conservative reading, since the
  /// rule then matches *more* paths rather than fewer. An empty rule
  /// (`Disallow:`) is the standard way to allow everything and is skipped.
  bool allowsPath(String path) {
    final String normalised = path.isEmpty ? '/' : path;
    for (final String rule in disallow) {
      if (rule.isEmpty) {
        continue;
      }
      if (normalised.startsWith(rule)) {
        return false;
      }
    }
    return true;
  }

  /// Parses [robotsTxt], collecting the `Disallow` prefixes of `User-agent: *`.
  ///
  /// The parser is forgiving, as robots.txt files in the wild are: comments are
  /// stripped, unknown fields are ignored, a blank or malformed line is skipped,
  /// and consecutive `User-agent` lines are treated as one group so the common
  /// `User-agent: *` / `User-agent: Googlebot` pairing still counts as a star
  /// group. A file with no star group yields [empty].
  static RobotsRules parse(String robotsTxt) {
    final List<String> disallow = <String>[];
    bool inStarGroup = false;
    bool lastLineWasUserAgent = false;
    for (final String rawLine in const LineSplitter().convert(robotsTxt)) {
      String line = rawLine;
      final int comment = line.indexOf('#');
      if (comment >= 0) {
        line = line.substring(0, comment);
      }
      line = line.trim();
      if (line.isEmpty) {
        continue;
      }
      final int colon = line.indexOf(':');
      if (colon < 0) {
        continue;
      }
      final String field = line.substring(0, colon).trim().toLowerCase();
      final String value = line.substring(colon + 1).trim();
      if (field == 'user-agent') {
        // A new group starts only when the previous line was something else;
        // consecutive user-agent lines share one group.
        if (!lastLineWasUserAgent) {
          inStarGroup = false;
        }
        if (value == '*') {
          inStarGroup = true;
        }
        lastLineWasUserAgent = true;
      } else {
        lastLineWasUserAgent = false;
        if (field == 'disallow' && inStarGroup) {
          disallow.add(value);
        }
      }
    }
    if (disallow.isEmpty) {
      return empty;
    }
    return RobotsRules(disallow: List<String>.unmodifiable(disallow));
  }
}

/// Fetches web pages subject to a [WebCorpusPolicy].
///
/// Create one collector per crawl so its robots.txt cache is shared across the
/// pages of a site. The injected [http.Client] is the only way this class talks
/// to the network, which keeps tests hermetic.
class WebCorpusCollector {
  /// Creates a collector using [client] for every request, including robots.txt.
  WebCorpusCollector({
    required http.Client client,
    this.policy = const WebCorpusPolicy(),
  }) : _client = client;

  /// The rules this collector enforces.
  final WebCorpusPolicy policy;

  final http.Client _client;

  /// Parsed robots.txt per host; `null` values are cached too, so a host that
  /// has no usable robots.txt is not fetched twice. `containsKey` distinguishes
  /// "cached negative" from "not yet fetched".
  final Map<String, RobotsRules?> _robotsCache = <String, RobotsRules?>{};

  /// The largest robots.txt this collector will read, in bytes. Real files are
  /// a few kilobytes; the cap keeps a malicious host from streaming forever.
  static const int _maxRobotsBytes = 64 * 1024;

  /// Returns the rules for [uri]'s host, fetching and caching them on first use.
  ///
  /// Returns `null` when no rules were fetched: because the policy disables
  /// robots.txt, because the file is missing, or because the request failed. A
  /// missing or unreachable robots.txt means "no rules were expressed", so the
  /// crawl proceeds (fail open); a *present* file that disallows the path stops
  /// the page in [collect]. Network errors for robots.txt are swallowed on
  /// purpose, since a broken robots.txt should not abort a whole crawl.
  Future<RobotsRules?> robotsFor(Uri uri) async {
    final String host = uri.host;
    if (_robotsCache.containsKey(host)) {
      return _robotsCache[host];
    }
    if (!policy.obeyRobotsTxt) {
      _robotsCache[host] = null;
      return null;
    }
    RobotsRules? rules;
    try {
      final Uri robotsUri = uri.replace(path: '/robots.txt', query: '', fragment: '');
      final http.Request request = http.Request('GET', robotsUri);
      request.followRedirects = false;
      request.headers['user-agent'] = policy.userAgent;
      final http.StreamedResponse response = await _client.send(request);
      if (response.statusCode >= 200 && response.statusCode < 300) {
        final Uint8List? body = await _readCapped(response, _maxRobotsBytes);
        if (body != null) {
          rules = RobotsRules.parse(utf8.decode(body, allowMalformed: true));
        }
      } else {
        await response.stream.drain<void>();
      }
    } on Exception {
      rules = null;
    }
    _robotsCache[host] = rules;
    return rules;
  }

  /// Fetches [uri] and converts it into a corpus document, or returns `null`.
  ///
  /// `null` means the policy or the page itself said no: the URI's scheme or
  /// host is not allowed, robots.txt disallows its path, the response is not a
  /// 2xx, the content type is not textual, a redirect leaves the allow-list or
  /// exceeds [WebCorpusPolicy.maxRedirects], the body exceeds
  /// [WebCorpusPolicy.maxBytes], or the extracted text is shorter than 32
  /// characters (a stub, an error page or a redirect notice — not training
  /// material).
  ///
  /// A transport or protocol failure throws [WebCorpusException] instead, so the
  /// caller can report a real error without mistaking it for a policy skip.
  Future<CorpusDocument?> collect(Uri uri) async {
    if (!policy.allows(uri)) {
      return null;
    }
    if (policy.obeyRobotsTxt) {
      final RobotsRules? rules = await robotsFor(uri);
      if (rules != null && !rules.allowsPath(uri.path.isEmpty ? '/' : uri.path)) {
        return null;
      }
    }

    final _FetchOutcome? outcome = await _fetchFollowingRedirects(uri);
    if (outcome == null) {
      return null;
    }
    final http.StreamedResponse response = outcome.response;
    if (response.statusCode < 200 || response.statusCode >= 300) {
      await response.stream.drain<void>();
      return null;
    }
    final String? contentType = response.headers['content-type'];
    if (contentType != null && !_isTextualContentType(contentType)) {
      await response.stream.drain<void>();
      return null;
    }
    final Uint8List? body = await _readCapped(response, policy.maxBytes);
    if (body == null) {
      return null;
    }

    final String raw = utf8.decode(body, allowMalformed: true);
    final bool isHtml = contentType == null ||
        contentType.toLowerCase().contains('html') ||
        contentType.toLowerCase().contains('xml');
    final String text = isHtml ? htmlToText(raw) : raw;
    if (text.trim().length < 32) {
      return null;
    }

    final Map<String, String> meta = <String, String>{
      'contentType': contentType ?? 'text/html',
    };
    final String? title = isHtml ? htmlTitle(raw) : null;
    if (title != null) {
      meta['title'] = title;
    }

    return CorpusDocument(
      id: CorpusDocument.hashOf(text),
      text: text,
      source: CorpusSource.web,
      uri: outcome.uri.toString(),
      collectedAt: DateTime.now().toUtc(),
      meta: meta,
    );
  }

  /// Requests [uri] and follows up to [WebCorpusPolicy.maxRedirects] redirects
  /// by hand, re-checking [WebCorpusPolicy.allows] at every hop.
  ///
  /// Returns the final response together with the URI it came from, or `null`
  /// when a redirect leaves the allow-list or the hop budget is exhausted. The
  /// effective URI is tracked here rather than read from `response.request`
  /// because test doubles and browser transports do not always populate that
  /// field, and provenance must not depend on the transport. Every redirect
  /// response is drained before the next request so the underlying connection
  /// can be reused. Throws [WebCorpusException] when the transport itself fails.
  Future<_FetchOutcome?> _fetchFollowingRedirects(Uri uri) async {
    Uri current = uri;
    for (int hop = 0; hop <= policy.maxRedirects; hop++) {
      final http.Request request = http.Request('GET', current);
      request.followRedirects = false;
      request.headers['user-agent'] = policy.userAgent;
      request.headers['accept'] =
          'text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1';
      final http.StreamedResponse response;
      try {
        response = await _client.send(request);
      } on Exception catch (error) {
        throw WebCorpusException('request failed: $error', current);
      }
      final String? location = response.headers['location'];
      if (!response.isRedirect || location == null) {
        return _FetchOutcome(response, current);
      }
      await response.stream.drain<void>();
      if (hop == policy.maxRedirects) {
        return null;
      }
      final Uri next = current.resolve(location);
      if (!policy.allows(next)) {
        return null;
      }
      current = next;
    }
    return null;
  }

  /// Reads at most [maxBytes] from [response]; returns `null` (and cancels the
  /// subscription) as soon as the body exceeds the cap.
  ///
  /// Reading incrementally rather than through `toBytes()` is what makes the cap
  /// a real defence: an oversized page is abandoned mid-stream instead of being
  /// buffered in full first.
  Future<Uint8List?> _readCapped(http.StreamedResponse response, int maxBytes) async {
    final BytesBuilder builder = BytesBuilder(copy: false);
    int total = 0;
    await for (final List<int> chunk in response.stream) {
      total += chunk.length;
      if (total > maxBytes) {
        return null;
      }
      builder.add(chunk);
    }
    return builder.takeBytes();
  }

  /// Whether [contentType] denotes text the collector can use.
  ///
  /// Accepts anything under `text/`, the XML and JSON families (including the
  /// `+xml` / `+json` suffixes), because those are readable and safe to feed to
  /// the extractor. Binary types are refused even if the extension or URL looks
  /// promising.
  static bool _isTextualContentType(String contentType) {
    final String type = contentType.split(';').first.trim().toLowerCase();
    if (type.startsWith('text/')) {
      return true;
    }
    if (type == 'application/xhtml+xml' ||
        type == 'application/xml' ||
        type == 'application/json') {
      return true;
    }
    return type.endsWith('+xml') || type.endsWith('+json');
  }
}

/// A completed fetch: the response plus the URI it was actually served from
/// after redirects.
///
/// Private because it only exists to carry the effective URL out of
/// [WebCorpusCollector._fetchFollowingRedirects] without trusting the transport
/// to fill in `response.request`.
class _FetchOutcome {
  /// Creates an outcome for [response], served from [uri].
  const _FetchOutcome(this.response, this.uri);

  /// The final, non-redirect response.
  final http.StreamedResponse response;

  /// The URI of the final request, used as document provenance.
  final Uri uri;
}

/// A real transport or protocol failure during a web collection.
///
/// Distinguishing this from a `null` result lets the caller show an error (and
/// retry later) for a dropped connection while treating a policy refusal as
/// normal operation.
class WebCorpusException implements Exception {
  /// Creates an exception describing [message] for the failing [uri].
  const WebCorpusException(this.message, this.uri);

  /// Human-readable description of what went wrong.
  final String message;

  /// The URI that was being fetched when the failure occurred.
  final Uri uri;

  /// Renders the failure for logs; includes no response body, which could hold
  /// sensitive page content.
  @override
  String toString() => 'WebCorpusException: $message ($uri)';
}