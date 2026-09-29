// The corpus screen: the one place a user can inspect, extend and prune the
// text the model learns from.
//
// It is deliberately honest rather than tidy. A screen that prints only a
// document count cannot answer the two questions a user actually has — where
// did the text come from, and why did that scan add nothing? — so the last-scan
// panel reports denied roots and rejected files, and the stats card carries the
// store's rejection counters instead of hiding them.

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:harbor_core/harbor_core.dart';

import '../../core/corpus/corpus_controller.dart';
import '../../core/corpus/device_collector.dart';

/// Shows the corpus and lets the user fill, inspect and prune it.
///
/// The corpus is the only input to learning, and on Android "why is my corpus
/// empty?" almost always has a platform answer rather than a bug: a shared
/// storage root was denied by the OS, or files were skipped as binary or
/// oversized. Reporting those outcomes instead of a bare count is therefore the
/// central design choice of this screen; the rest is the controls needed to
/// paste text, fetch a page, scan the device, and remove or clear documents.
class CorpusScreen extends StatefulWidget {
  /// Creates the screen.
  ///
  /// The [controller] is owned by the caller so the corpus survives this screen
  /// being rebuilt or swapped out by another tab.
  const CorpusScreen({super.key, required this.controller});

  /// The corpus state this screen renders and drives.
  final CorpusController controller;

  @override
  State<CorpusScreen> createState() => _CorpusScreenState();
}

class _CorpusScreenState extends State<CorpusScreen> {
  /// Draft text for the paste box.
  final TextEditingController _paste = TextEditingController();

  /// Draft URL for the fetch box.
  final TextEditingController _url = TextEditingController();

  @override
  void initState() {
    super.initState();
    // Loading the persisted corpus touches disk, so it is deferred past the
    // first frame: doing it inside initState would either block the first paint
    // or mark the widget dirty during build.
    WidgetsBinding.instance.addPostFrameCallback((Duration _) {
      unawaited(widget.controller.initialize());
    });
  }

  @override
  void dispose() {
    _paste.dispose();
    _url.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final CorpusController controller = widget.controller;
    return ListenableBuilder(
      listenable: controller,
      builder: (BuildContext context, Widget? child) {
        final CorpusStats stats = controller.stats;
        final String? error = controller.error;
        final String? message = controller.message;
        final DeviceScanReport? scan = controller.lastScan;
        return ListView(
          padding: const EdgeInsets.all(16),
          children: <Widget>[
            if (error != null)
              _notice(
                message: error,
                icon: Icons.error_outline,
                background: Theme.of(context).colorScheme.errorContainer,
                foreground: Theme.of(context).colorScheme.onErrorContainer,
              ),
            if (message != null)
              _notice(
                message: message,
                icon: Icons.info_outline,
                background: Theme.of(context).colorScheme.surfaceContainerHighest,
                foreground: Theme.of(context).colorScheme.primary,
              ),
            _card(_statsPanel(stats)),
            _card(_actionsPanel(controller)),
            if (scan != null) _card(_lastScanPanel(scan)),
            _documentsHeader(stats),
            ..._documentTiles(controller),
            _card(_trainingPreviewPanel()),
          ],
        );
      },
    );
  }

  /// Builds the volume, provenance and rejection summary.
  ///
  /// The rejection footnote is rendered only when something was refused, so a
  /// healthy corpus stays uncluttered while a mysteriously small one explains
  /// itself without the user having to guess.
  Widget _statsPanel(CorpusStats stats) {
    final bool hasRejections = stats.rejectedDuplicates != 0 ||
        stats.rejectedTooLarge != 0 ||
        stats.rejectedQuota != 0;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Text('Corpus', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 12),
        Wrap(
          spacing: 24,
          runSpacing: 12,
          children: <Widget>[
            _metric('Documents', _thousands(stats.documentCount)),
            _metric('Characters', _thousands(stats.totalChars)),
            _metric('Size', _bytes(stats.totalBytes)),
            for (final CorpusSource source in CorpusSource.values)
              _metric(source.name, _thousands(stats.bySource[source] ?? 0)),
          ],
        ),
        if (hasRejections) ...<Widget>[
          const SizedBox(height: 12),
          Text(
            'duplicates ${stats.rejectedDuplicates} · '
            'too large ${stats.rejectedTooLarge} · '
            'over quota ${stats.rejectedQuota}',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ],
      ],
    );
  }

  /// Builds the scan, paste, fetch and clear controls.
  ///
  /// Every button that starts asynchronous work uses [unawaited] and a busy
  /// guard where one exists, because dropping the future would silently lose
  /// the operation and its failure under this project's lint configuration.
  Widget _actionsPanel(CorpusController controller) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Text(
          'Add to the corpus',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: 12),
        FilledButton.icon(
          onPressed: controller.busy
              ? null
              : () {
                  unawaited(controller.scanDevice());
                },
          icon: controller.busy
              ? const SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Icon(Icons.folder_open),
          label: Text(controller.busy ? 'Scanning…' : 'Scan device'),
        ),
        const SizedBox(height: 16),
        TextField(
          controller: _paste,
          minLines: 4,
          maxLines: 8,
          decoration: const InputDecoration(
            hintText: 'Paste text to learn from',
            border: OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: 8),
        FilledButton.tonalIcon(
          onPressed: _addPasted,
          icon: const Icon(Icons.playlist_add),
          label: const Text('Add text'),
        ),
        const SizedBox(height: 16),
        TextField(
          controller: _url,
          decoration: const InputDecoration(
            hintText: 'https://example.com/article',
            border: OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: 8),
        FilledButton.tonalIcon(
          onPressed: () {
            unawaited(controller.fetchUrl(_url.text));
          },
          icon: const Icon(Icons.download_outlined),
          label: const Text('Fetch page'),
        ),
        const SizedBox(height: 8),
        TextButton.icon(
          onPressed: () {
            unawaited(_confirmClear());
          },
          icon: const Icon(Icons.delete_outline),
          label: const Text('Clear corpus'),
        ),
      ],
    );
  }

  /// Saves the pasted text and clears the box only when it was accepted.
  ///
  /// Clearing unconditionally would silently discard text the store refused as
  /// too short or as a duplicate, which is exactly the invisible data loss this
  /// screen exists to prevent; the mounted guard keeps the box alive across the
  /// asynchronous add.
  Future<void> _addPasted() async {
    final bool added = await widget.controller.addText(_paste.text);
    if (!mounted) {
      return;
    }
    if (added) {
      _paste.clear();
    }
  }

  /// Asks for confirmation before emptying the corpus.
  ///
  /// Clearing is destructive and cannot be undone, and the corpus can represent
  /// hours of scanning, so it is always confirmed; the dialog also states that
  /// device files are untouched, because "clear" would otherwise read as "delete
  /// my documents".
  Future<void> _confirmClear() async {
    final bool? confirmed = await showDialog<bool>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Clear the corpus?'),
          content: const Text(
            'Every stored document is deleted from Harbor. Your device files '
            'are not touched; only the text Harbor copied from them is removed.',
          ),
          actions: <Widget>[
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.of(context).pop(true),
              child: const Text('Clear'),
            ),
          ],
        );
      },
    );
    if (!mounted) {
      return;
    }
    if (confirmed ?? false) {
      unawaited(widget.controller.clear());
    }
  }

  /// Summarises the most recent device scan and its per-root outcome.
  ///
  /// The detail lists are the point of the panel: a denied root and a rejected
  /// file explain a smaller-than-expected corpus, which a bare total cannot.
  /// The footnote tells the user a denial is an Android storage-permission
  /// result rather than a Harbor defect, and that the private directory is
  /// always read.
  Widget _lastScanPanel(DeviceScanReport report) {
    final List<String> errors = report.errors.take(10).toList();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Text('Last scan', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        Text(report.summary),
        const SizedBox(height: 4),
        Text('${report.duration.inMilliseconds} ms'),
        _expansion(
          title: 'Roots scanned (${report.rootsScanned.length})',
          lines: report.rootsScanned,
        ),
        _expansion(
          title: 'Roots denied (${report.rootsDenied.length})',
          lines: report.rootsDenied,
        ),
        _expansion(
          title: 'Errors (${report.errors.length})',
          lines: errors,
        ),
        const SizedBox(height: 8),
        Text(
          'A denied root is an Android storage-permission outcome, not a bug: '
          'shared storage can be unreadable on this device or OS version. '
          'Harbor always reads its own private directory, so a scan can still '
          'find text even when shared roots are refused.',
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    );
  }

  /// Builds the "Documents (n)" heading above the document list.
  ///
  /// The count lives in the heading so the user can see the size of the list
  /// without expanding anything.
  Widget _documentsHeader(CorpusStats stats) {
    return Padding(
      padding: const EdgeInsets.only(top: 4, bottom: 8),
      child: Text(
        'Documents (${stats.documentCount})',
        style: Theme.of(context).textTheme.titleMedium,
      ),
    );
  }

  /// Builds the document list, or the empty state when nothing is stored.
  ///
  /// The empty state is the only place a first-time user learns that learning
  /// is local and that text — not photos or binaries — is the raw material, so
  /// it answers those questions before pointing at the controls above it.
  List<Widget> _documentTiles(CorpusController controller) {
    final List<CorpusDocument> documents = controller.documents;
    if (documents.isEmpty) {
      return <Widget>[
        _card(
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Text(
                'No documents yet',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              const Text(
                'The corpus is everything the model learns from. Nothing here '
                'is uploaded. Scanning reads text files, not images or '
                'binaries, and a few hundred kilobytes of prose is a '
                'reasonable start.',
              ),
            ],
          ),
        ),
      ];
    }
    return <Widget>[
      for (final CorpusDocument doc in documents) _documentCard(controller, doc),
    ];
  }

  /// Renders one stored document as an expandable excerpt with a delete action.
  ///
  /// The excerpt is capped at 1200 characters and the scroll area at 300 pixels
  /// so a half-megabyte document cannot turn the list into an unreadable wall of
  /// text, while the full text stays available by scrolling.
  Widget _documentCard(CorpusController controller, CorpusDocument doc) {
    final String body = _excerpt(doc.text, 1200);
    return _card(
      ExpansionTile(
        tilePadding: EdgeInsets.zero,
        title: Text(doc.uri, maxLines: 1, overflow: TextOverflow.ellipsis),
        subtitle: Text(
          '${doc.source.name} · ${_thousands(doc.charCount)} chars',
        ),
        trailing: IconButton(
          tooltip: 'Delete this document',
          icon: const Icon(Icons.delete_outline),
          onPressed: () {
            unawaited(controller.remove(doc.id));
          },
        ),
        children: <Widget>[
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Container(
              constraints: const BoxConstraints(maxHeight: 300),
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    SelectableText(body),
                    if (doc.text.length > 1200) const Text('…'),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  /// Offers a preview of the exact string the trainer will consume.
  ///
  /// The preview matters because the trainer truncates the corpus to a budget;
  /// showing the real text makes that truncation visible instead of surprising
  /// during training, and the subtitle says so explicitly.
  Widget _trainingPreviewPanel() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Text('Training text', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        const Text(
          'This is the exact text the trainer will consume, truncated to the '
          'preview budget.',
        ),
        const SizedBox(height: 8),
        TextButton(
          onPressed: _showTrainingPreview,
          child: const Text('Preview training text'),
        ),
      ],
    );
  }

  /// Opens a read-only dialog containing the training string.
  ///
  /// The navigator is captured before the dialog is shown so the Close action
  /// never has to reach back through a `BuildContext` that might be stale;
  /// showing a dialog is synchronous, so no await guard is needed here.
  void _showTrainingPreview() {
    final NavigatorState navigator = Navigator.of(context);
    final String preview = widget.controller.trainingText(maxChars: 4000);
    unawaited(
      showDialog<void>(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: const Text('Training text preview'),
            content: SizedBox(
              width: double.maxFinite,
              child: SingleChildScrollView(
                child: SelectableText(
                  preview,
                  style: const TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 13,
                  ),
                ),
              ),
            ),
            actions: <Widget>[
              TextButton(
                onPressed: () => navigator.pop(),
                child: const Text('Close'),
              ),
            ],
          );
        },
      ),
    );
  }

  /// Renders a message as a tinted notice card with a leading icon.
  ///
  /// Errors and status messages share one shape so neither can be mistaken for
  /// the other and neither is ever dropped; only the colour differs, and errors
  /// use the error palette so they read as failures rather than progress.
  Widget _notice({
    required String message,
    required IconData icon,
    required Color background,
    required Color foreground,
  }) {
    return Card(
      color: background,
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Icon(icon, color: foreground),
            const SizedBox(width: 12),
            Expanded(
              child: Text(message, style: TextStyle(color: foreground)),
            ),
          ],
        ),
      ),
    );
  }

  /// Builds one collapsible list of scan detail lines.
  ///
  /// A scan can touch dozens of absolute paths, so the details stay collapsed
  /// until asked for, and an empty list says 'None' rather than rendering an
  /// unexplained gap.
  Widget _expansion({required String title, required List<String> lines}) {
    return ExpansionTile(
      tilePadding: EdgeInsets.zero,
      title: Text(title),
      childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
      expandedCrossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        if (lines.isEmpty)
          const Text('None')
        else
          for (final String line in lines)
            Text(line, style: Theme.of(context).textTheme.bodySmall),
      ],
    );
  }

  /// Wraps [child] in the padded card used by every panel on this screen.
  ///
  /// A shared wrapper keeps the spacing and elevation consistent; the screen is
  /// a long list of unrelated panels, and ragged card metrics would make it look
  /// broken rather than information-dense.
  Widget _card(Widget child) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: child,
      ),
    );
  }

  /// Builds one value-over-label tile for the stats [Wrap].
  ///
  /// A method rather than a widget class because the tile is tiny, stateless and
  /// only ever built here.
  Widget _metric(String label, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        Text(
          value,
          style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
        ),
        Text(label, style: Theme.of(context).textTheme.bodySmall),
      ],
    );
  }
}

/// Formats [value] with thousands separators.
///
/// Corpus sizes run into the hundreds of thousands, and a bare "128394" is much
/// harder to sanity-check at a glance than "128,394".
String _thousands(int value) {
  final String digits = value.abs().toString();
  final StringBuffer buffer = StringBuffer();
  for (int i = 0; i < digits.length; i++) {
    if (i > 0 && (digits.length - i) % 3 == 0) {
      buffer.write(',');
    }
    buffer.write(digits[i]);
  }
  return value < 0 ? '-$buffer' : buffer.toString();
}

/// Formats [value] as a compact byte size.
///
/// The store counts bytes while the user thinks in kilobytes and megabytes, so
/// the stats card translates between the two instead of showing raw bytes.
String _bytes(int value) {
  const List<String> units = <String>['B', 'kB', 'MB', 'GB'];
  double size = value.toDouble();
  int unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit++;
  }
  final String rendered = unit == 0 || size >= 100
      ? size.toStringAsFixed(0)
      : size.toStringAsFixed(1);
  return '$rendered ${units[unit]}';
}

/// Returns at most [maxChars] characters of [text].
///
/// Expanding a document has to be instant even when the document is half a
/// megabyte, so the tile renders an excerpt and never the whole file.
String _excerpt(String text, int maxChars) =>
    text.length <= maxChars ? text : text.substring(0, maxChars);