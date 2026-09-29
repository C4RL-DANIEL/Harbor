// Corpus state for the UI.
//
// The corpus is the only input to learning, so its screen has to answer three
// questions honestly: how much text do I have, where did it come from, and what
// stopped the last attempt. That is why a scan reports denied roots and rejected
// files instead of just a count — "0 documents added" with no explanation is the
// single most confusing thing a data-gathering feature can say.

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:harbor_core/harbor_core.dart';

import 'device_collector.dart';

/// Owns the corpus store and the collectors that feed it.
class CorpusController extends ChangeNotifier {
  /// Creates a controller.
  CorpusController({
    required this.store,
    required this.collector,
    this.webCollector,
    this.deviceRoots = const <Directory>[],
  });

  /// The corpus itself.
  final CorpusStore store;

  /// The device file scanner.
  final DeviceCorpusCollector collector;

  /// The HTTP collector, if the platform allows network access.
  final WebCorpusCollector? webCollector;

  /// Directories [scanDevice] walks, in order.
  final List<Directory> deviceRoots;

  bool _busy = false;
  String? _error;
  String? _message;
  DeviceScanReport? _lastScan;

  /// Whether a scan or fetch is in progress.
  bool get busy => _busy;

  /// The last failure, if any.
  String? get error => _error;

  /// The last success message, if any.
  String? get message => _message;

  /// The result of the most recent device scan.
  DeviceScanReport? get lastScan => _lastScan;

  /// Current corpus statistics.
  CorpusStats get stats => store.stats;

  /// The stored documents.
  List<CorpusDocument> get documents => store.documents;

  /// Loads any previously persisted corpus.
  Future<void> initialize() async {
    await store.load();
    notifyListeners();
  }

  /// Walks the device roots and adds everything readable.
  Future<DeviceScanReport?> scanDevice() async {
    if (_busy) {
      return null;
    }
    _busy = true;
    _error = null;
    _message = 'Scanning device';
    notifyListeners();
    try {
      final DeviceScanReport report = await collector.scan(
        deviceRoots,
        onProgress: (String message) {
          _message = message;
          notifyListeners();
        },
      );
      _lastScan = report;
      _message = report.summary;
      if (report.documentsAdded > 0) {
        await store.save();
      }
      return report;
    } on Object catch (error) {
      _error = 'The device scan failed: $error';
      return null;
    } finally {
      _busy = false;
      notifyListeners();
    }
  }

  /// Adds text the user pasted or imported.
  Future<bool> addText(String text, {String uri = 'user-input'}) async {
    return _add(
      TextRedactor.redact(text),
      CorpusSource.user,
      uri,
      'Added ${text.length} characters',
    );
  }

  /// Fetches [url] and adds the readable text of the page.
  Future<bool> fetchUrl(String url) async {
    final WebCorpusCollector? fetch = webCollector;
    if (fetch == null) {
      _error = 'Web collection is not available in this build.';
      notifyListeners();
      return false;
    }
    final Uri? parsed = Uri.tryParse(url.trim());
    if (parsed == null || !parsed.hasScheme) {
      _error = '"$url" is not a valid URL.';
      notifyListeners();
      return false;
    }
    if (_busy) {
      return false;
    }
    _busy = true;
    _error = null;
    _message = 'Fetching $parsed';
    notifyListeners();
    try {
      final CorpusDocument? document = await fetch.collect(parsed);
      if (document == null) {
        _error = 'Nothing readable was found at $parsed '
            '(robots.txt, content type, or size).';
        return false;
      }
      final bool added = await store.add(document);
      _message = added
          ? 'Added ${document.charCount} characters from $parsed'
          : 'That page is already in the corpus.';
      if (added) {
        await store.save();
      }
      return added;
    } on Object catch (error) {
      _error = 'Fetching $parsed failed: $error';
      return false;
    } finally {
      _busy = false;
      notifyListeners();
    }
  }

  /// Removes one document by id.
  Future<void> remove(String id) async {
    await store.remove(id);
    await store.save();
    notifyListeners();
  }

  /// Empties the corpus.
  Future<void> clear() async {
    await store.clear();
    _lastScan = null;
    _message = 'Corpus cleared';
    notifyListeners();
  }

  /// Persists the corpus.
  Future<void> save() async {
    await store.save();
  }

  /// The training string this corpus would produce, for a size preview.
  String trainingText({int? maxChars}) =>
      store.trainingText(maxChars: maxChars ?? 200000);

  Future<bool> _add(
    String text,
    CorpusSource source,
    String uri,
    String successMessage,
  ) async {
    if (text.trim().length < 24) {
      _error = 'That text is too short to be useful for training.';
      notifyListeners();
      return false;
    }
    final bool added = await store.addText(text, source: source, uri: uri);
    _message = added ? successMessage : 'That text is already in the corpus.';
    _error = null;
    if (added) {
      await store.save();
    }
    notifyListeners();
    return added;
  }
}