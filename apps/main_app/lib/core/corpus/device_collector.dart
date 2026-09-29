// Device-side corpus gathering.
//
// This is the half of the corpus pipeline that needs a real file system, so it
// lives in the app rather than in `harbor_core`. It deliberately does *not*
// pretend to have access it may not have: Android's scoped storage means an
// app can always read its own private directory and may or may not be able to
// read shared storage depending on the OS version and what the user granted.
// The scan therefore reports per-root accessibility, and the UI shows which
// roots were readable, instead of silently returning nothing and letting the
// user conclude that learning is broken.

import 'dart:async';
import 'dart:io';

import 'package:harbor_core/harbor_core.dart';

/// What one device scan did.
class DeviceScanReport {
  /// Creates a report.
  const DeviceScanReport({
    required this.rootsScanned,
    required this.rootsDenied,
    required this.filesSeen,
    required this.filesRead,
    required this.filesSkipped,
    required this.bytesRead,
    required this.documentsAdded,
    required this.errors,
    required this.duration,
  });

  /// Directories that were walked successfully.
  final List<String> rootsScanned;

  /// Directories that could not be opened, with the reason.
  final List<String> rootsDenied;

  /// Files encountered, before any filtering.
  final int filesSeen;

  /// Files whose text was extracted and offered to the corpus.
  final int filesRead;

  /// Files rejected as binary, oversized, or of an unsupported type.
  final int filesSkipped;

  /// Bytes of text extracted.
  final int bytesRead;

  /// Documents the corpus actually accepted (after dedupe and quota checks).
  final int documentsAdded;

  /// Non-fatal errors, one line each.
  final List<String> errors;

  /// Wall-clock time the scan took.
  final Duration duration;

  /// A one-line summary for the UI.
  String get summary => '$filesSeen files seen · $filesRead read · '
      '$documentsAdded added · ${(bytesRead / 1024).toStringAsFixed(0)} kB'
      '${rootsDenied.isEmpty ? '' : ' · ${rootsDenied.length} root(s) denied'}';

  /// JSON form, written to the training log.
  Map<String, Object?> toJson() => <String, Object?>{
        'roots_scanned': rootsScanned,
        'roots_denied': rootsDenied,
        'files_seen': filesSeen,
        'files_read': filesRead,
        'files_skipped': filesSkipped,
        'bytes_read': bytesRead,
        'documents_added': documentsAdded,
        'errors': errors,
        'duration_ms': duration.inMilliseconds,
      };
}

/// Walks a bounded set of directories and feeds readable text into a corpus.
class DeviceCorpusCollector {
  /// Creates a collector writing into [store].
  ///
  /// The defaults are chosen so a scan finishes in a couple of seconds on a
  /// phone with a large private directory: at most [maxFilesPerRoot] files per
  /// root, [maxBytesPerFile] bytes per file, and [maxDepth] directory levels.
  /// A corpus is meant to be a representative sample, not a backup.
  DeviceCorpusCollector({
    required this.store,
    this.maxFilesPerRoot = 80,
    this.maxBytesPerFile = 96 * 1024,
    this.maxDepth = 3,
  });

  /// Destination corpus.
  final CorpusStore store;

  /// Cap on files read per root.
  final int maxFilesPerRoot;

  /// Cap on bytes read per file.
  final int maxBytesPerFile;

  /// Directory recursion depth.
  final int maxDepth;

  /// Scans every directory in [roots].
  ///
  /// Never throws: a permission failure on one root is recorded in the report
  /// and the remaining roots are still scanned. [onProgress] receives a short
  /// message per root so the UI can show what is happening during a long scan.
  Future<DeviceScanReport> scan(
    List<Directory> roots, {
    void Function(String message)? onProgress,
  }) async {
    final Stopwatch stopwatch = Stopwatch()..start();
    final List<String> scanned = <String>[];
    final List<String> denied = <String>[];
    final List<String> errors = <String>[];
    int filesSeen = 0;
    int filesRead = 0;
    int filesSkipped = 0;
    int bytesRead = 0;
    int added = 0;

    for (final Directory root in roots) {
      onProgress?.call('Scanning ${root.path}');
      final List<File> candidates = <File>[];
      try {
        if (!root.existsSync()) {
          denied.add('${root.path} (does not exist)');
          continue;
        }
        await _collect(root, candidates, depth: 0, errors: errors);
        scanned.add(root.path);
      } on FileSystemException catch (error) {
        // The common case on Android 10+ shared storage: the directory exists
        // but the platform denies the listing. Recorded, not fatal.
        denied.add('${root.path} (${error.osError?.message ?? error.message})');
        continue;
      }
      filesSeen += candidates.length;
      onProgress?.call(
        'Reading ${candidates.length} file(s) from ${root.path}',
      );
      int read = 0;
      for (final File file in candidates) {
        if (read >= maxFilesPerRoot) {
          break;
        }
        try {
          final int length = file.lengthSync();
          if (length == 0 || length > maxBytesPerFile) {
            filesSkipped++;
            continue;
          }
          final List<int> bytes = file.readAsBytesSync();
          final String? text = DeviceTextExtractor.extract(
            file.path,
            bytes,
            maxChars: maxBytesPerFile,
          );
          if (text == null || text.trim().length < 24) {
            filesSkipped++;
            continue;
          }
          final bool accepted = await store.addText(
            TextRedactor.redact(text),
            source: CorpusSource.device,
            uri: file.path,
            meta: <String, String>{
              'name': file.uri.pathSegments.isEmpty
                  ? file.path
                  : file.uri.pathSegments.last,
              'bytes': '$length',
            },
          );
          read++;
          filesRead++;
          bytesRead += text.length;
          if (accepted) {
            added++;
          }
        } on FileSystemException catch (error) {
          filesSkipped++;
          errors.add('${file.path}: ${error.message}');
        }
      }
    }

    stopwatch.stop();
    return DeviceScanReport(
      rootsScanned: scanned,
      rootsDenied: denied,
      filesSeen: filesSeen,
      filesRead: filesRead,
      filesSkipped: filesSkipped,
      bytesRead: bytesRead,
      documentsAdded: added,
      errors: errors,
      duration: stopwatch.elapsed,
    );
  }

  /// Breadth-first walk of [directory] up to [maxDepth].
  Future<void> _collect(
    Directory directory,
    List<File> out, {
    required int depth,
    required List<String> errors,
  }) async {
    if (depth > maxDepth) {
      return;
    }
    final List<FileSystemEntity> entries;
    try {
      entries = await directory.list(followLinks: false).toList();
    } on FileSystemException catch (error) {
      errors.add('${directory.path}: ${error.message}');
      return;
    }
    for (final FileSystemEntity entity in entries) {
      if (entity is File) {
        if (DeviceTextExtractor.looksLikeText(entity.path)) {
          out.add(entity);
        }
      } else if (entity is Directory) {
        final String name = entity.uri.pathSegments.isEmpty
            ? ''
            : entity.uri.pathSegments.last;
        // Skip dot-directories: they are almost always caches, version-control
        // metadata or build output, and they dominate the file count without
        // contributing prose.
        if (name.startsWith('.')) {
          continue;
        }
        await _collect(entity, out, depth: depth + 1, errors: errors);
      }
    }
  }
}