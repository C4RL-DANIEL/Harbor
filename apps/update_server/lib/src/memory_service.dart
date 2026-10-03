// File-backed memory storage for the update server.
//
// harbor_core defines the `MemoryStorage` seam and stays `dart:io`-free so the
// identical runtime compiles for the browser; this file is the server's side of
// that seam. It mirrors the Android app's implementation rather than sharing
// one, because the two runtimes are deliberately independent packages and a
// shared `dart:io` helper would drag the file system into code the browser has
// to compile.
//
// Writes go to a sibling temp file and are renamed over the target, so a crash
// mid-write leaves the previous memory intact rather than a truncated document.
// Reads swallow corruption: an unreadable memory file must not stop the update
// API, which does not need it at all.

import 'dart:io';

import 'package:harbor_core/harbor_core.dart';

/// A [MemoryStorage] backed by a single UTF-8 JSON file.
class FileMemoryStorage implements MemoryStorage {
  /// Creates storage over [file].
  FileMemoryStorage(this.file);

  /// The file the memory is written to.
  final File file;

  @override
  Future<String?> read() async {
    try {
      if (!await file.exists()) {
        return null;
      }
      return file.readAsString();
    } on Object {
      return null;
    }
  }

  @override
  Future<void> write(String payload) async {
    await file.parent.create(recursive: true);
    final File temporary = File('${file.path}.tmp');
    await temporary.writeAsString(payload, flush: true);
    await temporary.rename(file.path);
  }

  @override
  Future<void> delete() async {
    try {
      if (await file.exists()) {
        await file.delete();
      }
    } on Object {
      // Deleting is best-effort for the same reason reading is.
    }
  }
}
