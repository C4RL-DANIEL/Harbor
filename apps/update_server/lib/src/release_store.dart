// Durable, atomic JSON persistence for [ServerState].
//
// This library is Flutter-free: it runs on the pure Dart VM.

import 'dart:convert';
import 'dart:io';

import 'models.dart';

/// Persists [ServerState] to a JSON document.
///
/// Writes are atomic (`<file>.tmp` followed by a rename) and strictly
/// serialized through a future chain, so concurrent [save] calls can never
/// interleave their bytes. [ReleaseStore.inMemory] keeps the state in RAM,
/// which is what the unit tests and preview tooling use.
class ReleaseStore {
  /// Persists to [path].
  ReleaseStore(String path) : _file = File(path);

  /// Persists to an explicit [File].
  ReleaseStore.file(File file) : _file = file;

  /// Keeps everything in memory; nothing touches the filesystem.
  ReleaseStore.inMemory() : _file = null;

  static const JsonEncoder _encoder = JsonEncoder.withIndent('  ');

  final File? _file;
  ServerState? _cached;
  Future<ServerState>? _loadFuture;
  Future<void> _writeChain = Future<void>.value();
  late final ServerState _fallback = ServerState.seed();

  /// The path of the backing file, or `null` for an in-memory store.
  String? get path => _file?.path;

  /// Whether this store writes to disk.
  bool get isPersistent => _file != null;

  /// The most recently loaded or saved state.
  ///
  /// Before the first [load] this falls back to the seed state; callers that
  /// need the on-disk contents must `await load()` first.
  ServerState get cached => _cached ?? _fallback;

  /// Loads the state, seeding defaults when no file exists yet.
  ///
  /// Concurrent callers share a single load; the result is memoized.
  Future<ServerState> load() {
    final ServerState? existing = _cached;
    if (existing != null) {
      return Future<ServerState>.value(existing);
    }
    return _loadFuture ??= _loadFromDisk().whenComplete(() {
      _loadFuture = null;
    });
  }

  Future<ServerState> _loadFromDisk() async {
    final File? file = _file;
    if (file == null) {
      final ServerState seed = ServerState.seed();
      _cached = seed;
      return seed;
    }
    if (!await file.exists()) {
      final ServerState seed = ServerState.seed();
      await save(seed);
      return seed;
    }

    final String raw;
    try {
      raw = await file.readAsString();
    } on FileSystemException catch (error) {
      throw StateStoreException(
        'unable to read state file "${file.path}": ${error.message}',
      );
    }
    if (raw.trim().isEmpty) {
      final ServerState seed = ServerState.seed();
      _cached = seed;
      return seed;
    }

    final Object? decoded;
    try {
      decoded = jsonDecode(raw);
    } on FormatException catch (error) {
      throw StateStoreException(
        'state file "${file.path}" is not valid JSON: ${error.message}',
      );
    }
    if (decoded is! Map<String, Object?>) {
      throw StateStoreException(
        'state file "${file.path}" must contain a JSON object',
      );
    }

    final ServerState state;
    try {
      state = ServerState.fromJson(decoded);
    } on ApiException catch (error) {
      throw StateStoreException(
        'state file "${file.path}" is invalid: ${error.message}',
      );
    }
    _cached = state;
    return state;
  }

  /// Persists [state] and immediately refreshes [cached].
  ///
  /// The returned future completes when *this* write has landed on disk. A
  /// failure is reported to the caller but never poisons later writes.
  Future<void> save(ServerState state) {
    _cached = state;
    final File? file = _file;
    if (file == null) {
      return Future<void>.value();
    }
    final Future<void> write =
        _writeChain.then((_) => _writeAtomic(file, state));
    _writeChain = write.then<void>(
      (_) {},
      onError: (Object _) {},
    );
    return write;
  }

  /// Waits for every queued write to settle.
  Future<void> flush() => _writeChain;

  Future<void> _writeAtomic(File file, ServerState state) async {
    final Directory directory = file.parent;
    if (!await directory.exists()) {
      await directory.create(recursive: true);
    }
    final File temp = File('${file.path}.tmp');
    final String encoded = _encoder.convert(state.toJson());
    await temp.writeAsString(encoded, flush: true);
    await temp.rename(file.path);
  }
}