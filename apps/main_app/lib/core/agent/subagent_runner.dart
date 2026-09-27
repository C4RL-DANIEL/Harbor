// Subagent execution layer for concurrent local task fan-out.
//
// The extended-thinking agent decomposes a goal into steps, and each step is
// dispatched to a *subagent* — a small, single-purpose worker with its own
// typed contract. Because steps are frequently independent (parse three files,
// run two checks), the runner executes them concurrently on a bounded worker
// pool so the agent never blocks the UI isolate and never opens more work than
// the device can sustain.
//
// Safety properties that matter in production:
//
//   * [SystemCommandSubagent] runs executables directly (`runInShell: false`)
//     against an explicit allow-list. Any argument containing a shell
//     metacharacter is rejected before the process is spawned, which removes
//     command-injection as a class of bug rather than filtering for it.
//   * Every task has a deadline; a blown deadline yields a failed
//     [SubagentResult] with a clear reason instead of a hung agent.
//   * Failures are values, not exceptions: one failing step never aborts a
//     fan-out, so the verifier can reason about partial results.

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'file_readers.dart';

/// Thrown when a task cannot be dispatched (unknown kind, duplicate id).
class SubagentDispatchException implements Exception {
  SubagentDispatchException(this.message);

  final String message;

  @override
  String toString() => 'SubagentDispatchException: $message';
}

/// Thrown when a command violates the execution allow-list.
class CommandRejectedException implements Exception {
  CommandRejectedException(this.message);

  final String message;

  @override
  String toString() => 'CommandRejectedException: $message';
}

/// The capabilities a subagent can provide.
enum SubagentKind {
  fileParsing,
  systemCommand,
  codeVerification,
  custom,
}

/// A unit of work handed to a subagent.
class SubagentTask {
  const SubagentTask({
    required this.id,
    required this.description,
    required this.kind,
    this.args = const <String, Object?>{},
    this.timeout = const Duration(seconds: 30),
    this.priority = 0,
  });

  /// Unique identifier, referenced by results and the thinking trace.
  final String id;

  /// Human-readable intent, shown in the UI thinking panel.
  final String description;

  /// Which subagent should handle this task.
  final SubagentKind kind;

  /// Subagent-specific arguments.
  final Map<String, Object?> args;

  /// Hard deadline for the task.
  final Duration timeout;

  /// Higher values are dispatched first when the pool is saturated.
  final int priority;

  Map<String, Object?> toJson() => <String, Object?>{
        'id': id,
        'description': description,
        'kind': kind.name,
        'args': args,
        'timeout_ms': timeout.inMilliseconds,
        'priority': priority,
      };

  SubagentTask copyWith({
    String? id,
    String? description,
    SubagentKind? kind,
    Map<String, Object?>? args,
    Duration? timeout,
    int? priority,
  }) =>
      SubagentTask(
        id: id ?? this.id,
        description: description ?? this.description,
        kind: kind ?? this.kind,
        args: args ?? this.args,
        timeout: timeout ?? this.timeout,
        priority: priority ?? this.priority,
      );
}

/// Outcome of a dispatched task. Failures are represented, not thrown.
class SubagentResult {
  const SubagentResult({
    required this.taskId,
    required this.success,
    required this.summary,
    required this.duration,
    this.output,
    this.error,
    this.metadata = const <String, Object?>{},
  });

  /// Id of the originating task.
  final String taskId;

  /// Whether the subagent completed without error.
  final bool success;

  /// One-line, model-consumable description of what happened.
  final String summary;

  /// Wall-clock time spent on the task.
  final Duration duration;

  /// Structured payload on success.
  final Object? output;

  /// Error message on failure.
  final String? error;

  /// Extra structured facts (counts, paths, exit codes).
  final Map<String, Object?> metadata;

  /// Convenience failure constructor.
  factory SubagentResult.failure({
    required String taskId,
    required String summary,
    required String error,
    required Duration duration,
    Map<String, Object?> metadata = const <String, Object?>{},
  }) =>
      SubagentResult(
        taskId: taskId,
        success: false,
        summary: summary,
        duration: duration,
        error: error,
        metadata: metadata,
      );

  Map<String, Object?> toJson() => <String, Object?>{
        'task_id': taskId,
        'success': success,
        'summary': summary,
        'duration_ms': duration.inMilliseconds,
        'output': output,
        'error': error,
        'metadata': metadata,
      };
}

/// Cooperative cancellation token handed to subagents.
class CancellationToken {
  bool _cancelled = false;
  String? _reason;

  bool get isCancelled => _cancelled;
  String? get reason => _reason;

  void cancel([String reason = 'cancelled by orchestrator']) {
    _cancelled = true;
    _reason = reason;
  }

  /// Throws [SubagentDispatchException] if cancellation was requested.
  void throwIfCancelled() {
    if (_cancelled) {
      throw SubagentDispatchException(_reason ?? 'task cancelled');
    }
  }
}

/// Ambient services available to a subagent while it runs.
class SubagentContext {
  SubagentContext({
    required this.token,
    required this.taskId,
    this.onLog,
    this.reader,
  });

  final CancellationToken token;
  final String taskId;

  /// Optional progress sink, surfaced in the agent trace.
  final void Function(String message)? onLog;

  /// Shared universal file reader (limits are configured once).
  final UniversalFileReader? reader;

  void log(String message) => onLog?.call(message);
}

/// Contract implemented by every subagent.
abstract class Subagent {
  /// Stable name, e.g. `file_parsing`.
  String get name;

  /// Kind this subagent handles.
  SubagentKind get kind;

  /// One-line capability description used in planning prompts.
  String get capability;

  /// Executes [task]. Implementations may throw; the runner converts throws
  /// into failed [SubagentResult]s.
  Future<SubagentResult> execute(SubagentTask task, SubagentContext context);
}

/// `file_parsing` — reads files, directories, or a whole codebase through the
/// [UniversalFileReader], covering text, code, binary headers, archives and
/// structured data.
class FileParsingSubagent implements Subagent {
  FileParsingSubagent({UniversalFileReader? reader})
      : _reader = reader ?? UniversalFileReader();

  final UniversalFileReader _reader;

  @override
  String get name => 'file_parsing';

  @override
  SubagentKind get kind => SubagentKind.fileParsing;

  @override
  String get capability =>
      'Read and parse text, source code, binary headers, archives and '
      'structured data (json/yaml/csv/ini/xml) from disk.';

  @override
  Future<SubagentResult> execute(
    SubagentTask task,
    SubagentContext context,
  ) async {
    final Stopwatch sw = Stopwatch()..start();
    final Object? pathArg = task.args['path'];
    final Object? dirArg = task.args['directory'];
    final Object? pathsArg = task.args['paths'];
    context.token.throwIfCancelled();

    try {
      if (pathsArg is List) {
        final List<String> paths =
            pathsArg.map((Object? p) => p.toString()).toList(growable: false);
        final List<FileReadResult> results = <FileReadResult>[];
        for (final String p in paths) {
          context.token.throwIfCancelled();
          context.log('reading $p');
          results.add(await _reader.read(p));
        }
        sw.stop();
        return SubagentResult(
          taskId: task.id,
          success: true,
          summary: 'Parsed ${results.length} file(s)',
          duration: sw.elapsed,
          output: results.map((FileReadResult r) => r.toJson()).toList(growable: false),
          metadata: <String, Object?>{
            'file_count': results.length,
            'total_bytes':
                results.fold<int>(0, (int a, FileReadResult r) => a + r.sizeBytes),
          },
        );
      }

      if (dirArg is String && dirArg.isNotEmpty) {
        final Set<String>? extensions = _stringSet(task.args['extensions']);
        final CodebaseReadResult walk = await _reader.readCodebase(
          dirArg,
          extensions: extensions,
          maxFiles: _intArg(task.args['max_files'], 200),
        );
        sw.stop();
        return SubagentResult(
          taskId: task.id,
          success: true,
          summary: 'Walked ${walk.files.length} file(s) under $dirArg',
          duration: sw.elapsed,
          output: walk.toJson(),
          metadata: <String, Object?>{
            'file_count': walk.files.length,
            'total_bytes': walk.totalBytes,
            'languages': walk.languageHistogram,
            'warning_count': walk.warnings.length,
          },
        );
      }

      if (pathArg is String && pathArg.isNotEmpty) {
        context.log('reading $pathArg');
        final FileReadResult result = await _reader.read(pathArg);
        sw.stop();
        return SubagentResult(
          taskId: task.id,
          success: true,
          summary: result.toPromptSummary().trim(),
          duration: sw.elapsed,
          output: result.toJson(),
          metadata: <String, Object?>{
            'kind': result.kind.name,
            'size_bytes': result.sizeBytes,
            'language': result.language,
            'truncated': result.isTruncated,
          },
        );
      }

      throw SubagentDispatchException(
        'file_parsing requires one of "path", "paths" or "directory"',
      );
    } on FileReadException catch (e) {
      sw.stop();
      return SubagentResult.failure(
        taskId: task.id,
        summary: 'Failed to read file',
        error: e.toString(),
        duration: sw.elapsed,
      );
    } on ParseFailureException catch (e) {
      sw.stop();
      return SubagentResult.failure(
        taskId: task.id,
        summary: 'Failed to parse file',
        error: e.toString(),
        duration: sw.elapsed,
      );
    } on FileReadLimitException catch (e) {
      sw.stop();
      return SubagentResult.failure(
        taskId: task.id,
        summary: 'File exceeded safety limits',
        error: e.toString(),
        duration: sw.elapsed,
      );
    }
  }
}

/// `system_command` — runs allow-listed executables with arguments passed
/// directly to the OS (never through a shell).
class SystemCommandSubagent implements Subagent {
  SystemCommandSubagent({
    Set<String>? allowedExecutables,
    this.workingDirectory,
    this.inheritEnvironment = true,
  }) : allowedExecutables =
            allowedExecutables ?? defaultAllowedExecutables;

  /// Executables that may be invoked. Anything else is rejected.
  final Set<String> allowedExecutables;

  /// Default working directory when a task does not supply one.
  final String? workingDirectory;

  /// Whether the child inherits the parent environment.
  final bool inheritEnvironment;

  /// Conservative default allow-list of read-only / build tooling.
  static const Set<String> defaultAllowedExecutables = <String>{
    'ls',
    'pwd',
    'cat',
    'head',
    'tail',
    'wc',
    'file',
    'stat',
    'du',
    'find',
    'grep',
    'rg',
    'sed',
    'awk',
    'sort',
    'uniq',
    'diff',
    'echo',
    'env',
    'uname',
    'whoami',
    'date',
    'sha256sum',
    'md5sum',
    'unzip',
    'zipinfo',
    'tar',
    'gzip',
    'git',
    'dart',
    'flutter',
    'python3',
    'node',
    'java',
    'javac',
    'gradle',
    'adb',
  };

  /// Characters that would let an argument escape into shell interpretation.
  static final RegExp _shellMetacharacters =
      RegExp(r'[;&|`$><\n\r\\]|\$\(|&&|\|\|');

  @override
  String get name => 'system_command';

  @override
  SubagentKind get kind => SubagentKind.systemCommand;

  @override
  String get capability =>
      'Execute an allow-listed executable with explicit arguments (no shell). '
      'Useful for listing directories, inspecting archives and running builds.';

  /// Validates a command + argument vector against the allow-list.
  ///
  /// Throws [CommandRejectedException] with a precise reason on rejection.
  void validate(String executable, List<String> args) {
    final String base = executable.split(RegExp(r'[/\\]')).last;
    if (!allowedExecutables.contains(base)) {
      throw CommandRejectedException(
        'executable "$base" is not in the allow-list '
        '(${allowedExecutables.length} permitted)',
      );
    }
    if (executable.contains(RegExp(r'[/\\]')) &&
        (executable.contains('..') || executable.startsWith('/'))) {
      throw CommandRejectedException(
        'executable paths must be bare names, got "$executable"',
      );
    }
    for (final String a in args) {
      if (_shellMetacharacters.hasMatch(a)) {
        throw CommandRejectedException(
          'argument "$a" contains shell metacharacters and was rejected',
        );
      }
    }
  }

  @override
  Future<SubagentResult> execute(
    SubagentTask task,
    SubagentContext context,
  ) async {
    final Stopwatch sw = Stopwatch()..start();
    final Object? cmdArg = task.args['command'];
    if (cmdArg is! String || cmdArg.trim().isEmpty) {
      sw.stop();
      return SubagentResult.failure(
        taskId: task.id,
        summary: 'Missing "command" argument',
        error: 'system_command requires a non-empty "command" string',
        duration: sw.elapsed,
      );
    }
    final List<String> args = (task.args['args'] as List<Object?>? ?? const <Object?>[])
        .map((Object? a) => a.toString())
        .toList(growable: false);
    final Object? cwdArg = task.args['working_directory'];

    try {
      validate(cmdArg, args);
      context.token.throwIfCancelled();
      context.log('executing $cmdArg ${args.join(' ')}');

      final ProcessResult result = await Process.run(
        cmdArg,
        args,
        workingDirectory:
            (cwdArg is String && cwdArg.isNotEmpty) ? cwdArg : workingDirectory,
        runInShell: false,
        includeParentEnvironment: inheritEnvironment,
      );
      sw.stop();
      final String stdout = result.stdout is String
          ? result.stdout as String
          : utf8.decode(result.stdout as List<int>, allowMalformed: true);
      final String stderr = result.stderr is String
          ? result.stderr as String
          : utf8.decode(result.stderr as List<int>, allowMalformed: true);
      final bool ok = result.exitCode == 0;
      return SubagentResult(
        taskId: task.id,
        success: ok,
        summary: ok
            ? '$cmdArg exited 0 with ${stdout.length} bytes of stdout'
            : '$cmdArg exited ${result.exitCode}',
        duration: sw.elapsed,
        output: <String, Object?>{
          'stdout': _truncate(stdout, 8000),
          'stderr': _truncate(stderr, 4000),
        },
        error: ok ? null : 'exit code ${result.exitCode}',
        metadata: <String, Object?>{
          'executable': cmdArg,
          'args': args,
          'exit_code': result.exitCode,
          'stdout_bytes': stdout.length,
          'stderr_bytes': stderr.length,
        },
      );
    } on CommandRejectedException catch (e) {
      sw.stop();
      return SubagentResult.failure(
        taskId: task.id,
        summary: 'Command rejected by the execution allow-list',
        error: e.toString(),
        duration: sw.elapsed,
      );
    } on ProcessException catch (e) {
      sw.stop();
      return SubagentResult.failure(
        taskId: task.id,
        summary: 'Failed to spawn $cmdArg',
        error: e.toString(),
        duration: sw.elapsed,
        metadata: <String, Object?>{'executable': cmdArg},
      );
    }
  }

  static String _truncate(String s, int max) =>
      s.length <= max ? s : '${s.substring(0, max)}\n…[truncated]';
}

/// One static finding produced by [CodeVerificationSubagent].
class VerificationFinding {
  const VerificationFinding({
    required this.rule,
    required this.severity,
    required this.message,
    this.line,
    this.path,
  });

  final String rule;

  /// `error`, `warning` or `info`.
  final String severity;
  final String message;
  final int? line;
  final String? path;

  Map<String, Object?> toJson() => <String, Object?>{
        'rule': rule,
        'severity': severity,
        'message': message,
        'line': line,
        'path': path,
      };
}

/// `code_verification` — deterministic static checks over source text:
/// delimiter balance, trailing whitespace, tab indentation, missing final
/// newline, over-long lines, leftover conflict markers and TODO/FIXME markers.
class CodeVerificationSubagent implements Subagent {
  CodeVerificationSubagent({this.maxLineLength = 120});

  /// Lines longer than this are reported.
  final int maxLineLength;

  @override
  String get name => 'code_verification';

  @override
  SubagentKind get kind => SubagentKind.codeVerification;

  @override
  String get capability =>
      'Statically verify source files: balanced delimiters, whitespace, long '
      'lines, merge-conflict markers and TODO/FIXME markers.';

  @override
  Future<SubagentResult> execute(
    SubagentTask task,
    SubagentContext context,
  ) async {
    final Stopwatch sw = Stopwatch()..start();
    final UniversalFileReader reader = context.reader ?? UniversalFileReader();
    final Object? pathArg = task.args['path'];
    final Object? codeArg = task.args['code'];
    final Object? dirArg = task.args['directory'];

    try {
      final List<VerificationFinding> findings = <VerificationFinding>[];
      int filesChecked = 0;

      if (codeArg is String) {
        context.token.throwIfCancelled();
        findings.addAll(_verifySource(codeArg, task.args['path']?.toString()));
        filesChecked = 1;
      } else if (pathArg is String && pathArg.isNotEmpty) {
        context.token.throwIfCancelled();
        final FileReadResult r = await reader.read(pathArg);
        final String? text = r.text;
        if (text == null) {
          sw.stop();
          return SubagentResult.failure(
            taskId: task.id,
            summary: 'Not a text source file',
            error: 'code_verification requires decodable text, got '
                '${r.kind.name}',
            duration: sw.elapsed,
          );
        }
        findings.addAll(_verifySource(text, pathArg));
        filesChecked = 1;
      } else if (dirArg is String && dirArg.isNotEmpty) {
        final CodebaseReadResult walk = await reader.readCodebase(
          dirArg,
          extensions: _stringSet(task.args['extensions']),
          maxFiles: _intArg(task.args['max_files'], 100),
        );
        for (final FileReadResult f in walk.files) {
          context.token.throwIfCancelled();
          final String? text = f.text;
          if (text == null) {
            continue;
          }
          findings.addAll(_verifySource(text, f.path));
          filesChecked++;
        }
      } else {
        throw SubagentDispatchException(
          'code_verification requires "code", "path" or "directory"',
        );
      }

      final int errors = findings
          .where((VerificationFinding f) => f.severity == 'error')
          .length;
      final int warnings = findings
          .where((VerificationFinding f) => f.severity == 'warning')
          .length;
      sw.stop();
      return SubagentResult(
        taskId: task.id,
        success: errors == 0,
        summary: errors == 0
            ? 'Verified $filesChecked file(s): no errors, $warnings warning(s)'
            : 'Verified $filesChecked file(s): $errors error(s), '
                '$warnings warning(s)',
        duration: sw.elapsed,
        output: findings
            .map((VerificationFinding f) => f.toJson())
            .toList(growable: false),
        error: errors == 0 ? null : '$errors verification error(s) found',
        metadata: <String, Object?>{
          'files_checked': filesChecked,
          'error_count': errors,
          'warning_count': warnings,
          'finding_count': findings.length,
        },
      );
    } on SubagentDispatchException catch (e) {
      sw.stop();
      return SubagentResult.failure(
        taskId: task.id,
        summary: 'Invalid verification task',
        error: e.toString(),
        duration: sw.elapsed,
      );
    } on FileReadException catch (e) {
      sw.stop();
      return SubagentResult.failure(
        taskId: task.id,
        summary: 'Could not read source for verification',
        error: e.toString(),
        duration: sw.elapsed,
      );
    }
  }

  List<VerificationFinding> _verifySource(String source, String? path) {
    final List<VerificationFinding> out = <VerificationFinding>[];
    final List<String> lines = source.split('\n');

    // Strip string literals crudely so delimiters inside strings do not count.
    String sanitised = source
        .replaceAll(RegExp(r'"""(?:.|\n)*?"""'), '""')
        .replaceAll(RegExp(r"'''(?:.|\n)*?'''"), "''")
        .replaceAll(RegExp(r'"(?:[^"\\\n]|\\.)*"'), '""')
        .replaceAll(RegExp(r"'(?:[^'\\\n]|\\.)*'"), "''")
        .replaceAll(RegExp(r'//[^\n]*'), '')
        .replaceAll(RegExp(r'/\*(?:.|\n)*?\*/'), '');

    for (final MapEntry<String, String> pair in <String, String>{
      '()': '()',
      '[]': '[]',
      '{}': '{}',
    }.entries) {
      final String open = pair.key[0];
      final String close = pair.key[1];
      int depth = 0;
      int minDepth = 0;
      for (final int code in sanitised.codeUnits) {
        if (code == open.codeUnitAt(0)) {
          depth++;
        } else if (code == close.codeUnitAt(0)) {
          depth--;
          if (depth < minDepth) {
            minDepth = depth;
          }
        }
      }
      if (depth != 0 || minDepth < 0) {
        out.add(
          VerificationFinding(
            rule: 'balanced_delimiters',
            severity: 'error',
            message: 'unbalanced "$open$close" (net depth $depth)',
            path: path,
          ),
        );
      }
    }

    for (int i = 0; i < lines.length; i++) {
      final String line = lines[i];
      final int lineNo = i + 1;
      if (line.contains('<<<<<<<') ||
          line.contains('>>>>>>>') ||
          line.endsWith('=======')) {
        out.add(
          VerificationFinding(
            rule: 'merge_conflict_marker',
            severity: 'error',
            message: 'unresolved merge conflict marker',
            line: lineNo,
            path: path,
          ),
        );
      }
      if (line.length > maxLineLength) {
        out.add(
          VerificationFinding(
            rule: 'max_line_length',
            severity: 'warning',
            message: 'line is ${line.length} characters '
                '(limit $maxLineLength)',
            line: lineNo,
            path: path,
          ),
        );
      }
      if (line != line.trimRight()) {
        out.add(
          VerificationFinding(
            rule: 'trailing_whitespace',
            severity: 'info',
            message: 'trailing whitespace',
            line: lineNo,
            path: path,
          ),
        );
      }
      if (line.startsWith('\t')) {
        out.add(
          VerificationFinding(
            rule: 'tab_indentation',
            severity: 'info',
            message: 'tab-indented line',
            line: lineNo,
            path: path,
          ),
        );
      }
      if (RegExp(r'\b(TODO|FIXME|XXX|HACK)\b').hasMatch(line)) {
        out.add(
          VerificationFinding(
            rule: 'unresolved_marker',
            severity: 'warning',
            message: 'unresolved ${RegExp(r'\b(TODO|FIXME|XXX|HACK)\b').firstMatch(line)?.group(1)} marker',
            line: lineNo,
            path: path,
          ),
        );
      }
    }

    if (source.isNotEmpty && !source.endsWith('\n')) {
      out.add(
        VerificationFinding(
          rule: 'missing_final_newline',
          severity: 'info',
          message: 'file does not end with a newline',
          line: lines.length,
          path: path,
        ),
      );
    }
    return out;
  }
}

/// A task submitted to the runner together with its completion handle.
class _PendingTask {
  _PendingTask(this.task, this.completer, this.token);

  final SubagentTask task;
  final Completer<SubagentResult> completer;
  final CancellationToken token;
}

/// Bounded-concurrency dispatcher for subagents.
class SubagentRunner {
  SubagentRunner({
    this.maxConcurrency = 4,
    List<Subagent>? subagents,
    UniversalFileReader? reader,
    this.onEvent,
  }) : _reader = reader ?? UniversalFileReader() {
    if (maxConcurrency < 1) {
      throw SubagentDispatchException('maxConcurrency must be at least 1');
    }
    for (final Subagent s in subagents ?? _defaultSubagents()) {
      register(s);
    }
  }

  /// Maximum number of tasks executing at once.
  final int maxConcurrency;

  final UniversalFileReader _reader;

  /// Optional observer for lifecycle events (used by the thinking UI).
  final void Function(String message, Map<String, Object?> details)? onEvent;

  final Map<SubagentKind, Subagent> _subagents = <SubagentKind, Subagent>{};
  final List<_PendingTask> _queue = <_PendingTask>[];
  final Set<CancellationToken> _liveTokens = <CancellationToken>{};

  int _running = 0;
  int _completed = 0;
  int _failed = 0;
  int _sequence = 0;

  static List<Subagent> _defaultSubagents() => <Subagent>[
        FileParsingSubagent(),
        SystemCommandSubagent(),
        CodeVerificationSubagent(),
      ];

  /// Registers (or replaces) the subagent for a kind.
  void register(Subagent subagent) {
    _subagents[subagent.kind] = subagent;
  }

  /// Whether a subagent is registered for [kind].
  bool supports(SubagentKind kind) => _subagents.containsKey(kind);

  /// Registered subagents.
  List<Subagent> get subagents => _subagents.values.toList(growable: false);

  /// Number of tasks currently executing.
  int get activeCount => _running;

  /// Number of queued tasks.
  int get queuedCount => _queue.length;

  /// Tasks finished successfully.
  int get completedCount => _completed;

  /// Tasks finished with failure.
  int get failedCount => _failed;

  /// Generates a unique task id with the given prefix.
  String nextTaskId([String prefix = 'task']) {
    _sequence++;
    return '${prefix}_${_sequence.toString().padLeft(3, '0')}';
  }

  /// Dispatches a single task and awaits its result.
  Future<SubagentResult> run(SubagentTask task) {
    final CancellationToken token = CancellationToken();
    final Completer<SubagentResult> completer = Completer<SubagentResult>();
    _liveTokens.add(token);
    _queue.add(_PendingTask(task, completer, token));
    _pump();
    return completer.future;
  }

  /// Dispatches [tasks] concurrently and returns results in *submission* order.
  Future<List<SubagentResult>> runAll(List<SubagentTask> tasks) async {
    if (tasks.isEmpty) {
      return const <SubagentResult>[];
    }
    final List<Future<SubagentResult>> futures =
        tasks.map(run).toList(growable: false);
    return Future.wait<SubagentResult>(futures);
  }

  /// Dispatches [tasks] and emits each result as it finishes, in completion
  /// order. Each task's result is emitted exactly once.
  Stream<SubagentResult> runStreaming(List<SubagentTask> tasks) {
    late StreamController<SubagentResult> controller;
    int outstanding = 0;
    bool closed = false;

    void maybeClose() {
      if (!closed && outstanding == 0) {
        closed = true;
        unawaited(controller.close());
      }
    }

    controller = StreamController<SubagentResult>(
      onListen: () {
        if (tasks.isEmpty) {
          maybeClose();
          return;
        }
        outstanding = tasks.length;
        for (final SubagentTask task in tasks) {
          unawaited(
            run(task).then((SubagentResult result) {
              if (!controller.isClosed) {
                controller.add(result);
              }
              outstanding--;
              maybeClose();
            }),
          );
        }
      },
      onCancel: () {
        cancelAll('stream subscription cancelled');
      },
    );
    return controller.stream;
  }

  /// Cancels every queued and in-flight task.
  void cancelAll([String reason = 'cancelled by orchestrator']) {
    for (final CancellationToken t in _liveTokens) {
      t.cancel(reason);
    }
    for (final _PendingTask p in List<_PendingTask>.from(_queue)) {
      if (!p.completer.isCompleted) {
        p.completer.complete(
          SubagentResult.failure(
            taskId: p.task.id,
            summary: 'Task cancelled before execution',
            error: reason,
            duration: Duration.zero,
          ),
        );
      }
    }
    _queue.clear();
  }

  /// Resets the counters (does not cancel live work).
  void resetStatistics() {
    _completed = 0;
    _failed = 0;
  }

  void _pump() {
    while (_running < maxConcurrency && _queue.isNotEmpty) {
      // Highest priority first, stable on submission order.
      int bestIndex = 0;
      for (int i = 1; i < _queue.length; i++) {
        if (_queue[i].task.priority > _queue[bestIndex].task.priority) {
          bestIndex = i;
        }
      }
      final _PendingTask next = _queue.removeAt(bestIndex);
      _running++;
      unawaited(_execute(next));
    }
  }

  Future<void> _execute(_PendingTask pending) async {
    final SubagentTask task = pending.task;
    final Subagent? subagent = _subagents[task.kind];
    final Stopwatch sw = Stopwatch()..start();

    onEvent?.call('subagent_started', <String, Object?>{
      'task_id': task.id,
      'kind': task.kind.name,
      'description': task.description,
      'active': _running,
      'queued': _queue.length,
    });

    SubagentResult result;
    if (subagent == null) {
      sw.stop();
      result = SubagentResult.failure(
        taskId: task.id,
        summary: 'No subagent registered for ${task.kind.name}',
        error: 'unregistered subagent kind: ${task.kind.name}',
        duration: sw.elapsed,
        metadata: <String, Object?>{
          'available': _subagents.keys.map((SubagentKind k) => k.name).toList(),
        },
      );
    } else {
      final SubagentContext context = SubagentContext(
        token: pending.token,
        taskId: task.id,
        reader: _reader,
        onLog: (String message) => onEvent?.call('subagent_log', <String, Object?>{
          'task_id': task.id,
          'message': message,
        }),
      );
      try {
        result = await subagent
            .execute(task, context)
            .timeout(task.timeout, onTimeout: () {
          throw TimeoutException(
            'subagent ${subagent.name} exceeded ${task.timeout.inMilliseconds}ms',
          );
        });
        if (sw.elapsed > task.timeout && result.success) {
          result = SubagentResult(
            taskId: result.taskId,
            success: false,
            summary: 'Task finished after its deadline',
            duration: result.duration,
            output: result.output,
            error: 'deadline exceeded',
            metadata: result.metadata,
          );
        }
      } on TimeoutException catch (e) {
        result = SubagentResult.failure(
          taskId: task.id,
          summary: 'Task timed out after ${task.timeout.inSeconds}s',
          error: e.message ?? 'timeout',
          duration: sw.elapsed,
          metadata: <String, Object?>{'timeout_ms': task.timeout.inMilliseconds},
        );
      } on SubagentDispatchException catch (e) {
        result = SubagentResult.failure(
          taskId: task.id,
          summary: 'Task could not be dispatched',
          error: e.toString(),
          duration: sw.elapsed,
        );
      } on Object catch (e, stack) {
        result = SubagentResult.failure(
          taskId: task.id,
          summary: 'Subagent threw an unexpected error',
          error: '$e',
          duration: sw.elapsed,
          metadata: <String, Object?>{'stack': stack.toString().split('\n').take(5).join('\n')},
        );
      }
    }

    sw.stop();
    if (result.success) {
      _completed++;
    } else {
      _failed++;
    }
    _liveTokens.remove(pending.token);
    _running--;

    onEvent?.call('subagent_finished', <String, Object?>{
      'task_id': task.id,
      'success': result.success,
      'summary': result.summary,
      'duration_ms': result.duration.inMilliseconds,
    });

    if (!pending.completer.isCompleted) {
      pending.completer.complete(result);
    }
    _pump();
  }

  /// Diagnostic snapshot for the engine status screen.
  Map<String, Object?> describe() => <String, Object?>{
        'architecture': 'subagent_runner',
        'max_concurrency': maxConcurrency,
        'registered': _subagents.entries
            .map((MapEntry<SubagentKind, Subagent> e) => <String, Object?>{
                  'kind': e.key.name,
                  'name': e.value.name,
                  'capability': e.value.capability,
                })
            .toList(growable: false),
        'active': _running,
        'queued': _queue.length,
        'completed': _completed,
        'failed': _failed,
      };
}

Set<String>? _stringSet(Object? value) {
  if (value is List) {
    return value
        .map((Object? v) => v.toString().toLowerCase().replaceAll('.', ''))
        .toSet();
  }
  if (value is String && value.isNotEmpty) {
    return value
        .split(',')
        .map((String s) => s.trim().toLowerCase().replaceAll('.', ''))
        .where((String s) => s.isNotEmpty)
        .toSet();
  }
  return null;
}

int _intArg(Object? value, int fallback) {
  if (value is int) {
    return value;
  }
  if (value is num) {
    return value.toInt();
  }
  if (value is String) {
    return int.tryParse(value) ?? fallback;
  }
  return fallback;
}

/// Utility retained for callers that need a deterministic pseudo-random
/// ordering of tasks without pulling in a dependency.
int stableTaskHash(String id) => id.codeUnits.fold<int>(
      7,
      (int acc, int c) => (acc * 31 + c) & 0x7FFFFFFF,
    );

/// Clamps a concurrency request to the device's available parallelism.
int recommendedConcurrency({int? availableProcessors}) {
  final int cores = availableProcessors ?? Platform.numberOfProcessors;
  return math.max(2, math.min(8, cores));
}