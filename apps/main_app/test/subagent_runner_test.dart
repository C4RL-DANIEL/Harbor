// Unit tests for the subagent execution layer in
// core/agent/subagent_runner.dart.
//
// All fixtures live under a per-test Directory.systemTemp root. Command
// execution tests only use binaries that exist everywhere (`echo`, `true`) or
// an intentionally non-existent name on an injected allow-list, so they are
// hermetic.

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

import 'package:main_app/core/agent/subagent_runner.dart';

/// A custom subagent that records how many invocations overlap and in what
/// order they were entered, optionally pausing to simulate work.
class _RecordingSubagent implements Subagent {
  _RecordingSubagent({this.delay = Duration.zero});

  final Duration delay;

  int active = 0;
  int peak = 0;
  final List<String> order = <String>[];

  @override
  String get name => 'recording';

  @override
  SubagentKind get kind => SubagentKind.custom;

  @override
  String get capability => 'Test subagent that records concurrency.';

  @override
  Future<SubagentResult> execute(SubagentTask task, SubagentContext context) async {
    active++;
    if (active > peak) {
      peak = active;
    }
    order.add(task.id);
    if (delay > Duration.zero) {
      await Future<void>.delayed(delay);
    }
    active--;
    if (task.args['fail'] == true) {
      return SubagentResult.failure(
        taskId: task.id,
        summary: 'forced failure',
        error: 'forced',
        duration: Duration.zero,
      );
    }
    return SubagentResult(
      taskId: task.id,
      success: true,
      summary: 'ok',
      duration: Duration.zero,
      output: <String, Object?>{'id': task.id},
    );
  }
}

/// A custom subagent that always throws, to prove the runner converts throws.
class _ThrowingSubagent implements Subagent {
  @override
  String get name => 'throwing';

  @override
  SubagentKind get kind => SubagentKind.custom;

  @override
  String get capability => 'Test subagent that throws.';

  @override
  Future<SubagentResult> execute(SubagentTask task, SubagentContext context) async {
    throw StateError('boom');
  }
}

SubagentTask customTask(
  String id, {
  bool fail = false,
  int priority = 0,
  Duration timeout = const Duration(seconds: 5),
}) =>
    SubagentTask(
      id: id,
      description: 'custom task $id',
      kind: SubagentKind.custom,
      args: <String, Object?>{'fail': fail},
      priority: priority,
      timeout: timeout,
    );

List<Map<String, Object?>> findingsOf(SubagentResult result) =>
    (result.output! as List<Object?>)
        .map((Object? finding) => finding as Map<String, Object?>)
        .toList(growable: false);

Set<String> rulesOf(SubagentResult result) =>
    findingsOf(result).map((Map<String, Object?> f) => f['rule']! as String).toSet();

void main() {
  late Directory root;

  setUp(() async {
    root = await Directory.systemTemp.createTemp('harbor_subagent_');
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

  group('SubagentRunner defaults', () {
    test('supports the three built-in kinds and describes them', () {
      final SubagentRunner runner = SubagentRunner();

      expect(runner.supports(SubagentKind.fileParsing), isTrue);
      expect(runner.supports(SubagentKind.systemCommand), isTrue);
      expect(runner.supports(SubagentKind.codeVerification), isTrue);
      expect(runner.supports(SubagentKind.custom), isFalse);
      expect(runner.subagents, hasLength(3));

      final Map<String, Object?> info = runner.describe();
      expect(info['architecture'], 'subagent_runner');
      expect(info['max_concurrency'], 4);
      final List<Object?> registered = info['registered']! as List<Object?>;
      final Set<Object?> kinds = registered
          .map((Object? entry) => (entry as Map<String, Object?>)['kind'])
          .toSet();
      expect(
        kinds,
        containsAll(<String>['fileParsing', 'systemCommand', 'codeVerification']),
      );

      // `name` is the stable snake_case identifier documented on Subagent,
      // distinct from the SubagentKind enum name reported as `kind`.
      final Set<Object?> names = registered
          .map((Object? entry) => (entry as Map<String, Object?>)['name'])
          .toSet();
      expect(
        names,
        containsAll(
          <String>['file_parsing', 'system_command', 'code_verification'],
        ),
      );
    });

    test('nextTaskId produces monotonically increasing ids', () {
      final SubagentRunner runner = SubagentRunner();
      expect(runner.nextTaskId('plan'), 'plan_001');
      expect(runner.nextTaskId('plan'), 'plan_002');
      expect(runner.nextTaskId('other'), 'other_003');
    });
  });

  group('FileParsingSubagent', () {
    test('reads a single file and reports kind and size', () async {
      final File file = writeText('note.txt', 'alpha\nbeta\n');
      final SubagentResult result = await SubagentRunner().run(
        SubagentTask(
          id: 'fp_single',
          description: 'parse note.txt',
          kind: SubagentKind.fileParsing,
          args: <String, Object?>{'path': file.path},
        ),
      );

      expect(result.success, isTrue);
      expect(result.metadata['kind'], 'text');
      expect(result.metadata['size_bytes']! as int, greaterThan(0));
      expect(result.output, isNotNull);
      final Map<String, Object?> output = result.output! as Map<String, Object?>;
      expect(output['path'], file.path);
      expect(output['line_count'], 2);
    });

    test('reads every path in a paths list', () async {
      final File first = writeText('one.txt', 'one\n');
      final File second = writeText('two.txt', 'two\n');
      final SubagentResult result = await SubagentRunner().run(
        SubagentTask(
          id: 'fp_paths',
          description: 'parse two files',
          kind: SubagentKind.fileParsing,
          args: <String, Object?>{
            'paths': <String>[first.path, second.path],
          },
        ),
      );

      expect(result.success, isTrue);
      expect(result.metadata['file_count'], 2);
      expect(result.metadata['total_bytes'], first.lengthSync() + second.lengthSync());
      expect(result.output! as List<Object?>, hasLength(2));
    });

    test('walks a directory argument', () async {
      writeText('src/a.dart', 'void a() {}\n');
      writeText('src/b.dart', 'void b() {}\n');
      final SubagentResult result = await SubagentRunner().run(
        SubagentTask(
          id: 'fp_dir',
          description: 'walk the tree',
          kind: SubagentKind.fileParsing,
          args: <String, Object?>{'directory': root.path},
        ),
      );

      expect(result.success, isTrue);
      expect(result.metadata['file_count'], 2);
      expect(result.metadata['languages'], <String, int>{'dart': 2});
    });

    test('fails cleanly when no path, paths or directory is given', () async {
      final SubagentResult result = await SubagentRunner().run(
        const SubagentTask(
          id: 'fp_missing',
          description: 'no target',
          kind: SubagentKind.fileParsing,
        ),
      );

      expect(result.success, isFalse);
      expect(result.summary, contains('dispatched'));
      expect(result.error, contains('requires one of "path"'));
    });
  });

  group('SystemCommandSubagent allow-list', () {
    test('rejects a non-allow-listed executable', () {
      final SystemCommandSubagent command = SystemCommandSubagent();
      expect(
        () => command.validate('rm', <String>['-rf', '/']),
        throwsA(isA<CommandRejectedException>()),
      );
    });

    test('rejects every shell metacharacter argument before spawning', () {
      final SystemCommandSubagent command = SystemCommandSubagent();
      final List<String> dangerous = <String>[
        'a;b',
        'a|b',
        r'a$b',
        'a>b',
        'a`b`',
        'a&&b',
        'a\nb',
        r'$(id)',
      ];
      for (final String argument in dangerous) {
        expect(
          () => command.validate('ls', <String>[argument]),
          throwsA(isA<CommandRejectedException>()),
          reason: argument,
        );
      }
    });

    test('allows a plain argument vector', () {
      final SystemCommandSubagent command = SystemCommandSubagent();
      expect(() => command.validate('ls', <String>['-la']), returnsNormally);
      expect(() => command.validate('echo', <String>['hello']), returnsNormally);
    });

    test('rejects path-style executables as bare-name violations', () {
      final SystemCommandSubagent command = SystemCommandSubagent();
      expect(
        () => command.validate('/bin/ls', <String>[]),
        throwsA(isA<CommandRejectedException>()),
      );
      expect(
        () => command.validate('../ls', <String>[]),
        throwsA(isA<CommandRejectedException>()),
      );
    });

    test('runs an allow-listed command through the runner', () async {
      final SubagentResult result = await SubagentRunner().run(
        const SubagentTask(
          id: 'sc_echo',
          description: 'echo hello',
          kind: SubagentKind.systemCommand,
          args: <String, Object?>{
            'command': 'echo',
            'args': <String>['hello'],
          },
        ),
      );

      expect(result.success, isTrue);
      expect(result.metadata['exit_code'], 0);
      final Map<String, Object?> output = result.output! as Map<String, Object?>;
      expect(output['stdout'], contains('hello'));
    });

    test('rejects a disallowed command through the runner', () async {
      final SubagentResult result = await SubagentRunner().run(
        const SubagentTask(
          id: 'sc_rm',
          description: 'delete the world',
          kind: SubagentKind.systemCommand,
          args: <String, Object?>{'command': 'rm'},
        ),
      );

      expect(result.success, isFalse);
      expect(result.summary, contains('rejected'));
      expect(result.error, contains('not in the allow-list'));
    });

    test('rejects a metacharacter argument through the runner', () async {
      final SubagentResult result = await SubagentRunner().run(
        const SubagentTask(
          id: 'sc_meta',
          description: 'inject',
          kind: SubagentKind.systemCommand,
          args: <String, Object?>{
            'command': 'echo',
            'args': <String>['a;b'],
          },
        ),
      );

      expect(result.success, isFalse);
      expect(result.error, contains('metacharacters'));
    });

    test('reports a spawn failure for a missing allow-listed executable', () async {
      // Injected allow-list keeps this deterministic and independent of the
      // host: the name is allowed but does not exist on PATH.
      const String missing = 'harbor_missing_binary_42';
      final SubagentRunner runner = SubagentRunner(
        subagents: <Subagent>[
          SystemCommandSubagent(allowedExecutables: const <String>{missing}),
        ],
      );
      final SubagentResult result = await runner.run(
        const SubagentTask(
          id: 'sc_spawn',
          description: 'spawn a missing binary',
          kind: SubagentKind.systemCommand,
          args: <String, Object?>{'command': missing},
        ),
      );

      expect(result.success, isFalse);
      expect(result.summary, contains('Failed to spawn'));
      expect(result.metadata['executable'], missing);
    });
  });

  group('CodeVerificationSubagent', () {
    Future<SubagentResult> verifyCode(String code) => SubagentRunner().run(
          SubagentTask(
            id: 'cv',
            description: 'verify code',
            kind: SubagentKind.codeVerification,
            args: <String, Object?>{'code': code},
          ),
        );

    test('reports unbalanced delimiters as an error', () async {
      final SubagentResult result = await verifyCode('void main() {\n');
      expect(result.success, isFalse);
      expect(result.metadata['error_count']! as int, greaterThanOrEqualTo(1));
      expect(rulesOf(result), contains('balanced_delimiters'));
      final Map<String, Object?> finding = findingsOf(result)
          .firstWhere((Map<String, Object?> f) => f['rule'] == 'balanced_delimiters');
      expect(finding['severity'], 'error');
    });

    test('flags merge-conflict markers', () async {
      final SubagentResult result = await verifyCode(
        '<<<<<<< HEAD\nint x = 1;\n=======\nint y = 2;\n>>>>>>> other\n',
      );
      expect(result.success, isFalse);
      expect(rulesOf(result), contains('merge_conflict_marker'));
    });

    test('flags a line longer than maxLineLength', () async {
      final SubagentResult result = await verifyCode('final x = ${'a' * 130};\n');
      expect(rulesOf(result), contains('max_line_length'));
      final Map<String, Object?> finding = findingsOf(result)
          .firstWhere((Map<String, Object?> f) => f['rule'] == 'max_line_length');
      expect(finding['severity'], 'warning');
      expect(result.success, isTrue);
    });

    test('flags trailing whitespace as info', () async {
      final SubagentResult result = await verifyCode('int x = 1;   \n');
      expect(rulesOf(result), contains('trailing_whitespace'));
      final Map<String, Object?> finding = findingsOf(result)
          .firstWhere((Map<String, Object?> f) => f['rule'] == 'trailing_whitespace');
      expect(finding['severity'], 'info');
      expect(finding['line'], 1);
    });

    test('flags an unresolved TODO marker as a warning', () async {
      final SubagentResult result = await verifyCode('// TODO: fix this\n');
      expect(rulesOf(result), contains('unresolved_marker'));
      final Map<String, Object?> finding = findingsOf(result)
          .firstWhere((Map<String, Object?> f) => f['rule'] == 'unresolved_marker');
      expect(finding['severity'], 'warning');
    });

    test('flags a missing final newline as info', () async {
      final SubagentResult result = await verifyCode('int x = 1;');
      expect(rulesOf(result), contains('missing_final_newline'));
      expect(result.success, isTrue);
      expect(result.metadata['error_count'], 0);
    });

    test('accepts clean code with no findings', () async {
      final SubagentResult result = await verifyCode('void main() {\n  print(1);\n}\n');
      expect(result.success, isTrue);
      expect(result.metadata['error_count'], 0);
      expect(result.metadata['warning_count'], 0);
      expect(result.metadata['finding_count'], 0);
      expect(findingsOf(result), isEmpty);
    });

    test('ignores delimiters that live inside string literals', () async {
      final SubagentResult result = await verifyCode('final s = "{([])}";\n');
      expect(rulesOf(result), isNot(contains('balanced_delimiters')));
      expect(result.success, isTrue);
    });

    test('walks a directory argument and counts the files checked', () async {
      writeText('lib/clean.dart', 'void main() {\n  print(1);\n}\n');
      final SubagentResult result = await SubagentRunner().run(
        SubagentTask(
          id: 'cv_dir',
          description: 'verify a tree',
          kind: SubagentKind.codeVerification,
          args: <String, Object?>{'directory': root.path},
        ),
      );

      expect(result.success, isTrue);
      expect(result.metadata['files_checked']! as int, greaterThan(0));
    });

    test('fails cleanly when no target is supplied', () async {
      final SubagentResult result = await SubagentRunner().run(
        const SubagentTask(
          id: 'cv_missing',
          description: 'verify nothing',
          kind: SubagentKind.codeVerification,
        ),
      );

      expect(result.success, isFalse);
      expect(result.summary, contains('Invalid verification task'));
      expect(result.error, contains('requires'));
    });
  });

  group('concurrency and lifecycle', () {
    test('runAll returns results in submission order', () async {
      final SubagentRunner runner = SubagentRunner(
        maxConcurrency: 3,
        subagents: <Subagent>[_RecordingSubagent()],
      );
      final List<SubagentTask> tasks = <SubagentTask>[
        customTask('t1'),
        customTask('t2'),
        customTask('t3'),
        customTask('t4'),
        customTask('t5'),
      ];

      final List<SubagentResult> results = await runner.runAll(tasks);
      expect(results, hasLength(5));
      for (int i = 0; i < tasks.length; i++) {
        expect(results[i].taskId, tasks[i].id);
        expect(results[i].success, isTrue);
      }
    });

    test('maxConcurrency bounds the number of overlapping invocations', () async {
      final _RecordingSubagent recorder =
          _RecordingSubagent(delay: const Duration(milliseconds: 30));
      final SubagentRunner runner = SubagentRunner(
        maxConcurrency: 2,
        subagents: <Subagent>[recorder],
      );
      final List<SubagentTask> tasks = <SubagentTask>[
        for (int i = 0; i < 6; i++) customTask('c$i'),
      ];

      final List<SubagentResult> results = await runner.runAll(tasks);
      expect(results.every((SubagentResult r) => r.success), isTrue);
      expect(recorder.peak, lessThanOrEqualTo(2));
      expect(recorder.peak, greaterThanOrEqualTo(1));
      expect(recorder.peak, 2);
    });

    test('higher priority is dispatched first when the pool is saturated', () async {
      final _RecordingSubagent recorder =
          _RecordingSubagent(delay: const Duration(milliseconds: 60));
      final SubagentRunner runner = SubagentRunner(
        maxConcurrency: 1,
        subagents: <Subagent>[recorder],
      );

      final Future<SubagentResult> low1 = runner.run(customTask('low1'));
      final Future<SubagentResult> low2 = runner.run(customTask('low2'));
      final Future<SubagentResult> low3 = runner.run(customTask('low3'));
      final Future<SubagentResult> high = runner.run(customTask('high', priority: 10));
      await Future.wait(<Future<SubagentResult>>[low1, low2, low3, high]);

      expect(recorder.order, <String>['low1', 'high', 'low2', 'low3']);
    });

    test('a task that overruns its deadline fails with a deadline error', () async {
      final SubagentRunner runner = SubagentRunner(
        maxConcurrency: 1,
        subagents: <Subagent>[
          _RecordingSubagent(delay: const Duration(milliseconds: 200)),
        ],
      );
      final Stopwatch stopwatch = Stopwatch()..start();
      final SubagentResult result = await runner.run(
        customTask('slow', timeout: const Duration(milliseconds: 50)),
      );
      stopwatch.stop();

      expect(result.success, isFalse);
      expect(result.error, contains('exceeded'));
      expect(result.metadata['timeout_ms'], 50);
      expect(stopwatch.elapsedMilliseconds, greaterThanOrEqualTo(30));
      expect(stopwatch.elapsedMilliseconds, lessThan(400));
    });

    test('a subagent that throws becomes a failed result', () async {
      final SubagentRunner runner = SubagentRunner(
        subagents: <Subagent>[_ThrowingSubagent()],
      );
      final SubagentResult result = await runner.run(customTask('boom'));

      expect(result.success, isFalse);
      expect(result.summary, contains('unexpected error'));
      expect(result.error, contains('boom'));
    });

    test('an unregistered kind fails without throwing', () async {
      final SubagentRunner runner = SubagentRunner(subagents: <Subagent>[]);
      final SubagentResult result = await runner.run(customTask('none'));

      expect(result.success, isFalse);
      expect(result.summary, contains('No subagent registered'));
      expect(result.metadata['available'], isEmpty);
    });

    test('runStreaming emits exactly one result per task and completes', () async {
      final SubagentRunner runner = SubagentRunner(
        maxConcurrency: 2,
        subagents: <Subagent>[_RecordingSubagent()],
      );
      final List<SubagentTask> tasks = <SubagentTask>[
        customTask('s1'),
        customTask('s2'),
        customTask('s3'),
      ];

      final List<SubagentResult> results = await runner.runStreaming(tasks).toList();
      expect(results, hasLength(3));
      expect(
        results.map((SubagentResult r) => r.taskId).toSet(),
        <String>{'s1', 's2', 's3'},
      );
      expect(runner.completedCount, 3);
    });

    test('cancelAll completes queued tasks with a cancellation failure', () async {
      final SubagentRunner runner = SubagentRunner(
        maxConcurrency: 1,
        subagents: <Subagent>[
          _RecordingSubagent(delay: const Duration(milliseconds: 40)),
        ],
      );
      final Future<SubagentResult> running = runner.run(customTask('running'));
      final Future<SubagentResult> queued1 = runner.run(customTask('q1'));
      final Future<SubagentResult> queued2 = runner.run(customTask('q2'));
      expect(runner.queuedCount, 2);

      runner.cancelAll('test stop');
      final List<SubagentResult> cancelled =
          await Future.wait(<Future<SubagentResult>>[queued1, queued2]);
      for (final SubagentResult result in cancelled) {
        expect(result.success, isFalse);
        expect(result.summary, contains('cancelled'));
        expect(result.error, 'test stop');
      }
      expect(runner.queuedCount, 0);

      // The already-running task finishes normally; the token is advisory.
      final SubagentResult first = await running;
      expect(first.success, isTrue);
    });

    test('counters track active, queued, completed and failed work', () async {
      final SubagentRunner runner = SubagentRunner(
        maxConcurrency: 1,
        subagents: <Subagent>[
          _RecordingSubagent(delay: const Duration(milliseconds: 40)),
        ],
      );
      expect(runner.activeCount, 0);
      expect(runner.queuedCount, 0);
      expect(runner.completedCount, 0);
      expect(runner.failedCount, 0);

      final List<Future<SubagentResult>> futures = <Future<SubagentResult>>[
        runner.run(customTask('k1')),
      ];
      expect(runner.activeCount, 1);
      futures.add(runner.run(customTask('k2')));
      futures.add(runner.run(customTask('k3')));
      expect(runner.queuedCount, 2);

      final List<SubagentResult> results = await Future.wait(futures);
      expect(results.every((SubagentResult r) => r.success), isTrue);
      expect(runner.completedCount, 3);
      expect(runner.failedCount, 0);
      expect(runner.activeCount, 0);
      expect(runner.queuedCount, 0);

      runner.resetStatistics();
      expect(runner.completedCount, 0);
      expect(runner.failedCount, 0);
    });

    test('failedCount increments for a failing subagent', () async {
      final SubagentRunner runner = SubagentRunner(
        maxConcurrency: 1,
        subagents: <Subagent>[_RecordingSubagent()],
      );
      final SubagentResult ok = await runner.run(customTask('ok'));
      final SubagentResult bad = await runner.run(customTask('bad', fail: true));

      expect(ok.success, isTrue);
      expect(bad.success, isFalse);
      expect(runner.completedCount, 1);
      expect(runner.failedCount, 1);
    });

    test('recommendedConcurrency clamps between 2 and 8', () {
      expect(recommendedConcurrency(availableProcessors: 1), greaterThanOrEqualTo(2));
      expect(recommendedConcurrency(availableProcessors: 1), 2);
      expect(recommendedConcurrency(availableProcessors: 32), lessThanOrEqualTo(8));
      expect(recommendedConcurrency(availableProcessors: 32), 8);
      expect(recommendedConcurrency(availableProcessors: 4), 4);
    });

    test('stableTaskHash is deterministic and non-negative', () {
      final int first = stableTaskHash('plan_step_01');
      expect(first, stableTaskHash('plan_step_01'));
      expect(first, greaterThanOrEqualTo(0));
      expect(stableTaskHash('a'), isNot(stableTaskHash('b')));
    });
  });

  group('task and result primitives', () {
    test('SubagentTask.toJson exposes the task fields', () {
      const SubagentTask task = SubagentTask(
        id: 'x',
        description: 'do a thing',
        kind: SubagentKind.codeVerification,
        args: <String, Object?>{'code': 'x'},
        timeout: Duration(seconds: 3),
        priority: 7,
      );
      final Map<String, Object?> json = task.toJson();

      expect(json['id'], 'x');
      expect(json['description'], 'do a thing');
      expect(json['kind'], 'codeVerification');
      expect(json['args'], <String, Object?>{'code': 'x'});
      expect(json['timeout_ms'], 3000);
      expect(json['priority'], 7);
    });

    test('SubagentTask.copyWith overrides only the named fields', () {
      const SubagentTask task = SubagentTask(
        id: 'old',
        description: 'original',
        kind: SubagentKind.fileParsing,
      );
      final SubagentTask copy = task.copyWith(id: 'new', priority: 5);

      expect(copy.id, 'new');
      expect(copy.priority, 5);
      expect(copy.description, 'original');
      expect(copy.kind, SubagentKind.fileParsing);
      expect(copy.timeout, const Duration(seconds: 30));
    });

    test('CancellationToken cancels and throwIfCancelled reports the reason', () {
      final CancellationToken token = CancellationToken();
      expect(token.isCancelled, isFalse);
      expect(token.throwIfCancelled, returnsNormally);

      token.cancel('stop now');
      expect(token.isCancelled, isTrue);
      expect(token.reason, 'stop now');
      expect(token.throwIfCancelled, throwsA(isA<SubagentDispatchException>()));
    });

    test('SubagentResult.failure sets success false', () {
      final SubagentResult result = SubagentResult.failure(
        taskId: 'f',
        summary: 'nope',
        error: 'because',
        duration: Duration.zero,
      );
      expect(result.success, isFalse);
      expect(result.taskId, 'f');
      expect(result.error, 'because');
      expect(result.toJson()['success'], isFalse);
    });
  });
}