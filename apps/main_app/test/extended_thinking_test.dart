// Unit tests for the extended-thinking agent in
// core/agent/extended_thinking.dart.
//
// The planner is deterministic and model-free, so these tests exercise the
// real Plan -> Execute -> Verify -> Replan loop against real temp files and the
// real SubagentRunner with its built-in subagents.

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

import 'package:main_app/core/agent/extended_thinking.dart';
import 'package:main_app/core/agent/subagent_runner.dart';

/// A backend that always returns an empty (invalid) plan.
class _EmptyPlanBackend implements PlanningBackend {
  @override
  Future<Plan> plan(
    String goal,
    Map<String, Object?> context,
    List<StepOutcome> previousOutcomes,
  ) async =>
      Plan(goal: goal, steps: const <ThoughtStep>[], rationale: 'empty by design');

  @override
  Future<VerificationVerdict> verify(
    ThoughtStep step,
    SubagentResult result,
    Map<String, Object?> context,
  ) async =>
      const VerificationVerdict(passed: true, note: 'unused');
}

/// A backend whose plans always contain one critical step that is guaranteed to
/// fail, so every repair cycle fails too.
class _AlwaysFailingBackend implements PlanningBackend {
  _AlwaysFailingBackend(this.missingPath);

  final String missingPath;
  int planCalls = 0;

  @override
  Future<Plan> plan(
    String goal,
    Map<String, Object?> context,
    List<StepOutcome> previousOutcomes,
  ) async {
    planCalls++;
    return Plan(
      goal: goal,
      rationale: 'always fails',
      steps: <ThoughtStep>[
        ThoughtStep(
          index: 0,
          description: 'read a missing target',
          tool: SubagentKind.fileParsing,
          args: <String, Object?>{'path': missingPath},
        ),
      ],
    );
  }

  @override
  Future<VerificationVerdict> verify(
    ThoughtStep step,
    SubagentResult result,
    Map<String, Object?> context,
  ) async =>
      result.success
          ? const VerificationVerdict(passed: true, note: 'ok')
          : const VerificationVerdict(
              passed: false,
              note: 'failed',
              issues: <String>['failed'],
            );
}

/// Builds a system-command step, which needs no registered work to validate.
ThoughtStep planStep(int index, {List<int> dependsOn = const <int>[]}) =>
    ThoughtStep(
      index: index,
      description: 'step $index',
      tool: SubagentKind.systemCommand,
      dependsOn: dependsOn,
    );

void main() {
  late Directory root;

  setUp(() async {
    root = await Directory.systemTemp.createTemp('harbor_thinking_');
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

  File cleanSource(String relative) =>
      writeText(relative, 'void main() {\n  print(1);\n}\n');

  group('HeuristicPlanningBackend.extractPaths', () {
    test('extracts quoted paths, directories and bare extension paths', () {
      final HeuristicPlanningBackend backend = HeuristicPlanningBackend();
      final List<String> paths = backend.extractPaths(
        'Read "lib/main.dart", pubspec.yaml and the directory "src/core"; '
        'also assets/logo.png',
      );

      expect(
        paths.toSet(),
        <String>{'lib/main.dart', 'src/core', 'pubspec.yaml', 'assets/logo.png'},
      );
    });

    test('returns no paths for a goal that names none', () {
      final HeuristicPlanningBackend backend = HeuristicPlanningBackend();
      expect(backend.extractPaths('summarise the current situation'), isEmpty);
    });
  });

  group('HeuristicPlanningBackend.plan', () {
    Future<Plan> planFor(
      String goal, {
      Map<String, Object?> context = const <String, Object?>{},
    }) =>
        HeuristicPlanningBackend().plan(goal, context, const <StepOutcome>[]);

    test('a named source file yields a read step plus a verification step', () async {
      final Plan plan = await planFor('Read and verify "lib/main.dart"');

      expect(
        plan.steps.any((ThoughtStep s) =>
            s.tool == SubagentKind.fileParsing && s.args['path'] == 'lib/main.dart'),
        isTrue,
      );
      final ThoughtStep verifyStep = plan.steps
          .firstWhere((ThoughtStep s) => s.tool == SubagentKind.codeVerification);
      expect(verifyStep.dependsOn, contains(0));
      expect(plan.steps.first.tool, SubagentKind.fileParsing);
    });

    test('an audit goal produces a verification step', () async {
      final Plan plan = await planFor('audit this project');
      expect(
        plan.steps.any((ThoughtStep s) => s.tool == SubagentKind.codeVerification),
        isTrue,
      );
    });

    test('two named files produce two independent read steps', () async {
      final Plan plan = await planFor('Compare "a.dart" and "b.dart"');

      final List<ThoughtStep> reads = plan.steps
          .where((ThoughtStep s) => s.tool == SubagentKind.fileParsing)
          .toList(growable: false);
      expect(reads, hasLength(2));
      expect(reads.every((ThoughtStep s) => s.dependsOn.isEmpty), isTrue);
      expect(
        reads.map((ThoughtStep s) => s.args['path']).toSet(),
        <String>{'a.dart', 'b.dart'},
      );
    });

    test('an explicit run command becomes a systemCommand step', () async {
      final Plan plan = await planFor('run echo hello');

      expect(plan.steps, hasLength(1));
      final ThoughtStep command = plan.steps.single;
      expect(command.tool, SubagentKind.systemCommand);
      expect(command.args['command'], 'echo');
      expect(command.args['args'], <String>['hello']);
      expect(command.critical, isFalse);
    });

    test('a root context hint becomes a directory walk step', () async {
      final Plan plan = await planFor(
        'summarise the project',
        context: <String, Object?>{'root': root.path},
      );

      final ThoughtStep walk = plan.steps
          .singleWhere((ThoughtStep s) => s.tool == SubagentKind.fileParsing);
      expect(walk.args['directory'], root.path);
    });

    test('the produced plan validates and has at least one wave', () async {
      final Plan plan = await planFor('Read and verify "lib/main.dart"');
      expect(() => plan.validate(), returnsNormally);
      expect(plan.executionWaves().length, greaterThanOrEqualTo(1));
    });

    test('a goal with no recognisable target still yields a valid plan', () async {
      final Plan plan = await planFor('make everything better');
      expect(plan.steps, isNotEmpty);
      expect(() => plan.validate(), returnsNormally);
      expect(plan.executionWaves(), isNotEmpty);
    });
  });

  group('Plan.validate and executionWaves', () {
    test('rejects a missing step index', () {
      final Plan plan = Plan(
        goal: 'g',
        rationale: '',
        steps: <ThoughtStep>[planStep(0), planStep(2)],
      );
      expect(() => plan.validate(), throwsA(isA<PlanValidationException>()));
    });

    test('rejects an empty plan', () {
      const Plan plan = Plan(goal: 'g', rationale: '', steps: <ThoughtStep>[]);
      expect(() => plan.validate(), throwsA(isA<PlanValidationException>()));
    });

    test('rejects a self-dependency', () {
      final Plan plan = Plan(
        goal: 'g',
        rationale: '',
        steps: <ThoughtStep>[planStep(0, dependsOn: <int>[0])],
      );
      expect(() => plan.validate(), throwsA(isA<PlanValidationException>()));
    });

    test('rejects an out-of-range dependency', () {
      final Plan plan = Plan(
        goal: 'g',
        rationale: '',
        steps: <ThoughtStep>[planStep(0, dependsOn: <int>[7])],
      );
      expect(() => plan.validate(), throwsA(isA<PlanValidationException>()));
    });

    test('rejects a two-step cycle and names the cycle', () {
      final Plan plan = Plan(
        goal: 'g',
        rationale: '',
        steps: <ThoughtStep>[
          planStep(0, dependsOn: <int>[1]),
          planStep(1, dependsOn: <int>[0]),
        ],
      );
      expect(
        () => plan.validate(),
        throwsA(
          isA<PlanValidationException>()
              .having((PlanValidationException e) => e.message, 'message', contains('cycle')),
        ),
      );
    });

    test('a diamond dependency yields waves [0], [1,2], [3]', () {
      final Plan plan = Plan(
        goal: 'diamond',
        rationale: '',
        steps: <ThoughtStep>[
          planStep(0),
          planStep(1, dependsOn: <int>[0]),
          planStep(2, dependsOn: <int>[0]),
          planStep(3, dependsOn: <int>[1, 2]),
        ],
      );

      final List<List<ThoughtStep>> waves = plan.executionWaves();
      expect(waves, hasLength(3));
      expect(
        waves[0].map((ThoughtStep s) => s.index).toList(growable: false),
        <int>[0],
      );
      expect(
        waves[1].map((ThoughtStep s) => s.index).toList(growable: false),
        <int>[1, 2],
      );
      expect(
        waves[2].map((ThoughtStep s) => s.index).toList(growable: false),
        <int>[3],
      );
    });
  });

  group('HeuristicPlanningBackend.verify', () {
    late HeuristicPlanningBackend backend;

    setUp(() {
      backend = HeuristicPlanningBackend();
    });

    test('passes a successful file read with a kind and a size', () async {
      const ThoughtStep step = ThoughtStep(
        index: 0,
        description: 'read',
        tool: SubagentKind.fileParsing,
        args: <String, Object?>{'path': 'x.txt'},
      );
      const SubagentResult result = SubagentResult(
        taskId: 't',
        success: true,
        summary: 'read x.txt',
        duration: Duration.zero,
        output: <String, Object?>{'path': 'x.txt'},
        metadata: <String, Object?>{'kind': 'text', 'size_bytes': 10},
      );

      final VerificationVerdict verdict =
          await backend.verify(step, result, const <String, Object?>{});
      expect(verdict.passed, isTrue);
      expect(verdict.issues, isEmpty);
    });

    test('fails a failed result and records an issue', () async {
      const ThoughtStep step = ThoughtStep(
        index: 0,
        description: 'read',
        tool: SubagentKind.fileParsing,
        args: <String, Object?>{'path': 'x.txt'},
      );
      final SubagentResult result = SubagentResult.failure(
        taskId: 't',
        summary: 'no',
        error: 'boom',
        duration: Duration.zero,
      );

      final VerificationVerdict verdict =
          await backend.verify(step, result, const <String, Object?>{});
      expect(verdict.passed, isFalse);
      expect(verdict.issues, isNotEmpty);
      expect(verdict.issues.first, contains('boom'));
    });

    test('fails a code verification result with errors', () async {
      const ThoughtStep step = ThoughtStep(
        index: 1,
        description: 'verify',
        tool: SubagentKind.codeVerification,
      );
      const SubagentResult result = SubagentResult(
        taskId: 't',
        success: true,
        summary: 'verified',
        duration: Duration.zero,
        output: <Object?>[],
        metadata: <String, Object?>{'error_count': 2, 'files_checked': 1},
      );

      final VerificationVerdict verdict =
          await backend.verify(step, result, const <String, Object?>{});
      expect(verdict.passed, isFalse);
      expect(verdict.issues.first, contains('structural error'));
    });

    test('fails a system command result with a non-zero exit code', () async {
      const ThoughtStep step = ThoughtStep(
        index: 2,
        description: 'run',
        tool: SubagentKind.systemCommand,
      );
      const SubagentResult result = SubagentResult(
        taskId: 't',
        success: true,
        summary: 'ran',
        duration: Duration.zero,
        metadata: <String, Object?>{'exit_code': 1},
      );

      final VerificationVerdict verdict =
          await backend.verify(step, result, const <String, Object?>{});
      expect(verdict.passed, isFalse);
      expect(verdict.issues.first, contains('exited with code 1'));
    });
  });

  group('ExtendedThinkingAgent.solve', () {
    test('plans, executes and verifies a real file end to end', () async {
      final File file = cleanSource('main.dart');
      final List<ThinkingPhase> phases = <ThinkingPhase>[];
      final ExtendedThinkingAgent agent = ExtendedThinkingAgent(
        runner: SubagentRunner(),
        onEvent: (ThinkingEvent event) {
          phases.add(event.phase);
        },
      );

      final ThinkingTrace trace = await agent.solve('read "${file.path}" and verify');

      expect(trace.success, isTrue);
      expect(trace.plans, isNotEmpty);
      expect(trace.toolCallCount, greaterThanOrEqualTo(1));
      expect(trace.thoughts, isNotEmpty);
      expect(trace.duration.inMilliseconds, greaterThanOrEqualTo(0));
      expect(
        trace.outcomes.every((StepOutcome o) => o.verdict != null),
        isTrue,
      );
      expect(trace.outcomes.every((StepOutcome o) => o.verified), isTrue);
      expect(phases, contains(ThinkingPhase.planning));
      expect(phases, contains(ThinkingPhase.executing));
      expect(phases, contains(ThinkingPhase.verifying));
      expect(phases, contains(ThinkingPhase.complete));

      final String markdown = trace.toMarkdown();
      expect(markdown, contains('## Thinking'));
      expect(markdown, contains(file.path));
      expect(markdown, contains('### Steps'));
    });

    test('truncates a multi-step plan to maxSteps and still succeeds', () async {
      final File first = cleanSource('a.dart');
      cleanSource('b.dart');
      final ExtendedThinkingAgent agent = ExtendedThinkingAgent(
        runner: SubagentRunner(),
        config: const ExtendedThinkingConfig(maxSteps: 1),
      );

      final ThinkingTrace trace =
          await agent.solve('read "${first.path}" and "${root.path}/b.dart" and verify');

      expect(trace.plans.first.steps, hasLength(1));
      expect(() => trace.plans.first.validate(), returnsNormally);
      expect(trace.success, isTrue);
      expect(trace.outcomes, hasLength(1));
    });

    test('failFastOnCriticalError aborts on a failing critical step', () async {
      final String missing = '${root.path}/missing_${DateTime.now().microsecondsSinceEpoch}.dart';
      final ExtendedThinkingAgent agent = ExtendedThinkingAgent(
        runner: SubagentRunner(),
        config: const ExtendedThinkingConfig(failFastOnCriticalError: true),
      );

      final ThinkingTrace trace = await agent.solve('read "$missing"');

      expect(trace.success, isFalse);
      expect(trace.failureReason, isNotNull);
      expect(trace.failureReason, contains('failed'));
      expect(trace.plans, hasLength(1));
    });

    test('an invalid stub plan returns a failure instead of throwing', () async {
      final ExtendedThinkingAgent agent = ExtendedThinkingAgent(
        runner: SubagentRunner(),
        backend: _EmptyPlanBackend(),
      );

      final ThinkingTrace trace = await agent.solve('anything at all');

      expect(trace.success, isFalse);
      expect(trace.failureReason, isNotNull);
      expect(trace.failureReason, contains('plan'));
      expect(trace.plans, isEmpty);
      expect(trace.outcomes, isEmpty);
    });

    test('replanning stops at maxReplanCycles when repair also fails', () async {
      final _AlwaysFailingBackend backend =
          _AlwaysFailingBackend('${root.path}/nope.dart');
      final ExtendedThinkingAgent agent = ExtendedThinkingAgent(
        runner: SubagentRunner(),
        backend: backend,
        config: const ExtendedThinkingConfig(maxReplanCycles: 1),
      );

      final ThinkingTrace trace = await agent.solve('read the missing file');

      expect(trace.success, isFalse);
      expect(trace.replans, 1);
      expect(trace.replans, lessThanOrEqualTo(1));
      expect(trace.failureReason, contains('replan'));
      expect(trace.plans, hasLength(2));
      expect(backend.planCalls, 2);
    });

    test('explain returns a markdown thinking trace', () async {
      final File file = cleanSource('explain.dart');
      final String markdown = await ExtendedThinkingAgent(runner: SubagentRunner())
          .explain('read "${file.path}" and verify');

      expect(markdown, contains('## Thinking'));
      expect(markdown, contains('### Steps'));
    });
  });

  group('ThinkingTrace getters', () {
    StepOutcome outcome({bool? passed}) {
      const ThoughtStep step = ThoughtStep(
        index: 0,
        description: 'x',
        tool: SubagentKind.custom,
      );
      return StepOutcome(
        step: step,
        result: const SubagentResult(
          taskId: 't',
          success: true,
          summary: 's',
          duration: Duration.zero,
        ),
        verdict: passed == null
            ? null
            : VerificationVerdict(passed: passed, note: 'n'),
      );
    }

    test('problems counts every non-verified outcome', () {
      final ThinkingTrace trace = ThinkingTrace(
        goal: 'g',
        thoughts: <String>['thought'],
        plans: const <Plan>[],
        outcomes: <StepOutcome>[
          outcome(passed: true),
          outcome(passed: false),
          outcome(),
        ],
        replans: 0,
        success: false,
        duration: const Duration(milliseconds: 5),
        failureReason: 'nope',
      );

      expect(trace.problems, hasLength(2));
      expect(trace.toolCallCount, 3);
      expect(trace.toolCallCount, trace.outcomes.length);
    });

    test('toJson exposes the documented keys', () {
      final ThinkingTrace trace = ThinkingTrace(
        goal: 'g',
        thoughts: <String>['thought'],
        plans: const <Plan>[],
        outcomes: <StepOutcome>[outcome(passed: true)],
        replans: 1,
        success: true,
        duration: const Duration(milliseconds: 5),
      );
      final Map<String, Object?> json = trace.toJson();

      expect(
        json.keys,
        containsAll(<String>[
          'goal',
          'success',
          'replans',
          'duration_ms',
          'tool_calls',
          'failure_reason',
          'thoughts',
          'plans',
          'outcomes',
        ]),
      );
      expect(json['goal'], 'g');
      expect(json['success'], isTrue);
      expect(json['tool_calls'], 1);
      expect(json['replans'], 1);
      expect(json['failure_reason'], isNull);
    });
  });
}