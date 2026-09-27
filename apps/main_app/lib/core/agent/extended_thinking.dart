// Extended-thinking agent: explicit Plan -> Execute -> Verify -> Replan.
//
// Claude-Code-style agency has three properties this implementation makes
// literal:
//
//   1. PLAN BEFORE ACTING. The agent first produces a numbered plan whose steps
//      each declare the subagent they need and the arguments to pass. Nothing
//      touches the device until the plan exists and is scored.
//   2. FAN OUT, DON'T SERIALISE. Steps declare `dependsOn` edges. The executor
//      computes dependency *waves* and runs every step in a wave concurrently
//      through [SubagentRunner], so three independent file parses cost one
//      round-trip, not three.
//   3. VERIFY, THEN REPLAN. A separate verification pass — never the executor —
//      judges each result. Failures trigger bounded repair planning rather than
//      blind retries, and the whole exchange is captured in an immutable
//      [ThinkingTrace] that the UI renders as the collapsible thinking panel.
//
// The model itself is behind [PlanningBackend]: [HeuristicPlanningBackend] is a
// real, deterministic planner (no model weights required) used on device and in
// tests, and a model-backed backend drops into the same interface.

import 'dart:async';
import 'dart:math' as math;

import 'file_readers.dart';
import 'subagent_runner.dart';

/// Thrown when a plan is structurally invalid (missing steps, cycles, unknown
/// dependency indices, unknown subagent kinds).
class PlanValidationException implements Exception {
  PlanValidationException(this.message);

  final String message;

  @override
  String toString() => 'PlanValidationException: $message';
}

/// Phase of the reasoning loop the agent is currently in.
enum ThinkingPhase {
  planning,
  executing,
  verifying,
  replanning,
  complete,
  failed;

  /// Human label for the UI.
  String get label {
    switch (this) {
      case ThinkingPhase.planning:
        return 'Planning';
      case ThinkingPhase.executing:
        return 'Executing';
      case ThinkingPhase.verifying:
        return 'Verifying';
      case ThinkingPhase.replanning:
        return 'Replanning';
      case ThinkingPhase.complete:
        return 'Complete';
      case ThinkingPhase.failed:
        return 'Failed';
    }
  }
}

/// A single planned step with declared dependencies.
class ThoughtStep {
  const ThoughtStep({
    required this.index,
    required this.description,
    required this.tool,
    this.args = const <String, Object?>{},
    this.dependsOn = const <int>[],
    this.critical = true,
    this.timeout = const Duration(seconds: 30),
  });

  /// Position of this step within its plan.
  final int index;

  /// Natural-language intent, shown in the thinking trace.
  final String description;

  /// Subagent kind required to execute the step.
  final SubagentKind tool;

  /// Arguments handed to the subagent.
  final Map<String, Object?> args;

  /// Indices of steps that must finish before this one starts.
  final List<int> dependsOn;

  /// Whether a failure of this step is fatal to the overall goal.
  final bool critical;

  /// Deadline for this step.
  final Duration timeout;

  /// A stable task id derived from the plan position.
  String taskId(String prefix) => '${prefix}_step_${index.toString().padLeft(2, '0')}';

  /// Converts this step into a runnable [SubagentTask].
  SubagentTask toTask(String prefix) => SubagentTask(
        id: taskId(prefix),
        description: description,
        kind: tool,
        args: args,
        timeout: timeout,
        priority: critical ? 10 : 0,
      );

  Map<String, Object?> toJson() => <String, Object?>{
        'index': index,
        'description': description,
        'tool': tool.name,
        'args': args,
        'depends_on': dependsOn,
        'critical': critical,
        'timeout_ms': timeout.inMilliseconds,
      };
}

/// An ordered plan of steps for a goal.
class Plan {
  const Plan({
    required this.goal,
    required this.steps,
    required this.rationale,
    this.replanIndex = 0,
  });

  /// The user-facing goal this plan pursues.
  final String goal;

  /// Steps in index order.
  final List<ThoughtStep> steps;

  /// Why the planner chose this decomposition.
  final String rationale;

  /// 0 for the initial plan, 1+ for repair plans.
  final int replanIndex;

  /// Validates structure: non-empty, dense indices, known tools, acyclic.
  void validate() {
    if (steps.isEmpty) {
      throw PlanValidationException('plan for "$goal" has no steps');
    }
    for (int i = 0; i < steps.length; i++) {
      if (steps[i].index != i) {
        throw PlanValidationException(
          'step at position $i declares index ${steps[i].index}',
        );
      }
      for (final int dep in steps[i].dependsOn) {
        if (dep < 0 || dep >= steps.length) {
          throw PlanValidationException(
            'step $i depends on out-of-range step $dep',
          );
        }
        if (dep == i) {
          throw PlanValidationException('step $i depends on itself');
        }
      }
    }
    _detectCycles();
  }

  void _detectCycles() {
    const int white = 0;
    const int grey = 1;
    const int black = 2;
    final List<int> colour = List<int>.filled(steps.length, white);

    void visit(int node, List<int> path) {
      colour[node] = grey;
      path.add(node);
      for (final int dep in steps[node].dependsOn) {
        if (colour[dep] == grey) {
          throw PlanValidationException(
            'dependency cycle detected: ${<int>[...path, dep].join(' -> ')}',
          );
        }
        if (colour[dep] == white) {
          visit(dep, path);
        }
      }
      path.removeLast();
      colour[node] = black;
    }

    for (int i = 0; i < steps.length; i++) {
      if (colour[i] == white) {
        visit(i, <int>[]);
      }
    }
  }

  /// Groups steps into dependency waves. Every step in a wave can run
  /// concurrently.
  List<List<ThoughtStep>> executionWaves() {
    validate();
    final List<List<ThoughtStep>> waves = <List<ThoughtStep>>[];
    final Set<int> done = <int>{};
    final int total = steps.length;
    while (done.length < total) {
      final List<ThoughtStep> wave = <ThoughtStep>[];
      for (final ThoughtStep s in steps) {
        if (done.contains(s.index)) {
          continue;
        }
        if (s.dependsOn.every(done.contains)) {
          wave.add(s);
        }
      }
      if (wave.isEmpty) {
        throw PlanValidationException(
          'no executable step found; dependency graph is unsatisfiable',
        );
      }
      waves.add(wave);
      for (final ThoughtStep s in wave) {
        done.add(s.index);
      }
    }
    return waves;
  }

  Map<String, Object?> toJson() => <String, Object?>{
        'goal': goal,
        'rationale': rationale,
        'replan_index': replanIndex,
        'step_count': steps.length,
        'steps': steps.map((ThoughtStep s) => s.toJson()).toList(growable: false),
      };
}

/// Result of verifying one executed step.
class VerificationVerdict {
  const VerificationVerdict({
    required this.passed,
    required this.note,
    this.issues = const <String>[],
  });

  final bool passed;
  final String note;
  final List<String> issues;

  Map<String, Object?> toJson() => <String, Object?>{
        'passed': passed,
        'note': note,
        'issues': issues,
      };
}

/// An executed step together with its verification outcome.
class StepOutcome {
  const StepOutcome({
    required this.step,
    required this.result,
    this.verdict,
    this.attempt = 1,
  });

  final ThoughtStep step;
  final SubagentResult result;

  /// Null until the verification pass runs.
  final VerificationVerdict? verdict;

  /// Which attempt produced this outcome (1 = first try).
  final int attempt;

  bool get succeeded => result.success;

  bool get verified => verdict?.passed ?? false;

  StepOutcome withVerdict(VerificationVerdict v) => StepOutcome(
        step: step,
        result: result,
        verdict: v,
        attempt: attempt,
      );

  Map<String, Object?> toJson() => <String, Object?>{
        'step': step.toJson(),
        'result': result.toJson(),
        'verdict': verdict?.toJson(),
        'attempt': attempt,
      };
}

/// Immutable record of a completed reasoning episode.
class ThinkingTrace {
  const ThinkingTrace({
    required this.goal,
    required this.thoughts,
    required this.plans,
    required this.outcomes,
    required this.replans,
    required this.success,
    required this.duration,
    this.failureReason,
  });

  final String goal;

  /// Ordered narration of what the agent was thinking at each phase.
  final List<String> thoughts;

  /// Every plan produced, including repair plans.
  final List<Plan> plans;

  /// Every executed step, in execution order, including retries.
  final List<StepOutcome> outcomes;

  /// Number of repair cycles performed.
  final int replans;

  /// Whether every critical step succeeded and verified.
  final bool success;

  /// Total wall-clock time of the episode.
  final Duration duration;

  /// Populated when [success] is false.
  final String? failureReason;

  /// Steps that failed or failed verification.
  List<StepOutcome> get problems =>
      outcomes.where((StepOutcome o) => !o.verified).toList(growable: false);

  /// Total subagent invocations across all attempts.
  int get toolCallCount => outcomes.length;

  Map<String, Object?> toJson() => <String, Object?>{
        'goal': goal,
        'success': success,
        'replans': replans,
        'duration_ms': duration.inMilliseconds,
        'tool_calls': toolCallCount,
        'failure_reason': failureReason,
        'thoughts': thoughts,
        'plans': plans.map((Plan p) => p.toJson()).toList(growable: false),
        'outcomes':
            outcomes.map((StepOutcome o) => o.toJson()).toList(growable: false),
      };

  /// Markdown rendering for the UI's thinking panel.
  String toMarkdown() {
    final StringBuffer b = StringBuffer()
      ..writeln('## Thinking: ${success ? 'succeeded' : 'failed'}')
      ..writeln()
      ..writeln('**Goal:** $goal')
      ..writeln()
      ..writeln('**Duration:** ${duration.inMilliseconds}ms · '
          '**Tool calls:** $toolCallCount · **Replans:** $replans');
    if (failureReason != null) {
      b
        ..writeln()
        ..writeln('**Failure:** $failureReason');
    }
    b
      ..writeln()
      ..writeln('### Reasoning')
      ..writeln();
    for (final String thought in thoughts) {
      b.writeln('- $thought');
    }
    b
      ..writeln()
      ..writeln('### Steps')
      ..writeln();
    for (final StepOutcome o in outcomes) {
      final String mark = o.verified ? '✓' : '✗';
      b.writeln(
        '$mark **${o.step.index}. ${o.step.description}** '
        '(`${o.step.tool.name}`, attempt ${o.attempt}) — ${o.result.summary}',
      );
      final VerificationVerdict? v = o.verdict;
      if (v != null && v.issues.isNotEmpty) {
        for (final String issue in v.issues) {
          b.writeln('    - $issue');
        }
      }
    }
    return b.toString();
  }
}

/// Limits and tuning for an agent episode.
class ExtendedThinkingConfig {
  const ExtendedThinkingConfig({
    this.maxSteps = 24,
    this.maxReplanCycles = 2,
    this.maxParallel = 4,
    this.stepTimeout = const Duration(seconds: 30),
    this.requireVerification = true,
    this.failFastOnCriticalError = false,
  });

  /// Hard cap on steps in a single plan.
  final int maxSteps;

  /// How many repair cycles are allowed before giving up.
  final int maxReplanCycles;

  /// Upper bound on concurrent step execution.
  final int maxParallel;

  /// Default per-step deadline.
  final Duration stepTimeout;

  /// Whether a verification pass runs at all.
  final bool requireVerification;

  /// Whether a failed critical step aborts the remaining waves.
  final bool failFastOnCriticalError;
}

/// Progress event emitted during an episode, for live UI updates.
class ThinkingEvent {
  const ThinkingEvent({
    required this.phase,
    required this.message,
    this.stepIndex,
    this.data = const <String, Object?>{},
  });

  final ThinkingPhase phase;
  final String message;
  final int? stepIndex;
  final Map<String, Object?> data;

  Map<String, Object?> toJson() => <String, Object?>{
        'phase': phase.name,
        'message': message,
        'step_index': stepIndex,
        'data': data,
      };
}

/// Backend that produces plans and verifies results.
///
/// A model-backed implementation would prompt the on-device LLM here; the
/// heuristic implementation below is fully functional without model weights.
abstract class PlanningBackend {
  /// Produces a plan for [goal].
  ///
  /// [previousOutcomes] is empty for the initial plan and carries the failed
  /// steps for repair plans.
  Future<Plan> plan(
    String goal,
    Map<String, Object?> context,
    List<StepOutcome> previousOutcomes,
  );

  /// Judges whether an executed step achieved its intent.
  Future<VerificationVerdict> verify(
    ThoughtStep step,
    SubagentResult result,
    Map<String, Object?> context,
  );
}

/// Deterministic, model-free planner and verifier.
///
/// It performs genuine analysis of the goal text: quoted paths, bare paths with
/// recognised extensions, directories, verification/listing intents and
/// explicit commands are all extracted and turned into real subagent steps.
class HeuristicPlanningBackend implements PlanningBackend {
  HeuristicPlanningBackend();

  static final RegExp _quoted =
      RegExp(r'''["']([^"']+)["']''');

  static final RegExp _barePath = RegExp(
    r'(?<![\w./-])((?:\.{0,2}/)?[\w.@-]+(?:/[\w.@-]+)*\.'
    r'(?:dart|kt|java|swift|c|h|cpp|cs|go|rs|py|rb|php|js|ts|tsx|jsx|json|'
    r'jsonl|yaml|yml|csv|tsv|xml|ini|toml|properties|txt|md|log|zip|tar|gz|'
    r'tgz|jar|apk|pdf|png|jpg|jpeg|elf|so|bin|db|sqlite|wasm))',
    caseSensitive: false,
  );

  static final RegExp _directoryHint = RegExp(
    r'\b(?:directory|folder|codebase|repo|repository|project|tree)\b',
    caseSensitive: false,
  );

  static final RegExp _verifyHint = RegExp(
    r'\b(?:verify|validate|lint|check|audit|review|inspect for|quality)\b',
    caseSensitive: false,
  );

  static final RegExp _listHint = RegExp(
    r'\b(?:list|enumerate|show|display|what files|contents of)\b',
    caseSensitive: false,
  );

  static final RegExp _commandHint = RegExp(
    r'\b(?:run|execute|invoke)\s+([A-Za-z0-9_.-]+)((?:\s+[^\s;&|`$><]+)*)',
    caseSensitive: false,
  );

  static final RegExp _archiveHint = RegExp(
    r'\b(?:archive|zip|extract|contents|unpack)\b',
    caseSensitive: false,
  );

  /// Extracts candidate file paths mentioned in [goal].
  List<String> extractPaths(String goal) {
    final Set<String> paths = <String>{};
    for (final RegExpMatch m in _quoted.allMatches(goal)) {
      final String candidate = m.group(1)!.trim();
      if (candidate.isEmpty || candidate.contains(' ')) {
        continue;
      }
      // Only treat a quoted token as a path if it looks like one.
      if (candidate.contains('/') ||
          candidate.contains('.') ||
          UniversalFileReader.classifyExtension(candidate) != FileKind.unknown) {
        paths.add(candidate);
      }
    }
    for (final RegExpMatch m in _barePath.allMatches(goal)) {
      paths.add(m.group(1)!);
    }
    return paths.toList(growable: false);
  }

  @override
  Future<Plan> plan(
    String goal,
    Map<String, Object?> context,
    List<StepOutcome> previousOutcomes,
  ) async {
    if (previousOutcomes.isNotEmpty) {
      return _repairPlan(goal, context, previousOutcomes);
    }

    final List<ThoughtStep> steps = <ThoughtStep>[];
    final List<String> rationale = <String>[];

    final List<String> paths = extractPaths(goal);
    final String? explicitDirectory = _explicitDirectory(context, paths);
    final bool wantsVerification = _verifyHint.hasMatch(goal);
    final bool wantsListing =
        _listHint.hasMatch(goal) || _directoryHint.hasMatch(goal);
    final bool mentionsArchive = _archiveHint.hasMatch(goal);
    final RegExpMatch? command = _commandHint.firstMatch(goal);

    // 1. Reading steps: one per path so they can fan out concurrently.
    for (final String path in paths.take(math.max(0, 6))) {
      if (path == explicitDirectory) {
        continue;
      }
      final bool archive = UniversalFileReader.classifyExtension(path) ==
              FileKind.archive ||
          (mentionsArchive && path.endsWith('.tar.gz'));
      steps.add(
        ThoughtStep(
          index: steps.length,
          description: archive
              ? 'Open archive "$path" and list its entries'
              : 'Read and parse "$path"',
          tool: SubagentKind.fileParsing,
          args: <String, Object?>{'path': path},
          timeout: const Duration(seconds: 20),
          critical: true,
        ),
      );
      rationale.add('"$path" was named in the goal, so it must be read.');
    }

    // 2. Directory / codebase walk.
    if (explicitDirectory != null) {
      final List<int> deps = <int>[];
      steps.add(
        ThoughtStep(
          index: steps.length,
          description: 'Walk the codebase under "$explicitDirectory"',
          tool: SubagentKind.fileParsing,
          args: <String, Object?>{
            'directory': explicitDirectory,
            if (context['extensions'] != null)
              'extensions': context['extensions'] as Object,
          },
          dependsOn: deps,
          timeout: const Duration(seconds: 45),
          critical: true,
        ),
      );
      rationale.add(
        'A directory was referenced, so a recursive walk collects the sources.',
      );
    }

    // 3. Explicit command execution.
    if (command != null) {
      final String exe = command.group(1)!;
      final List<String> args = (command.group(2) ?? '')
          .trim()
          .split(RegExp(r'\s+'))
          .where((String s) => s.isNotEmpty)
          .toList(growable: false);
      steps.add(
        ThoughtStep(
          index: steps.length,
          description: 'Run allow-listed command "$exe ${args.join(' ')}"'
              .trim(),
          tool: SubagentKind.systemCommand,
          args: <String, Object?>{
            'command': exe,
            'args': args,
            if (explicitDirectory != null) 'working_directory': explicitDirectory,
          },
          timeout: const Duration(seconds: 60),
          critical: false,
        ),
      );
      rationale.add(
        'The goal asked to run a command; it will be validated against the '
        'execution allow-list before spawning.',
      );
    }

    // 4. Listing intent without an explicit target still warrants a listing.
    if (wantsListing && explicitDirectory == null && paths.isEmpty) {
      steps.add(
        ThoughtStep(
          index: steps.length,
          description: 'List the working directory contents',
          tool: SubagentKind.systemCommand,
          args: const <String, Object?>{'command': 'ls', 'args': <String>['-la']},
          timeout: const Duration(seconds: 15),
          critical: false,
        ),
      );
      rationale.add('The goal asks for a listing but named no path.');
    }

    // 5. Verification step, dependent on everything it inspects.
    if (wantsVerification || steps.isEmpty) {
      final List<int> deps = <int>[];
      // Verify the sources produced by read/walk steps.
      for (final ThoughtStep s in steps) {
        if (s.tool == SubagentKind.fileParsing) {
          deps.add(s.index);
        }
      }
      final bool canVerifyDirectory = explicitDirectory != null &&
          _looksLikeSourceDirectory(explicitDirectory);
      if (steps.isEmpty) {
        // Nothing was recognisable; still produce a useful, safe first step.
        steps.add(
          const ThoughtStep(
            index: 0,
            description: 'List the working directory to discover targets',
            tool: SubagentKind.systemCommand,
            args: <String, Object?>{
              'command': 'ls',
              'args': <String>['-la'],
            },
            timeout: Duration(seconds: 15),
            critical: false,
          ),
        );
        deps.clear();
        rationale.add(
          'No concrete target was recognised, so the first step is discovery.',
        );
      }
      steps.add(
        ThoughtStep(
          index: steps.length,
          description: canVerifyDirectory
              ? 'Statically verify sources under "$explicitDirectory"'
              : 'Statically verify the parsed sources for structural defects',
          tool: SubagentKind.codeVerification,
          args: <String, Object?>{
            if (canVerifyDirectory) 'directory': explicitDirectory,
            if (!canVerifyDirectory && paths.isNotEmpty)
              'path': _firstSourcePath(paths) ?? paths.first,
          },
          dependsOn: deps,
          timeout: const Duration(seconds: 45),
          critical: false,
        ),
      );
      rationale.add(
        'A deterministic verification pass catches defects the model should '
        'not have to re-derive.',
      );
    }

    if (steps.isEmpty) {
      steps.add(
        const ThoughtStep(
          index: 0,
          description: 'List the working directory',
          tool: SubagentKind.systemCommand,
          args: <String, Object?>{'command': 'ls', 'args': <String>['-la']},
          timeout: Duration(seconds: 15),
          critical: false,
        ),
      );
    }

    final Plan plan = Plan(
      goal: goal,
      steps: List<ThoughtStep>.unmodifiable(steps),
      rationale: rationale.isEmpty
          ? 'Single-step plan: the goal mapped directly to one tool call.'
          : rationale.join(' '),
    );
    plan.validate();
    return plan;
  }

  String? _explicitDirectory(
    Map<String, Object?> context,
    List<String> paths,
  ) {
    final Object? explicit = context['root'];
    if (explicit is String && explicit.isNotEmpty) {
      return explicit;
    }
    for (final String p in paths) {
      if (UniversalFileReader.classifyExtension(p) == FileKind.unknown) {
        return p;
      }
    }
    if (context['directory'] is String) {
      return context['directory'] as String;
    }
    return null;
  }

  bool _looksLikeSourceDirectory(String path) {
    final FileKind kind = UniversalFileReader.classifyExtension(path);
    return kind == FileKind.unknown || kind == FileKind.text;
  }

  String? _firstSourcePath(List<String> paths) {
    for (final String p in paths) {
      if (UniversalFileReader.classifyExtension(p) == FileKind.code) {
        return p;
      }
    }
    return null;
  }

  /// Builds a repair plan for the failed steps in [previousOutcomes].
  Future<Plan> _repairPlan(
    String goal,
    Map<String, Object?> context,
    List<StepOutcome> previousOutcomes,
  ) {
    final List<ThoughtStep> steps = <ThoughtStep>[];
    final List<String> rationale = <String>[];

    for (final StepOutcome outcome in previousOutcomes) {
      final ThoughtStep failed = outcome.step;
      final String error = outcome.result.error ?? 'unknown failure';
      rationale.add(
        'Step ${failed.index} ("${failed.description}") failed: $error',
      );

      final Object? path = failed.args['path'];
      switch (failed.tool) {
        case SubagentKind.fileParsing:
          if (path is String && path.isNotEmpty) {
            final String? parent = _parentDirectory(path);
            // Recover by locating the file rather than re-reading blindly.
            steps.add(
              ThoughtStep(
                index: steps.length,
                description: 'Locate "$path" on disk before retrying',
                tool: SubagentKind.systemCommand,
                args: <String, Object?>{
                  'command': 'find',
                  'args': <String>[
                    parent ?? '.',
                    '-maxdepth',
                    '3',
                    '-name',
                    _basename(path),
                  ],
                },
                timeout: const Duration(seconds: 20),
                critical: false,
              ),
            );
            rationale.add(
              'Repair: locate the missing target with find before retrying the '
              'read.',
            );
          } else {
            steps.add(
              ThoughtStep(
                index: steps.length,
                description: 'List the working directory to re-establish targets',
                tool: SubagentKind.systemCommand,
                args: const <String, Object?>{
                  'command': 'ls',
                  'args': <String>['-la'],
                },
                timeout: const Duration(seconds: 15),
                critical: false,
              ),
            );
          }
          break;
        case SubagentKind.systemCommand:
          final Object? exe = failed.args['command'];
          if (exe is String) {
            steps.add(
              ThoughtStep(
                index: steps.length,
                description: 'Confirm "$exe" is available via file lookup',
                tool: SubagentKind.systemCommand,
                args: <String, Object?>{'command': 'which', 'args': <String>[exe]},
                timeout: const Duration(seconds: 10),
                critical: false,
              ),
            );
            rationale.add(
              'Repair: check the executable exists instead of retrying the '
              'same failing invocation.',
            );
          }
          break;
        case SubagentKind.codeVerification:
          final Object? verifyPath = failed.args['path'];
          if (verifyPath is String && verifyPath.isNotEmpty) {
            steps.add(
              ThoughtStep(
                index: steps.length,
                description: 'Re-read "$verifyPath" as raw text to inspect '
                    'the reported defect',
                tool: SubagentKind.fileParsing,
                args: <String, Object?>{'path': verifyPath},
                timeout: const Duration(seconds: 20),
                critical: false,
              ),
            );
            rationale.add(
              'Repair: re-read the offending file so the caller can show the '
              'defect in context.',
            );
          }
          break;
        case SubagentKind.custom:
          break;
      }
    }

    if (steps.isEmpty) {
      rationale.add(
        'No automated repair is available for the reported failure; the agent '
        'will report it rather than retry blindly.',
      );
      steps.add(
        const ThoughtStep(
          index: 0,
          description: 'Collect diagnostic context for the unresolved failure',
          tool: SubagentKind.systemCommand,
          args: <String, Object?>{
            'command': 'ls',
            'args': <String>['-la'],
          },
          timeout: Duration(seconds: 15),
          critical: false,
        ),
      );
    }

    final Plan plan = Plan(
      goal: goal,
      steps: List<ThoughtStep>.unmodifiable(steps),
      rationale: rationale.join(' '),
      replanIndex: previousOutcomes
              .map((StepOutcome o) => o.attempt)
              .fold<int>(0, math.max) +
          1,
    );
    plan.validate();
    return Future<Plan>.value(plan);
  }

  static String? _parentDirectory(String path) {
    final int slash = path.lastIndexOf('/');
    if (slash <= 0) {
      return null;
    }
    return path.substring(0, slash);
  }

  static String _basename(String path) {
    final int slash = path.lastIndexOf('/');
    return slash < 0 ? path : path.substring(slash + 1);
  }

  @override
  Future<VerificationVerdict> verify(
    ThoughtStep step,
    SubagentResult result,
    Map<String, Object?> context,
  ) async {
    final List<String> issues = <String>[];

    if (!result.success) {
      issues.add('tool reported failure: ${result.error ?? result.summary}');
      return VerificationVerdict(
        passed: false,
        note: 'Step failed during execution',
        issues: issues,
      );
    }

    final Map<String, Object?> meta = result.metadata;
    switch (step.tool) {
      case SubagentKind.fileParsing:
        final Object? path = step.args['path'];
        final Object? directory = step.args['directory'];
        if (path is String) {
          final Object? kind = meta['kind'];
          if (kind == null) {
            issues.add('read produced no file classification');
          }
          final Object? bytes = meta['size_bytes'];
          if (bytes is num && bytes <= 0) {
            issues.add('file "$path" is empty (0 bytes)');
          }
        } else if (directory is String) {
          final Object? count = meta['file_count'];
          if (count is num && count == 0) {
            issues.add('no readable files found under "$directory"');
          }
        } else if (step.args['paths'] is List) {
          final Object? count = meta['file_count'];
          if (count is num && count == 0) {
            issues.add('none of the requested paths could be parsed');
          }
        }
        if (result.output == null) {
          issues.add('parser returned no structured output');
        }
        break;

      case SubagentKind.systemCommand:
        final Object? exit = meta['exit_code'];
        if (exit is num && exit != 0) {
          issues.add('command exited with code $exit');
        }
        break;

      case SubagentKind.codeVerification:
        final Object? errors = meta['error_count'];
        if (errors is num && errors > 0) {
          issues.add('$errors structural error(s) reported');
        }
        final Object? checked = meta['files_checked'];
        if (checked is num && checked == 0) {
          issues.add('verification inspected zero files');
        }
        break;

      case SubagentKind.custom:
        break;
    }

    final bool passed = issues.isEmpty;
    return VerificationVerdict(
      passed: passed,
      note: passed
          ? 'Verified: ${result.summary}'
          : 'Verification found ${issues.length} issue(s)',
      issues: List<String>.unmodifiable(issues),
    );
  }
}

/// The extended-thinking agent: plans, fans out, verifies, replans.
class ExtendedThinkingAgent {
  ExtendedThinkingAgent({
    required this.runner,
    PlanningBackend? backend,
    this.config = const ExtendedThinkingConfig(),
    this.onEvent,
  }) : backend = backend ?? HeuristicPlanningBackend();

  /// Executor used to run steps.
  final SubagentRunner runner;

  /// Planner + verifier.
  final PlanningBackend backend;

  /// Episode limits.
  final ExtendedThinkingConfig config;

  /// Live progress observer for the UI.
  final void Function(ThinkingEvent event)? onEvent;

  final List<String> _thoughts = <String>[];

  void _think(ThinkingPhase phase, String message, {int? stepIndex}) {
    _thoughts.add('[${phase.label}] $message');
    onEvent?.call(
      ThinkingEvent(phase: phase, message: message, stepIndex: stepIndex),
    );
  }

  /// Runs a full Plan -> Execute -> Verify episode for [goal].
  ///
  /// [context] may carry `root`, `directory`, `extensions` and any other hints
  /// the planner or verifier should consider.
  Future<ThinkingTrace> solve(
    String goal, {
    Map<String, Object?> context = const <String, Object?>{},
  }) async {
    final Stopwatch sw = Stopwatch()..start();
    _thoughts.clear();

    final List<Plan> plans = <Plan>[];
    final List<StepOutcome> allOutcomes = <StepOutcome>[];
    int replans = 0;
    String? failureReason;

    _think(ThinkingPhase.planning, 'Analysing goal: "$goal"');
    Plan plan;
    try {
      plan = await backend.plan(goal, context, const <StepOutcome>[]);
      plan.validate();
    } on PlanValidationException catch (e) {
      sw.stop();
      _think(ThinkingPhase.failed, 'Planning failed: ${e.message}');
      return ThinkingTrace(
        goal: goal,
        thoughts: List<String>.unmodifiable(_thoughts),
        plans: const <Plan>[],
        outcomes: const <StepOutcome>[],
        replans: 0,
        success: false,
        duration: sw.elapsed,
        failureReason: 'invalid plan: ${e.message}',
      );
    }

    if (plan.steps.length > config.maxSteps) {
      _think(
        ThinkingPhase.planning,
        'Plan had ${plan.steps.length} steps; truncating to ${config.maxSteps}',
      );
      plan = Plan(
        goal: plan.goal,
        steps: List<ThoughtStep>.unmodifiable(
          plan.steps.take(config.maxSteps).map(
                (ThoughtStep s) => ThoughtStep(
                  index: s.index,
                  description: s.description,
                  tool: s.tool,
                  args: s.args,
                  dependsOn: s.dependsOn
                      .where((int d) => d < config.maxSteps)
                      .toList(growable: false),
                  critical: s.critical,
                  timeout: s.timeout,
                ),
              ),
        ),
        rationale: plan.rationale,
        replanIndex: plan.replanIndex,
      );
      plan.validate();
    }

    plans.add(plan);
    _think(
      ThinkingPhase.planning,
      'Produced ${plan.steps.length} step(s): ${plan.rationale}',
    );

    while (true) {
      final _EpisodeResult episode = await _executePlan(plan, context);
      allOutcomes.addAll(episode.outcomes);

      if (episode.criticalFailure != null) {
        failureReason = episode.criticalFailure;
        _think(ThinkingPhase.failed, 'Critical step failed: $failureReason');
        break;
      }

      final List<StepOutcome> problems = episode.outcomes
          .where((StepOutcome o) => !o.verified && o.step.critical)
          .toList(growable: false);

      if (problems.isEmpty) {
        _think(ThinkingPhase.complete, 'All critical steps verified.');
        break;
      }

      if (replans >= config.maxReplanCycles) {
        failureReason =
            '${problems.length} critical step(s) failed and the replan budget '
            'of ${config.maxReplanCycles} is exhausted';
        _think(ThinkingPhase.failed, failureReason);
        break;
      }

      replans++;
      _think(
        ThinkingPhase.replanning,
        'Replan cycle $replans of ${config.maxReplanCycles} for '
        '${problems.length} unresolved step(s)',
      );

      Plan repair;
      try {
        repair = await backend.plan(goal, context, problems);
        repair.validate();
      } on PlanValidationException catch (e) {
        failureReason = 'repair plan was invalid: ${e.message}';
        _think(ThinkingPhase.failed, failureReason);
        break;
      }
      plans.add(repair);
      _think(
        ThinkingPhase.replanning,
        'Repair plan: ${repair.rationale}',
      );
      plan = repair;
    }

    sw.stop();
    final bool success = failureReason == null;
    final ThinkingTrace trace = ThinkingTrace(
      goal: goal,
      thoughts: List<String>.unmodifiable(_thoughts),
      plans: List<Plan>.unmodifiable(plans),
      outcomes: List<StepOutcome>.unmodifiable(allOutcomes),
      replans: replans,
      success: success,
      duration: sw.elapsed,
      failureReason: failureReason,
    );
    onEvent?.call(
      ThinkingEvent(
        phase: success ? ThinkingPhase.complete : ThinkingPhase.failed,
        message: success
            ? 'Episode complete in ${sw.elapsedMilliseconds}ms'
            : 'Episode failed: $failureReason',
      ),
    );
    return trace;
  }

  Future<_EpisodeResult> _executePlan(
    Plan plan,
    Map<String, Object?> context,
  ) async {
    final List<List<ThoughtStep>> waves = plan.executionWaves();
    final List<StepOutcome> outcomes = <StepOutcome>[];
    String? criticalFailure;

    for (int w = 0; w < waves.length; w++) {
      final List<ThoughtStep> wave = waves[w];
      _think(
        ThinkingPhase.executing,
        'Wave ${w + 1}/${waves.length}: dispatching ${wave.length} '
        'step(s) concurrently',
      );

      // Respect the parallelism ceiling by chunking the wave if needed.
      final int chunkSize = math.max(1, config.maxParallel);
      for (int i = 0; i < wave.length; i += chunkSize) {
        final List<ThoughtStep> chunk = wave.sublist(
          i,
          math.min(i + chunkSize, wave.length),
        );
        final List<SubagentTask> tasks = chunk
            .map((ThoughtStep s) => s.toTask('plan${plan.replanIndex}'))
            .toList(growable: false);
        final List<SubagentResult> results = await runner.runAll(tasks);

        _think(ThinkingPhase.verifying, 'Verifying ${results.length} result(s)');

        for (int r = 0; r < chunk.length; r++) {
          final ThoughtStep step = chunk[r];
          final SubagentResult result = results[r];
          _think(
            ThinkingPhase.executing,
            'Step ${step.index} "${step.description}" -> '
            '${result.success ? 'ok' : 'failed'}: ${result.summary}',
            stepIndex: step.index,
          );

          VerificationVerdict verdict;
          if (config.requireVerification) {
            verdict = await backend.verify(step, result, context);
          } else {
            verdict = VerificationVerdict(
              passed: result.success,
              note: result.success ? 'Execution succeeded' : 'Execution failed',
            );
          }

          final StepOutcome outcome = StepOutcome(
            step: step,
            result: result,
            verdict: verdict,
          );
          outcomes.add(outcome);
          if (!verdict.passed) {
            _think(
              ThinkingPhase.verifying,
              'Step ${step.index} did not verify: ${verdict.issues.join('; ')}',
              stepIndex: step.index,
            );
          }
        }
      }

      if (config.failFastOnCriticalError) {
        StepOutcome? fatal;
        for (final StepOutcome o in outcomes) {
          if (o.step.critical && !o.verified) {
            fatal = o;
            break;
          }
        }
        if (fatal != null) {
          criticalFailure =
              'step ${fatal.step.index} ("${fatal.step.description}") failed: '
              '${fatal.result.error ?? 'verification failed'}';
          break;
        }
      }
    }

    return _EpisodeResult(
      outcomes: outcomes,
      criticalFailure: criticalFailure,
    );
  }

  /// Convenience: solves [goal] and returns only the markdown trace.
  Future<String> explain(String goal, {Map<String, Object?>? context}) async {
    final ThinkingTrace trace = await solve(
      goal,
      context: context ?? const <String, Object?>{},
    );
    return trace.toMarkdown();
  }
}

class _EpisodeResult {
  const _EpisodeResult({required this.outcomes, this.criticalFailure});

  final List<StepOutcome> outcomes;
  final String? criticalFailure;
}