// Tests for the reasoning layer: planning, depth selection and verification.
//
// These exercise pure functions over the planner, so every case is exact: no
// model, no tokenizer, no clock beyond the timestamps the tests pass in. A
// reasoning layer that is not deterministic is not debuggable, and pinning the
// outputs here is what keeps it that way.

import 'package:harbor_core/harbor_core.dart';
import 'package:test/test.dart';

DateTime _at(int hoursAgo) =>
    DateTime.now().toUtc().subtract(Duration(hours: hoursAgo));

void main() {
  group('ThinkingStrategy', () {
    test('only none is disabled', () {
      expect(ThinkingStrategy.none.isEnabled, isFalse);
      expect(ThinkingStrategy.concise.isEnabled, isTrue);
      expect(ThinkingStrategy.thorough.isEnabled, isTrue);
    });

    test('fromWire round-trips every value and defaults on junk', () {
      for (final ThinkingStrategy strategy in ThinkingStrategy.values) {
        expect(ThinkingStrategy.fromWire(strategy.wire), strategy);
      }
      expect(ThinkingStrategy.fromWire('banana'), ThinkingStrategy.none);
      expect(ThinkingStrategy.fromWire(null), ThinkingStrategy.none);
    });
  });

  group('ReasoningPlanner.suggest', () {
    test('a greeting or acknowledgement gets no reasoning', () {
      for (final String trivial in <String>[
        'hi',
        'hello there',
        'thanks',
        'ok',
        'good morning',
      ]) {
        expect(ReasoningPlanner.suggest(trivial), ThinkingStrategy.none, reason: trivial);
      }
      expect(ReasoningPlanner.suggest('   '), ThinkingStrategy.none);
      expect(ReasoningPlanner.suggest(''), ThinkingStrategy.none);
    });

    test('an ordinary question gets a single check', () {
      expect(
        ReasoningPlanner.suggest('What is the capital of France?'),
        ThinkingStrategy.concise,
      );
    });

    test('an analysis or multi-part ask gets the full plan', () {
      expect(
        ReasoningPlanner.suggest(
          'Compare the trade-offs of MLA versus standard attention for '
          'on-device inference, then explain which wins when memory is the '
          'limit, and say what you would measure',
        ),
        ThinkingStrategy.thorough,
      );
      expect(
        ReasoningPlanner.suggest('Debug this crash step by step'),
        ThinkingStrategy.thorough,
      );
      // Two question marks alone is one signal, not two — this asks for a
      // full plan only because the content is genuinely multi-part.
      expect(
        ReasoningPlanner.suggest(
          'Why is the build red, and how do I fix it? Compare the log lines '
          'between step 1 and step 2 and design the storage migration, then '
          'debug the crash and implement the rollback',
        ),
        ThinkingStrategy.thorough,
      );
    });
  });

  group('ReasoningPlanner.plan', () {
    test('concise yields a dense, all-pending plan with a rationale', () {
      final ThinkingPlan plan = const ReasoningPlanner().plan(
        'What is the capital of France?',
        strategy: ThinkingStrategy.concise,
        at: _at(0),
      );
      expect(plan.strategy, ThinkingStrategy.concise);
      expect(plan.length, greaterThanOrEqualTo(3));
      for (int i = 0; i < plan.length; i++) {
        expect(plan.steps[i].index, i);
        expect(plan.steps[i].status, ThinkStepStatus.pending);
        expect(plan.steps[i].title, isNotEmpty);
      }
      expect(plan.rationale, isNotEmpty);
      expect(plan.isEmpty, isFalse);
    });

    test('thorough decomposes into more steps than concise', () {
      const ReasoningPlanner planner = ReasoningPlanner();
      final ThinkingPlan concise = planner.plan(
        'Explain X',
        strategy: ThinkingStrategy.concise,
      );
      final ThinkingPlan thorough = planner.plan(
        'Explain X',
        strategy: ThinkingStrategy.thorough,
      );
      expect(thorough.length, greaterThan(concise.length));
      expect(
        thorough.steps.any((ThinkStep s) =>
            s.title.toLowerCase().contains('decompose'),
          ),
        isTrue,
      );
    });

    test('a memory step appears only when memory actually contributed', () {
      const ReasoningPlanner planner = ReasoningPlanner();
      bool hasRecall(ThinkingPlan p) => p.steps
          .any((ThinkStep s) => s.title.toLowerCase().contains('recall'));
      expect(
        hasRecall(
          planner.plan('q', strategy: ThinkingStrategy.concise, hasMemory: true),
        ),
        isTrue,
      );
      expect(
        hasRecall(
          planner.plan('q', strategy: ThinkingStrategy.concise, hasMemory: false),
        ),
        isFalse,
      );
    });

    test('the tool step wording changes with tool availability', () {
      const ReasoningPlanner planner = ReasoningPlanner();
      final String withTools = planner
          .plan(
            'q',
            strategy: ThinkingStrategy.thorough,
            hasTools: true,
          )
          .steps
          .firstWhere((ThinkStep s) => s.title.toLowerCase().contains('facts'))
          .detail;
      final String without = planner
          .plan(
            'q',
            strategy: ThinkingStrategy.thorough,
            hasTools: false,
          )
          .steps
          .firstWhere((ThinkStep s) => s.title.toLowerCase().contains('facts'))
          .detail;
      expect(withTools, isNot(equals(without)));
    });

    test('a long question is truncated for display', () {
      final ThinkingPlan plan = const ReasoningPlanner().plan(
        'a' * 400,
        strategy: ThinkingStrategy.concise,
      );
      expect(plan.question.length, lessThanOrEqualTo(201));
      expect(plan.question, endsWith('…'));
    });

    test('toJson serialises the plan and every step', () {
      final ThinkingPlan plan = const ReasoningPlanner().plan(
        'hi there friend',
        strategy: ThinkingStrategy.thorough,
      );
      final Map<String, Object?> json = plan.toJson();
      expect(json['strategy'], 'thorough');
      expect(json['created_at'], isA<String>());
      expect(json['steps'], isA<List<Object?>>());
      final Map<String, Object?> first =
          (json['steps']! as List<Object?>).first! as Map<String, Object?>;
      expect(first['status'], 'pending');
      expect(first['index'], 0);
    });

    test('withSteps keeps everything but the steps', () {
      const ReasoningPlanner planner = ReasoningPlanner();
      final ThinkingPlan original =
          planner.plan('question', strategy: ThinkingStrategy.concise);
      final ThinkingPlan updated = original.withSteps(<ThinkStep>[
        original.steps.first.copyWith(status: ThinkStepStatus.done),
      ]);
      expect(updated.question, original.question);
      expect(updated.strategy, original.strategy);
      expect(updated.rationale, original.rationale);
      expect(updated.length, 1);
      expect(updated.steps.first.status, ThinkStepStatus.done);
    });
  });

  group('ReasoningPlanner.verify', () {
    const ReasoningPlanner planner = ReasoningPlanner();

    test('an on-topic draft passes with no findings', () {
      final VerificationResult result = planner.verify(
        question: 'What is the capital of France?',
        draft: 'The capital of France is Paris, in Europe.',
        usedTool: false,
      );
      expect(result.verdict, VerificationVerdict.ok);
      expect(result.findings, isEmpty);
      expect(result.needsRevision, isFalse);
    });

    test('a near-empty draft is flagged', () {
      final VerificationResult result = planner.verify(
        question: 'Explain photosynthesis in a few sentences.',
        draft: 'ok',
        usedTool: false,
      );
      expect(result.needsRevision, isTrue);
      expect(
        result.findings.map((ReasoningFinding f) => f.code),
        contains('empty_draft'),
      );
    });

    test('a repeating draft is flagged as degenerate', () {
      for (final String draft in <String>[
        'the the the the the',
        'I am I am I am I am',
        '!!! ??? !!! ??? !!!',
      ]) {
        final VerificationResult result = planner.verify(
          question: 'What is the weather in Lisbon today?',
          draft: draft,
          usedTool: false,
        );
        expect(
          result.findings.map((ReasoningFinding f) => f.code),
          contains('repetition'),
          reason: draft,
        );
      }
    });

    test('an off-topic draft is flagged', () {
      final VerificationResult result = planner.verify(
        question: 'How do I tune a bicycle gearbox?',
        draft: 'The mitochondria is the powerhouse of the cell, generally.',
        usedTool: false,
      );
      expect(
        result.findings.map((ReasoningFinding f) => f.code),
        contains('off_topic'),
      );
    });

    test('claiming a tool result that never ran is flagged', () {
      final VerificationResult lied = planner.verify(
        question: 'What is my battery level right now?',
        draft: 'According to the tool your battery is at sixty percent full.',
        usedTool: false,
      );
      expect(
        lied.findings.map((ReasoningFinding f) => f.code),
        contains('unsupported_tool_claim'),
      );

      final VerificationResult told = planner.verify(
        question: 'What is my battery level right now?',
        draft: 'According to the tool your battery is at sixty percent full.',
        usedTool: true,
      );
      expect(
        told.findings.map((ReasoningFinding f) => f.code),
        isNot(contains('unsupported_tool_claim')),
      );
    });

    test('a hypothetical tool mention is not a false claim', () {
      final VerificationResult result = planner.verify(
        question: 'Will it rain in my city tomorrow?',
        draft: 'If you tell me your city I can check the weather for you.',
        usedTool: false,
      );
      expect(
        result.findings.map((ReasoningFinding f) => f.code),
        isNot(contains('unsupported_tool_claim')),
      );
    });

    test('a tool name in the catalogue trips the claim check', () {
      final VerificationResult result = planner.verify(
        question: 'How much storage do I have left on device?',
        draft: 'The storage figure on this device looks pretty healthy today.',
        usedTool: false,
        toolNames: <String>{'device.storage'},
      );
      expect(
        result.findings.map((ReasoningFinding f) => f.code),
        contains('unsupported_tool_claim'),
      );
    });

    test('minAnswerChars is honoured', () {
      final VerificationResult result = planner.verify(
        question: 'Name a prime number larger than seven.',
        draft: 'Eleven is prime.',
        usedTool: false,
        minAnswerChars: 4,
      );
      expect(result.findings, isEmpty);
    });
  });

  group('ReasoningPlanner helpers', () {
    test('isDegenerate separates loops from sentences', () {
      expect(ReasoningPlanner.isDegenerate(''), isTrue);
      expect(ReasoningPlanner.isDegenerate('!!! ???'), isTrue);
      expect(ReasoningPlanner.isDegenerate('the the the the'), isTrue);
      expect(
        ReasoningPlanner.isDegenerate('The harbor light kept the ships safe.'),
        isFalse,
      );
    });

    test('repairPrompt quotes the question, the draft and the fault', () {
      const ReasoningPlanner planner = ReasoningPlanner();
      final String prompt = planner.repairPrompt(
        question: 'What is the capital of France?',
        draft: 'the the the the',
        findings: const <ReasoningFinding>[
          ReasoningFinding('repetition', 'The draft repeats itself.'),
        ],
      );
      expect(prompt, contains('capital of France'));
      expect(prompt, contains('the the the the'));
      expect(prompt, contains('repeats itself'));
    });
  });

  group('ThinkingTrace', () {
    test('summary reports the strategy, the step tally and the outcome', () {
      const ReasoningPlanner planner = ReasoningPlanner();
      final ThinkingPlan drafted = planner.plan(
        'Explain X',
        strategy: ThinkingStrategy.thorough,
      );
      final ThinkingPlan plan = drafted.withSteps(<ThinkStep>[
        for (final ThinkStep step in drafted.steps)
          step.copyWith(status: ThinkStepStatus.done),
      ]);
      final ThinkingTrace accepted = ThinkingTrace(
        plan: plan,
        verdict: VerificationVerdict.ok,
        findings: const <ReasoningFinding>[],
        checkedDraft: 'a draft',
        finalText: 'a draft',
        revisions: 0,
        elapsed: const Duration(milliseconds: 12),
      );
      expect(accepted.summary, contains('Thorough'));
      expect(accepted.summary, contains('${plan.length}/${plan.length}'));
      expect(accepted.summary, contains('accepted'));
      expect(accepted.revised, isFalse);

      final ThinkingTrace revised = ThinkingTrace(
        plan: plan,
        verdict: VerificationVerdict.revise,
        findings: const <ReasoningFinding>[
          ReasoningFinding('off_topic', 'nope'),
        ],
        checkedDraft: 'nope',
        finalText: 'better',
        revisions: 1,
        elapsed: const Duration(milliseconds: 20),
      );
      expect(revised.revised, isTrue);
      expect(revised.summary, contains('revised'));
    });

    test('toJson exposes plan, verdict, findings and revision state', () {
      const ReasoningPlanner planner = ReasoningPlanner();
      final ThinkingTrace trace = ThinkingTrace(
        plan: planner.plan('q', strategy: ThinkingStrategy.concise),
        verdict: VerificationVerdict.revise,
        findings: const <ReasoningFinding>[
          ReasoningFinding('empty_draft', 'too short'),
        ],
        checkedDraft: 'x',
        finalText: 'y',
        revisions: 1,
        elapsed: const Duration(milliseconds: 5),
      );
      final Map<String, Object?> json = trace.toJson();
      expect(json['verdict'], 'revise');
      expect(json['revisions'], 1);
      expect(json['revised'], isTrue);
      expect(json['plan'], isA<Map<String, Object?>>());
      expect(json['findings'], hasLength(1));
    });
  });
}
