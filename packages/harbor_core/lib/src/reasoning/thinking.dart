// Extended thinking for a model too small to think on its own.
//
// The reasoning here is *explicit* rather than emergent: the engine already
// controls the generation loop, so it can do what a large model is trained to do
// implicitly — decompose the question, answer, then check the answer against the
// question before showing it. Every part of that process is materialised as a
// data structure, so the UI can render the reasoning, the tests can assert it,
// and a failure is attributable to a named step rather than to "the model".
//
// The design rule is that thinking must never make an answer worse. It may add
// steps to the *record* and at most one extra generation pass to *repair* a
// draft that fails a check; when the checks pass, the draft the user sees is
// exactly what a non-thinking turn would have produced.

/// How much reasoning a turn performs.
enum ThinkingStrategy {
  /// Answer directly. No thinking events, no extra pass.
  ///
  /// This is the default so a caller that knows nothing about this file keeps
  /// the behaviour it had before thinking existed.
  none('none', 'Direct'),

  /// A short plan and one verification pass, for ordinary questions.
  concise('concise', 'Concise'),

  /// A fuller decomposition, for questions with several parts or a tool step.
  thorough('thorough', 'Thorough');

  const ThinkingStrategy(this.wire, this.label);

  /// Stable name written to the wire form and to storage.
  final String wire;

  /// Human label for the UI.
  final String label;

  /// Whether this strategy produces any thinking events at all.
  bool get isEnabled => this != ThinkingStrategy.none;

  /// Parses [wire], defaulting to [none] for an unknown value.
  static ThinkingStrategy fromWire(String? value) {
    for (final ThinkingStrategy strategy in ThinkingStrategy.values) {
      if (strategy.wire == value) {
        return strategy;
      }
    }
    return ThinkingStrategy.none;
  }
}

/// Lifecycle of one planned step.
enum ThinkStepStatus {
  /// Not started.
  pending('pending'),

  /// Currently being worked on.
  running('running'),

  /// Finished successfully.
  done('done'),

  /// Finished and failed.
  failed('failed'),

  /// Skipped because a dependency failed or the step was unnecessary.
  skipped('skipped');

  const ThinkStepStatus(this.wire);

  /// Stable name written to the wire form.
  final String wire;
}

/// One step in a reasoning plan.
class ThinkStep {
  /// Creates a step.
  const ThinkStep({
    required this.index,
    required this.title,
    required this.detail,
    this.status = ThinkStepStatus.pending,
    this.result,
  });

  /// Position within the plan, from zero.
  final int index;

  /// Short imperative title, e.g. `Recall what is known`.
  final String title;

  /// One sentence explaining what this step does and why.
  final String detail;

  /// Current lifecycle state.
  final ThinkStepStatus status;

  /// What the step concluded, once it has run.
  final String? result;

  /// A copy with selected fields replaced.
  ThinkStep copyWith({ThinkStepStatus? status, String? result}) => ThinkStep(
        index: index,
        title: title,
        detail: detail,
        status: status ?? this.status,
        result: result ?? this.result,
      );

  /// Wire form for the SSE payload and the dashboard.
  Map<String, Object?> toJson() => <String, Object?>{
        'index': index,
        'title': title,
        'detail': detail,
        'status': status.wire,
        if (result != null) 'result': result,
      };
}

/// A reasoning plan: an ordered, costed decomposition of one turn.
class ThinkingPlan {
  /// Creates a plan.
  const ThinkingPlan({
    required this.question,
    required this.strategy,
    required this.steps,
    required this.rationale,
    required this.createdAt,
  });

  /// The user turn this plan is for, trimmed for display.
  final String question;

  /// The depth of reasoning requested.
  final ThinkingStrategy strategy;

  /// Steps in index order.
  final List<ThinkStep> steps;

  /// One sentence on why this decomposition was chosen.
  final String rationale;

  /// When the plan was made, in UTC.
  final DateTime createdAt;

  /// Number of steps.
  int get length => steps.length;

  /// Whether the plan contains any step at all.
  bool get isEmpty => steps.isEmpty;

  /// A copy with [steps] replaced, e.g. once statuses advance.
  ThinkingPlan withSteps(List<ThinkStep> steps) => ThinkingPlan(
        question: question,
        strategy: strategy,
        steps: List<ThinkStep>.unmodifiable(steps),
        rationale: rationale,
        createdAt: createdAt,
      );

  /// Wire form for the SSE payload and the dashboard.
  Map<String, Object?> toJson() => <String, Object?>{
        'question': question,
        'strategy': strategy.wire,
        'rationale': rationale,
        'created_at': createdAt.toUtc().toIso8601String(),
        'steps': <Object?>[
          for (final ThinkStep step in steps) step.toJson(),
        ],
      };
}

/// The result of checking a draft answer.
enum VerificationVerdict {
  /// The draft addressed the question; it is shown unchanged.
  ok('ok'),

  /// The draft failed a check and one repair pass is worth running.
  revise('revise');

  const VerificationVerdict(this.wire);

  /// Stable name written to the wire form.
  final String wire;
}

/// One thing a verification pass observed.
class ReasoningFinding {
  /// Creates a finding.
  const ReasoningFinding(this.code, this.message);

  /// Stable code, e.g. `empty_draft`, for a test to assert on.
  final String code;

  /// Human explanation shown in the thinking trace.
  final String message;

  /// Wire form.
  Map<String, Object?> toJson() => <String, Object?>{
        'code': code,
        'message': message,
      };
}

/// The complete record of one reasoned turn: what was planned, what was checked,
/// and what changed as a result.
class ThinkingTrace {
  /// Creates a trace.
  const ThinkingTrace({
    required this.plan,
    required this.verdict,
    required this.findings,
    required this.checkedDraft,
    required this.finalText,
    required this.revisions,
    required this.elapsed,
  });

  /// The plan the turn followed.
  final ThinkingPlan plan;

  /// Whether the draft was accepted or repaired.
  final VerificationVerdict verdict;

  /// What the check observed.
  final List<ReasoningFinding> findings;

  /// A short prefix of the draft that was checked.
  final String checkedDraft;

  /// A short prefix of the answer finally returned.
  final String finalText;

  /// How many repair passes ran (0 or 1 today).
  final int revisions;

  /// Wall-clock time spent reasoning, excluding generation.
  final Duration elapsed;

  /// Whether the answer changed because of the check.
  bool get revised => revisions > 0;

  /// A one-line summary for a log or a status chip.
  String get summary {
    final int done = plan.steps
        .where((ThinkStep s) => s.status == ThinkStepStatus.done)
        .length;
    return '${plan.strategy.label} · $done/${plan.length} steps · '
        '${revised ? 'revised' : 'accepted'}';
  }

  /// Wire form for the SSE payload and the dashboard.
  Map<String, Object?> toJson() => <String, Object?>{
        'plan': plan.toJson(),
        'verdict': verdict.wire,
        'findings': <Object?>[
          for (final ReasoningFinding finding in findings) finding.toJson(),
        ],
        'revisions': revisions,
        'revised': revised,
        'summary': summary,
        'elapsed_ms': elapsed.inMilliseconds,
      };
}

/// Builds a [ThinkingPlan] and checks a draft, with no model involved.
///
/// Both operations are deliberately pure and deterministic: given the same
/// question and the same hints they produce the same plan and the same verdict,
/// which is what makes "why did it answer that" a question with an answer.
class ReasoningPlanner {
  /// Creates a planner.
  const ReasoningPlanner();

  /// Maximum characters of the question kept in a plan.
  static const int _questionLimit = 200;

  /// Chooses a strategy for [question] when the caller has not pinned one.
  ///
  /// This is what makes automatic mode feel considered: a greeting costs nothing,
  /// a one-line question gets one check, and a multi-part request gets a plan.
  static ThinkingStrategy suggest(String question) {
    final String trimmed = question.trim();
    if (trimmed.isEmpty) {
      return ThinkingStrategy.none;
    }
    final List<String> words = trimmed.split(RegExp(r'\s+'));
    final String lower = trimmed.toLowerCase();
    // A greeting or acknowledgement has nothing to reason about.
    final bool isTrivial = words.length <= 4 &&
        RegExp(r'^(hi|hey|hello|yo|thanks|thank you|ok|okay|yes|no|sure|'
                r'good morning|good evening|good night|bye|cheers)\b')
            .hasMatch(lower);
    if (isTrivial) {
      return ThinkingStrategy.none;
    }

    int complexity = 0;
    if (words.length > 18) {
      complexity += 1;
    }
    if (words.length > 45) {
      complexity += 1;
    }
    if (trimmed.length > 240) {
      complexity += 1;
    }
    if (RegExp(r'\b(compare|contrast|analyse|analyze|evaluate|design|plan|'
            r'debug|implement|refactor|explain why|step by step|trade-?offs?|'
            r'pros and cons|architecture|strategy|migrate|optimise|optimize)\b')
        .hasMatch(lower)) {
      complexity += 2;
    }
    // Several enumerated asks, or more than one question mark, is a multi-part
    // request even when it is short.
    final int questionMarks = '?'.allMatches(trimmed).length;
    if (questionMarks > 1) {
      complexity += 1;
    }
    if (RegExp(r'(\b1[.)]\s)|(\b2[.)]\s)|(^|\n)\s*[-*]\s').hasMatch(trimmed)) {
      complexity += 1;
    }
    if (RegExp(r'\band\b.{0,40}\band\b').hasMatch(lower)) {
      complexity += 1;
    }

    if (complexity >= 2) {
      return ThinkingStrategy.thorough;
    }
    return ThinkingStrategy.concise;
  }

  /// Builds the plan for [question] at [strategy].
  ///
  /// [hasTools] and [hasMemory] add the conditional steps, because planning to
  /// "check memory" when the store is empty is a step that can only ever be
  /// noise in the trace.
  ThinkingPlan plan(
    String question, {
    required ThinkingStrategy strategy,
    bool hasTools = false,
    bool hasMemory = false,
    DateTime? at,
  }) {
    final DateTime moment = at ?? DateTime.now().toUtc();
    final String shown = _trim(question);
    final List<ThinkStep> steps = <ThinkStep>[];

    void add(String title, String detail) {
      steps.add(
        ThinkStep(index: steps.length, title: title, detail: detail),
      );
    }

    if (strategy == ThinkingStrategy.thorough) {
      add(
        'Decompose the request',
        'Identify each distinct part of the question so none is answered '
            'implicitly and forgotten.',
      );
      if (hasMemory) {
        add(
          'Recall what is known',
          'Check stored memories for a preference, fact or goal that changes '
              'the answer.',
        );
      }
      add(
        'Decide on facts and tools',
        hasTools
            ? 'Decide whether the answer needs a tool result, and which one, '
                'before writing prose.'
            : 'Separate what can be answered from what would need a fact that '
                'is not available.',
      );
      add(
        'Draft the answer',
        'Write the shortest response that covers every part identified in the '
            'first step.',
      );
      add(
        'Check the draft',
        'Re-read the draft against the request: does it answer the actual '
            'question, and does it invent anything?',
      );
      add(
        'Revise if the check objected',
        'Repair only what the check flagged, then return the result.',
      );
      return ThinkingPlan(
        question: shown,
        strategy: strategy,
        steps: List<ThinkStep>.unmodifiable(steps),
        rationale: 'The request has several parts or a non-trivial ask, so a '
            'full decomposition and a verification pass are worth the cost.',
        createdAt: moment,
      );
    }

    if (strategy == ThinkingStrategy.concise) {
      add(
        'Identify the ask',
        'Work out exactly what is being asked before answering.',
      );
      if (hasMemory) {
        add(
          'Recall what is known',
          'Pull in any stored preference or fact that applies.',
        );
      }
      add(
        'Answer',
        'Reply directly and briefly.',
      );
      add(
        'Sanity-check the answer',
        'Make sure the reply addresses the question and states no invented '
            'fact.',
      );
      return ThinkingPlan(
        question: shown,
        strategy: strategy,
        steps: List<ThinkStep>.unmodifiable(steps),
        rationale: 'A single clear question: one short plan and one check keep '
            'the answer fast without skipping verification.',
        createdAt: moment,
      );
    }

    // `none` still produces a one-step plan so the trace is never structurally
    // empty; the engine does not emit it, but a caller may plan for a UI label.
    add(
      'Answer directly',
      'No decomposition or verification for this turn.',
    );
    return ThinkingPlan(
      question: shown,
      strategy: ThinkingStrategy.none,
      steps: List<ThinkStep>.unmodifiable(steps),
      rationale: 'Thinking is disabled for this turn.',
      createdAt: moment,
    );
  }

  /// Checks a generated [draft] against [question].
  ///
  /// The checks are the failures a small model actually makes: an empty or
  /// degenerate reply, a reply that ignores the question's subject, or a claim
  /// to have used a tool that never ran. Anything subtler than that cannot be
  /// judged without a second, better model, and pretending otherwise would mean
  /// rewriting correct answers.
  VerificationResult verify({
    required String question,
    required String draft,
    required bool usedTool,
    Set<String> toolNames = const <String>{},
    int minAnswerChars = 12,
  }) {
    final List<ReasoningFinding> findings = <ReasoningFinding>[];
    final String trimmed = draft.trim();

    if (trimmed.length < minAnswerChars) {
      findings.add(
        const ReasoningFinding(
          'empty_draft',
          'The draft is too short to be an answer.',
        ),
      );
    } else {
      if (isDegenerate(trimmed)) {
        findings.add(
          const ReasoningFinding(
            'repetition',
            'The draft repeats itself instead of answering.',
          ),
        );
      }
      final List<String> asked = _topicTokens(question);
      if (asked.isNotEmpty && !_mentionsAny(trimmed, asked)) {
        findings.add(
          const ReasoningFinding(
            'off_topic',
            'The draft shares no subject matter with the question.',
          ),
        );
      }
      if (!usedTool &&
          _claimsToolUse(trimmed, toolNames) &&
          !_isHypothetical(trimmed)) {
        findings.add(
          const ReasoningFinding(
            'unsupported_tool_claim',
            'The draft implies a tool result that was never produced.',
          ),
        );
      }
    }

    return VerificationResult(
      verdict: findings.isEmpty
          ? VerificationVerdict.ok
          : VerificationVerdict.revise,
      findings: List<ReasoningFinding>.unmodifiable(findings),
    );
  }

  /// The instruction handed to the model for a repair pass.
  ///
  /// Written as a directive the *user* is asking, because that is the only shape
  /// a completion model reliably follows, and deliberately narrow: it names the
  /// specific fault so the repair cannot wander into rewriting a correct answer.
  String repairPrompt({
    required String question,
    required String draft,
    required List<ReasoningFinding> findings,
  }) {
    final String problems =
        findings.map((ReasoningFinding f) => f.message).join(' ');
    return 'Your previous draft did not pass its check. $problems '
        'Answer the question again, correctly and briefly, without repeating '
        'the mistake.\n'
        'Question: $question\n'
        'Previous draft: ${_trim(draft, 400)}';
  }

  /// Whether [text] is a short loop rather than a sentence.
  static bool isDegenerate(String text) {
    final List<String> words = text
        .toLowerCase()
        .split(RegExp(r'[^a-z0-9]+'))
        .where((String w) => w.isNotEmpty)
        .toList(growable: false);
    if (words.isEmpty) {
      return true;
    }
    // One word repeated: "the the the the".
    if (words.length >= 4 && words.toSet().length == 1) {
      return true;
    }
    // A period-two loop: "I am I am I am". The pairs are taken at even offsets
    // only, because scanning every position sees both "i am" and "am i" and so
    // reports two distinct bigrams for a text that is repeating exactly one —
    // which made this branch dead on arrival for the loop it was written to
    // catch.
    if (words.length >= 6) {
      final Set<String> pairs = <String>{};
      for (int i = 0; i + 1 < words.length; i += 2) {
        pairs.add('${words[i]} ${words[i + 1]}');
      }
      if (pairs.length == 1) {
        return true;
      }
    }
    // No letter at all: punctuation soup.
    if (!RegExp(r'[a-zA-Z]').hasMatch(text)) {
      return true;
    }
    return false;
  }

  /// The meaningful tokens of the question, used for the on-topic check.
  static List<String> _topicTokens(String question) {
    final List<String> tokens = <String>[];
    for (final String token
        in question.toLowerCase().split(RegExp(r'[^a-z0-9]+'))) {
      if (token.length < 3 || _stopWords.contains(token)) {
        continue;
      }
      tokens.add(token);
    }
    return tokens;
  }

  /// Whether [text] contains any of [tokens], by stem prefix.
  static bool _mentionsAny(String text, List<String> tokens) {
    final String lower = text.toLowerCase();
    for (final String token in tokens) {
      final String stem = token.length > 5 ? token.substring(0, 5) : token;
      if (lower.contains(stem)) {
        return true;
      }
    }
    return false;
  }

  /// Whether [text] implies a tool produced a result.
  static bool _claimsToolUse(String text, Set<String> toolNames) {
    final String lower = text.toLowerCase();
    if (RegExp(r'\b(the tool|the device|i checked|i looked it up|'
            r'according to the (?:tool|device)|the result (?:is|was)|'
            r'i (?:ran|used|called) the)\b')
        .hasMatch(lower)) {
      return true;
    }
    for (final String name in toolNames) {
      final String leaf = name.contains('.') ? name.split('.').last : name;
      if (leaf.length >= 3 && lower.contains(leaf.toLowerCase())) {
        return true;
      }
    }
    return false;
  }

  /// Whether a tool-looking sentence is explicitly conditional.
  ///
  /// "If you tell me your city I can check the weather" mentions a tool and
  /// correctly did *not* call one; treating that as a false claim would punish
  /// the right answer.
  static bool _isHypothetical(String text) {
    return RegExp(
      r'\b(if you|i can|could|would|you could|let me know|once you|'
      r"i would need|i do not have|i don't have|i cannot|i can't|is not "
      r"available|isn't available)\b",
    ).hasMatch(text.toLowerCase());
  }

  static String _trim(String text, [int limit = _questionLimit]) {
    final String cleaned = text.trim().replaceAll(RegExp(r'\s+'), ' ');
    return cleaned.length <= limit ? cleaned : '${cleaned.substring(0, limit)}…';
  }

  /// Words ignored by the on-topic check.
  static const Set<String> _stopWords = <String>{
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'your', 'with', 'that',
    'this', 'have', 'has', 'had', 'was', 'were', 'will', 'would', 'can',
    'could', 'should', 'about', 'into', 'from', 'they', 'them', 'their',
    'what', 'when', 'where', 'which', 'who', 'how', 'why', 'there', 'here',
    'please', 'tell', 'give', 'make', 'does', 'did', 'its', 'explain',
    'write', 'help', 'need', 'want', 'know', 'like', 'some', 'any',
  };
}

/// The outcome of [ReasoningPlanner.verify].
class VerificationResult {
  /// Creates a result.
  const VerificationResult({required this.verdict, required this.findings});

  /// Whether the draft may be shown as-is.
  final VerificationVerdict verdict;

  /// What was observed.
  final List<ReasoningFinding> findings;

  /// Whether a repair pass should run.
  bool get needsRevision => verdict == VerificationVerdict.revise;
}