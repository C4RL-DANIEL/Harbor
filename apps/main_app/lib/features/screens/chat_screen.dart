// The chat surface for the Harbor app.
//
// The screen is deliberately thin: every piece of turn state (the committed
// transcript, the text arriving token by token, the active tool, the reasoning
// plan and trace, the memories a turn taught the assistant) belongs to
// [ChatController], and this file only decides how to paint each of those
// states. That keeps the "half-generated sentence is not a reply yet" rule in
// one place instead of scattered across widget code.

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:harbor_core/harbor_core.dart';

import '../../core/chat/chat_controller.dart';
import 'memory_screen.dart';

/// The chat surface: transcript, live turn, reasoning card, and composer.
///
/// The in-progress answer is kept separate from the committed transcript on
/// purpose. [ChatController.streamingText] is half a sentence — not yet a
/// reply the user can act on or respond to — so it renders as its own
/// disposable newest item while [ChatController.busy] is true. Keeping the two
/// apart means a cancelled or failed turn can throw the partial text away
/// without stranding it in the history, and the transcript only ever contains
/// complete turns the model actually finished.
class ChatScreen extends StatefulWidget {
  /// Creates the chat screen.
  const ChatScreen({super.key, required this.controller});

  /// The controller that owns the transcript and streams answers.
  final ChatController controller;

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

/// State for [ChatScreen].
class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _input = TextEditingController();

  @override
  void initState() {
    super.initState();
    widget.controller.seedGreeting(
      'Harbor answers with a small language model that was trained on this '
      'device from whatever text you gathered on the Corpus tab. It can be '
      'wrong. The Tools tab shows exact device readings taken directly from '
      'the hardware, without the model.',
    );
  }

  @override
  void dispose() {
    _input.dispose();
    super.dispose();
  }

  /// Sends the composer text, if any, without blocking on the whole answer.
  ///
  /// The future is intentionally not awaited: progress arrives through the
  /// controller's listener, and waiting here would freeze the very live
  /// transcript this screen exists to show.
  void _send() {
    final String text = _input.text.trim();
    if (text.isEmpty) {
      return;
    }
    _input.clear();
    unawaited(widget.controller.send(text));
  }

  /// Opens the memory screen over the same controller.
  ///
  /// The navigator is captured before the route is scheduled, exactly as
  /// [_showPrompt] does for the dialog, so the push lands on the navigator
  /// this screen was built against even if a controller rebuild happens
  /// first. The controller is captured for the same reason: the pushed route
  /// must not resolve `widget.controller` across the async gap.
  void _openMemory() {
    final NavigatorState navigator = Navigator.of(context);
    final ChatController controller = widget.controller;
    unawaited(
      navigator.push<void>(
        MaterialPageRoute<void>(
          builder: (BuildContext context) => MemoryScreen(controller: controller),
        ),
      ),
    );
  }

  /// Shows the exact prompt the model would read for the current transcript.
  ///
  /// The navigator is captured before the dialog opens so the Close action can
  /// dismiss it without reaching back through a context that may since have
  /// gone away.
  void _showPrompt(BuildContext context) {
    final NavigatorState navigator = Navigator.of(context);
    unawaited(
      showDialog<void>(
        context: context,
        builder: (BuildContext context) => AlertDialog(
          title: const Text('Prompt sent to the model'),
          content: SizedBox(
            width: double.maxFinite,
            child: SingleChildScrollView(
              child: SelectableText(
                widget.controller.renderedPrompt,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 11),
              ),
            ),
          ),
          actions: <Widget>[
            TextButton(
              onPressed: () => navigator.pop(),
              child: const Text('Close'),
            ),
          ],
        ),
      ),
    );
  }

  /// Asks for confirmation, then clears the transcript.
  Future<void> _confirmClear() async {
    final bool? confirmed = await showDialog<bool>(
      context: context,
      builder: (BuildContext context) => AlertDialog(
        title: const Text('Clear the conversation?'),
        content: const Text('This removes every message from the transcript.'),
        actions: <Widget>[
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Clear'),
          ),
        ],
      ),
    );
    if (!mounted) {
      return;
    }
    if (confirmed ?? false) {
      widget.controller.clear();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: ListenableBuilder(
          listenable: widget.controller,
          builder: (BuildContext context, Widget? child) {
            final ChatController controller = widget.controller;
            final ToolCall? activeTool = controller.activeTool;
            final String? error = controller.error;
            final Widget? thinking = _thinkingCard(context);
            return Column(
              children: <Widget>[
                _toolbar(context),
                if (activeTool != null) _toolChip(context, activeTool),
                if (error != null) _errorCard(context, error),
                if (thinking != null) thinking,
                if (controller.remembered.isNotEmpty) _rememberedStrip(context),
                Expanded(child: _transcript(context)),
                _composer(context),
              ],
            );
          },
        ),
      ),
    );
  }

  /// The status row above the transcript: counts, prompt preview, and clear.
  Widget _toolbar(BuildContext context) {
    final ChatController controller = widget.controller;
    final ThemeData theme = Theme.of(context);
    final String tokens = controller.generatedTokens > 0
        ? ' · ${controller.generatedTokens} tokens in '
            '${controller.lastTurnDuration.inMilliseconds} ms'
        : '';
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 8, 4, 0),
      child: Row(
        children: <Widget>[
          Expanded(
            child: Text(
              '${controller.messages.length} messages$tokens',
              style: theme.textTheme.labelMedium,
              overflow: TextOverflow.ellipsis,
            ),
          ),
          IconButton(
            key: const ValueKey<String>('toolbar-memory'),
            tooltip: 'Memory',
            onPressed: _openMemory,
            icon: const Icon(Icons.psychology_outlined),
          ),
          TextButton(
            onPressed: () => _showPrompt(context),
            child: const Text('Prompt'),
          ),
          IconButton(
            tooltip: 'Clear',
            onPressed: () => unawaited(_confirmClear()),
            icon: const Icon(Icons.delete_outline),
          ),
        ],
      ),
    );
  }

  /// The pinned reasoning card, or nothing when no turn has planned.
  ///
  /// It sits above the transcript rather than inside it: a trace belongs to the
  /// turn as a whole, and giving it a message-list index would make the
  /// reversed [ListView] recycle it like one more bubble the user can reply
  /// to — the exact element-reuse confusion every other row here keys against.
  Widget? _thinkingCard(BuildContext context) {
    final ChatController controller = widget.controller;
    final ThinkingPlan? livePlan = controller.thinkingPlan;
    final ThinkingTrace? trace = controller.thinkingTrace;
    if (livePlan == null && trace == null) {
      return null;
    }
    // The card is "running" only in the spec's exact window: a plan exists,
    // no trace yet, and the turn is still in flight. A settled turn keeps its
    // completed card visible without pretending to still work.
    final bool running = controller.busy && livePlan != null && trace == null;
    // Once a trace exists, `trace.plan` is the authoritative view: the engine
    // advances step statuses onto the trace's copy, while the controller's raw
    // `thinkingPlan` may still be the pre-flight snapshot with every step
    // pending. When the trace is null, the guard above proved `livePlan`
    // non-null.
    final ThinkingPlan plan = trace?.plan ?? livePlan!;
    // Keyed on the plan instant rather than the controller: every turn's plan
    // carries a fresh `createdAt`, so a new plan builds a new card — and with
    // it a fresh collapse default — while status updates within one turn
    // reuse this element and keep whatever state the user tapped into.
    return _ThinkingCard(
      key: ValueKey<int>(plan.createdAt.microsecondsSinceEpoch),
      running: running,
      plan: plan,
      trace: trace,
    );
  }

  /// The thin strip reporting what the assistant has learned this session.
  ///
  /// Learning must never be silent: a memory the user cannot see is a memory
  /// the user cannot correct, so each turn's new memories announce themselves
  /// and the whole row opens the memory screen to review or delete them.
  Widget _rememberedStrip(BuildContext context) {
    final List<MemoryEntry> remembered = widget.controller.remembered;
    if (remembered.isEmpty) {
      return const SizedBox.shrink();
    }
    final ThemeData theme = Theme.of(context);
    final ColorScheme scheme = theme.colorScheme;
    return Material(
      key: const ValueKey<String>('remembered-strip'),
      color: scheme.secondaryContainer,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: remembered.map((MemoryEntry entry) {
          return InkWell(
            key: ValueKey<String>('memory-${entry.id}'),
            onTap: _openMemory,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              child: Row(
                children: <Widget>[
                  Icon(
                    Icons.school_outlined,
                    size: 16,
                    color: scheme.onSecondaryContainer,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Semantics(
                      label: 'Remembered: ${entry.label}',
                      child: Text(
                        entry.label,
                        style: theme.textTheme.labelMedium?.copyWith(
                          color: scheme.onSecondaryContainer,
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ),
                  IconButton(
                    icon: Icon(
                      Icons.close,
                      size: 16,
                      color: scheme.onSecondaryContainer,
                    ),
                    tooltip: 'Forget this memory',
                    onPressed: () => widget.controller.forgetMemory(entry.id),
                  ),
                ],
              ),
            ),
          );
        }).toList(),
      ),
    );
  }
  /// A progress chip naming the tool call that is running right now.
  Widget _toolChip(BuildContext context, ToolCall tool) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 2),
      child: Align(
        alignment: Alignment.centerLeft,
        child: Chip(
          avatar: const SizedBox(
            width: 16,
            height: 16,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
          label: Text('Running ${tool.name}...'),
        ),
      ),
    );
  }

  /// A red-tinted card reporting the last turn's failure.
  Widget _errorCard(BuildContext context, String message) {
    final ColorScheme scheme = Theme.of(context).colorScheme;
    return Card(
      color: scheme.errorContainer,
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: Row(
          children: <Widget>[
            Icon(Icons.error_outline, color: scheme.onErrorContainer),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                message,
                style: TextStyle(color: scheme.onErrorContainer),
              ),
            ),
            TextButton(
              onPressed: () {},
              child: const Text('Dismiss'),
            ),
          ],
        ),
      ),
    );
  }

  /// The reversed transcript: newest message first, live turn on top.
  ///
  /// [ListView.reverse] puts index 0 at the bottom of the viewport, so the list
  /// is built newest-first to keep the newest turn pinned above the composer.
  Widget _transcript(BuildContext context) {
    final ChatController controller = widget.controller;
    final List<ChatMessage> visible = <ChatMessage>[
      for (final ChatMessage message in controller.messages)
        if (message.role != ChatRole.system) message,
    ].reversed.toList();
    final String streaming = controller.streamingText;
    final bool showStreaming = controller.busy && streaming.isNotEmpty;
    final bool showWaiting = controller.busy && streaming.isEmpty;
    final int leading = showStreaming || showWaiting ? 1 : 0;
    return ListView.builder(
      reverse: true,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      itemCount: visible.length + leading,
      itemBuilder: (BuildContext context, int index) {
        if (showWaiting && index == 0) {
          return _waitingBubble(context);
        }
        if (showStreaming && index == 0) {
          return _assistantBubble(context, streaming);
        }
        return _messageRow(context, visible[index - leading]);
      },
    );
  }

  /// The composer: a bounded multi-line field and the send/stop control.
  Widget _composer(BuildContext context) {
    final bool busy = widget.controller.busy;
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 4, 12, 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: <Widget>[
          Expanded(
            child: TextField(
              controller: _input,
              minLines: 1,
              maxLines: 4,
              textInputAction: TextInputAction.send,
              onSubmitted: (_) => _send(),
              decoration: const InputDecoration(
                hintText:
                    'Ask about this device, or anything you have gathered text about',
                border: OutlineInputBorder(),
                isDense: true,
              ),
            ),
          ),
          const SizedBox(width: 8),
          if (busy)
            OutlinedButton.icon(
              onPressed: widget.controller.stop,
              icon: const Icon(Icons.stop),
              label: const Text('Stop'),
            )
          else
            FilledButton.icon(
              onPressed: _send,
              icon: const Icon(Icons.send),
              label: const Text('Send'),
            ),
        ],
      ),
    );
  }

  /// Renders one committed message according to its role.
  ///
  /// System messages are prompt plumbing rather than conversation, so they are
  /// skipped instead of being shown as bubbles.
  Widget _messageRow(BuildContext context, ChatMessage message) {
    return switch (message.role) {
      ChatRole.system => const SizedBox.shrink(),
      ChatRole.user => _userBubble(context, message.content),
      ChatRole.assistant => _assistantBubble(context, message.content),
      ChatRole.tool => _toolCard(context, message),
    };
  }

  /// A right-aligned user bubble filled with the primary container colour.
  Widget _userBubble(BuildContext context, String content) {
    final ColorScheme scheme = Theme.of(context).colorScheme;
    return Align(
      alignment: Alignment.centerRight,
      child: Container(
        constraints: const BoxConstraints(maxWidth: 560),
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: scheme.primaryContainer,
          borderRadius: BorderRadius.circular(14),
        ),
        child: GestureDetector(
          onLongPress: () {
            widget.controller.remember(content);
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text('Remembered: ${content.length > 60 ? content.substring(0, 60) : content}'),
                duration: const Duration(seconds: 2),
              ),
            );
          },
          child: SelectableText(
            content,
            semanticsLabel: 'User message: $content — long-press to remember',
            style: TextStyle(color: scheme.onPrimaryContainer),
          ),
        ),
      ),
    );
  }

  /// Wraps [child] in the left-aligned assistant bubble shell.
  Widget _assistantShell(BuildContext context, Widget child) {
    final ColorScheme scheme = Theme.of(context).colorScheme;
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        constraints: const BoxConstraints(maxWidth: 560),
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: scheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(14),
        ),
        child: child,
      ),
    );
  }

  /// A left-aligned assistant bubble around [content].
  Widget _assistantBubble(BuildContext context, String content) {
    final ColorScheme scheme = Theme.of(context).colorScheme;
    return _assistantShell(
      context,
      Semantics(
        label: 'Assistant message: $content',
        child: SelectableText(content, style: TextStyle(color: scheme.onSurface)),
      ),
    );
  }

  /// A left-aligned assistant bubble holding a small progress indicator.
  ///
  /// Shown only while the turn is running but no text has arrived yet, so the
  /// user can tell the model is working rather than stalled.
  Widget _waitingBubble(BuildContext context) {
    return _assistantShell(
      context,
      const SizedBox(
        width: 18,
        height: 18,
        child: CircularProgressIndicator(strokeWidth: 2),
      ),
    );
  }

  /// A compact monospace card for the output of a tool call.
  ///
  /// The body is clamped to [maxHeight] and scrolls internally so one long
  /// reading (a file listing, say) cannot push the rest of the conversation off
  /// screen.
  Widget _toolCard(BuildContext context, ChatMessage message) {
    final ThemeData theme = Theme.of(context);
    final ColorScheme scheme = theme.colorScheme;
    return Align(
      alignment: Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 560),
        child: Card(
          margin: const EdgeInsets.symmetric(vertical: 4),
          color: scheme.surfaceContainerHighest,
          child: Padding(
            padding: const EdgeInsets.all(10),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Row(
                  children: <Widget>[
                    Icon(
                      Icons.terminal,
                      size: 14,
                      color: scheme.onSurfaceVariant,
                    ),
                    const SizedBox(width: 6),
                    Flexible(
                      child: Semantics(
                        label: 'Tool result: ${message.name ?? 'tool'}',
                        child: Text(
                          message.name ?? 'tool',
                          style: theme.textTheme.labelMedium?.copyWith(
                            fontFamily: 'monospace',
                          ),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    const Chip(
                      label: Text('result'),
                      visualDensity: VisualDensity.compact,
                      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                ConstrainedBox(
                  constraints: const BoxConstraints(maxHeight: 220),
                  child: SingleChildScrollView(
                    child: SelectableText(
                      message.content,
                      style: const TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 12,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// The collapsible reasoning card for the current or most recent thought turn.
///
/// Claude-style: while the turn is in flight the card shows the plan stepping
/// along — pending steps dim, the running step spins — so a slow reasoning
/// pass reads as progress rather than a freeze. Once the trace lands, the card
/// shows what each step concluded and collapses by default, because the user
/// asked for an answer, not an essay about the answer.
///
/// The open/closed state lives here, in widget state, not on the controller:
/// collapse is a view concern, the controller has no opinion about it, and
/// keeping it local is what lets the user's explicit choice survive the dozens
/// of rebuilds a streaming turn fires.
class _ThinkingCard extends StatefulWidget {
  /// Creates the card over one [plan] and, once settled, its [trace].
  const _ThinkingCard({
    super.key,
    required this.running,
    required this.plan,
    required this.trace,
  });

  /// Whether the turn this card belongs to is still in flight.
  final bool running;

  /// The plan whose steps are rendered. After settlement this is the trace's
  /// own plan, which carries the final step statuses.
  final ThinkingPlan plan;

  /// The completed trace, or null while the turn has not been verified yet.
  final ThinkingTrace? trace;

  @override
  State<_ThinkingCard> createState() => _ThinkingCardState();
}

/// State for [_ThinkingCard]: the user's manual open/closed choice.
class _ThinkingCardState extends State<_ThinkingCard> {
  /// True once the user has explicitly opened or closed this card.
  ///
  /// Until then the card follows its own default (expanded while running,
  /// collapsed once settled). After that the user's choice wins, even when the
  /// turn finishes — auto-collapsing a card someone just tapped open is how
  /// detail views become impossible to read.
  bool? _userChoice;

  /// Whether the card was running when this state object was created.
  ///
  /// Captured in `initState` so a *later* completion can auto-collapse the
  /// card while a plan that lands already finished still opens collapsed.
  late bool _wasRunning;

  /// The open state as last known by this widget, mirroring the tile.
  late bool _open;

  /// Whether the card should show itself open absent a user choice.
  bool get _defaultOpen => widget.running || (_wasRunning && _userChoice == null);

  @override
  void initState() {
    super.initState();
    _wasRunning = widget.running;
    _open = _defaultOpen;
  }

  /// Moves the [ExpansionTile] to [open], driving its own hidden controller.
  ///
  /// `initiallyExpanded` cannot be used as a live value — the tile reads it
  /// only when its state is created — so a toggle has to go through the
  /// controller while `initiallyExpanded` is re-derived on every build to stay
  /// correct if the element is ever recreated.
  void _setOpen(BuildContext tileContext, bool open) {
    final ExpansionTileController controller =
        ExpansionTileController.of(tileContext);
    if (open) {
      controller.expand();
    } else {
      controller.collapse();
    }
  }

  /// Flips the card and records that the user, not the turn lifecycle, chose.
  void _toggle(BuildContext tileContext) {
    final bool open = !_open;
    setState(() {
      _userChoice = open;
      _open = open;
    });
    _setOpen(tileContext, open);
  }

  /// The leading indicator for one step, by lifecycle status.
  Widget _stepIcon(BuildContext context, ThinkStep step) {
    final ThemeData theme = Theme.of(context);
    return switch (step.status) {
      ThinkStepStatus.running => const SizedBox(
          width: 14,
          height: 14,
          child: CircularProgressIndicator(strokeWidth: 2),
        ),
      ThinkStepStatus.done =>
        Icon(Icons.check_circle, size: 16, color: theme.colorScheme.primary),
      ThinkStepStatus.failed =>
        Icon(Icons.cancel, size: 16, color: theme.colorScheme.error),
      ThinkStepStatus.skipped =>
        Icon(Icons.remove_circle_outline, size: 16, color: theme.disabledColor),
      ThinkStepStatus.pending =>
        Icon(Icons.circle_outlined, size: 16, color: theme.disabledColor),
    };
  }

  /// One plan step: its title, detail, and — once settled — what it concluded.
  ///
  /// The `result` line only appears in the completed view: a running step has
  /// nothing to report, and the detail sentence already explains what the step
  /// *will* do.
  Widget _stepRow(BuildContext context, ThinkStep step) {
    final ThemeData theme = Theme.of(context);
    final bool showResult = widget.trace != null && (step.result ?? '').isNotEmpty;
    return Padding(
      key: ValueKey<String>('think-step-${step.index}'),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          _stepIcon(context, step),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Text(
                  step.title,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                    color: step.status == ThinkStepStatus.pending
                        ? theme.disabledColor
                        : null,
                  ),
                ),
                Text(
                  step.detail,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
                if (showResult)
                  Text(
                    step.result!,
                    style: theme.textTheme.bodySmall?.copyWith(
                      fontFamily: 'monospace',
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// One objection the verification pass raised, listed under a revise verdict.
  Widget _findingRow(BuildContext context, ReasoningFinding finding) {
    final ThemeData theme = Theme.of(context);
    return Padding(
      key: ValueKey<String>('think-finding-${finding.code}'),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Icon(Icons.report_problem_outlined, size: 16, color: theme.colorScheme.error),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              '${finding.code}: ${finding.message}',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.error,
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    final ThinkingTrace? trace = widget.trace;
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      color: theme.colorScheme.surfaceContainerHighest.withAlpha(120),
      child: Theme(
        // ExpansionTile paints its dividers at the list-tile inset by default,
        // which floats them away from this card's own padding; zeroing the
        // tile's decoration insets pins the hairlines to the card edges.
        data: theme.copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          key: const ValueKey<String>('thinking-tile'),
          initiallyExpanded: _open,
          onExpansionChanged: (bool expanded) {
            // Mirrors both user taps and this widget's own expand/collapse
            // calls; the user's choice was already recorded in [_toggle], so
            // the sequence stays consistent no matter which fired.
            _open = expanded;
          },
          tilePadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 0),
          childrenPadding: const EdgeInsets.only(bottom: 8),
          leading: Icon(
            Icons.psychology_outlined,
            color: widget.running
                ? theme.colorScheme.primary
                : theme.colorScheme.onSurfaceVariant,
          ),
          title: Row(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              Flexible(
                child: Text(
                  widget.running ? 'Thinking' : 'Thought it through',
                  style: const TextStyle(fontWeight: FontWeight.w600),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              const SizedBox(width: 8),
              if (trace != null && trace.revisions > 0)
                Container(
                  key: const ValueKey<String>('think-revised-badge'),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 6,
                    vertical: 1,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.amber.withAlpha(60),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(
                    'Revised',
                    style: theme.textTheme.labelSmall?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
            ],
          ),
          subtitle: Text(
            // The trace's own summary already names the strategy, the step
            // count and the outcome; while running there is no outcome to
            // summarise yet, so show what is being attempted instead.
            trace?.summary ??
                'Plan · ${widget.plan.strategy.label} · '
                    '${widget.plan.steps.length} steps',
            style: theme.textTheme.bodySmall,
            overflow: TextOverflow.ellipsis,
          ),
          trailing: IconButton(
            key: ValueKey<String>('think-toggle-${_open ? 'open' : 'closed'}'),
            tooltip: _open ? 'Collapse thinking' : 'Expand thinking',
            onPressed: () => _toggle(context),
            icon: Icon(_open ? Icons.expand_less : Icons.expand_more),
          ),
          children: <Widget>[
            // The question the plan answers, so a reopened card after several
            // turns still says which turn it belongs to.
            Padding(
              key: const ValueKey<String>('think-question'),
              padding: const EdgeInsets.fromLTRB(16, 4, 16, 2),
              child: Text(
                widget.plan.question,
                style: theme.textTheme.bodySmall?.copyWith(
                  fontStyle: FontStyle.italic,
                  color: theme.colorScheme.onSurfaceVariant,
                ),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ),
            for (final ThinkStep step in widget.plan.steps) _stepRow(context, step),
            if (trace != null &&
                trace.verdict == VerificationVerdict.revise &&
                trace.findings.isNotEmpty) ...<Widget>[
              const Padding(
                key: ValueKey<String>('think-findings-label'),
                padding: EdgeInsets.fromLTRB(16, 8, 16, 2),
                child: Text('The check objected:'),
              ),
              for (final ReasoningFinding finding in trace.findings)
                _findingRow(context, finding),
            ],
            Padding(
              key: const ValueKey<String>('think-rationale'),
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
              child: Text(
                widget.plan.rationale,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
