// The chat surface for the Harbor app.
//
// The screen is deliberately thin: every piece of turn state (the committed
// transcript, the text arriving token by token, the active tool) belongs to
// [ChatController], and this file only decides how to paint each of those
// states. That keeps the "half-generated sentence is not a reply yet" rule in
// one place instead of scattered across widget code.

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:harbor_core/harbor_core.dart';

import '../../core/chat/chat_controller.dart';

/// The chat surface: transcript, live turn, and composer.
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
            final ToolCall? activeTool = widget.controller.activeTool;
            final String? error = widget.controller.error;
            return Column(
              children: <Widget>[
                _toolbar(context),
                if (activeTool != null) _toolChip(context, activeTool),
                if (error != null) _errorCard(context, error),
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
          label: Text('Running ${tool.name}…'),
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
        child: SelectableText(
          content,
          style: TextStyle(color: scheme.onPrimaryContainer),
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
      SelectableText(content, style: TextStyle(color: scheme.onSurface)),
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
                      child: Text(
                        message.name ?? 'tool',
                        style: theme.textTheme.labelMedium?.copyWith(
                          fontFamily: 'monospace',
                        ),
                        overflow: TextOverflow.ellipsis,
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