// The memory screen: what the assistant knows about you, and how to change it.
//
// A persistent memory that the user cannot see is not a feature, it is a secret.
// Everything the extractor stored has to be listable, searchable and deletable
// from one place, because "it remembered the wrong thing" is the failure mode
// that makes people stop trusting an assistant — and the only honest answer to
// that is to show the list and hand over the delete key.
//
// The screen reads straight through [ChatController.memory], so it always shows
// the same store the engine recalls from; there is no second copy to drift.

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:harbor_core/harbor_core.dart';

import '../../core/chat/chat_controller.dart';

/// Lists, searches and edits the assistant's long-term memory.
class MemoryScreen extends StatefulWidget {
  /// Creates the screen over [controller].
  const MemoryScreen({super.key, required this.controller});

  /// Owns the memory store and the mutations performed here.
  final ChatController controller;

  @override
  State<MemoryScreen> createState() => _MemoryScreenState();
}

class _MemoryScreenState extends State<MemoryScreen> {
  final TextEditingController _search = TextEditingController();
  String _query = '';

  ChatController get _controller => widget.controller;

  @override
  void dispose() {
    _search.dispose();
    super.dispose();
  }

  Future<void> _forget(MemoryEntry entry) async {
    // Forgetting a name or a stated preference by one stray tap is the worst
    // mistake this screen can make, so only that class of entry is confirmed.
    if (entry.kind == MemoryKind.identity) {
      final bool? confirmed = await showDialog<bool>(
        context: context,
        builder: (BuildContext context) => AlertDialog(
          title: const Text('Forget this?'),
          content: Text('Harbor will stop using:\n\n"${entry.text}"'),
          actions: <Widget>[
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: const Text('Keep'),
            ),
            FilledButton(
              onPressed: () => Navigator.of(context).pop(true),
              child: const Text('Forget'),
            ),
          ],
        ),
      );
      if (confirmed != true) {
        return;
      }
    }
    await _controller.forgetMemory(entry.id);
  }

  Future<void> _clearEverything() async {
    final bool? confirmed = await showDialog<bool>(
      context: context,
      builder: (BuildContext context) => AlertDialog(
        title: const Text('Forget everything?'),
        content: const Text(
          'This deletes every memory the assistant has stored on this device. '
          'Your chat transcript is not affected.',
        ),
        actions: <Widget>[
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Forget everything'),
          ),
        ],
      ),
    );
    if (confirmed ?? false) {
      await _controller.clearMemory();
    }
  }

  Future<void> _teach() async {
    final TextEditingController text = TextEditingController();
    MemoryKind kind = MemoryKind.fact;
    final bool? saved = await showDialog<bool>(
      context: context,
      builder: (BuildContext context) => StatefulBuilder(
        builder: (BuildContext context, StateSetter rebuild) => AlertDialog(
          title: const Text('Tell Harbor to remember'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              TextField(
                key: const ValueKey<String>('memory.teach.text'),
                controller: text,
                autofocus: true,
                maxLines: 3,
                decoration: const InputDecoration(
                  labelText: 'What should it remember?',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 12),
              DropdownButtonFormField<MemoryKind>(
                key: const ValueKey<String>('memory.teach.kind'),
                value: kind,
                decoration: const InputDecoration(
                  labelText: 'Kind',
                  border: OutlineInputBorder(),
                ),
                items: <DropdownMenuItem<MemoryKind>>[
                  for (final MemoryKind option in MemoryKind.values)
                    DropdownMenuItem<MemoryKind>(
                      value: option,
                      child: Text(_MemoryLabels.of(option)),
                    ),
                ],
                onChanged: (MemoryKind? value) {
                  if (value == null) {
                    return;
                  }
                  rebuild(() {
                    kind = value;
                  });
                },
              ),
            ],
          ),
          actions: <Widget>[
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.of(context).pop(true),
              child: const Text('Remember'),
            ),
          ],
        ),
      ),
    );
    if (saved == true && text.text.trim().isNotEmpty) {
      _controller.remember(text.text, kind: kind);
    }
    text.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final MemoryStore? store = _controller.memory;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Memory'),
        actions: <Widget>[
          if (store != null && store.length > 0)
            IconButton(
              tooltip: 'Forget everything',
              onPressed: () => unawaited(_clearEverything()),
              icon: const Icon(Icons.delete_sweep_outlined),
            ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => unawaited(_teach()),
        icon: const Icon(Icons.add),
        label: const Text('Remember'),
      ),
      body: ListenableBuilder(
        listenable: _controller,
        builder: (BuildContext context, Widget? child) {
          if (store == null) {
            return const _EmptyState(
              icon: Icons.psychology_outlined,
              title: 'Memory is off',
              message: 'This assistant was started without a memory store, so '
                  'nothing is kept between conversations.',
            );
          }
          final List<MemoryEntry> shown = _query.trim().isEmpty
              ? store.entries
              : <MemoryEntry>[
                  for (final MemoryMatch match in store.recall(_query))
                    match.entry,
                ];
          return ListView(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 88),
            children: <Widget>[
              _SummaryCard(store: store, shown: shown.length, searching: _query.trim().isNotEmpty),
              const SizedBox(height: 12),
              TextField(
                key: const ValueKey<String>('memory.search'),
                controller: _search,
                decoration: InputDecoration(
                  hintText: 'Search memories',
                  prefixIcon: const Icon(Icons.search),
                  border: const OutlineInputBorder(),
                  isDense: true,
                  suffixIcon: _query.isEmpty
                      ? null
                      : IconButton(
                          tooltip: 'Clear search',
                          icon: const Icon(Icons.close),
                          onPressed: () => setState(() {
                            _search.clear();
                            _query = '';
                          }),
                        ),
                ),
                onChanged: (String value) => setState(() => _query = value),
              ),
              const SizedBox(height: 12),
              if (shown.isEmpty)
                _EmptyState(
                  icon: Icons.inbox_outlined,
                  title: _query.isEmpty ? 'Nothing remembered yet' : 'No match',
                  message: _query.isEmpty
                      ? 'Harbor learns what you tell it — "my name is…", "I '
                          'prefer…", "remember that…" — and you can add an '
                          'entry with the button below.'
                      : 'No memory matches "$_query".',
                )
              else
                for (final MemoryEntry entry in shown)
                  _MemoryTile(
                    key: ValueKey<String>('memory.row.${entry.id}'),
                    entry: entry,
                    score: _query.trim().isEmpty
                        ? null
                        : _scoreFor(store, entry.id),
                    onForget: () => unawaited(_forget(entry)),
                  ),
            ],
          );
        },
      ),
    );
  }

  double? _scoreFor(MemoryStore store, String id) {
    for (final MemoryMatch match in store.recall(_query, limit: 64)) {
      if (match.entry.id == id) {
        return match.score;
      }
    }
    return null;
  }
}

/// Human label per kind, shared by the list header and the add dialog.
class _MemoryLabels {
  static String of(MemoryKind kind) {
    switch (kind) {
      case MemoryKind.identity:
        return 'Identity';
      case MemoryKind.preference:
        return 'Preference';
      case MemoryKind.fact:
        return 'Fact';
      case MemoryKind.goal:
        return 'Goal';
      case MemoryKind.topic:
        return 'Topic';
      case MemoryKind.episode:
        return 'Episode';
    }
  }

  static IconData icon(MemoryKind kind) {
    switch (kind) {
      case MemoryKind.identity:
        return Icons.badge_outlined;
      case MemoryKind.preference:
        return Icons.tune;
      case MemoryKind.fact:
        return Icons.info_outline;
      case MemoryKind.goal:
        return Icons.flag_outlined;
      case MemoryKind.topic:
        return Icons.sell_outlined;
      case MemoryKind.episode:
        return Icons.menu_book_outlined;
    }
  }
}

class _SummaryCard extends StatelessWidget {
  const _SummaryCard({
    required this.store,
    required this.shown,
    required this.searching,
  });

  final MemoryStore store;
  final int shown;
  final bool searching;

  @override
  Widget build(BuildContext context) {
    final TextTheme text = Theme.of(context).textTheme;
    final Map<String, Object?> kinds =
        (store.describe()['kinds'] as Map<String, Object?>?) ??
            const <String, Object?>{};
    final List<Widget> chips = <Widget>[
      for (final MemoryKind kind in MemoryKind.values)
        if ((kinds[kind.wire] as int? ?? 0) > 0)
          Chip(
            visualDensity: VisualDensity.compact,
            label: Text('${_MemoryLabels.of(kind)} · ${kinds[kind.wire]}'),
          ),
    ];
    return Card(
      elevation: 0,
      color: Theme.of(context).colorScheme.surfaceContainerHigh,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              searching ? '$shown matching · ${store.length} stored'
                        : '${store.length} memories',
              style: text.titleSmall,
            ),
            const SizedBox(height: 4),
            Text(
              'Everything here is stored on this device and stays here. '
              'Harbor reads these back into a prompt only when they are '
              'relevant to what you just said.',
              style: text.bodySmall,
            ),
            if (chips.isNotEmpty) ...<Widget>[
              const SizedBox(height: 10),
              Wrap(spacing: 8, runSpacing: 8, children: chips),
            ],
          ],
        ),
      ),
    );
  }
}

class _MemoryTile extends StatelessWidget {
  const _MemoryTile({
    super.key,
    required this.entry,
    required this.score,
    required this.onForget,
  });

  final MemoryEntry entry;
  final double? score;
  final VoidCallback onForget;

  @override
  Widget build(BuildContext context) {
    final TextTheme text = Theme.of(context).textTheme;
    return Card(
      elevation: 0,
      margin: const EdgeInsets.only(bottom: 8),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 10, 4, 10),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Icon(_MemoryLabels.icon(entry.kind), size: 20),
            const SizedBox(width: 12),
            Expanded(
              child: Semantics(
                label: 'Memory ${entry.text}',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    Text(entry.text, style: text.bodyMedium),
                    const SizedBox(height: 4),
                    Text(
                      <String>[
                        _MemoryLabels.of(entry.kind),
                        _ago(entry.createdAt),
                        if (entry.uses > 0) 'used ${entry.uses}×',
                        if (score != null)
                          'match ${(score! * 100).round()}%',
                      ].join(' · '),
                      style: text.bodySmall,
                    ),
                  ],
                ),
              ),
            ),
            IconButton(
              key: ValueKey<String>('memory.forget.${entry.id}'),
              tooltip: 'Forget',
              onPressed: onForget,
              icon: const Icon(Icons.delete_outline),
            ),
          ],
        ),
      ),
    );
  }

  static String _ago(DateTime at) {
    final Duration diff = DateTime.now().toUtc().difference(at);
    if (diff.inMinutes < 60) {
      return '${diff.inMinutes}m ago';
    }
    if (diff.inHours < 24) {
      return '${diff.inHours}h ago';
    }
    if (diff.inDays < 30) {
      return '${diff.inDays}d ago';
    }
    return '${diff.inDays ~/ 30}mo ago';
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState({
    required this.icon,
    required this.title,
    required this.message,
  });

  final IconData icon;
  final String title;
  final String message;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 24),
      child: Column(
        children: <Widget>[
          Icon(icon, size: 40),
          const SizedBox(height: 12),
          Text(title, style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 6),
          Text(message, textAlign: TextAlign.center),
        ],
      ),
    );
  }
}
