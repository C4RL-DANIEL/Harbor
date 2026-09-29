// Dynamic UI module registry.
//
// The update server's layout document names module *types*. This registry is the
// client-side half of that contract: it maps a type string to a real widget. A
// module whose type is unknown is not silently dropped — it renders a visible
// placeholder describing what the server asked for, so a schema drift is
// obvious during development instead of manifesting as a mysteriously missing
// feature.
//
// Adding a server-injectable feature is therefore a two-line change here plus a
// layout edit on the server, with no release required.

import 'dart:convert';

import 'package:flutter/material.dart';

import '../core/agent/extended_thinking.dart';
import '../core/agent/file_readers.dart';
import '../core/chat/chat_controller.dart';
import '../core/corpus/corpus_controller.dart';
import '../core/model/harbor_model_runtime.dart';
import '../core/tools/tools_controller.dart';
import '../core/training/training_controller.dart';
import '../core/update_engine/dynamic_feature_flag_provider.dart';
import '../core/update_engine/update_service.dart';
import 'screens/chat_screen.dart';
import 'screens/corpus_screen.dart';
import 'screens/tools_screen.dart';
import 'screens/training_screen.dart';

/// Signature every dynamic module widget implements.
typedef DynamicModuleBuilder = Widget Function(
  BuildContext context,
  DynamicModule module,
  DynamicModuleContext moduleContext,
);

/// Shared services handed to every dynamic module.
class DynamicModuleContext {
  const DynamicModuleContext({
    required this.flags,
    required this.reader,
    required this.agent,
    required this.updateService,
    required this.currentUpdateResult,
    this.chat,
    this.tools,
    this.training,
    this.corpus,
    this.modelRuntime,
  });

  /// Live feature-flag store, usable for additional in-module gating.
  final FeatureFlagProvider flags;

  /// Universal file reader.
  final UniversalFileReader reader;

  /// Extended-thinking agent.
  final ExtendedThinkingAgent agent;

  /// OTA update service.
  final UpdateService updateService;

  /// Result of the most recent update check, if any.
  final UpdateCheckResult? currentUpdateResult;

  // The four fields below are optional so that a layout can still be built by a
  // caller that has no model at all — a preview, a test, a widget gallery. A
  // panel whose controller is absent renders an explanation instead of a
  // silently empty frame.

  /// The chat controller, when a model is available.
  final ChatController? chat;

  /// The tool controller.
  final ToolsController? tools;

  /// The training controller.
  final TrainingController? training;

  /// The corpus controller.
  final CorpusController? corpus;

  /// The model runtime the training panel reports on.
  final HarborModelRuntime? modelRuntime;
}

/// Resolves module types to widgets and renders whole layouts.
class DynamicModuleRegistry {
  DynamicModuleRegistry([Map<String, DynamicModuleBuilder>? builders])
      : _builders = <String, DynamicModuleBuilder>{
          ...defaultBuilders,
          ...?builders,
        };

  final Map<String, DynamicModuleBuilder> _builders;

  /// Types this registry can render.
  Set<String> get supportedTypes => _builders.keys.toSet();

  /// Registers (or overrides) a builder for [type].
  void register(String type, DynamicModuleBuilder builder) {
    _builders[type] = builder;
  }

  /// Whether [type] has a renderer.
  bool supports(String type) => _builders.containsKey(type);

  /// Builds the widget for [module], falling back to a visible placeholder.
  Widget build(
    BuildContext context,
    DynamicModule module,
    DynamicModuleContext moduleContext,
  ) {
    final DynamicModuleBuilder? builder = _builders[module.type];
    if (builder == null) {
      return _UnknownModuleCard(
        module: module,
        supportedTypes: supportedTypes.toList(growable: false)..sort(),
      );
    }
    return KeyedSubtree(
      key: ValueKey<String>('module_${module.id}'),
      child: builder(context, module, moduleContext),
    );
  }

  /// Renders every section and module the flags currently allow.
  Widget buildLayout(
    BuildContext context,
    DynamicLayout layout,
    DynamicModuleContext moduleContext,
  ) {
    return ListenableBuilder(
      listenable: moduleContext.flags,
      builder: (BuildContext context, Widget? _) {
        final List<DynamicSection> sections = moduleContext.flags.visibleSections;
        if (sections.isEmpty) {
          return const _EmptyLayoutNotice();
        }
        return ListView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 96),
          children: <Widget>[
            for (final DynamicSection section in sections) ...<Widget>[
              _SectionHeader(section: section),
              const SizedBox(height: 8),
              for (final DynamicModule module in section.modules) ...<Widget>[
                build(context, module, moduleContext),
                const SizedBox(height: 12),
              ],
              const SizedBox(height: 12),
            ],
          ],
        );
      },
    );
  }

  /// The module types the app ships with.
  static final Map<String, DynamicModuleBuilder> defaultBuilders =
      <String, DynamicModuleBuilder>{
    'engine_status': (BuildContext c, DynamicModule m, DynamicModuleContext ctx) =>
        _EngineStatusCard(module: m),
    'feature_flags': (BuildContext c, DynamicModule m, DynamicModuleContext ctx) =>
        _FeatureFlagCard(module: m, context: ctx),
    'update_status': (BuildContext c, DynamicModule m, DynamicModuleContext ctx) =>
        _UpdateStatusCard(module: m, context: ctx),
    'file_inspector': (BuildContext c, DynamicModule m, DynamicModuleContext ctx) =>
        _FileInspectorCard(module: m, context: ctx),
    'agent_console': (BuildContext c, DynamicModule m, DynamicModuleContext ctx) =>
        _AgentConsoleCard(module: m, context: ctx),
    'thinking_panel': (BuildContext c, DynamicModule m, DynamicModuleContext ctx) =>
        _ThinkingPanelCard(module: m, context: ctx),
    'banner': (BuildContext c, DynamicModule m, DynamicModuleContext ctx) =>
        _ServerBannerCard(module: m),
    // The four panels below let the server place the model's own surfaces
    // anywhere in the workspace. They are registered but not part of the
    // default layout: the app already has dedicated tabs for them, and a
    // workspace that duplicates every tab by default is noise, not features.
    'chat_panel': (BuildContext c, DynamicModule m, DynamicModuleContext ctx) =>
        _EmbeddedPanel(
      module: m,
      missing: 'This build has no chat controller.',
      child: ctx.chat == null ? null : ChatScreen(controller: ctx.chat!),
    ),
    'tools_panel': (BuildContext c, DynamicModule m, DynamicModuleContext ctx) =>
        _EmbeddedPanel(
      module: m,
      missing: 'This build has no tool controller.',
      child: ctx.tools == null ? null : ToolsScreen(controller: ctx.tools!),
    ),
    'training_panel': (
      BuildContext c,
      DynamicModule m,
      DynamicModuleContext ctx,
    ) =>
        _EmbeddedPanel(
      module: m,
      missing: 'This build has no training controller.',
      child: ctx.training == null || ctx.modelRuntime == null
          ? null
          : TrainingScreen(
              controller: ctx.training!,
              runtime: ctx.modelRuntime!,
            ),
    ),
    'corpus_panel': (BuildContext c, DynamicModule m, DynamicModuleContext ctx) =>
        _EmbeddedPanel(
      module: m,
      missing: 'This build has no corpus controller.',
      child: ctx.corpus == null ? null : CorpusScreen(controller: ctx.corpus!),
    ),
  };
}

class _SectionHeader extends StatelessWidget {
  const _SectionHeader({required this.section});

  final DynamicSection section;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: <Widget>[
        Text(
          section.title,
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Divider(color: Theme.of(context).colorScheme.outlineVariant),
        ),
      ],
    );
  }
}

/// Base card shared by every module, honouring server-supplied titles.
/// A full screen embedded inside the server-driven workspace.
///
/// The height is fixed because the workspace is a `ListView`: an embedded
/// screen that grows without bound would make the list scroll inside itself.
/// 560 logical pixels is roughly one phone screen, which is enough to use the
/// panel without turning the workspace into a maze.
class _EmbeddedPanel extends StatelessWidget {
  const _EmbeddedPanel({
    required this.module,
    required this.child,
    required this.missing,
  });

  /// Roughly one phone screen, which is enough to use a panel without turning
  /// the workspace into a maze. It is a constant because every panel wants the
  /// same value; a per-panel height would be a knob with no reason to exist.
  static const double _height = 560;

  final DynamicModule module;

  /// The screen to embed, or null when the controller is absent.
  final Widget? child;

  /// What to say when there is nothing to embed.
  final String missing;

  @override
  Widget build(BuildContext context) {
    final Widget? embedded = child;
    return _ModuleCard(
      module: module,
      child: embedded == null
          ? Text(
              missing,
              style: Theme.of(context).textTheme.bodySmall,
            )
          : SizedBox(height: _height, child: embedded),
    );
  }
}

class _ModuleCard extends StatelessWidget {
  const _ModuleCard({
    required this.module,
    required this.child,
    this.trailing,
  });

  final DynamicModule module;
  final Widget child;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      color: Theme.of(context).colorScheme.surfaceContainerHigh,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Row(
              children: <Widget>[
                Expanded(
                  child: Text(
                    module.title ?? _humanise(module.type),
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.w600,
                        ),
                  ),
                ),
                if (trailing != null) trailing!,
              ],
            ),
            const SizedBox(height: 12),
            child,
          ],
        ),
      ),
    );
  }

  static String _humanise(String type) => type
      .split('_')
      .map((String w) => w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}')
      .join(' ');
}

class _EngineStatusCard extends StatelessWidget {
  const _EngineStatusCard({required this.module});

  final DynamicModule module;

  @override
  Widget build(BuildContext context) {
    final Map<String, Object?> report =
        (module.props['report'] as Map<String, Object?>?) ?? const <String, Object?>{};
    return _ModuleCard(
      module: module,
      child: report.isEmpty
          ? const Text('No engine report supplied.')
          : _KeyValueTable(values: report),
    );
  }
}

class _FeatureFlagCard extends StatelessWidget {
  const _FeatureFlagCard({required this.module, required this.context});

  final DynamicModule module;
  final DynamicModuleContext context;

  @override
  Widget build(BuildContext ctx) {
    return _ModuleCard(
      module: module,
      trailing: IconButton(
        tooltip: 'Refresh flags',
        icon: const Icon(Icons.refresh, size: 18),
        onPressed: () => context.flags.refresh(),
      ),
      child: ListenableBuilder(
        listenable: context.flags,
        builder: (BuildContext c, Widget? _) {
          final List<String> keys = context.flags.matrix.knownKeys.toList()
            ..sort();
          if (keys.isEmpty) {
            return const Text('The server has not published any flags yet.');
          }
          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Text(
                'Matrix v${context.flags.version} · '
                'source: ${context.flags.source}'
                '${context.flags.lastSyncedAt == null ? '' : ' · synced '
                    '${context.flags.lastSyncedAt!.toLocal()} '}',
                style: Theme.of(c).textTheme.bodySmall,
              ),
              const SizedBox(height: 8),
              for (final String key in keys)
                SwitchListTile(
                  dense: true,
                  contentPadding: EdgeInsets.zero,
                  title: Text(key, style: Theme.of(c).textTheme.bodyMedium),
                  subtitle: Text(
                    'resolves to '
                    '${context.flags.rawValue(key) ?? 'null'} '
                    '(${context.flags.typeOf(key).label})'
                    '${context.flags.isOverridden(key) ? ' · overridden locally' : ''}',
                    style: Theme.of(c).textTheme.bodySmall,
                  ),
                  value: context.flags.getFlag(key),
                  onChanged: context.flags.typeOf(key) ==
                              FlagValueType.boolean ||
                          context.flags.rawValue(key) == null
                      ? (bool v) => context.flags.setOverride(key, v)
                      : null,
                ),
            ],
          );
        },
      ),
    );
  }
}

class _UpdateStatusCard extends StatelessWidget {
  const _UpdateStatusCard({required this.module, required this.context});

  final DynamicModule module;
  final DynamicModuleContext context;

  @override
  Widget build(BuildContext ctx) {
    final UpdateCheckResult? result = context.currentUpdateResult;
    return _ModuleCard(
      module: module,
      trailing: IconButton(
        tooltip: 'Check for updates',
        icon: const Icon(Icons.system_update_alt, size: 18),
        onPressed: () => context.updateService.checkForUpdate(),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          _KeyValueTable(
            values: <String, Object?>{
              'installed': context.updateService.currentVersion,
              'platform': context.updateService.platform,
              'server': context.updateService.baseUrl,
              'strategy': context.updateService.strategy.name,
              if (result != null) 'decision': result.decision.name,
              if (result?.release != null)
                'latest': result!.release!.latestVersion,
              if (result?.release != null)
                'min_supported': result!.release!.minSupportedVersion,
              if (result?.error != null) 'last_error': result!.error,
            },
          ),
        ],
      ),
    );
  }
}

class _FileInspectorCard extends StatefulWidget {
  const _FileInspectorCard({required this.module, required this.context});

  final DynamicModule module;
  final DynamicModuleContext context;

  @override
  State<_FileInspectorCard> createState() => _FileInspectorCardState();
}

class _FileInspectorCardState extends State<_FileInspectorCard> {
  late final TextEditingController _path = TextEditingController(
    text: widget.module.prop<String>('initialPath', ''),
  );
  FileReadResult? _result;
  String? _error;
  bool _busy = false;

  @override
  void dispose() {
    _path.dispose();
    super.dispose();
  }

  Future<void> _read() async {
    final String path = _path.text.trim();
    if (path.isEmpty) {
      setState(() => _error = 'Enter a file or directory path first.');
      return;
    }
    setState(() {
      _busy = true;
      _error = null;
      _result = null;
    });
    try {
      final FileReadResult result = await widget.context.reader.read(path);
      if (!mounted) {
        return;
      }
      setState(() => _result = result);
    } on FileReadException catch (e) {
      if (mounted) {
        setState(() => _error = e.toString());
      }
    } on ParseFailureException catch (e) {
      if (mounted) {
        setState(() => _error = e.toString());
      }
    } on FileReadLimitException catch (e) {
      if (mounted) {
        setState(() => _error = e.toString());
      }
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return _ModuleCard(
      module: widget.module,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Row(
            children: <Widget>[
              Expanded(
                child: TextField(
                  controller: _path,
                  decoration: const InputDecoration(
                    labelText: 'Path',
                    hintText: '/sdcard/Download/report.csv',
                    isDense: true,
                    border: OutlineInputBorder(),
                  ),
                  onSubmitted: (_) => _read(),
                ),
              ),
              const SizedBox(width: 8),
              FilledButton(
                onPressed: _busy ? null : _read,
                child: _busy
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text('Read'),
              ),
            ],
          ),
          if (_error != null) ...<Widget>[
            const SizedBox(height: 12),
            Text(
              _error!,
              style: TextStyle(color: Theme.of(context).colorScheme.error),
            ),
          ],
          if (_result != null) ...<Widget>[
            const SizedBox(height: 12),
            _KeyValueTable(
              values: <String, Object?>{
                'kind': _result!.kind.name,
                'size_bytes': _result!.sizeBytes,
                'language': _result!.language,
                'encoding': _result!.encoding,
                'lines': _result!.lineCount,
                'truncated': _result!.isTruncated,
                'format': _result!.magicDescription,
                'archive_entries': _result!.archiveEntries?.length,
                'warnings': _result!.warnings,
              },
            ),
            if (_result!.binaryHeader != null) ...<Widget>[
              const SizedBox(height: 12),
              _MonoBlock(_result!.binaryHeader!),
            ],
            if (_result!.structuredData != null) ...<Widget>[
              const SizedBox(height: 12),
              _MonoBlock(
                const JsonEncoder.withIndent('  ')
                    .convert(_result!.structuredData),
                maxLines: 20,
              ),
            ] else if (_result!.text != null) ...<Widget>[
              const SizedBox(height: 12),
              _MonoBlock(_result!.text!, maxLines: 12),
            ],
          ],
        ],
      ),
    );
  }
}

class _AgentConsoleCard extends StatefulWidget {
  const _AgentConsoleCard({required this.module, required this.context});

  final DynamicModule module;
  final DynamicModuleContext context;

  @override
  State<_AgentConsoleCard> createState() => _AgentConsoleCardState();
}

class _AgentConsoleCardState extends State<_AgentConsoleCard> {
  final TextEditingController _goal = TextEditingController();
  final List<ThinkingEvent> _live = <ThinkingEvent>[];
  ThinkingTrace? _trace;
  String? _error;
  bool _busy = false;

  @override
  void dispose() {
    _goal.dispose();
    super.dispose();
  }

  Future<void> _solve() async {
    final String goal = _goal.text.trim();
    if (goal.isEmpty) {
      setState(() => _error = 'Describe a goal for the agent first.');
      return;
    }
    setState(() {
      _busy = true;
      _error = null;
      _trace = null;
      _live.clear();
    });
    try {
      final ThinkingTrace trace = await widget.context.agent.solve(goal);
      if (mounted) {
        setState(() => _trace = trace);
      }
    } on Object catch (e) {
      if (mounted) {
        setState(() => _error = 'Agent run failed: $e');
      }
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return _ModuleCard(
      module: widget.module,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          TextField(
            controller: _goal,
            minLines: 2,
            maxLines: 4,
            decoration: const InputDecoration(
              labelText: 'Goal',
              hintText: 'Read pubspec.yaml and verify lib/main.dart',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: <Widget>[
              FilledButton.icon(
                onPressed: _busy ? null : _solve,
                icon: const Icon(Icons.play_arrow),
                label: const Text('Plan & run'),
              ),
              const SizedBox(width: 12),
              if (_busy)
                const SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
            ],
          ),
          if (_error != null) ...<Widget>[
            const SizedBox(height: 12),
            Text(
              _error!,
              style: TextStyle(color: Theme.of(context).colorScheme.error),
            ),
          ],
          if (_live.isNotEmpty && _trace == null) ...<Widget>[
            const SizedBox(height: 12),
            for (final ThinkingEvent e in _live)
              Text(
                '${e.phase.label}: ${e.message}',
                style: Theme.of(context).textTheme.bodySmall,
              ),
          ],
          if (_trace != null) ...<Widget>[
            const SizedBox(height: 12),
            _KeyValueTable(
              values: <String, Object?>{
                'success': _trace!.success,
                'plans': _trace!.plans.length,
                'tool_calls': _trace!.toolCallCount,
                'replans': _trace!.replans,
                'duration_ms': _trace!.duration.inMilliseconds,
                'failure': _trace!.failureReason,
              },
            ),
            const SizedBox(height: 12),
            _MonoBlock(_trace!.toMarkdown(), maxLines: 24),
          ],
        ],
      ),
    );
  }
}

class _ThinkingPanelCard extends StatefulWidget {
  const _ThinkingPanelCard({required this.module, required this.context});

  final DynamicModule module;
  final DynamicModuleContext context;

  @override
  State<_ThinkingPanelCard> createState() => _ThinkingPanelCardState();
}

class _ThinkingPanelCardState extends State<_ThinkingPanelCard> {
  late bool _expanded =
      !widget.module.prop<bool>('collapsedByDefault', true);

  @override
  Widget build(BuildContext context) {
    return _ModuleCard(
      module: widget.module,
      trailing: IconButton(
        tooltip: _expanded ? 'Collapse' : 'Expand',
        icon: Icon(_expanded ? Icons.expand_less : Icons.expand_more, size: 18),
        onPressed: () => setState(() => _expanded = !_expanded),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            'Extended-thinking episodes surface their full plan, tool calls and '
            'verification verdicts here.',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          if (_expanded) ...<Widget>[
            const SizedBox(height: 12),
            Text(
              'Open the agent console module, run a goal, and the markdown '
              'trace is rendered inline in that card.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ],
      ),
    );
  }
}

class _ServerBannerCard extends StatelessWidget {
  const _ServerBannerCard({required this.module});

  final DynamicModule module;

  @override
  Widget build(BuildContext context) {
    final String message = module.prop<String>('message', '');
    final String severity = module.prop<String>('severity', 'info');
    final ColorScheme colors = Theme.of(context).colorScheme;
    final Color background = switch (severity) {
      'error' => colors.errorContainer,
      'warning' => colors.tertiaryContainer,
      _ => colors.secondaryContainer,
    };
    return Card(
      elevation: 0,
      color: background,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: <Widget>[
            const Icon(Icons.campaign_outlined),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                message.isEmpty ? 'Server banner' : message,
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _UnknownModuleCard extends StatelessWidget {
  const _UnknownModuleCard({
    required this.module,
    required this.supportedTypes,
  });

  final DynamicModule module;
  final List<String> supportedTypes;

  @override
  Widget build(BuildContext context) {
    final ColorScheme colors = Theme.of(context).colorScheme;
    return Card(
      elevation: 0,
      color: colors.errorContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Row(
              children: <Widget>[
                Icon(Icons.extension_off_outlined, color: colors.onErrorContainer),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Unsupported module type "${module.type}"',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          color: colors.onErrorContainer,
                        ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              'The server asked this build to render a module the client does '
              'not implement. Supported types: ${supportedTypes.join(', ')}.',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: colors.onErrorContainer,
                  ),
            ),
            const SizedBox(height: 8),
            _MonoBlock(jsonEncode(module.toJson()), maxLines: 10),
          ],
        ),
      ),
    );
  }
}

class _EmptyLayoutNotice extends StatelessWidget {
  const _EmptyLayoutNotice();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            const Icon(Icons.dashboard_customize_outlined, size: 40),
            const SizedBox(height: 12),
            Text(
              'No dynamic modules are currently enabled.',
              style: Theme.of(context).textTheme.bodyMedium,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 4),
            Text(
              'Publish a layout from the update server admin dashboard, or '
              'enable a feature flag to reveal its module.',
              style: Theme.of(context).textTheme.bodySmall,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}

class _KeyValueTable extends StatelessWidget {
  const _KeyValueTable({required this.values});

  final Map<String, Object?> values;

  @override
  Widget build(BuildContext context) {
    final TextTheme text = Theme.of(context).textTheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: values.entries
          .map(
            (MapEntry<String, Object?> e) => Padding(
              padding: const EdgeInsets.symmetric(vertical: 2),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  SizedBox(
                    width: 140,
                    child: Text(
                      e.key,
                      style: text.bodySmall?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  Expanded(
                    child: Text(
                      _render(e.value),
                      style: text.bodySmall,
                    ),
                  ),
                ],
              ),
            ),
          )
          .toList(growable: false),
    );
  }

  static String _render(Object? value) {
    if (value == null) {
      return '—';
    }
    if (value is List) {
      return value.isEmpty ? '[]' : value.join(', ');
    }
    if (value is Map) {
      return jsonEncode(value);
    }
    return value.toString();
  }
}

class _MonoBlock extends StatelessWidget {
  const _MonoBlock(this.content, {this.maxLines});

  final String content;
  final int? maxLines;

  @override
  Widget build(BuildContext context) {
    final List<String> lines = content.split('\n');
    final String shown = maxLines == null || lines.length <= maxLines!
        ? content
        : '${lines.take(maxLines!).join('\n')}\n…[${lines.length - maxLines!} more lines]';

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(10),
      ),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Text(
          shown,
          style: const TextStyle(
            fontFamily: 'monospace',
            fontSize: 11.5,
            height: 1.35,
          ),
        ),
      ),
    );
  }
}