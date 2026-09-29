// The interactive tool console: run any registered capability by hand.
//
// This screen sits next to the chat screen on purpose. Chat goes through the
// model, so when the model misreads a tool result, or refuses to call a tool at
// all, the user has no way to see what the device actually said. The console
// removes the model from the loop: it renders each tool's declared schema as a
// form, invokes the tool directly through [ToolsController], and shows the raw
// payload plus a history of every invocation. A wrong model answer can then be
// checked against a reading the user produced themselves.

import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:harbor_core/harbor_core.dart';

import '../../core/tools/tools_controller.dart';

/// Formats [time] as a wall-clock `HH:MM:SS` stamp for the history rows.
///
/// Kept local and dependency-free on purpose: the web build ships without
/// `intl` locale data, and a history stamp only has to be recognisable, not
/// localised.
String _clock(DateTime time) {
  String pad(int value) => value.toString().padLeft(2, '0');
  return '${pad(time.hour)}:${pad(time.minute)}:${pad(time.second)}';
}

/// A developer-facing console for every tool the assistant can call.
///
/// It exists next to the chat screen because the model is not always right:
/// when a chat answer about the device looks wrong, the user needs the raw
/// reading, produced by their own explicit invocation, without the model in the
/// loop to interpret it.
class ToolsScreen extends StatefulWidget {
  /// Creates the console over [controller].
  const ToolsScreen({super.key, required this.controller});

  /// The controller that owns the tool registry and the invocation history.
  final ToolsController controller;

  @override
  State<ToolsScreen> createState() => _ToolsScreenState();
}

/// State for [ToolsScreen]: the filter text and the tile owning the last result.
///
/// Both pieces live here rather than in the controller because they are view
/// concerns: the controller knows *what* ran, while the screen decides which
/// tile should display it and which tools the user wants to see.
class _ToolsScreenState extends State<ToolsScreen> {
  /// The current filter text, lower-cased on comparison rather than on entry so
  /// the field keeps exactly what the user typed.
  String _filter = '';

  /// Which tool produced the result currently shown.
  ///
  /// `controller.last` is a single shared slot, so without this the same result
  /// would be rendered beneath every expanded tile. Tracking the name keeps the
  /// result attached to the tile that asked for it.
  String? _lastRunTool;

  /// Applies [_filter] to the catalogue, matching name and description alike.
  ///
  /// Description matching matters because a user often remembers what a tool
  /// does, not what it is called.
  List<Tool> _visibleTools() {
    final String query = _filter.trim().toLowerCase();
    final List<Tool> all = widget.controller.tools;
    if (query.isEmpty) {
      return all;
    }
    return all.where((Tool tool) {
      return tool.name.toLowerCase().contains(query) ||
          tool.description.toLowerCase().contains(query);
    }).toList();
  }

  /// Runs [name] with [arguments] and tags the result with its owning tile.
  ///
  /// The name is recorded before the await so the running indicator appears in
  /// the right tile immediately; the controller's `notifyListeners` is what
  /// repaints the list once the call settles.
  Future<void> _run(String name, Map<String, Object?> arguments) async {
    setState(() {
      _lastRunTool = name;
    });
    await widget.controller.invoke(name, arguments);
    if (!mounted) {
      return;
    }
  }

  /// Explains the platform split when device-backed tools cannot run here.
  ///
  /// Without this banner a web visitor sees every device tool fail and reads it
  /// as a broken app; the banner reframes a "not available" result as an
  /// environment fact rather than a bug in the tool.
  Widget _buildBanner(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            const Icon(Icons.phone_android),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  const Text(
                    'Device readings need the Android build. '
                    'The web tools work here.',
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'A tool that returns "not available" is a platform limit, '
                    'not a failure.',
                    style: theme.textTheme.bodySmall,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Builds one expandable tool: schema form, run button and this run's result.
  Widget _buildToolTile(BuildContext context, Tool tool) {
    return ExpansionTile(
      key: PageStorageKey<String>(tool.name),
      title: Text(
        tool.name,
        style: const TextStyle(fontFamily: 'monospace'),
      ),
      subtitle: Text(tool.description),
      trailing: tool.mutating ? const Chip(label: Text('writes')) : null,
      children: <Widget>[
        _ToolForm(
          tool: tool,
          busy: widget.controller.busy,
          // The future is deliberately detached: the button's job is to start
          // the call, and the controller broadcasts completion by itself.
          onRun: (Map<String, Object?> arguments) {
            unawaited(_run(tool.name, arguments));
          },
        ),
        _buildResult(context, tool.name),
      ],
    );
  }

  /// Renders the most recent run, but only beneath the tool that produced it.
  ///
  /// While a call is in flight the tile shows progress instead of a stale
  /// result, so a slow device read cannot be mistaken for the previous answer.
  Widget _buildResult(BuildContext context, String toolName) {
    if (_lastRunTool != toolName) {
      return const SizedBox.shrink();
    }
    final ToolsController controller = widget.controller;
    if (controller.busy) {
      return const Padding(
        padding: EdgeInsets.symmetric(vertical: 12),
        child: Row(
          children: <Widget>[
            SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            SizedBox(width: 12),
            Text('Running…'),
          ],
        ),
      );
    }
    final ToolInvocation? invocation = controller.last;
    if (invocation == null) {
      return const SizedBox.shrink();
    }
    final ToolResult result = invocation.result;
    final ThemeData theme = Theme.of(context);
    return Card(
      color: result.ok ? Colors.green.withAlpha(20) : Colors.red.withAlpha(20),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Expanded(child: Text(result.summary)),
                const SizedBox(width: 8),
                Chip(label: Text('${invocation.elapsed.inMilliseconds} ms')),
              ],
            ),
            const SizedBox(height: 8),
            if (!result.ok)
              Text(
                result.error ?? 'failed',
                style: TextStyle(color: theme.colorScheme.error),
              )
            else if (result.data == null)
              const Text('(no payload)')
            else
              ConstrainedBox(
                constraints: const BoxConstraints(maxHeight: 260),
                child: SingleChildScrollView(
                  child: SelectableText(
                    const JsonEncoder.withIndent('  ').convert(result.data),
                    style: const TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 11,
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  /// Renders recent invocations newest first, with a one-tap clear.
  ///
  /// History sits below the tools rather than inside each tile because its
  /// value is comparative: seeing a run next to the one before it is how a user
  /// notices a reading change between two invocations.
  Widget _buildHistory(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    final List<ToolInvocation> history = widget.controller.history;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: <Widget>[
            Text('History', style: theme.textTheme.titleMedium),
            TextButton(
              onPressed: widget.controller.clearHistory,
              child: const Text('Clear'),
            ),
          ],
        ),
        if (history.isEmpty)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 12),
            child: Text('Nothing has been run yet.'),
          )
        else
          for (final ToolInvocation invocation in history)
            ListTile(
              leading: Icon(
                invocation.result.ok ? Icons.check_circle : Icons.error,
                color: invocation.result.ok ? Colors.green : Colors.red,
              ),
              title: Text(invocation.call.name),
              subtitle: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: <Widget>[
                  Text(
                    invocation.result.summary,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                  Text('${invocation.elapsed.inMilliseconds} ms'),
                ],
              ),
              trailing: Text(_clock(invocation.at)),
            ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Tools')),
      body: ListenableBuilder(
        listenable: widget.controller,
        builder: (BuildContext context, Widget? child) {
          final List<Tool> visible = _visibleTools();
          return ListView(
            padding: const EdgeInsets.all(16),
            children: <Widget>[
              if (!widget.controller.platformSupported) _buildBanner(context),
              TextField(
                decoration: const InputDecoration(
                  prefixIcon: Icon(Icons.search),
                  hintText: 'Filter tools',
                  border: OutlineInputBorder(),
                ),
                onChanged: (String value) {
                  setState(() {
                    _filter = value;
                  });
                },
              ),
              const SizedBox(height: 8),
              Text(
                '${visible.length} of ${widget.controller.tools.length} tools',
              ),
              const SizedBox(height: 8),
              for (final Tool tool in visible) _buildToolTile(context, tool),
              const Divider(height: 32),
              _buildHistory(context),
            ],
          );
        },
      ),
    );
  }
}

/// A form generated from one tool's declared parameters.
///
/// Building the controls from the schema rather than hand-writing a form per
/// tool means a newly registered tool is immediately runnable here, and the
/// console can never drift out of sync with the catalogue the model sees.
class _ToolForm extends StatefulWidget {
  const _ToolForm({
    required this.tool,
    required this.busy,
    required this.onRun,
  });

  /// The tool whose schema the form renders.
  final Tool tool;

  /// Whether a call is already in flight, which disables the run button.
  final bool busy;

  /// Called with the collected arguments when the user runs the tool.
  final void Function(Map<String, Object?> arguments) onRun;

  @override
  State<_ToolForm> createState() => _ToolFormState();
}

/// State for [_ToolForm]: one entry per schema field, plus validation errors.
///
/// Field controls keep their own maps so `_collect` can read a field by name
/// without walking the widget tree, which is what makes a schema-driven form
/// practical to validate and submit.
class _ToolFormState extends State<_ToolForm> {
  final Map<String, TextEditingController> _text =
      <String, TextEditingController>{};
  final Map<String, bool> _bools = <String, bool>{};
  final Map<String, String> _enums = <String, String>{};
  final Map<String, Map<String, Object?>> _specs =
      <String, Map<String, Object?>>{};
  final Set<String> _required = <String>{};

  /// Per-field validation messages, keyed by argument name.
  Map<String, String> _errors = <String, String>{};

  @override
  void initState() {
    super.initState();
    _indexProperties();
  }

  @override
  void dispose() {
    for (final TextEditingController controller in _text.values) {
      controller.dispose();
    }
    super.dispose();
  }

  /// Indexes the tool's JSON-Schema-style `parameters` block once.
  ///
  /// Doing this in `initState` instead of `build` keeps controller identity
  /// stable across the rebuilds the controller triggers, so text the user has
  /// typed is not discarded when a run starts.
  void _indexProperties() {
    final Object? rawRequired = widget.tool.parameters['required'];
    if (rawRequired is List<Object?>) {
      _required.addAll(rawRequired.whereType<String>());
    }
    final Object? rawProperties = widget.tool.parameters['properties'];
    if (rawProperties is! Map<String, Object?>) {
      return;
    }
    for (final MapEntry<String, Object?> entry in rawProperties.entries) {
      final Object? rawSpec = entry.value;
      final Map<String, Object?> spec =
          rawSpec is Map<String, Object?> ? rawSpec : <String, Object?>{};
      _specs[entry.key] = spec;
      final Object? type = spec['type'];
      if (type == 'boolean') {
        _bools[entry.key] = false;
        continue;
      }
      if (spec['enum'] is List<Object?>) {
        continue;
      }
      _text[entry.key] = TextEditingController();
    }
  }

  /// Turns the form into an argument map, or null when validation fails.
  ///
  /// Optional blanks are omitted rather than sent as empty strings or zeros, so
  /// each tool's own defaults survive; an unparseable number is reported next
  /// to the offending field and the call is abandoned.
  Map<String, Object?>? _collect() {
    final Map<String, Object?> arguments = <String, Object?>{};
    final Map<String, String> errors = <String, String>{};
    for (final MapEntry<String, Map<String, Object?>> entry in _specs.entries) {
      final String key = entry.key;
      final Map<String, Object?> spec = entry.value;
      final bool required = _required.contains(key);
      final Object? type = spec['type'];
      if (type == 'boolean') {
        arguments[key] = _bools[key] ?? false;
        continue;
      }
      final Object? enumValues = spec['enum'];
      if (enumValues is List<Object?>) {
        final String? selected = _enums[key];
        if (selected == null) {
          if (required) {
            errors[key] = 'Required';
          }
        } else {
          arguments[key] = selected;
        }
        continue;
      }
      final String raw = _text[key]?.text.trim() ?? '';
      if (raw.isEmpty) {
        if (required) {
          errors[key] = 'Required';
        }
        continue;
      }
      if (type == 'integer') {
        final int? parsed = int.tryParse(raw);
        if (parsed == null) {
          errors[key] = 'Enter a whole number';
        } else {
          arguments[key] = parsed;
        }
      } else if (type == 'number') {
        final double? parsed = double.tryParse(raw);
        if (parsed == null) {
          errors[key] = 'Enter a number';
        } else {
          arguments[key] = parsed;
        }
      } else {
        arguments[key] = raw;
      }
    }
    setState(() {
      _errors = errors;
    });
    return errors.isEmpty ? arguments : null;
  }

  /// Validates the form and hands the arguments to the parent.
  void _handleRun() {
    final Map<String, Object?>? arguments = _collect();
    if (arguments == null) {
      return;
    }
    widget.onRun(arguments);
  }

  /// Renders one schema entry as the control that suits its declared type.
  ///
  /// Booleans and enums read and write their dedicated maps because they never
  /// need a text controller; text and number fields share [_text] so that
  /// [_collect] has one place to read their values from.
  Widget _buildField(
    BuildContext context,
    ThemeData theme,
    String key,
    Map<String, Object?> spec,
  ) {
    final bool required = _required.contains(key);
    final String label = required ? '$key *' : key;
    final Object? description = spec['description'];
    final String? helper = description is String ? description : null;
    final Object? type = spec['type'];
    final Object? enumValues = spec['enum'];
    final Widget field;
    if (type == 'boolean') {
      field = SwitchListTile(
        key: ValueKey<String>('${widget.tool.name}.$key.switch'),
        contentPadding: EdgeInsets.zero,
        title: Text(label),
        subtitle: helper == null ? null : Text(helper),
        value: _bools[key] ?? false,
        onChanged: (bool value) {
          setState(() {
            _bools[key] = value;
          });
        },
      );
    } else if (enumValues is List<Object?>) {
      final List<String> options = enumValues.whereType<String>().toList();
      field = DropdownButtonFormField<String>(
        key: ValueKey<String>('${widget.tool.name}.$key.enum'),
        value: _enums[key],
        decoration: InputDecoration(labelText: label, helperText: helper),
        items: <DropdownMenuItem<String>>[
          const DropdownMenuItem<String>(value: null, child: Text('—')),
          for (final String option in options)
            DropdownMenuItem<String>(value: option, child: Text(option)),
        ],
        onChanged: (String? value) {
          setState(() {
            if (value == null) {
              _enums.remove(key);
            } else {
              _enums[key] = value;
            }
          });
        },
      );
    } else if (type == 'integer' || type == 'number') {
      field = TextFormField(
        key: ValueKey<String>('${widget.tool.name}.$key.number'),
        controller: _text[key],
        keyboardType: TextInputType.number,
        decoration: InputDecoration(labelText: label, helperText: helper),
      );
    } else {
      field = TextFormField(
        key: ValueKey<String>('${widget.tool.name}.$key.text'),
        controller: _text[key],
        decoration: InputDecoration(labelText: label, helperText: helper),
      );
    }
    final String? message = _errors[key];
    return Padding(
      key: ValueKey<String>('${widget.tool.name}.$key.row'),
      padding: const EdgeInsets.only(bottom: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          field,
          if (message != null)
            Padding(
              key: ValueKey<String>('${widget.tool.name}.$key.error'),
              padding: const EdgeInsets.only(top: 4, left: 12),
              child: Text(
                message,
                style: TextStyle(color: theme.colorScheme.error),
              ),
            ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        for (final MapEntry<String, Map<String, Object?>> entry
            in _specs.entries)
          _buildField(context, theme, entry.key, entry.value),
        Padding(
          padding: const EdgeInsets.only(top: 8),
          child: Align(
            alignment: Alignment.centerLeft,
            child: FilledButton.icon(
              onPressed: widget.busy ? null : _handleRun,
              icon: const Icon(Icons.play_arrow),
              label: Text('Run ${widget.tool.name}'),
            ),
          ),
        ),
      ],
    );
  }
}