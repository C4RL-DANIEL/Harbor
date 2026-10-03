// Harbor update server — Flutter Web admin dashboard.
//
// This is the only Flutter-dependent file in the package. The server and all
// of lib/src/** run on the pure Dart VM.
//
// Build:  flutter build web --release --dart-define=HARBOR_API_BASE_URL=<url>

import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;

import 'package:flutter/material.dart';
import 'package:harbor_core/harbor_core.dart';
import 'package:intl/intl.dart';
import 'package:pub_semver/pub_semver.dart';
// `package:web` is the supported replacement for the deprecated `dart:html`.
// This dashboard only ever builds for Flutter Web, so it uses the browser DOM
// bindings directly rather than a platform abstraction.
import 'package:web/web.dart' as web;

import 'src/admin_api_client.dart';
import 'src/browser_model.dart';
import 'src/models.dart';

void main() {
  runApp(const HarborAdminApp());
}

/// Root application: Material 3, seeded light/dark themes.
class HarborAdminApp extends StatelessWidget {
  const HarborAdminApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Harbor Update Admin',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF3F51B5),
        ),
      ),
      darkTheme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF3F51B5),
          brightness: Brightness.dark,
        ),
      ),
      home: const AdminShell(),
    );
  }
}

// ---------------------------------------------------------------------------
// Shell: connection settings + navigation
// ---------------------------------------------------------------------------

class AdminShell extends StatefulWidget {
  const AdminShell({super.key});

  @override
  State<AdminShell> createState() => _AdminShellState();
}

class _AdminShellState extends State<AdminShell> {
  static const String _defaultBaseUrl = String.fromEnvironment(
    'HARBOR_API_BASE_URL',
    defaultValue: 'http://localhost:8080',
  );

  static const List<String> _destinations = <String>[
    'Overview',
    'Releases',
    'Update Control',
    'Feature Flags',
    'Preview',
    'Chat',
  ];

  static const List<IconData> _icons = <IconData>[
    Icons.dashboard_outlined,
    Icons.rocket_launch_outlined,
    Icons.system_update_alt_outlined,
    Icons.toggle_on_outlined,
    Icons.preview_outlined,
    Icons.chat_bubble_outline,
  ];

  late final TextEditingController _baseUrlController;
  late final TextEditingController _tokenController;
  AdminApiClient? _client;
  int _index = 0;
  int _generation = 0;
  bool _tokenVisible = false;

  @override
  void initState() {
    super.initState();
    _baseUrlController = TextEditingController(
      text: _readStored('harbor.baseUrl', _defaultBaseUrl),
    );
    _tokenController = TextEditingController(
      text: _readStored('harbor.adminToken', ''),
    );
    _client = AdminApiClient(
      baseUrl: _baseUrlController.text,
      adminToken: _tokenController.text,
    );
  }

  @override
  void dispose() {
    _baseUrlController.dispose();
    _tokenController.dispose();
    _client?.close();
    super.dispose();
  }

  void _applyConnection() {
    final String baseUrl = _baseUrlController.text.trim();
    _writeStored('harbor.baseUrl', baseUrl);
    _writeStored('harbor.adminToken', _tokenController.text);
    _client?.close();
    setState(() {
      _client = AdminApiClient(
        baseUrl: baseUrl,
        adminToken: _tokenController.text,
      );
      _generation++;
    });
    _snack('Now targeting $baseUrl');
  }

  void _refreshPane() => setState(() => _generation++);

  void _snack(String message) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Harbor Update Admin'),
        actions: <Widget>[
          SizedBox(
            width: 240,
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 10),
              child: TextField(
                controller: _baseUrlController,
                decoration: const InputDecoration(
                  labelText: 'Base URL',
                  isDense: true,
                  border: OutlineInputBorder(),
                ),
                style: theme.textTheme.bodySmall,
              ),
            ),
          ),
          const SizedBox(width: 8),
          SizedBox(
            width: 220,
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 10),
              child: TextField(
                controller: _tokenController,
                obscureText: !_tokenVisible,
                decoration: InputDecoration(
                  labelText: 'Admin token',
                  isDense: true,
                  border: const OutlineInputBorder(),
                  suffixIcon: IconButton(
                    icon: Icon(
                      _tokenVisible
                          ? Icons.visibility_off
                          : Icons.visibility,
                    ),
                    onPressed: () =>
                        setState(() => _tokenVisible = !_tokenVisible),
                  ),
                ),
                style: theme.textTheme.bodySmall,
              ),
            ),
          ),
          const SizedBox(width: 8),
          FilledButton.icon(
            onPressed: _applyConnection,
            icon: const Icon(Icons.link),
            label: const Text('Connect'),
          ),
          IconButton(
            tooltip: 'Reload current pane',
            onPressed: _refreshPane,
            icon: const Icon(Icons.refresh),
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: Row(
        children: <Widget>[
          NavigationRail(
            selectedIndex: _index,
            onDestinationSelected: (int value) =>
                setState(() => _index = value),
            labelType: NavigationRailLabelType.all,
            destinations: <NavigationRailDestination>[
              for (int i = 0; i < _destinations.length; i++)
                NavigationRailDestination(
                  icon: Icon(_icons[i]),
                  label: Text(_destinations[i]),
                ),
            ],
          ),
          const VerticalDivider(width: 1),
          Expanded(child: _buildPane(context)),
        ],
      ),
    );
  }

  Widget _buildPane(BuildContext context) {
    final AdminApiClient? client = _client;
    if (client == null) {
      return const Center(child: CircularProgressIndicator());
    }
    final Key key = ValueKey<int>(_generation);
    if (_index == 0) {
      return OverviewPane(key: key, client: client);
    }
    if (_index == 1) {
      return ReleasesPane(key: key, client: client);
    }
    if (_index == 2) {
      return UpdateControlPane(key: key, client: client);
    }
    if (_index == 3) {
      return FeatureFlagsPane(key: key, client: client);
    }
    if (_index == 4) {
      return PreviewPane(key: key, client: client);
    }
    return ChatPane(key: key, client: client);
  }
}

// ---------------------------------------------------------------------------
// Overview
// ---------------------------------------------------------------------------

class _OverviewData {
  const _OverviewData({
    required this.health,
    required this.flags,
    required this.releases,
    required this.state,
  });

  final Map<String, Object?> health;
  final FeatureFlagMatrix flags;
  final Map<String, ReleaseInfo> releases;
  final ServerState state;
}

class OverviewPane extends StatefulWidget {
  const OverviewPane({super.key, required this.client});

  final AdminApiClient client;

  @override
  State<OverviewPane> createState() => _OverviewPaneState();
}

class _OverviewPaneState extends State<OverviewPane> {
  late Future<_OverviewData> _future;

  @override
  void initState() {
    super.initState();
    _future = _load();
  }

  Future<_OverviewData> _load() async {
    final Map<String, Object?> health = await widget.client.health();
    final FeatureFlagMatrix flags = await widget.client.fetchFlags();
    final Map<String, ReleaseInfo> releases =
        await widget.client.listReleases();
    final ServerState state = await widget.client.adminState();
    return _OverviewData(
      health: health,
      flags: flags,
      releases: releases,
      state: state,
    );
  }

  /// Flips `force_update` for one platform and reloads the overview.
  ///
  /// The switch is the only admin action worth having on this pane: it is the
  /// one that can lock every device out, so an operator should be able to see
  /// and change it without leaving the page that shows service health.
  Future<void> _toggleForceUpdate(String platform, bool value) async {
    try {
      await widget.client.setForceUpdate(platform: platform, forceUpdate: value);
      if (!mounted) {
        return;
      }
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          SnackBar(
            content:
                Text('Force update ${value ? 'enabled' : 'disabled'} for $platform'),
          ),
        );
      _reload();
    } on Object catch (error) {
      if (!mounted) {
        return;
      }
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          SnackBar(content: Text('Could not change force_update: $error')),
        );
    }
  }

  void _reload() => setState(() => _future = _load());

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<_OverviewData>(
      future: _future,
      builder: (BuildContext context, AsyncSnapshot<_OverviewData> snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const Center(child: CircularProgressIndicator());
        }
        if (snapshot.hasError) {
          return ErrorPanel(message: '${snapshot.error}', onRetry: _reload);
        }
        final _OverviewData data = snapshot.data!;
        return _ScrollPage(
          title: 'Overview',
          subtitle: 'Live service health, stored releases and flag revision.',
          onRefresh: _reload,
          children: <Widget>[
            _Panel(
              title: 'Service health',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  _InfoRow('Status', '${data.health['status'] ?? 'unknown'}'),
                  _InfoRow(
                    'Version count',
                    '${data.health['version_count'] ?? 0}',
                  ),
                  _InfoRow(
                    'Server time',
                    _formatTimestamp(data.health['time'] as String?),
                  ),
                  _InfoRow('Base URL', widget.client.baseUrl),
                ],
              ),
            ),
            _Panel(
              title: 'Stored releases (${data.releases.length})',
              child: data.releases.isEmpty
                  ? const Text('No releases stored yet.')
                  : DataTable(
                      columns: const <DataColumn>[
                        DataColumn(label: Text('Platform')),
                        DataColumn(label: Text('Latest')),
                        DataColumn(label: Text('Min supported')),
                        DataColumn(label: Text('Force')),
                        DataColumn(label: Text('Size')),
                      ],
                      rows: <DataRow>[
                        for (final ReleaseInfo release
                            in data.releases.values)
                          DataRow(cells: <DataCell>[
                            DataCell(Text(release.platform)),
                            DataCell(Text(release.version)),
                            DataCell(Text(release.minSupportedVersion)),
                            DataCell(
                              Text(release.forceUpdate ? 'yes' : 'no'),
                            ),
                            DataCell(Text(_formatBytes(release.sizeBytes))),
                          ]),
                      ],
                    ),
            ),
            _Panel(
              title: 'Force-update control',
              child: data.releases.isEmpty
                  ? const Text(
                      'No releases stored yet, so there is nothing to force.',
                    )
                  : Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: <Widget>[
                        const Text(
                          'Forcing an update makes the client dialog '
                          'non-dismissible for every device below the latest '
                          'version of that platform.',
                        ),
                        const SizedBox(height: 8),
                        for (final ReleaseInfo release in data.releases.values)
                          SwitchListTile(
                            contentPadding: EdgeInsets.zero,
                            title: Text(
                              '${release.platform} · ${release.version}',
                            ),
                            subtitle: Text(
                              release.forceUpdate
                                  ? 'Devices below ${release.version} are '
                                      'blocked until they install.'
                                  : 'Update prompt is dismissible.',
                            ),
                            value: release.forceUpdate,
                            onChanged: (bool value) => unawaited(
                              _toggleForceUpdate(release.platform, value),
                            ),
                          ),
                      ],
                    ),
            ),
            _Panel(
              title: 'Feature flag matrix v${data.flags.version}',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  _InfoRow('Flags', '${data.flags.flags.length} keys'),
                  _InfoRow(
                    'Remote defaults',
                    '${data.flags.remoteDefaults.length} keys',
                  ),
                  _InfoRow(
                    'Updated at',
                    data.flags.updatedAt == null
                        ? 'unknown'
                        : DateFormat.yMMMd()
                            .add_Hms()
                            .format(data.flags.updatedAt!.toLocal()),
                  ),
                  const Divider(),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: <Widget>[
                      for (final MapEntry<String, Object?> entry
                          in data.flags.flags.entries)
                        Chip(
                          label: Text('${entry.key} = ${entry.value}'),
                        ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        );
      },
    );
  }
}

// ---------------------------------------------------------------------------
// Releases
// ---------------------------------------------------------------------------

class ReleasesPane extends StatefulWidget {
  const ReleasesPane({super.key, required this.client});

  final AdminApiClient client;

  @override
  State<ReleasesPane> createState() => _ReleasesPaneState();
}

class _ReleasesPaneState extends State<ReleasesPane> {
  final TextEditingController _version = TextEditingController();
  final TextEditingController _minSupported = TextEditingController();
  final TextEditingController _downloadUrl = TextEditingController();
  final TextEditingController _sha256 = TextEditingController();
  final TextEditingController _sizeBytes = TextEditingController();
  final TextEditingController _changelog = TextEditingController();
  final TextEditingController _releaseNotes = TextEditingController();
  String _platform = 'android';
  bool _forceUpdate = false;
  bool _busy = false;
  String? _message;
  bool _messageIsError = false;
  late Future<Map<String, ReleaseInfo>> _releases;

  @override
  void initState() {
    super.initState();
    _releases = widget.client.listReleases();
  }

  @override
  void dispose() {
    _version.dispose();
    _minSupported.dispose();
    _downloadUrl.dispose();
    _sha256.dispose();
    _sizeBytes.dispose();
    _changelog.dispose();
    _releaseNotes.dispose();
    super.dispose();
  }

  Map<String, Object?> get _requestBody => <String, Object?>{
        'platform': _platform,
        'version': _version.text.trim(),
        'min_supported_version': _minSupported.text.trim(),
        'download_url': _downloadUrl.text.trim(),
        'sha256': _sha256.text.trim().toLowerCase(),
        'size_bytes': int.tryParse(_sizeBytes.text.trim()) ?? 0,
        'changelog': _changelog.text,
        'force_update': _forceUpdate,
        'release_notes_url': _releaseNotes.text.trim(),
      };

  Future<void> _publish() async {
    setState(() {
      _busy = true;
      _message = null;
    });
    try {
      final ReleaseInfo release = ReleaseInfo.fromJson(
        _requestBody,
        defaultPublishedAt: DateTime.now().toUtc(),
      );
      final ReleaseInfo stored =
          await widget.client.publishRelease(release);
      setState(() {
        _message = 'Published ${stored.platform} ${stored.version}';
        _messageIsError = false;
        _releases = widget.client.listReleases();
      });
    } on ApiException catch (error) {
      setState(() {
        _message = 'Error ${error.statusCode}: ${error.message}';
        _messageIsError = true;
      });
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return _ScrollPage(
      title: 'Releases',
      subtitle: 'Publish release metadata; the client OTA engine consumes it.',
      onRefresh: () => setState(() {
        _releases = widget.client.listReleases();
      }),
      children: <Widget>[
        _Panel(
          title: 'Publish metadata',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              Row(
                children: <Widget>[
                  Expanded(
                    child: DropdownButtonFormField<String>(
                      value: _platform,
                      decoration: const InputDecoration(
                        labelText: 'Platform',
                        border: OutlineInputBorder(),
                      ),
                      items: _platformItems(),
                      onChanged: (String? value) {
                        if (value != null) {
                          setState(() => _platform = value);
                        }
                      },
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _field(_version, 'Version (semver)'),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child:
                        _field(_minSupported, 'Min supported version'),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                children: <Widget>[
                  Expanded(
                    flex: 3,
                    child: _field(_downloadUrl, 'Download URL'),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    flex: 3,
                    child: _field(_sha256, 'SHA-256 (64 hex)'),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    flex: 2,
                    child: _field(
                      _sizeBytes,
                      'Size (bytes)',
                      keyboardType: TextInputType.number,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              _field(_changelog, 'Changelog', maxLines: 4),
              const SizedBox(height: 12),
              Row(
                children: <Widget>[
                  Expanded(
                    child: _field(_releaseNotes, 'Release notes URL'),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: SwitchListTile(
                      title: const Text('Force update'),
                      value: _forceUpdate,
                      onChanged: (bool value) =>
                          setState(() => _forceUpdate = value),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Row(
                children: <Widget>[
                  FilledButton.icon(
                    onPressed: _busy ? null : _publish,
                    icon: _busy
                        ? const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.upload),
                    label: const Text('Publish release'),
                  ),
                  const SizedBox(width: 12),
                  if (_message != null)
                    Expanded(
                      child: Text(
                        _message!,
                        style: TextStyle(
                          color: _messageIsError
                              ? Theme.of(context).colorScheme.error
                              : Theme.of(context).colorScheme.primary,
                        ),
                      ),
                    ),
                ],
              ),
            ],
          ),
        ),
        _Panel(
          title: 'POST body preview',
          child: JsonText(value: _requestBody),
        ),
        _Panel(
          title: 'Existing releases',
          child: FutureBuilder<Map<String, ReleaseInfo>>(
            future: _releases,
            builder: (
              BuildContext context,
              AsyncSnapshot<Map<String, ReleaseInfo>> snapshot,
            ) {
              if (snapshot.connectionState != ConnectionState.done) {
                return const Padding(
                  padding: EdgeInsets.all(16),
                  child: Center(child: CircularProgressIndicator()),
                );
              }
              if (snapshot.hasError) {
                return Text('${snapshot.error}');
              }
              final Map<String, ReleaseInfo> releases =
                  snapshot.data ?? const <String, ReleaseInfo>{};
              if (releases.isEmpty) {
                return const Text('No releases stored yet.');
              }
              return DataTable(
                columns: const <DataColumn>[
                  DataColumn(label: Text('Platform')),
                  DataColumn(label: Text('Version')),
                  DataColumn(label: Text('Min')),
                  DataColumn(label: Text('Force')),
                  DataColumn(label: Text('URL')),
                ],
                rows: <DataRow>[
                  for (final ReleaseInfo release in releases.values)
                    DataRow(cells: <DataCell>[
                      DataCell(Text(release.platform)),
                      DataCell(Text(release.version)),
                      DataCell(Text(release.minSupportedVersion)),
                      DataCell(Text(release.forceUpdate ? 'yes' : 'no')),
                      DataCell(
                        ConstrainedBox(
                          constraints: const BoxConstraints(maxWidth: 320),
                          child: Text(
                            release.downloadUrl,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ),
                    ]),
                ],
              );
            },
          ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Update control
// ---------------------------------------------------------------------------

class UpdateControlPane extends StatefulWidget {
  const UpdateControlPane({super.key, required this.client});

  final AdminApiClient client;

  @override
  State<UpdateControlPane> createState() => _UpdateControlPaneState();
}

class _UpdateControlPaneState extends State<UpdateControlPane> {
  final TextEditingController _minSupported = TextEditingController();
  Map<String, ReleaseInfo> _releases = <String, ReleaseInfo>{};
  String? _platform;
  bool _loading = true;
  String? _error;
  String? _message;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _minSupported.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final Map<String, ReleaseInfo> releases =
          await widget.client.listReleases();
      setState(() {
        _releases = releases;
        _platform = _platform != null && releases.containsKey(_platform)
            ? _platform
            : (releases.keys.isNotEmpty ? releases.keys.first : 'android');
        _minSupported.text =
            releases[_platform]?.minSupportedVersion ?? '';
        _loading = false;
      });
    } on ApiException catch (error) {
      setState(() {
        _error = 'Error ${error.statusCode}: ${error.message}';
        _loading = false;
      });
    }
  }

  ReleaseInfo? get _current =>
      _platform == null ? null : _releases[_platform];

  Future<void> _applyMinSupported() async {
    final ReleaseInfo? current = _current;
    if (current == null) {
      return;
    }
    final String value = _minSupported.text.trim();
    try {
      final Version candidate = parseSemver(
        value,
        field: 'min_supported_version',
      );
      if (candidate > current.semver) {
        setState(() {
          _message = 'min_supported_version must not exceed '
              '${current.version}';
        });
        return;
      }
    } on ValidationException catch (error) {
      setState(() => _message = error.message);
      return;
    }
    setState(() {
      _busy = true;
      _message = null;
    });
    try {
      await widget.client.setMinSupported(
        platform: current.platform,
        minSupportedVersion: value,
      );
      await _load();
      setState(() => _message = 'Updated ${current.platform} minimum');
    } on ApiException catch (error) {
      setState(() => _message = 'Error ${error.statusCode}: ${error.message}');
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  Future<void> _toggleForce(bool value) async {
    final ReleaseInfo? current = _current;
    if (current == null) {
      return;
    }
    final bool? confirmed = await showDialog<bool>(
      context: context,
      builder: (BuildContext ctx) => AlertDialog(
        title: Text(value ? 'Force this update?' : 'Stop forcing updates?'),
        content: Text(
          value
              ? 'Every device below the latest version will be blocked by a '
                  'non-dismissible update dialog until it installs ${current.version}.'
              : 'The update dialog becomes dismissible again for devices on '
                  '${current.platform}.',
        ),
        actions: <Widget>[
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Confirm'),
          ),
        ],
      ),
    );
    if (confirmed != true) {
      return;
    }
    setState(() => _busy = true);
    try {
      await widget.client.setForceUpdate(
        platform: current.platform,
        forceUpdate: value,
      );
      await _load();
      setState(() =>
          _message = '${current.platform} force_update = $value');
    } on ApiException catch (error) {
      setState(() => _message = 'Error ${error.statusCode}: ${error.message}');
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    final ReleaseInfo? current = _current;
    return _ScrollPage(
      title: 'Update Control',
      subtitle: 'Constrain the minimum supported version and force updates.',
      onRefresh: _load,
      children: <Widget>[
        if (_error != null)
          _Panel(title: 'Connection error', child: Text(_error!)),
        _Panel(
          title: 'Per-platform policy',
          child: current == null
              ? const Text('No releases stored yet. Publish one first.')
              : Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: <Widget>[
                    DropdownButtonFormField<String>(
                      value: _platform,
                      decoration: const InputDecoration(
                        labelText: 'Platform',
                        border: OutlineInputBorder(),
                      ),
                      items: <DropdownMenuItem<String>>[
                        for (final String platform
                            in _releases.keys.toList()..sort())
                          DropdownMenuItem<String>(
                            value: platform,
                            child: Text(platform),
                          ),
                      ],
                      onChanged: (String? value) {
                        if (value == null) {
                          return;
                        }
                        setState(() {
                          _platform = value;
                          _minSupported.text =
                              _releases[value]?.minSupportedVersion ?? '';
                          _message = null;
                        });
                      },
                    ),
                    const SizedBox(height: 16),
                    _InfoRow('Latest version', current.version),
                    _InfoRow(
                      'Minimum supported',
                      current.minSupportedVersion,
                    ),
                    _InfoRow(
                      'Force update',
                      current.forceUpdate ? 'enabled' : 'disabled',
                    ),
                    const SizedBox(height: 16),
                    Row(
                      children: <Widget>[
                        Expanded(
                          child: TextField(
                            controller: _minSupported,
                            decoration: const InputDecoration(
                              labelText:
                                  'New minimum supported version (<= latest)',
                              border: OutlineInputBorder(),
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        FilledButton(
                          onPressed: _busy ? null : _applyMinSupported,
                          child: const Text('Apply minimum'),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    SwitchListTile(
                      title: const Text('Force update'),
                      subtitle: const Text(
                        'Makes the client update dialog non-dismissible.',
                      ),
                      value: current.forceUpdate,
                      onChanged: _busy ? null : _toggleForce,
                    ),
                    if (_message != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: Text(_message!),
                      ),
                  ],
                ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Feature flags
// ---------------------------------------------------------------------------

class _FlagEntry {
  _FlagEntry({String key = '', this.type = 'bool', String value = 'false'})
      : keyController = TextEditingController(text: key),
        valueController = TextEditingController(text: value);

  final TextEditingController keyController;
  final TextEditingController valueController;
  String type;

  void dispose() {
    keyController.dispose();
    valueController.dispose();
  }
}

class FeatureFlagsPane extends StatefulWidget {
  const FeatureFlagsPane({super.key, required this.client});

  final AdminApiClient client;

  @override
  State<FeatureFlagsPane> createState() => _FeatureFlagsPaneState();
}

class _FeatureFlagsPaneState extends State<FeatureFlagsPane> {
  final TextEditingController _layout = TextEditingController();
  List<_FlagEntry> _flags = <_FlagEntry>[];
  List<_FlagEntry> _defaults = <_FlagEntry>[];
  bool _merge = true;
  bool _loading = true;
  bool _busy = false;
  String? _message;
  bool _messageIsError = false;
  int _version = 0;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _layout.dispose();
    for (final _FlagEntry entry in _flags) {
      entry.dispose();
    }
    for (final _FlagEntry entry in _defaults) {
      entry.dispose();
    }
    super.dispose();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final FeatureFlagMatrix matrix = await widget.client.fetchFlags();
      for (final _FlagEntry entry in _flags) {
        entry.dispose();
      }
      for (final _FlagEntry entry in _defaults) {
        entry.dispose();
      }
      setState(() {
        _version = matrix.version;
        _flags = _entriesFrom(matrix.flags);
        _defaults = _entriesFrom(matrix.remoteDefaults);
        _layout.text =
            const JsonEncoder.withIndent('  ').convert(matrix.layout);
        _loading = false;
      });
    } on ApiException catch (error) {
      setState(() {
        _message = 'Error ${error.statusCode}: ${error.message}';
        _messageIsError = true;
        _loading = false;
      });
    }
  }

  void _formatLayout() {
    try {
      final Object? decoded = jsonDecode(_layout.text);
      _layout.text = const JsonEncoder.withIndent('  ').convert(decoded);
      setState(() {
        _message = 'Layout formatted';
        _messageIsError = false;
      });
    } on FormatException catch (error) {
      setState(() {
        _message = 'Layout is not valid JSON: ${error.message}';
        _messageIsError = true;
      });
    }
  }

  Future<void> _apply() async {
    setState(() {
      _busy = true;
      _message = null;
    });
    try {
      final Map<String, Object?> flags = _mapFrom(_flags);
      final Map<String, Object?> defaults = _mapFrom(_defaults);
      final Object? layout = _layout.text.trim().isEmpty
          ? <String, Object?>{}
          : jsonDecode(_layout.text);
      if (layout is! Map<String, Object?>) {
        throw const ValidationException('layout must be a JSON object');
      }
      FeatureFlagMatrix.validateLayout(layout);
      final int version = await widget.client.putFlags(
        flags: flags,
        remoteDefaults: defaults,
        layout: layout,
        merge: _merge,
      );
      setState(() {
        _version = version;
        _message = 'Saved matrix as version $version';
        _messageIsError = false;
      });
    } on ApiException catch (error) {
      setState(() {
        _message = 'Error ${error.statusCode}: ${error.message}';
        _messageIsError = true;
      });
    } on FormatException catch (error) {
      setState(() {
        _message = 'Invalid JSON: ${error.message}';
        _messageIsError = true;
      });
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    return _ScrollPage(
      title: 'Feature Flags (v$_version)',
      subtitle: 'Edit flag values, remote defaults and the dynamic layout.',
      onRefresh: _load,
      children: <Widget>[
        _Panel(
          title: 'Flags',
          trailing: OutlinedButton.icon(
            onPressed: () => setState(
              () => _flags = <_FlagEntry>[..._flags, _FlagEntry()],
            ),
            icon: const Icon(Icons.add),
            label: const Text('Add flag'),
          ),
          child: _entriesEditor(_flags),
        ),
        _Panel(
          title: 'Remote defaults',
          trailing: OutlinedButton.icon(
            onPressed: () => setState(
              () => _defaults = <_FlagEntry>[..._defaults, _FlagEntry()],
            ),
            icon: const Icon(Icons.add),
            label: const Text('Add default'),
          ),
          child: _entriesEditor(_defaults),
        ),
        _Panel(
          title: 'Layout JSON',
          trailing: Wrap(
            spacing: 8,
            children: <Widget>[
              OutlinedButton.icon(
                onPressed: _formatLayout,
                icon: const Icon(Icons.format_align_left),
                label: const Text('Format'),
              ),
            ],
          ),
          child: TextField(
            controller: _layout,
            maxLines: 14,
            style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
            decoration: const InputDecoration(
              border: OutlineInputBorder(),
              hintText: '{"sections": []}',
            ),
          ),
        ),
        _Panel(
          title: 'Apply',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              SwitchListTile(
                title: const Text('Merge with stored matrix'),
                subtitle: const Text(
                  'Off replaces flags, defaults and layout wholesale.',
                ),
                value: _merge,
                onChanged: (bool value) => setState(() => _merge = value),
              ),
              const SizedBox(height: 12),
              Row(
                children: <Widget>[
                  FilledButton.icon(
                    onPressed: _busy ? null : _apply,
                    icon: const Icon(Icons.save),
                    label: const Text('Save matrix'),
                  ),
                  const SizedBox(width: 12),
                  if (_message != null)
                    Expanded(
                      child: Text(
                        _message!,
                        style: TextStyle(
                          color: _messageIsError
                              ? Theme.of(context).colorScheme.error
                              : Theme.of(context).colorScheme.primary,
                        ),
                      ),
                    ),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _entriesEditor(List<_FlagEntry> entries) {
    if (entries.isEmpty) {
      return const Text('No keys. Use “Add” to create one.');
    }
    return Column(
      children: <Widget>[
        for (int i = 0; i < entries.length; i++)
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Row(
              children: <Widget>[
                Expanded(
                  flex: 3,
                  child: TextField(
                    controller: entries[i].keyController,
                    decoration: const InputDecoration(
                      labelText: 'Key',
                      isDense: true,
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                SizedBox(
                  width: 130,
                  child: DropdownButtonFormField<String>(
                    value: entries[i].type,
                    decoration: const InputDecoration(
                      labelText: 'Type',
                      isDense: true,
                      border: OutlineInputBorder(),
                    ),
                    items: const <DropdownMenuItem<String>>[
                      DropdownMenuItem<String>(
                        value: 'bool',
                        child: Text('bool'),
                      ),
                      DropdownMenuItem<String>(
                        value: 'string',
                        child: Text('string'),
                      ),
                      DropdownMenuItem<String>(
                        value: 'number',
                        child: Text('number'),
                      ),
                      DropdownMenuItem<String>(
                        value: 'json',
                        child: Text('json'),
                      ),
                    ],
                    onChanged: (String? value) {
                      if (value != null) {
                        setState(() => entries[i].type = value);
                      }
                    },
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  flex: 4,
                  child: TextField(
                    controller: entries[i].valueController,
                    decoration: const InputDecoration(
                      labelText: 'Value',
                      isDense: true,
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                IconButton(
                  tooltip: 'Remove',
                  onPressed: () => setState(() {
                    final _FlagEntry removed = entries.removeAt(i);
                    removed.dispose();
                  }),
                  icon: const Icon(Icons.delete_outline),
                ),
              ],
            ),
          ),
      ],
    );
  }

  static List<_FlagEntry> _entriesFrom(Map<String, Object?> map) {
    final List<_FlagEntry> entries = <_FlagEntry>[];
    for (final MapEntry<String, Object?> entry in map.entries) {
      entries.add(
        _FlagEntry(
          key: entry.key,
          type: _typeOf(entry.value),
          value: _valueText(entry.value),
        ),
      );
    }
    return entries;
  }

  static String _typeOf(Object? value) {
    if (value is bool) {
      return 'bool';
    }
    if (value is num) {
      return 'number';
    }
    if (value is String) {
      return 'string';
    }
    return 'json';
  }

  static String _valueText(Object? value) {
    if (value is String) {
      return value;
    }
    if (value == null) {
      return 'null';
    }
    if (value is num || value is bool) {
      return value.toString();
    }
    return jsonEncode(value);
  }

  static Map<String, Object?> _mapFrom(List<_FlagEntry> entries) {
    final Map<String, Object?> result = <String, Object?>{};
    for (final _FlagEntry entry in entries) {
      final String key = entry.keyController.text.trim();
      if (key.isEmpty) {
        continue;
      }
      result[key] = _parseValue(entry.type, entry.valueController.text);
    }
    return result;
  }

  static Object? _parseValue(String type, String raw) {
    switch (type) {
      case 'bool':
        return raw.trim().toLowerCase() == 'true';
      case 'number':
        return num.tryParse(raw.trim()) ?? 0;
      case 'json':
        return jsonDecode(raw);
      case 'string':
      default:
        return raw;
    }
  }
}

// ---------------------------------------------------------------------------
// Preview
// ---------------------------------------------------------------------------

class _PreviewData {
  const _PreviewData({
    required this.updateCheck,
    required this.flags,
  });

  final UpdateCheckResponse updateCheck;
  final FeatureFlagMatrix flags;
}

class PreviewPane extends StatefulWidget {
  const PreviewPane({super.key, required this.client});

  final AdminApiClient client;

  @override
  State<PreviewPane> createState() => _PreviewPaneState();
}

class _PreviewPaneState extends State<PreviewPane> {
  final TextEditingController _installed = TextEditingController(text: '1.0.0');
  String _platform = 'android';
  Future<_PreviewData>? _future;

  @override
  void initState() {
    super.initState();
    _future = _load();
  }

  @override
  void dispose() {
    _installed.dispose();
    super.dispose();
  }

  Future<_PreviewData> _load() async {
    final UpdateCheckResponse updateCheck = await widget.client.updateCheck(
      installedVersion: _installed.text.trim(),
      platform: _platform,
    );
    final FeatureFlagMatrix flags = await widget.client.fetchFlags();
    return _PreviewData(updateCheck: updateCheck, flags: flags);
  }

  @override
  Widget build(BuildContext context) {
    return _ScrollPage(
      title: 'Preview',
      subtitle: 'Exactly what a device at a given version receives.',
      onRefresh: () => setState(() => _future = _load()),
      children: <Widget>[
        _Panel(
          title: 'Device',
          child: Row(
            children: <Widget>[
              Expanded(
                child: DropdownButtonFormField<String>(
                  value: _platform,
                  decoration: const InputDecoration(
                    labelText: 'Platform',
                    border: OutlineInputBorder(),
                  ),
                  items: _platformItems(),
                  onChanged: (String? value) {
                    if (value != null) {
                      setState(() => _platform = value);
                    }
                  },
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: TextField(
                  controller: _installed,
                  decoration: const InputDecoration(
                    labelText: 'Installed version',
                    border: OutlineInputBorder(),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              FilledButton.icon(
                onPressed: () => setState(() => _future = _load()),
                icon: const Icon(Icons.play_arrow),
                label: const Text('Fetch'),
              ),
            ],
          ),
        ),
        FutureBuilder<_PreviewData>(
          future: _future,
          builder: (
            BuildContext context,
            AsyncSnapshot<_PreviewData> snapshot,
          ) {
            if (snapshot.connectionState != ConnectionState.done) {
              return const Padding(
                padding: EdgeInsets.all(24),
                child: Center(child: CircularProgressIndicator()),
              );
            }
            if (snapshot.hasError) {
              return _Panel(
                title: 'Error',
                child: Text('${snapshot.error}'),
              );
            }
            final _PreviewData data = snapshot.data!;
            return Column(
              children: <Widget>[
                _Panel(
                  title: 'GET /api/v1/update-check',
                  child: JsonText(value: data.updateCheck.toJson()),
                ),
                _Panel(
                  title: 'GET /api/v1/flags',
                  child: JsonText(value: data.flags.toJson()),
                ),
              ],
            );
          },
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Shared widgets / helpers
// ---------------------------------------------------------------------------

class _ScrollPage extends StatelessWidget {
  const _ScrollPage({
    required this.title,
    required this.subtitle,
    required this.children,
    required this.onRefresh,
  });

  final String title;
  final String subtitle;
  final List<Widget> children;
  final VoidCallback onRefresh;

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          Row(
            children: <Widget>[
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    Text(
                      title,
                      style: Theme.of(context).textTheme.headlineSmall,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      subtitle,
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                  ],
                ),
              ),
              OutlinedButton.icon(
                onPressed: onRefresh,
                icon: const Icon(Icons.refresh),
                label: const Text('Reload'),
              ),
            ],
          ),
          const SizedBox(height: 16),
          ...children,
        ],
      ),
    );
  }
}

class _Panel extends StatelessWidget {
  const _Panel({
    required this.title,
    required this.child,
    this.trailing,
  });

  final String title;
  final Widget child;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            Row(
              children: <Widget>[
                Expanded(
                  child: Text(
                    title,
                    style: Theme.of(context).textTheme.titleMedium,
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
}

class ErrorPanel extends StatelessWidget {
  const ErrorPanel({super.key, required this.message, required this.onRetry});

  final String message;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          Icon(
            Icons.error_outline,
            color: Theme.of(context).colorScheme.error,
            size: 42,
          ),
          const SizedBox(height: 12),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 32),
            child: Text(message, textAlign: TextAlign.center),
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: onRetry,
            icon: const Icon(Icons.refresh),
            label: const Text('Retry'),
          ),
        ],
      ),
    );
  }
}

class JsonText extends StatelessWidget {
  const JsonText({super.key, required this.value});

  final Object? value;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: SelectableText(
        const JsonEncoder.withIndent('  ').convert(value),
        style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
      ),
    );
  }
}

class _InfoRow extends StatelessWidget {
  const _InfoRow(this.label, this.value);

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          SizedBox(
            width: 160,
            child: Text(
              label,
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
          ),
          Expanded(child: SelectableText(value)),
        ],
      ),
    );
  }
}

Widget _field(
  TextEditingController controller,
  String label, {
  int maxLines = 1,
  TextInputType? keyboardType,
}) {
  return TextField(
    controller: controller,
    maxLines: maxLines,
    keyboardType: keyboardType,
    decoration: InputDecoration(
      labelText: label,
      border: const OutlineInputBorder(),
      isDense: true,
    ),
  );
}

List<DropdownMenuItem<String>> _platformItems() {
  final List<String> platforms = kSupportedPlatforms.toList()..sort();
  return <DropdownMenuItem<String>>[
    for (final String platform in platforms)
      DropdownMenuItem<String>(value: platform, child: Text(platform)),
  ];
}

String _formatBytes(int bytes) {
  if (bytes <= 0) {
    return '0 B';
  }
  final NumberFormat format = NumberFormat.decimalPattern();
  if (bytes < 1024) {
    return '${format.format(bytes)} B';
  }
  if (bytes < 1024 * 1024) {
    return '${(bytes / 1024).toStringAsFixed(1)} KiB';
  }
  return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MiB';
}

String _formatTimestamp(String? value) {
  if (value == null) {
    return 'unknown';
  }
  final DateTime? parsed = DateTime.tryParse(value);
  if (parsed == null) {
    return value;
  }
  return DateFormat.yMMMd().add_Hms().format(parsed.toLocal());
}

// ---- localStorage (web only) ---------------------------------------------

String _readStored(String key, String fallback) {
  try {
    final String? value = web.window.localStorage.getItem(key);
    if (value == null || value.isEmpty) {
      return fallback;
    }
    return value;
  } catch (_) {
    return fallback;
  }
}

void _writeStored(String key, String value) {
  try {
    web.window.localStorage.setItem(key, value);
  } catch (_) {
    // Storage may be unavailable in private browsing; settings stay in RAM.
  }
}

// ---------------------------------------------------------------------------
// Chat (in-browser model)
// ---------------------------------------------------------------------------

/// The chat pane: a model, its transcript and its tools, all inside the tab.
///
/// It deliberately takes no [AdminApiClient]: this pane runs with no server at
/// all, which is what makes the static GitHub Pages build useful. The pane is a
/// thin view over [BrowserModel], and the honesty card carries the explanation
/// of what the model is and is not, so the rest of the UI cannot quietly
/// overstate it.
class ChatPane extends StatefulWidget {
  /// Creates the pane.
  const ChatPane({super.key, this.client});

  /// The admin client supplying the base URL the chat endpoint is reached at.
  ///
  /// Null means the dashboard has no server configured, and the pane falls back
  /// to the in-browser model rather than failing every send.
  final AdminApiClient? client;

  @override
  State<ChatPane> createState() => _ChatPaneState();
}

class _ChatPaneState extends State<ChatPane> {
  _ChatPaneState();

  /// In-browser model, kept for the honesty card, the prompt preview and the
  /// "In-browser" source where it answers without a server at all.
  final BrowserModel _model = BrowserModel();
  final TextEditingController _composer = TextEditingController();
  final ScrollController _transcript = ScrollController();
  final List<ChatMessage> _messages = <ChatMessage>[];

  /// One client for every turn and memory call, created lazily and closed in
  /// [dispose]. A client per send would leak a TLS pool per message.
  late final http.Client _http = http.Client();

  StreamSubscription<String>? _sse;
  StreamSubscription<ChatEvent>? _turn;
  String _streaming = '';
  String? _runningTool;
  bool _loading = true;
  bool _busy = false;
  String? _error;

  /// Where the current turn goes: the server's `/api/v1/chat` (the model with
  /// memory and training) or this tab's own [BrowserModel]. Persisted, because
  /// an operator who runs the server should not re-pick it every reload.
  bool _useServer = _readStored('harbor.chatSource', 'server') != 'browser';
  bool _memoryOn = _readStored('harbor.chatMemory', 'true') == 'true';
  String _thinkingMode = _readStored('harbor.chatThinking', 'auto');

  /// One entry per assistant message: the reasoning trace that produced it.
  ///
  /// [ChatMessage] has no field for a trace and the wire format must not be
  /// bent to carry one, so traces live in a parallel list appended in lockstep
  /// with every assistant bubble. The transcript widget pairs them by index.
  final List<_MessageMeta> _metas = <_MessageMeta>[];

  /// The plan for the turn currently streaming, shown live above the answer.
  Map<String, Object?>? _livePlan;

  /// Memories learned this session, newest first.
  final List<Map<String, Object?>> _remembered = <Map<String, Object?>>[];

  static const List<String> _thinkingChoices = <String>[
    'auto',
    'none',
    'concise',
    'thorough',
  ];

  @override
  void initState() {
    super.initState();
    unawaited(_bootstrap());
  }

  @override
  void dispose() {
    // The stream first: dropping the subscription before the client closes
    // means the cancel's own error path can still deliver if it fails.
    unawaited(_sse?.cancel());
    unawaited(_turn?.cancel());
    _composer.dispose();
    _transcript.dispose();
    _model.dispose();
    _http.close();
    super.dispose();
  }

  /// Loads the model while keeping the progress bar repainting.
  ///
  /// [BrowserModel.load] reports progress only through [BrowserModel.status],
  /// and nothing calls `setState` between its three stages, so a ticker re-reads
  /// the status until loading finishes. Without it the bar would jump from 0 to
  /// 1 and no stage name would ever be visible.
  Future<void> _bootstrap() async {
    final Future<void> ticker = _tickProgress();
    try {
      await _model.load();
    } on Object catch (error) {
      if (!mounted) {
        await ticker;
        return;
      }
      setState(() {
        _loading = false;
        _error = 'Could not build the in-browser model: $error';
      });
      await ticker;
      return;
    }
    if (!mounted) {
      await ticker;
      return;
    }
    setState(() => _loading = false);
    await ticker;
  }

  /// Rebuilds the pane while [BrowserModel.load] is running.
  ///
  /// The short interval is below the per-stage yield inside the model, so at
  /// least one intermediate progress value reaches the screen.
  Future<void> _tickProgress() async {
    while (_loading && mounted) {
      await Future<void>.delayed(const Duration(milliseconds: 8));
      if (!mounted || !_loading) {
        return;
      }
      setState(() {});
    }
  }

  // ---- Turns ---------------------------------------------------------------

  /// Starts a turn over the current transcript and streams its events.
  ///
  /// Both sources speak in the same little [_TurnDelta] records so the state
  /// machine below has exactly one place where "a token arrived" is decided,
  /// whether the text came from an SSE frame or from a [ChatEvent].
  void _send() {
    final String text = _composer.text.trim();
    if (text.isEmpty || _busy) {
      return;
    }
    final ScaffoldMessengerState messenger = ScaffoldMessenger.of(context);
    setState(() {
      _messages.add(ChatMessage.user(text));
      _composer.clear();
      _streaming = '';
      _runningTool = null;
      _livePlan = null;
      _busy = true;
      _error = null;
    });
    _scrollToBottom();
    unawaited(
      _useServer
          ? _sendViaServer(messenger)
          : _sendViaBrowser(messenger),
    );
  }

  void _onDelta(_TurnDelta delta, ScaffoldMessengerState messenger) {
    if (!mounted) {
      return;
    }
    switch (delta.kind) {
      case _DeltaKind.token:
        setState(() => _streaming += delta.text);
      case _DeltaKind.toolStarted:
        setState(() => _runningTool = delta.name);
      case _DeltaKind.toolFinished:
        setState(() {
          // The one-line summary, not the full JSON: a `web.fetch` body can be
          // hundreds of kilobytes, and re-encoding it into the transcript would
          // push every later prompt past the context window.
          _messages.add(ChatMessage.tool(delta.name ?? 'tool', delta.text));
          _runningTool = null;
        });
      case _DeltaKind.thinking:
        setState(() {
          if (delta.plan != null) {
            _livePlan = delta.plan;
          }
          if (delta.trace != null) {
            _metas.add(_MessageMeta(trace: delta.trace));
            _livePlan = null;
          }
        });
      case _DeltaKind.memoryUpdated:
        setState(() {
          for (final Map<String, Object?> entry in delta.entries) {
            _remembered.insert(0, entry);
          }
          if (delta.entries.isNotEmpty) {
            _remembered.removeRange(
              12,
              _remembered.length > 12 ? _remembered.length : 12,
            );
          }
        });
        messenger
          ..hideCurrentSnackBar()
          ..showSnackBar(
            SnackBar(
              content: Text(
                'Remembered: ${_truncate(delta.entries.first['text'] ?? '')}',
              ),
            ),
          );
      case _DeltaKind.finished:
        setState(() {
          _messages.add(
            ChatMessage.assistant(
              delta.text.isEmpty ? '(the model produced no text)' : delta.text,
            ),
          );
          // A turn that emitted only a tool call carries no trace; keep the
          // lists aligned by recording an empty meta so indices stay honest.
          if (_metas.length < _assistantCount()) {
            _metas.add(const _MessageMeta());
          }
          _streaming = '';
          _runningTool = null;
          _livePlan = null;
          _busy = false;
        });
      case _DeltaKind.failed:
        setState(() {
          _streaming = '';
          _runningTool = null;
          _busy = false;
          if (delta.message != null) {
            _error = delta.message;
          }
        });
        if (delta.message != null) {
          messenger
            ..hideCurrentSnackBar()
            ..showSnackBar(SnackBar(content: Text(delta.message!)));
        }
    }
    _scrollToBottom();
  }

  int _assistantCount() => _messages
      .where((ChatMessage m) => m.role == ChatRole.assistant)
      .length;

  /// Runs the turn against `POST {base}/api/v1/chat?stream=true`.
  ///
  /// The body is consumed as a stream rather than with `client.post`, because
  /// buffering the whole response would deliver every token at the end of the
  /// turn and destroy the point of SSE: seeing the answer arrive.
  Future<void> _sendViaServer(ScaffoldMessengerState messenger) async {
    final Uri uri = _uri('/api/v1/chat', query: <String, String>{
      'stream': 'true',
    });
    try {
      final http.Request request = http.Request('POST', uri)
        ..headers['Content-Type'] = 'application/json'
        ..body = jsonEncode(<String, Object?>{
          'messages': _messages
              .map((ChatMessage m) => m.toJson())
              .toList(growable: false),
          'memory': _memoryOn,
          'thinking': _thinkingMode,
        });
      final http.StreamedResponse response = await _http.send(request);
      if (response.statusCode != 200) {
        _onDelta(
          _TurnDelta(
            kind: _DeltaKind.failed,
            message: 'Server returned ${response.statusCode}',
          ),
          messenger,
        );
        return;
      }
      final StreamController<_TurnDelta> deltas =
          StreamController<_TurnDelta>();
      _sse = response.stream
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .listen(
            (String line) => _handleSseLine(line, deltas),
            onError: (Object error) {
              deltas.add(_TurnDelta(
                kind: _DeltaKind.failed,
                message: 'Stream error: $error',
              ));
              unawaited(deltas.close());
            },
            onDone: () {
              if (!deltas.isClosed) {
                unawaited(deltas.close());
              }
            },
            cancelOnError: true,
          );
      await for (final _TurnDelta delta in deltas.stream) {
        _onDelta(delta, messenger);
        if (delta.kind == _DeltaKind.finished ||
            delta.kind == _DeltaKind.failed) {
          break;
        }
      }
    } on Object catch (error) {
      _onDelta(
        _TurnDelta(
          kind: _DeltaKind.failed,
          message: 'Server connection failed: $error',
        ),
        messenger,
      );
    }
  }

  /// Turns one SSE line into a delta, ignoring keep-alives and comments.
  void _handleSseLine(String line, StreamController<_TurnDelta> out) {
    if (!line.startsWith('data:')) {
      return;
    }
    final String payload = line.substring(5).trim();
    if (payload.isEmpty) {
      return;
    }
    if (payload == '[DONE]') {
      // A well-behaved server ends with a `finished` frame before [DONE]; if
      // the stream cut out early, close the busy state instead of spinning.
      out.add(const _TurnDelta(kind: _DeltaKind.finished, text: ''));
      return;
    }
    final Object? decoded;
    try {
      decoded = jsonDecode(payload);
    } on FormatException {
      return;
    }
    if (decoded is! Map<String, Object?>) {
      return;
    }
    final String type = decoded['type'] is String ? decoded['type']! as String : '';
    switch (type) {
      case 'token':
        out.add(_TurnDelta(
          kind: _DeltaKind.token,
          text: decoded['text'] is String ? decoded['text']! as String : '',
        ));
      case 'tool_started':
        out.add(_TurnDelta(
          kind: _DeltaKind.toolStarted,
          name: decoded['name'] is String ? decoded['name']! as String : 'tool',
        ));
      case 'tool_finished':
        out.add(_TurnDelta(
          kind: _DeltaKind.toolFinished,
          name: decoded['name'] is String ? decoded['name']! as String : 'tool',
          text: decoded['summary'] is String
              ? decoded['summary']! as String
              : '',
        ));
      case 'thinking':
        final Object? plan = decoded['plan'];
        final Object? trace = decoded['trace'];
        out.add(_TurnDelta(
          kind: _DeltaKind.thinking,
          plan: plan is Map<String, Object?> ? plan : null,
          trace: trace is Map<String, Object?> ? trace : null,
        ));
      case 'memory_updated':
        final Object? entries = decoded['entries'];
        out.add(_TurnDelta(
          kind: _DeltaKind.memoryUpdated,
          entries: <Map<String, Object?>>[
            if (entries is List<Object?>)
              for (final Object? e in entries)
                if (e is Map<String, Object?>) e,
          ],
        ));
      case 'finished':
        out.add(_TurnDelta(
          kind: _DeltaKind.finished,
          text: decoded['text'] is String ? decoded['text']! as String : '',
        ));
      case 'failed':
        out.add(_TurnDelta(
          kind: _DeltaKind.failed,
          message: decoded['message'] is String
              ? decoded['message']! as String
              : 'chat failed',
        ));
      default:
        break;
    }
  }

  /// Runs the turn against the tab's own [BrowserModel].
  Future<void> _sendViaBrowser(ScaffoldMessengerState messenger) async {
    _turn = _model.respond(List<ChatMessage>.of(_messages)).listen(
      (ChatEvent event) {
        switch (event) {
          case ChatToken(:final String text):
            _onDelta(
              _TurnDelta(kind: _DeltaKind.token, text: text),
              messenger,
            );
          case ChatToolStarted(:final ToolCall call):
            _onDelta(
              _TurnDelta(kind: _DeltaKind.toolStarted, name: call.name),
              messenger,
            );
          case ChatToolFinished(:final ToolCall call, :final ToolResult result):
            _onDelta(
              _TurnDelta(
                kind: _DeltaKind.toolFinished,
                name: call.name,
                text: result.summary,
              ),
              messenger,
            );
          case ChatThinking(:final ThinkingPlan plan, :final ThinkingTrace? trace):
            _onDelta(
              _TurnDelta(
                kind: _DeltaKind.thinking,
                plan: plan.toJson(),
                trace: trace?.toJson(),
              ),
              messenger,
            );
          case ChatMemoryUpdated(:final List<MemoryEntry> entries, :final bool explicit):
            _onDelta(
              _TurnDelta(
                kind: _DeltaKind.memoryUpdated,
                entries: <Map<String, Object?>>[
                  for (final MemoryEntry entry in entries) entry.toJson(),
                ],
                explicit: explicit,
              ),
              messenger,
            );
          case ChatFinished(:final String text):
            _onDelta(
              _TurnDelta(kind: _DeltaKind.finished, text: text),
              messenger,
            );
          case ChatFailed(:final String message):
            _onDelta(
              _TurnDelta(kind: _DeltaKind.failed, message: message),
              messenger,
            );
        }
      },
      onError: (Object error) {
        _onDelta(
          _TurnDelta(kind: _DeltaKind.failed, message: 'The turn failed: $error'),
          messenger,
        );
      },
      onDone: () {
        if (mounted && _busy) {
          setState(() {
            _streaming = '';
            _runningTool = null;
            _busy = false;
          });
        }
      },
    );
  }

  // ---- Turn controls -------------------------------------------------------

  /// Cancels the in-flight turn, keeping whatever text already arrived.
  void _stop() {
    unawaited(_sse?.cancel());
    _sse = null;
    unawaited(_turn?.cancel());
    _turn = null;
    if (!mounted) {
      return;
    }
    setState(() {
      _streaming = '';
      _runningTool = null;
      _livePlan = null;
      _busy = false;
    });
  }

  /// Empties the transcript without rebuilding the model.
  void _clear() {
    unawaited(_sse?.cancel());
    unawaited(_turn?.cancel());
    _sse = null;
    _turn = null;
    setState(() {
      _messages.clear();
      _metas.clear();
      _streaming = '';
      _runningTool = null;
      _livePlan = null;
      _busy = false;
      _error = null;
    });
  }

  /// Switches the answer source and remembers the choice.
  void _setSource({required bool server}) {
    setState(() {
      _useServer = server;
      _writeStored('harbor.chatSource', server ? 'server' : 'browser');
    });
  }

  void _setMemory(bool on) {
    setState(() {
      _memoryOn = on;
      _writeStored('harbor.chatMemory', on ? 'true' : 'false');
    });
  }

  void _setThinking(String mode) {
    setState(() {
      _thinkingMode = mode;
      _writeStored('harbor.chatThinking', mode);
    });
  }

  /// Scrolls the transcript to the newest message after the frame is laid out.
  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((Duration elapsed) {
      if (!_transcript.hasClients) {
        return;
      }
      unawaited(
        _transcript.animateTo(
          _transcript.position.maxScrollExtent,
          duration: const Duration(milliseconds: 180),
          curve: Curves.easeOut,
        ),
      );
    });
  }

  /// Shows the exact prompt the engine would send for the current transcript.
  void _showPrompt() {
    final String prompt = _model.renderPrompt(_messages);
    unawaited(
      showDialog<void>(
        context: context,
        builder: (BuildContext context) => AlertDialog(
          title: const Text('Rendered prompt'),
          content: SizedBox(
            width: 640,
            child: SingleChildScrollView(
              child: SelectableText(
                prompt,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
              ),
            ),
          ),
          actions: <Widget>[
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Close'),
            ),
          ],
        ),
      ),
    );
  }

  // ---- Layout ---------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    final Map<String, Object?> status = _model.status();
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          _toolbar(context, status),
          _capabilityRow(context),
          if (_loading) ...<Widget>[
            const SizedBox(height: 12),
            _progressRow(context, status),
          ],
          if (_error case final String message) ...<Widget>[
            const SizedBox(height: 12),
            _errorBanner(context, message),
          ],
          const SizedBox(height: 12),
          Expanded(child: _body()),
        ],
      ),
    );
  }

  /// The status strip: what the model is, plus the Prompt and Clear actions.
  Widget _toolbar(BuildContext context, Map<String, Object?> status) {
    final Object? vocabulary = status['vocabulary'];
    final Object? contextLength = status['context_length'];
    final Object? layers = status['layers'];
    return Wrap(
      spacing: 12,
      runSpacing: 8,
      crossAxisAlignment: WrapCrossAlignment.center,
      children: <Widget>[
        Text(
          _useServer ? 'Server chat' : 'In-browser chat',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        Text('${_formatCount(status['parameters'])} parameters'),
        Text('vocabulary $vocabulary'),
        Text('context $contextLength'),
        Text('$layers layers'),
        OutlinedButton.icon(
          onPressed: _showPrompt,
          icon: const Icon(Icons.code),
          label: const Text('Prompt'),
        ),
        OutlinedButton.icon(
          onPressed: _clear,
          icon: const Icon(Icons.delete_sweep_outlined),
          label: const Text('Clear'),
        ),
      ],
    );
  }

  /// The three switches that define a turn: which model answers, whether it
  /// remembers, and how hard it thinks. Kept beside the composer because an
  /// operator judging an answer needs to know what mode produced it.
  Widget _capabilityRow(BuildContext context) {
    return Wrap(
      key: const ValueKey<String>('chat.capabilities'),
      spacing: 16,
      runSpacing: 8,
      crossAxisAlignment: WrapCrossAlignment.center,
      children: <Widget>[
        Row(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            ChoiceChip(
              key: const ValueKey<String>('chat.source.server'),
              label: const Text('Server model'),
              selected: _useServer,
              onSelected: (_) => _setSource(server: true),
            ),
            const SizedBox(width: 8),
            ChoiceChip(
              key: const ValueKey<String>('chat.source.browser'),
              label: const Text('In-browser'),
              selected: !_useServer,
              onSelected: (_) => _setSource(server: false),
            ),
          ],
        ),
        Row(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            Switch(
              key: const ValueKey<String>('chat.memory.switch'),
              value: _memoryOn,
              onChanged: _setMemory,
            ),
            Text('Memory', style: Theme.of(context).textTheme.bodyMedium),
          ],
        ),
        Row(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            Text('Thinking', style: Theme.of(context).textTheme.bodyMedium),
            const SizedBox(width: 8),
            DropdownButton<String>(
              key: const ValueKey<String>('chat.thinking.dropdown'),
              value: _thinkingMode,
              items: <DropdownMenuItem<String>>[
                for (final String mode in _thinkingChoices)
                  DropdownMenuItem<String>(value: mode, child: Text(mode)),
              ],
              onChanged: (String? value) {
                if (value != null) {
                  _setThinking(value);
                }
              },
            ),
          ],
        ),
      ],
    );
  }

  /// The determinate progress bar shown while the model is being built.
  Widget _progressRow(BuildContext context, Map<String, Object?> status) {
    final Object? rawStage = status['stage'];
    final String stage = rawStage is String ? rawStage : 'loading';
    final Object? rawProgress = status['progress'];
    final double progress = rawProgress is double ? rawProgress : 0;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        const LinearProgressIndicator(),
        const SizedBox(height: 4),
        Text('$stage (${(progress * 100).round()}%)'),
      ],
    );
  }

  /// A compact inline error, used when a whole-pane failure has no snackbar.
  Widget _errorBanner(BuildContext context, String message) {
    final ColorScheme colors = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: colors.errorContainer,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(message, style: TextStyle(color: colors.onErrorContainer)),
    );
  }

  /// Formats a parameter count with separators, which six digits need.
  String _formatCount(Object? value) {
    if (value is int) {
      return NumberFormat.decimalPattern().format(value);
    }
    return '$value';
  }

  /// Chooses the two-column or single-column layout for the available width.
  Widget _body() {
    return LayoutBuilder(
      builder: (BuildContext context, BoxConstraints constraints) {
        if (constraints.maxWidth >= 860) {
          return Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              Expanded(flex: 3, child: _conversation()),
              const SizedBox(width: 16),
              Expanded(
                flex: 2,
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: <Widget>[
                      if (_remembered.isNotEmpty) _rememberedCard(context),
                      const _HonestyCard(),
                      _ToolsCard(tools: _model.tools, onRun: _model.runTool),
                    ],
                  ),
                ),
              ),
            ],
          );
        }
        return SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              SizedBox(height: 420, child: _conversation()),
              const SizedBox(height: 16),
              if (_remembered.isNotEmpty) _rememberedCard(context),
              const _HonestyCard(),
              _ToolsCard(tools: _model.tools, onRun: _model.runTool),
            ],
          ),
        );
      },
    );
  }

  /// The transcript and composer, shared by both responsive layouts.
  Widget _conversation() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        if (_livePlan != null)
          _ThinkingCard(
            key: const ValueKey<String>('chat.thinking.live'),
            plan: _livePlan!,
            running: true,
          ),
        Expanded(child: _transcriptCard()),
        const SizedBox(height: 8),
        _composerRow(),
      ],
    );
  }

  /// A bordered surface holding the transcript, so it reads as one chat log.
  Widget _transcriptCard() {
    final ColorScheme colors = Theme.of(context).colorScheme;
    return Container(
      decoration: BoxDecoration(
        color: colors.surfaceContainerLow,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: colors.outlineVariant),
      ),
      child: _ChatTranscript(
        messages: _messages,
        metas: _metas,
        streaming: _streaming,
        runningTool: _runningTool,
        controller: _transcript,
      ),
    );
  }

  /// What the assistant picked up during this session.
  Widget _rememberedCard(BuildContext context) {
    return _Panel(
      title: 'Picked up this session (${_remembered.length})',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          for (final Map<String, Object?> entry in _remembered)
            Padding(
              key: ValueKey<String>('chat.remembered.${entry['id']}'),
              padding: const EdgeInsets.symmetric(vertical: 2),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Chip(
                    visualDensity: VisualDensity.compact,
                    label: Text('${entry['kind'] ?? 'fact'}'),
                  ),
                  const SizedBox(width: 8),
                  Expanded(child: Text('${entry['text'] ?? ''}')),
                ],
              ),
            ),
        ],
      ),
    );
  }

  /// The multiline composer with its Send and Stop controls.
  Widget _composerRow() {
    final bool sendDisabled = _busy;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        TextField(
          controller: _composer,
          minLines: 1,
          maxLines: 4,
          textInputAction: TextInputAction.newline,
          onSubmitted: (String _) => _send(),
          decoration: InputDecoration(
            hintText: _useServer
                ? 'Ask the server model something...'
                : 'Ask the in-browser model something...',
            border: const OutlineInputBorder(),
            isDense: true,
          ),
        ),
        const SizedBox(height: 8),
        Row(
          children: <Widget>[
            FilledButton.icon(
              onPressed: sendDisabled ? null : _send,
              icon: const Icon(Icons.send),
              label: const Text('Send'),
            ),
            const SizedBox(width: 8),
            if (_busy)
              OutlinedButton.icon(
                onPressed: _stop,
                icon: const Icon(Icons.stop),
                label: const Text('Stop'),
              ),
          ],
        ),
      ],
    );
  }

  // ---- URLs ---------------------------------------------------------------

  Uri _uri(String path, {Map<String, String>? query}) {
    final String base = _baseUrl;
    if (base.isEmpty) {
      // Same-origin: the Pages-hosted dashboard calling the API it was built
      // beside. Uri.parse alone would return a relative URI that http rejects.
      final Uri page = Uri.base;
      return page.replace(path: path, query: query == null ? null : Uri(queryParameters: query).query);
    }
    final Uri parsed = Uri.parse(base.endsWith('/')
        ? '${base.substring(0, base.length - 1)}$path'
        : '$base$path');
    return query == null ? parsed : parsed.replace(queryParameters: query);
  }

  String get _baseUrl {
    final AdminApiClient? client = widget.client;
    return client?.baseUrl ?? '';
  }
}

/// One parsed step of a streamed turn, shared by both sources.
class _TurnDelta {
  const _TurnDelta({
    required this.kind,
    this.text = '',
    this.name,
    this.plan,
    this.trace,
    this.entries = const <Map<String, Object?>>[],
    this.explicit = false,
    this.message,
  });

  final _DeltaKind kind;
  final String text;
  final String? name;
  final Map<String, Object?>? plan;
  final Map<String, Object?>? trace;
  final List<Map<String, Object?>> entries;
  final bool explicit;
  final String? message;
}

enum _DeltaKind {
  token,
  toolStarted,
  toolFinished,
  thinking,
  memoryUpdated,
  finished,
  failed,
}

/// Side-channel state attached to an assistant message, by index.
class _MessageMeta {
  const _MessageMeta({this.trace});

  /// The reasoning trace that produced the paired answer.
  final Map<String, Object?>? trace;
}

String _truncate(Object? value, [int limit = 60]) {
  final String text = '$value';
  return text.length <= limit ? text : '${text.substring(0, limit)}…';
}

/// The collapsible reasoning card: what the assistant planned to do, and what
/// its check found. Expanded while a turn runs — a visible plan is the
/// difference between "thinking" and "frozen" — collapsed once settled.
class _ThinkingCard extends StatelessWidget {
  const _ThinkingCard({
    super.key,
    required this.plan,
    this.trace,
    this.running = false,
  });

  /// The plan map (ThinkingPlan.toJson) this card renders.
  final Map<String, Object?> plan;

  /// The finished trace (ThinkingTrace.toJson), or null while streaming.
  final Map<String, Object?>? trace;

  /// Whether the turn this describes is still in flight.
  final bool running;

  @override
  Widget build(BuildContext context) {
    final ColorScheme colors = Theme.of(context).colorScheme;
    final TextTheme text = Theme.of(context).textTheme;
    final String strategy = plan['strategy'] is String
        ? plan['strategy']! as String
        : 'thinking';
    final String summary = trace != null && trace!['summary'] is String
        ? trace!['summary']! as String
        : 'Planning the answer…';
    final Object? rawSteps = plan['steps'];
    final List<Object?> steps =
        rawSteps is List<Object?> ? rawSteps : const <Object?>[];
    final bool revised = trace?['revised'] == true;
    final Object? rawFindings = trace?['findings'];
    final List<Object?> findings =
        rawFindings is List<Object?> ? rawFindings : const <Object?>[];

    return Card(
      elevation: 0,
      color: colors.surfaceContainerHighest,
      child: ExpansionTile(
        shape: const Border(),
        initiallyExpanded: running,
        leading: Icon(
          running ? Icons.psychology_outlined : Icons.fact_check_outlined,
          size: 20,
          color: colors.primary,
        ),
        title: Text('Thinking', style: text.titleSmall),
        subtitle: Text(summary, style: text.bodySmall),
        trailing: Wrap(
          spacing: 6,
          children: <Widget>[
            Chip(
              visualDensity: VisualDensity.compact,
              label: Text(strategy),
            ),
            if (revised)
              const Chip(
                visualDensity: VisualDensity.compact,
                label: Text('revised'),
              ),
          ],
        ),
        childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
        children: <Widget>[
          for (final Object? rawStep in steps)
            if (rawStep is Map<String, Object?>)
              _ThinkingStep(row: rawStep)
            else
              Text('$rawStep', style: text.bodySmall),
          if (findings.isNotEmpty) ...<Widget>[
            const SizedBox(height: 4),
            Text(
              'Check found ${findings.length} issue(s):',
              style: text.bodySmall?.copyWith(fontWeight: FontWeight.w600),
            ),
            for (final Object? rawFinding in findings)
              if (rawFinding is Map<String, Object?>)
                Text(
                  '• ${rawFinding['message'] ?? rawFinding['code'] ?? ''}',
                  style: text.bodySmall,
                ),
          ],
        ],
      ),
    );
  }
}

class _ThinkingStep extends StatelessWidget {
  const _ThinkingStep({required this.row});

  final Map<String, Object?> row;

  @override
  Widget build(BuildContext context) {
    final TextTheme text = Theme.of(context).textTheme;
    final String status = row['status'] is String ? row['status']! as String : 'pending';
    final IconData icon = switch (status) {
      'done' => Icons.check_circle_outline,
      'running' => Icons.pending_outlined,
      'failed' => Icons.error_outline,
      'skipped' => Icons.remove_circle_outline,
      _ => Icons.radio_button_unchecked,
    };
    final String? result = row['result'] is String ? row['result']! as String : null;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Icon(icon, size: 16),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Text(
                  row['title'] is String ? row['title']! as String : '',
                  style: text.bodySmall?.copyWith(fontWeight: FontWeight.w600),
                ),
                if (row['detail'] is String)
                  Text(row['detail']! as String, style: text.bodySmall),
                if (result != null)
                  Text(result, style: text.bodySmall?.copyWith(fontStyle: FontStyle.italic)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// The scrolling transcript, with one bubble per role.
///
/// Tool results get their own monospace card rather than a chat bubble because
/// they are structured data, not something a person or the model said.
class _ChatTranscript extends StatelessWidget {
  const _ChatTranscript({
    required this.messages,
    required this.metas,
    required this.streaming,
    required this.runningTool,
    required this.controller,
  });

  /// Completed messages, oldest first.
  final List<ChatMessage> messages;

  /// Reasoning traces paired with assistant messages by index.
  final List<_MessageMeta> metas;

  /// Text generated so far in the running turn, or an empty string.
  final String streaming;

  /// Name of the tool currently running, or null when none is.
  final String? runningTool;

  /// Drives programmatic scrolling to the newest bubble.
  final ScrollController controller;

  @override
  Widget build(BuildContext context) {
    final String? tool = runningTool;
    return ListView(
      controller: controller,
      padding: const EdgeInsets.all(12),
      children: <Widget>[
        if (messages.isEmpty && streaming.isEmpty)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 32),
            child: Text(
              'No messages yet. The model answers from its seed corpus and '
              'from nothing else.',
              textAlign: TextAlign.center,
            ),
          ),
        for (int i = 0; i < messages.length; i++)
          _bubble(context, messages[i], _metaFor(messages[i], i)),
        if (streaming.isNotEmpty) _bubbleText(context, streaming, false),
        if (tool != null)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 6),
            child: Row(
              children: <Widget>[
                const SizedBox(
                  width: 14,
                  height: 14,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
                const SizedBox(width: 8),
                Text('Running $tool…'),
              ],
            ),
          ),
      ],
    );
  }

  /// The trace attached to an assistant message, if one was recorded.
  ///
  /// Traces are matched by walking both lists in send order rather than stored
  /// on the message: [ChatMessage] is a portable value type and a dashboard
  /// decoration must not leak into the wire format.
  _MessageMeta? _metaFor(ChatMessage message, int index) {
    if (message.role != ChatRole.assistant) {
      return null;
    }
    int seen = 0;
    for (int i = 0; i <= index && i < messages.length; i++) {
      if (messages[i].role == ChatRole.assistant) {
        seen++;
      }
    }
    final int metaIndex = seen - 1;
    if (metaIndex < 0 || metaIndex >= metas.length) {
      return null;
    }
    final _MessageMeta meta = metas[metaIndex];
    return meta.trace == null ? null : meta;
  }

  /// Renders one finished message according to its role.
  Widget _bubble(BuildContext context, ChatMessage message, _MessageMeta? meta) {
    switch (message.role) {
      case ChatRole.user:
        return _bubbleText(context, message.content, true);
      case ChatRole.assistant:
        final Map<String, Object?>? trace = meta?.trace;
        if (trace == null) {
          return _bubbleText(context, message.content, false);
        }
        // The answer and the reasoning that produced it stay visually bound:
        // the card sits directly above its bubble, collapsed, so the transcript
        // reads as a conversation with an expandable "how" on each turn.
        final Object? plan = trace['plan'];
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            if (plan is Map<String, Object?>)
              _ThinkingCard(plan: plan, trace: trace),
            _bubbleText(context, message.content, false),
          ],
        );
      case ChatRole.tool:
        return _toolCard(context, message);
      case ChatRole.system:
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 6),
          child: Text(
            message.content,
            style: Theme.of(context).textTheme.bodySmall,
            textAlign: TextAlign.center,
          ),
        );
    }
  }

  /// One aligned, coloured bubble; the user's are right-aligned.
  Widget _bubbleText(BuildContext context, String text, bool fromUser) {
    final ColorScheme colors = Theme.of(context).colorScheme;
    return Align(
      alignment: fromUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: const BoxConstraints(maxWidth: 560),
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: fromUser
              ? colors.primaryContainer
              : colors.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(12),
        ),
        child: SelectableText(
          text,
          style: TextStyle(
            color: fromUser ? colors.onPrimaryContainer : colors.onSurface,
          ),
        ),
      ),
    );
  }

  /// A monospace card for a tool result, which is data rather than prose.
  Widget _toolCard(BuildContext context, ChatMessage message) {
    final ColorScheme colors = Theme.of(context).colorScheme;
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        width: double.infinity,
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: colors.surface,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: colors.outlineVariant),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              'tool · ${message.name ?? 'unknown'}',
              style: Theme.of(context).textTheme.labelSmall,
            ),
            const SizedBox(height: 4),
            SelectableText(
              message.content,
              style: const TextStyle(fontFamily: 'monospace', fontSize: 11),
            ),
          ],
        ),
      ),
    );
  }
}

/// The required honesty card.
///
/// A from-scratch model with no pretrained checkpoint can produce fluent text
/// that sounds authoritative, so the pane states plainly what it is, what it was
/// trained on, and which capabilities live in the Android app instead. The
/// corpus button exists so the claim is checkable rather than merely asserted.
class _HonestyCard extends StatelessWidget {
  const _HonestyCard();

  @override
  Widget build(BuildContext context) {
    return _Panel(
      title: 'How this model works',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          const Text(
            'This model has no pretrained checkpoint. It is trained from '
            'scratch, in this tab, on only the bundled seed corpus — a few '
            'thousand characters of English about Harbor. It has never read '
            'anything else, so it will produce fluent-looking sentences whose '
            'facts are often unreliable.',
          ),
          const SizedBox(height: 8),
          const Text(
            'Device tools and on-device learning do not run here. Battery, '
            'clipboard, notifications and LoRA training run in the Android app '
            'through its platform channel. This browser tab can only fetch '
            'public web pages over HTTPS.',
          ),
          const SizedBox(height: 12),
          Align(
            alignment: Alignment.centerLeft,
            child: OutlinedButton.icon(
              onPressed: () => _showCorpusDialog(context),
              icon: const Icon(Icons.article_outlined),
              label: const Text('Show the seed corpus'),
            ),
          ),
        ],
      ),
    );
  }
}

/// The catalogue of tools this browser build can actually run.
///
/// The list comes from [BrowserModel.tools], so the pane and the chat engine can
/// never disagree about what exists. Forms are generated from each tool's JSON
/// schema rather than hard-coded, which keeps a new tool usable without a
/// matching UI change.
class _ToolsCard extends StatelessWidget {
  const _ToolsCard({required this.tools, required this.onRun});

  /// The tools to list.
  final List<Tool> tools;

  /// Invokes a tool by name with the collected arguments.
  final Future<ToolResult> Function(String name, Map<String, Object?> arguments)
      onRun;

  @override
  Widget build(BuildContext context) {
    return _Panel(
      title: 'Tools (${tools.length})',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          if (tools.isEmpty)
            const Text('No tools are available in this build.')
          else
            ...<Widget>[
              for (final Tool tool in tools)
                _ToolTile(tool: tool, onRun: onRun),
            ],
        ],
      ),
    );
  }
}

/// One tool with a schema-generated argument form and its latest result.
///
/// Each tile owns its own inputs and result, so expanding and running one tool
/// never disturbs another's form.
class _ToolTile extends StatefulWidget {
  const _ToolTile({required this.tool, required this.onRun});

  /// The tool being exposed.
  final Tool tool;

  /// Invokes the tool with the collected arguments.
  final Future<ToolResult> Function(String name, Map<String, Object?> arguments)
      onRun;

  @override
  State<_ToolTile> createState() => _ToolTileState();
}

class _ToolTileState extends State<_ToolTile> {
  late final List<_ToolArgument> _arguments;

  bool _running = false;
  bool _ok = false;
  String? _result;

  @override
  void initState() {
    super.initState();
    _arguments = _parseArguments(widget.tool.parameters);
  }

  @override
  void dispose() {
    for (final _ToolArgument argument in _arguments) {
      argument.dispose();
    }
    super.dispose();
  }

  /// Runs the tool and stores its JSON result.
  ///
  /// [BrowserModel.runTool] converts every failure into a [ToolResult] rather
  /// than throwing, so the only await here is the tool itself; the widget is
  /// still re-checked for `mounted` before it rebuilds.
  Future<void> _run() async {
    setState(() {
      _running = true;
      _ok = false;
      _result = null;
    });
    final ToolResult result = await widget.onRun(
      widget.tool.name,
      _collect(),
    );
    if (!mounted) {
      return;
    }
    setState(() {
      _running = false;
      _ok = result.ok;
      _result = const JsonEncoder.withIndent('  ').convert(result.toJson());
    });
  }

  /// Collects the non-empty argument values from the generated form.
  Map<String, Object?> _collect() {
    final Map<String, Object?> arguments = <String, Object?>{};
    for (final _ToolArgument argument in _arguments) {
      final Object? value = argument.value();
      if (value != null) {
        arguments[argument.name] = value;
      }
    }
    return arguments;
  }

  @override
  Widget build(BuildContext context) {
    final String? result = _result;
    return ExpansionTile(
      tilePadding: EdgeInsets.zero,
      title: Text(
        widget.tool.name,
        style: const TextStyle(fontFamily: 'monospace'),
      ),
      subtitle: Text(widget.tool.description),
      childrenPadding: const EdgeInsets.only(bottom: 12),
      children: <Widget>[
        for (final _ToolArgument argument in _arguments) _field(argument),
        const SizedBox(height: 8),
        Row(
          children: <Widget>[
            FilledButton.icon(
              onPressed: _running ? null : _run,
              icon: const Icon(Icons.play_arrow),
              label: const Text('Run'),
            ),
            const SizedBox(width: 12),
            if (_running)
              const SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
          ],
        ),
        if (result != null) ...<Widget>[
          const SizedBox(height: 8),
          Container(
            constraints: const BoxConstraints(maxHeight: 220),
            width: double.infinity,
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: _ok
                  ? Theme.of(context).colorScheme.surfaceContainerHighest
                  : Theme.of(context).colorScheme.errorContainer,
              borderRadius: BorderRadius.circular(8),
            ),
            child: SingleChildScrollView(
              child: SelectableText(
                result,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 11),
              ),
            ),
          ),
        ],
      ],
    );
  }

  /// Builds the right control for one schema-declared argument.
  Widget _field(_ToolArgument argument) {
    final String label =
        argument.required ? '${argument.name} (required)' : argument.name;
    if (argument.choices.isNotEmpty) {
      return Padding(
        padding: const EdgeInsets.only(bottom: 8),
        child: DropdownButtonFormField<String>(
          value: argument.selected,
          decoration: InputDecoration(
            labelText: label,
            isDense: true,
            border: const OutlineInputBorder(),
          ),
          items: <DropdownMenuItem<String>>[
            for (final String choice in argument.choices)
              DropdownMenuItem<String>(value: choice, child: Text(choice)),
          ],
          onChanged: (String? value) =>
              setState(() => argument.selected = value),
        ),
      );
    }
    if (argument.type == 'boolean') {
      return SwitchListTile(
        contentPadding: EdgeInsets.zero,
        title: Text(label),
        value: argument.boolean,
        onChanged: (bool value) => setState(() => argument.boolean = value),
      );
    }
    final bool numeric =
        argument.type == 'integer' || argument.type == 'number';
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: TextField(
        controller: argument.controller,
        keyboardType: numeric
            ? TextInputType.numberWithOptions(decimal: argument.type == 'number')
            : TextInputType.text,
        decoration: InputDecoration(
          labelText: label,
          hintText: argument.type,
          isDense: true,
          border: const OutlineInputBorder(),
        ),
      ),
    );
  }

  /// Parses a tool's JSON-schema `parameters` into editable arguments.
  ///
  /// Only the types the tool layer actually emits are interpreted; anything
  /// unrecognised falls back to a text field, so a schema written against a
  /// newer draft still renders instead of disappearing.
  static List<_ToolArgument> _parseArguments(Map<String, Object?> parameters) {
    final Object? rawProperties = parameters['properties'];
    if (rawProperties is! Map<Object?, Object?>) {
      return <_ToolArgument>[];
    }
    final Set<String> required = <String>{};
    final Object? rawRequired = parameters['required'];
    if (rawRequired is List<Object?>) {
      for (final Object? key in rawRequired) {
        if (key is String) {
          required.add(key);
        }
      }
    }
    final List<_ToolArgument> arguments = <_ToolArgument>[];
    for (final MapEntry<Object?, Object?> entry in rawProperties.entries) {
      final Object? key = entry.key;
      if (key is! String) {
        continue;
      }
      final Object? rawSpec = entry.value;
      final Map<String, Object?> spec = rawSpec is Map<Object?, Object?>
          ? _stringKeyed(rawSpec)
          : const <String, Object?>{};
      final Object? rawType = spec['type'];
      final List<String> choices = <String>[];
      final Object? rawEnum = spec['enum'];
      if (rawEnum is List<Object?>) {
        for (final Object? choice in rawEnum) {
          if (choice is String) {
            choices.add(choice);
          }
        }
      }
      arguments.add(
        _ToolArgument(
          name: key,
          type: rawType is String ? rawType : 'string',
          choices: choices,
          required: required.contains(key),
        ),
      );
    }
    return arguments;
  }

  /// Copies a dynamically decoded JSON object into a string-keyed map.
  static Map<String, Object?> _stringKeyed(Map<Object?, Object?> source) {
    final Map<String, Object?> out = <String, Object?>{};
    for (final MapEntry<Object?, Object?> entry in source.entries) {
      final Object? key = entry.key;
      if (key is String) {
        out[key] = entry.value;
      }
    }
    return out;
  }
}

/// One argument parsed from a tool schema, plus its edit state.
///
/// Holding the controller and the typed value together keeps the generated form
/// and the collected arguments in one place, so a field cannot be rendered
/// without also being readable when the tool is submitted.
class _ToolArgument {
  _ToolArgument({
    required this.name,
    required this.type,
    required this.choices,
    required this.required,
  });

  /// Argument name as it appears in the schema and the tool call.
  final String name;

  /// JSON-schema type: `string`, `integer`, `number` or `boolean`.
  final String type;

  /// Allowed values when the schema declares an enum; empty otherwise.
  final List<String> choices;

  /// Whether the schema lists this argument as required.
  final bool required;

  /// Backing text for string and numeric arguments.
  final TextEditingController controller = TextEditingController();

  /// Backing value for boolean arguments.
  bool boolean = false;

  /// Backing value for enum arguments.
  String? selected;

  /// The value to send, or null when an optional field was left blank.
  ///
  /// Returning null rather than an empty string matters because the tool layer
  /// treats a null argument as absent and an empty string as a real value.
  Object? value() {
    if (choices.isNotEmpty) {
      return selected;
    }
    if (type == 'boolean') {
      return boolean;
    }
    final String text = controller.text.trim();
    if (text.isEmpty) {
      return null;
    }
    switch (type) {
      case 'integer':
        return int.tryParse(text);
      case 'number':
        return double.tryParse(text);
      default:
        return controller.text;
    }
  }

  /// Releases the text controller.
  void dispose() => controller.dispose();
}

/// Shows the complete seed corpus, because the honesty claim is only checkable
/// if the user can read every word the model was trained on.
void _showCorpusDialog(BuildContext context) {
  unawaited(
    showDialog<void>(
      context: context,
      builder: (BuildContext context) => AlertDialog(
        title: const Text(
          'Bundled seed corpus (${kBrowserSeedCorpus.length} characters)',
        ),
        content: const SizedBox(
          width: 720,
          child: SingleChildScrollView(
            child: SelectableText(
              kBrowserSeedCorpus,
              style: TextStyle(fontFamily: 'monospace', fontSize: 12),
            ),
          ),
        ),
        actions: <Widget>[
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Close'),
          ),
        ],
      ),
    ),
  );
}