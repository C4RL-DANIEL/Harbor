// Harbor update server — Flutter Web admin dashboard.
//
// This is the only Flutter-dependent file in the package. The server and all
// of lib/src/** run on the pure Dart VM.
//
// Build:  flutter build web --release --dart-define=HARBOR_API_BASE_URL=<url>

import 'dart:convert';
import 'dart:html' as html;

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import 'src/admin_api_client.dart';
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
  ];

  static const List<IconData> _icons = <IconData>[
    Icons.dashboard_outlined,
    Icons.rocket_launch_outlined,
    Icons.system_update_alt_outlined,
    Icons.toggle_on_outlined,
    Icons.preview_outlined,
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
    return PreviewPane(key: key, client: client);
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
    final String? value = html.window.localStorage[key];
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
    html.window.localStorage[key] = value;
  } catch (_) {
    // Storage may be unavailable in private browsing; settings stay in RAM.
  }
}