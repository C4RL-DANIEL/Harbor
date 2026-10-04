// Harbor — application entry point and composition root.
//
// Startup sequence (deliberately non-blocking):
//
//   1. Build the on-device engines (MLA attention, fine-grained MoE, LoRA
//      trainer + idle scheduler) and the agent stack.
//   2. Restore the cached feature-flag matrix so the very first frame already
//      renders the correct dynamic layout, with no flash of the wrong UI.
//   3. Run a *silent* update check. A network failure is recorded and ignored;
//      the app never blocks on the update server.
//   4. Present the forced-update modal when the server requires it, otherwise a
//      dismissible banner for optional updates.
//
// Configuration arrives exclusively through `--dart-define`, so no secret or
// environment-specific URL is ever committed to the repository.

import 'dart:async';
import 'dart:io' show Platform;

import 'package:flutter/material.dart';
import 'package:harbor_core/harbor_core.dart' as hc;
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

import 'core/agent/extended_thinking.dart';
import 'core/agent/file_readers.dart';
import 'core/agent/subagent_runner.dart';
import 'core/auto/auto_pilot.dart';
import 'core/chat/assistant_settings.dart';
import 'core/chat/chat_controller.dart';
import 'core/chat/chat_engine_factory.dart';
import 'core/corpus/corpus_controller.dart';
import 'core/corpus/corpus_persistence.dart';
import 'core/corpus/device_collector.dart';
import 'core/engine/fine_grained_moe.dart';
import 'core/engine/mla_attention.dart';
import 'core/engine/on_device_lora.dart';
import 'core/memory/harbor_memory_storage.dart';
import 'core/model/harbor_model_runtime.dart';
import 'core/platform/platform_bridge.dart';
import 'core/platform/platform_device_state.dart';
import 'core/tools/tool_catalog.dart';
import 'core/tools/tools_controller.dart';
import 'core/training/training_controller.dart';
import 'core/update_engine/dynamic_feature_flag_provider.dart';
import 'core/update_engine/update_service.dart';
import 'core/update_engine/widgets/update_dialog.dart';
import 'features/dynamic_module_registry.dart';
import 'features/screens/chat_screen.dart';
import 'features/screens/corpus_screen.dart';
import 'features/screens/tools_screen.dart';
import 'features/screens/memory_screen.dart';
import 'features/screens/training_screen.dart';
import 'features/widgets/one_tap_setup_card.dart';

/// Material 3 seed colour for the Harbor brand.
const Color kHarborSeedColor = Color(0xFF2F6FED);

/// Default feature-flag matrix used before any network sync succeeds.
///
/// It mirrors `apps/update_server/flags/default_flags.json`, which the Pages
/// builder publishes as `/api/v1/flags.json`, so the app boots with a sensible
/// layout even when no update server is hosted. Every `type` below must have a
/// renderer in [DynamicModuleRegistry.defaultBuilders]; the client test suite
/// asserts that, because an unregistered type renders a placeholder card
/// instead of a working module.
const FeatureFlagMatrix kDefaultFlagMatrix = FeatureFlagMatrix(
  version: 1,
  updatedAt: null,
  flags: <String, Object?>{
    'dynamic_ui': true,
    'agent.thinking': true,
    'agent.subagents': true,
    'labs.voice_mode': false,
    'labs.lora_training': false,
  },
  remoteDefaults: <String, Object?>{
    'dynamic_ui': true,
    'agent.thinking': true,
    'agent.subagents': true,
    'labs.voice_mode': false,
    'labs.lora_training': false,
  },
  layout: DynamicLayout(
    sections: <DynamicSection>[
      DynamicSection(
        id: 'home',
        title: 'Home',
        order: 0,
        modules: <DynamicModule>[
          DynamicModule(
            id: 'engine_status',
            type: 'engine_status',
            flag: '',
            order: 0,
          ),
          DynamicModule(
            id: 'thinking_panel',
            type: 'thinking_panel',
            flag: 'agent.thinking',
            order: 1,
            props: <String, Object?>{'collapsedByDefault': true},
          ),
          DynamicModule(
            id: 'agent_console',
            type: 'agent_console',
            flag: 'agent.subagents',
            order: 2,
          ),
          DynamicModule(
            id: 'file_inspector',
            type: 'file_inspector',
            flag: '',
            order: 3,
          ),
          DynamicModule(
            id: 'update_status',
            type: 'update_status',
            flag: '',
            order: 4,
          ),
          DynamicModule(
            id: 'feature_flags',
            type: 'feature_flags',
            flag: 'dynamic_ui',
            order: 5,
          ),
        ],
      ),
      DynamicSection(
        id: 'labs',
        title: 'Labs',
        order: 1,
        modules: <DynamicModule>[
          DynamicModule(
            id: 'labs_announcement',
            type: 'banner',
            flag: 'labs.voice_mode',
            order: 0,
            props: <String, Object?>{
              'message': 'Voice mode is enabled for this account.',
              'severity': 'info',
            },
          ),
        ],
      ),
    ],
  ),
);

/// Compile-time configuration, supplied with `--dart-define`.
@immutable
class HarborConfig {
  const HarborConfig({
    required this.apiBaseUrl,
    required this.appVersion,
    required this.platform,
    required this.verboseLogging,
    required this.autoCheckUpdates,
  });

  /// Base URL of the update server.
  final String apiBaseUrl;

  /// Version of the running build, compared against the server's semver.
  final String appVersion;

  /// Platform identifier sent to the update server.
  final String platform;

  /// Whether engine diagnostics are logged to stdout.
  final bool verboseLogging;

  /// Whether the launch-time update check runs at all.
  final bool autoCheckUpdates;

  /// Reads configuration from the compile-time environment.
  ///
  /// The default API base URL targets the Android emulator's host loopback,
  /// which is the correct default for local development.
  factory HarborConfig.fromEnvironment() {
    const String apiBaseUrl = String.fromEnvironment(
      'HARBOR_API_BASE_URL',
      defaultValue: 'http://10.0.2.2:8080',
    );
    const String appVersion = String.fromEnvironment(
      'HARBOR_APP_VERSION',
      defaultValue: '1.0.0',
    );
    const String platform = String.fromEnvironment(
      'HARBOR_PLATFORM',
      defaultValue: 'android',
    );
    const bool verboseLogging = bool.fromEnvironment(
      'HARBOR_VERBOSE_LOGGING',
      defaultValue: false,
    );
    const bool autoCheckUpdates = bool.fromEnvironment(
      'HARBOR_AUTO_CHECK_UPDATES',
      defaultValue: true,
    );
    return const HarborConfig(
      apiBaseUrl: apiBaseUrl,
      appVersion: appVersion,
      platform: platform,
      verboseLogging: verboseLogging,
      autoCheckUpdates: autoCheckUpdates,
    );
  }

  Map<String, Object?> toJson() => <String, Object?>{
        'api_base_url': apiBaseUrl,
        'app_version': appVersion,
        'platform': platform,
        'verbose_logging': verboseLogging,
        'auto_check_updates': autoCheckUpdates,
      };
}

/// Manually-driven [DeviceStateSource], used for development and testing.
///
/// It is the fallback, not the default: on Android the composition root builds a
/// [PlatformDeviceStateSource] and this one only drives the engine tab's
/// simulator, so the training gate can be exercised without draining a battery.
class ManualDeviceStateSource implements DeviceStateSource {
  DeviceState _state = DeviceState.unknown;

  /// Current simulated state.
  DeviceState get state => _state;

  /// Replaces the simulated state.
  void update(DeviceState state) => _state = state;

  @override
  Future<DeviceState> read() async => _state;
}

/// Every long-lived service the UI needs, built once and shared.
class HarborServices {
  HarborServices({
    required this.config,
    required this.attention,
    required this.moe,
    required this.adapter,
    required this.trainer,
    required this.scheduler,
    required this.adapterStore,
    required this.deviceStateSource,
    required this.reader,
    required this.runner,
    required this.agent,
    required this.flags,
    required this.updateService,
    required this.flow,
    required this.registry,
    required this.bridge,
    required this.httpClient,
    required this.storage,
    required this.modelRuntime,
    required this.corpus,
    required this.tools,
    required this.chat,
    required this.memory,
    required this.assistantSettings,
    required this.autoPilot,
    required this.training,
    required this.platformDeviceState,
  });

  final HarborConfig config;
  final MultiHeadLatentAttention attention;
  final FineGrainedMoE moe;
  final LoraAdapter adapter;
  final OnDeviceLoRATrainer trainer;
  final IdleTrainingScheduler scheduler;
  final LoraAdapterStore adapterStore;
  final ManualDeviceStateSource deviceStateSource;
  final UniversalFileReader reader;
  final SubagentRunner runner;
  final ExtendedThinkingAgent agent;
  final FeatureFlagProvider flags;
  final UpdateService updateService;
  final UpdateFlowController flow;
  final DynamicModuleRegistry registry;
  final PlatformBridge bridge;

  /// The HTTP client shared by the web tools and the corpus collector. One
  /// client for both keeps the connection pool warm, which matters on a phone
  /// where a fresh TLS handshake per fetch is a visible delay.
  final http.Client httpClient;

  /// Where the corpus, tokenizer and checkpoint live.
  final HarborStorageLayout storage;

  /// The on-device model and its tokenizer.
  final HarborModelRuntime modelRuntime;

  /// Gathered training text.
  final CorpusController corpus;

  /// Runnable capabilities.
  final ToolsController tools;

  /// The assistant.
  final ChatController chat;

  /// What the assistant remembers between launches.
  ///
  /// The engine reads and writes this store during a turn and the controller
  /// holds the same reference for the UI, so both are exposed here rather than
  /// reconstructed — two stores over one file would each overwrite the other's
  /// write-behind save.
  final hc.MemoryStore memory;

  /// The user's memory / learning / reasoning switches.
  ///
  /// Persisted, so a person who turns thinking off is still not thinking on the
  /// next launch. Changing a value rebuilds the chat engine through its
  /// [AssistantSettings.onChange] hook wired in [_buildServices].
  final AssistantSettings assistantSettings;

  /// The one-tap setup and background self-maintenance loop.
  final AutoPilotController autoPilot;

  /// The training pipeline.
  final TrainingController training;

  /// Real battery and thermal readings, or null on a platform that cannot
  /// provide them. The manual source stays available for the engine tab's
  /// simulator either way.
  final DeviceStateSource? platformDeviceState;

  /// Result of the launch-time update check, once it completes.
  UpdateCheckResult? lastUpdateCheck;

  /// The most recent completed training round, if any.
  TrainingRoundReport? lastTrainingRound;

  /// Frees every resource owned by the composition root.
  Future<void> dispose() async {
    await scheduler.dispose();
    flow.dispose();
    flags.dispose();
    updateService.dispose();
    httpClient.close();
    chat.dispose();
    autoPilot.dispose();
    training.dispose();
    corpus.dispose();
    tools.dispose();
    modelRuntime.dispose();
  }
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final HarborConfig config = HarborConfig.fromEnvironment();
  final HarborServices services = await _buildServices(config);
  runApp(HarborApp(services: services));
}

/// Constructs the whole object graph.
Future<HarborServices> _buildServices(HarborConfig config) async {
  // ---- On-device model engines -------------------------------------------
  // A compact but structurally faithful configuration: the KV-compression ratio
  // is a property of the geometry, not of the absolute sizes.
  const MlaConfig mlaConfig = MlaConfig(
    hiddenSize: 128,
    numHeads: 8,
    headDim: 32,
    kvLoraRank: 32,
    ropeHeadDim: 8,
    qLoraRank: 64,
    maxSeqLen: 512,
  );
  final MultiHeadLatentAttention attention =
      MultiHeadLatentAttention(mlaConfig, seed: 0x4D4C41);

  const MoeConfig moeConfig = MoeConfig(
    hiddenSize: 128,
    intermediateSize: 256,
    numRoutedExperts: 8,
    topK: 2,
    numSharedExperts: 1,
    capacityFactor: 1.25,
  );
  final FineGrainedMoE moe = FineGrainedMoE(moeConfig, seed: 0x00E1);

  // ---- Legacy engine (diagnostics only) ----------------------------------
  // These objects are what the Engine tab plots, so the KV-compression ratio
  // and the expert routing stay inspectable. They do not produce the model's
  // answers any more. In particular the previous "self-distillation" seeding —
  // which trained the adapter to imitate a randomly initialised attention
  // module — has been removed rather than relabelled: fitting noise is worse
  // than not fitting anything, and the real pipeline learns from text.
  final LoraConfig loraConfig = LoraConfig(
    rank: 8,
    inFeatures: mlaConfig.hiddenSize,
    outFeatures: mlaConfig.hiddenSize,
    alpha: 16,
    targetModule: 'mla_output_projection',
  );

  Directory supportDirectory;
  try {
    supportDirectory = await getApplicationSupportDirectory();
  } on Object {
    // Without a support directory the app still runs; adapters and flags simply
    // do not persist across launches.
    supportDirectory = Directory.systemTemp;
  }

  final LoraAdapterStore adapterStore =
      FileLoraAdapterStore(Directory('${supportDirectory.path}/lora'));
  LoraAdapter adapter =
      await adapterStore.load('harbor_default') ??
          LoraAdapter.initialized(loraConfig, seed: 0xA11CE);

  final OnDeviceLoRATrainer trainer = OnDeviceLoRATrainer(
    adapter: adapter,
    learningRate: 0.004,
    weightDecay: 0.01,
    maxGradientNorm: 1.0,
  );

  final ManualDeviceStateSource deviceStateSource = ManualDeviceStateSource();
  final IdleTrainingScheduler scheduler = IdleTrainingScheduler(
    trainer: trainer,
    guards: const TrainingGuards(
      minBatteryLevel: 0.35,
      maxThermalState: ThermalState.fair,
      dailyComputeBudget: Duration(minutes: 20),
    ),
    batchSize: 8,
    stepsPerRound: 12,
  );

  // ---- Agent stack -------------------------------------------------------
  final UniversalFileReader reader = UniversalFileReader();
  final SubagentRunner runner = SubagentRunner(
    maxConcurrency: recommendedConcurrency(),
    reader: reader,
  );
  final ExtendedThinkingAgent agent = ExtendedThinkingAgent(
    runner: runner,
    backend: HeuristicPlanningBackend(),
  );

  // ---- Update engine -----------------------------------------------------
  final FeatureFlagProvider flags = FeatureFlagProvider(
    baseUrl: config.apiBaseUrl,
    seed: kDefaultFlagMatrix,
  );
  final UpdateService updateService = UpdateService(
    baseUrl: config.apiBaseUrl,
    currentVersion: config.appVersion,
    platform: config.platform,
  );
  final UpdateFlowController flow = UpdateFlowController(service: updateService);

  // ---- Corpus, model, tools and learning ---------------------------------
  // The order below is a dependency chain, not a preference: the storage layout
  // locates the corpus, the corpus trains the tokenizer, the tokenizer sizes the
  // model, and the model is what chat and training operate on.
  final PlatformBridge bridge = PlatformBridge();
  final HarborStorageLayout storage =
      HarborStorageLayout(Directory('${supportDirectory.path}/harbor'));

  final hc.CorpusStore corpusStore = hc.CorpusStore(
    storage: FileCorpusStorage(storage.corpusFile),
  );
  await corpusStore.load();

  final http.Client httpClient = http.Client();
  final CorpusController corpus = CorpusController(
    store: corpusStore,
    collector: DeviceCorpusCollector(store: corpusStore),
    webCollector: hc.WebCorpusCollector(
      client: httpClient,
      // An empty host allow-list means "any host"; the policy still enforces
      // https, a byte ceiling, a redirect ceiling and robots.txt.
      policy: const hc.WebCorpusPolicy(),
    ),
    deviceRoots: _corpusRoots(supportDirectory),
  );
  await corpus.initialize();

  final HarborModelRuntime modelRuntime =
      HarborModelRuntime(layout: storage, lowResource: platformIsLowResource());
  await modelRuntime.bootstrap(corpusStore.trainingText(maxChars: 24000));

  final hc.ToolRegistry toolRegistry = buildToolRegistry(
    bridge: bridge,
    client: httpClient,
  );

  final ToolsController tools = ToolsController(
    registry: toolRegistry,
    bridge: bridge,
  );

  // A vocabulary mismatch between the model and the tokenizer would produce
  // token ids the embedding table does not have, so the fallback pair is built
  // to match. It exists only so chat has *something* to run when bootstrapping
  // failed; the home screen says so rather than letting the app look broken.
  final hc.ByteTokenizer fallbackTokenizer = hc.ByteTokenizer();
  final hc.LanguageModelRuntime chatModel = modelRuntime.model ??
      hc.TinyLm(
        hc.TinyLmConfig.testPreset
            .copyWith(vocabSize: fallbackTokenizer.vocabSize),
      );
  final hc.Tokenizer chatTokenizer = modelRuntime.tokenizer ?? fallbackTokenizer;

  // ---- Memory, reasoning and the switches that control them --------------
  // Memory lives beside the corpus and the checkpoint because it is the same
  // kind of artefact: learned locally, worth keeping across launches, safe to
  // delete. The defaults are ON — an assistant that only remembers after you
  // find a settings page makes the user do the assistant's job.
  final hc.MemoryStore memory = hc.MemoryStore(
    storage: FileMemoryStorage(storage.memoryFile),
  );
  await memory.load();

  final AssistantSettings assistantSettings =
      AssistantSettings(file: File('${storage.root.path}/assistant.json'));
  await assistantSettings.load();

  final ChatController chat = ChatController(
    engine: buildChatEngine(
      settings: assistantSettings,
      model: chatModel,
      tokenizer: chatTokenizer,
      tools: toolRegistry,
      memory: memory,
      contextLength: modelRuntime.ready
          ? modelRuntime.config.contextLength
          : hc.TinyLmConfig.testPreset.contextLength,
    ),
    tokenizer: chatTokenizer,
    memory: memory,
    transcript: ChatTranscriptStore(storage.transcriptFile),
  );

  // Toggling a switch rebuilds the engine rather than mutating one, because the
  // engine's prompt shape is immutable by design: half a turn must not change
  // what "memory on" means halfway through generating it.
  assistantSettings.onChange = () {
    chat.replaceEngine(
      buildChatEngine(
        settings: assistantSettings,
        model: chatModel,
        tokenizer: chatTokenizer,
        tools: toolRegistry,
        memory: memory,
        contextLength: modelRuntime.ready
            ? modelRuntime.config.contextLength
            : hc.TinyLmConfig.testPreset.contextLength,
      ),
    );
  };
  await chat.restore();

  final TrainingController training = TrainingController(
    runtime: modelRuntime,
    corpus: corpus,
    bridge: bridge,
    log: TrainingLog(storage.trainingLogFile),
  );

  final DeviceStateSource? platformDeviceState =
      bridge.isSupported ? PlatformDeviceStateSource(bridge) : null;

  // Auto-Pilot: the one-tap path that runs gather → build → train, and the
  // background poll that keeps learning while the phone charges. It gates on the
  // real device state, so a user who taps "Get started" gets a working assistant
  // without having to learn the pipeline's order first.
  final AutoPilotController autoPilot = AutoPilotController(
    corpus: corpus,
    runtime: modelRuntime,
    training: training,
    deviceState: platformDeviceState ?? deviceStateSource,
    memory: memory,
  );

  return HarborServices(
    config: config,
    bridge: bridge,
    httpClient: httpClient,
    storage: storage,
    modelRuntime: modelRuntime,
    corpus: corpus,
    tools: tools,
    chat: chat,
    memory: memory,
    assistantSettings: assistantSettings,
    autoPilot: autoPilot,
    training: training,
    platformDeviceState: platformDeviceState,
    attention: attention,
    moe: moe,
    adapter: adapter,
    trainer: trainer,
    scheduler: scheduler,
    adapterStore: adapterStore,
    deviceStateSource: deviceStateSource,
    reader: reader,
    runner: runner,
    agent: agent,
    flags: flags,
    updateService: updateService,
    flow: flow,
    registry: DynamicModuleRegistry(),
  );
}

/// The directories a device scan may read.
///
/// The app's own support directory always works. The shared storage roots are
/// best-effort: on Android 11 and later a plain `Directory.list` there returns
/// permission errors for most subdirectories, which the collector records as
/// denied roots instead of pretending the scan found nothing.
List<Directory> _corpusRoots(Directory supportDirectory) {
  final List<Directory> roots = <Directory>[supportDirectory];
  if (!Platform.isAndroid) {
    return roots;
  }
  roots.addAll(<Directory>[
    Directory('/storage/emulated/0/Download'),
    Directory('/storage/emulated/0/Documents'),
  ]);
  final String? external = Platform.environment['EXTERNAL_STORAGE'];
  if (external != null && external.isNotEmpty) {
    roots.add(Directory(external));
  }
  return roots;
}

/// Detects whether the device is low-resource.
///
/// Uses the runtime environment to check CPU cores. A device with
/// 2 or fewer cores is treated as low-resource so the model
/// shrinks automatically.
bool platformIsLowResource() {
  if (!Platform.isAndroid) {
    return false;
  }
  return Platform.numberOfProcessors <= 2;
}

/// Provides [HarborServices] to the widget tree.
class HarborScope extends InheritedWidget {
  const HarborScope({
    super.key,
    required this.services,
    required super.child,
  });

  final HarborServices services;

  /// Looks up the nearest scope, throwing when the app was not wrapped in one.
  static HarborServices of(BuildContext context) {
    final HarborScope? scope =
        context.dependOnInheritedWidgetOfExactType<HarborScope>();
    if (scope == null) {
      throw FlutterError(
        'HarborScope.of() was called with a context that does not contain a '
        'HarborScope. Wrap the widget tree in HarborApp.',
      );
    }
    return scope.services;
  }

  @override
  bool updateShouldNotify(HarborScope oldWidget) =>
      services != oldWidget.services;
}

/// Root widget.
class HarborApp extends StatefulWidget {
  const HarborApp({super.key, required this.services});

  final HarborServices services;

  @override
  State<HarborApp> createState() => _HarborAppState();
}

class _HarborAppState extends State<HarborApp> {
  ThemeMode _themeMode = ThemeMode.system;

  @override
  Widget build(BuildContext context) {
    return HarborScope(
      services: widget.services,
      child: MaterialApp(
        title: 'Harbor',
        debugShowCheckedModeBanner: false,
        themeMode: _themeMode,
        theme: _buildTheme(Brightness.light),
        darkTheme: _buildTheme(Brightness.dark),
        home: HarborHomePage(
          onToggleTheme: () {
            setState(() {
              _themeMode = _themeMode == ThemeMode.dark
                  ? ThemeMode.light
                  : ThemeMode.dark;
            });
          },
        ),
      ),
    );
  }

  /// Builds a Material 3 colour scheme from the Harbor seed.
  ThemeData _buildTheme(Brightness brightness) {
    return ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(
        seedColor: kHarborSeedColor,
        brightness: brightness,
      ),
      visualDensity: VisualDensity.adaptivePlatformDensity,
      cardTheme: const CardTheme(clipBehavior: Clip.antiAlias),
    );
  }
}

/// Main shell: dynamic-module workspace, engine diagnostics and settings.
class HarborHomePage extends StatefulWidget {
  const HarborHomePage({super.key, required this.onToggleTheme});

  /// Toggles between the light and dark schemes.
  final VoidCallback onToggleTheme;

  @override
  State<HarborHomePage> createState() => _HarborHomePageState();
}

class _HarborHomePageState extends State<HarborHomePage> {
  int _tabIndex = 0;
  bool _forced = false;
  bool _bootstrapped = false;
  Timer? _idleTimer;
  String? _bootstrapError;

  HarborServices get _services => HarborScope.of(context);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => unawaited(_bootstrap()));
  }

  @override
  void dispose() {
    _idleTimer?.cancel();
    super.dispose();
  }

  /// Restores cached flags, runs the silent update check, kicks off
  /// automatic learning if the model is not ready yet, and starts the
  /// idle-training poll. Every step is failure-tolerant.
  Future<void> _bootstrap() async {
    if (_bootstrapped || !mounted) {
      return;
    }
    _bootstrapped = true;
    final HarborServices services = _services;

    await services.flags.initialize();
    if (!mounted) {
      return;
    }

    // Start the idle-training poll before the (possibly slow) network call so
    // learning is not gated on connectivity.
    _idleTimer = Timer.periodic(
      const Duration(seconds: 30),
      (_) => unawaited(_pollIdleTraining()),
    );

    // Automatic setup: if the model is not ready yet, start the
    // pipeline in the background so the app is usable immediately.
    // No manual "Get started" tap needed.
    if (!services.modelRuntime.ready && !services.autoPilot.busy) {
      unawaited(services.autoPilot.runSetup());
    }

    final FlagSyncResult sync = await services.flags.refresh();
    if (!mounted) {
      return;
    }
    if (!sync.ok && services.config.verboseLogging) {
      debugPrint('Feature flag refresh failed: ${sync.error}');
    }

    if (!services.config.autoCheckUpdates) {
      return;
    }

    final UpdateCheckResult result =
        await services.updateService.checkForUpdate(silent: true);
    if (!mounted) {
      return;
    }
    setState(() {
      services.lastUpdateCheck = result;
      _forced = result.isForced;
    });

    if (result.shouldPrompt && result.release != null) {
      if (result.isForced) {
        await _showForcedDialog(result);
      }
      // The soft banner renders inline from `lastUpdateCheck`.
    }
  }

  Future<void> _showForcedDialog(UpdateCheckResult result) async {
    if (!mounted || result.release == null) {
      return;
    }
    await ForceUpdateDialog.show(
      context,
      release: result.release!,
      installedVersion: result.installedVersion.toString(),
      controller: _services.flow,
    );
  }

  /// Reads device state into the scheduler and runs a gated round when allowed.
  Future<void> _pollIdleTraining() async {
    final HarborServices services = _services;
    if (services.training.isRunning) {
      return;
    }
    // The real pipeline, gated on the real phone. `platformDeviceState` reads
    // the battery and the thermal status through the platform channel; the
    // manual source is only the fallback for a platform that reports neither,
    // where the gate then fails closed on unknown power.
    final DeviceStateSource source =
        services.platformDeviceState ?? services.deviceStateSource;
    final DeviceState state = await source.read();
    services.scheduler.updateDeviceState(state);
    if (!services.scheduler.evaluateGate().allowed) {
      return;
    }
    if (services.training.readiness != null) {
      return;
    }
    // A deliberately short round. Background learning is worth having only if
    // it stops before it heats the phone, and because the session checkpoints
    // as it goes, an interrupted round is still saved progress.
    await services.training.start(
      override: services.training.config.copyWith(
        totalSteps: 40,
        checkpointEvery: 10,
      ),
    );
    if (mounted) {
      setState(() {});
    }
  }

  Future<void> _manualUpdateCheck() async {
    final HarborServices services = _services;
    final UpdateCheckResult result =
        await services.updateService.checkForUpdate(silent: false);
    if (!mounted) {
      return;
    }
    setState(() {
      services.lastUpdateCheck = result;
      _forced = result.isForced;
    });
    if (result.isForced && result.release != null) {
      await _showForcedDialog(result);
    }
  }

  @override
  Widget build(BuildContext context) {
    final HarborServices services = _services;

    // A forced update locks the UI: no back navigation, no tab switching.
    return PopScope(
      canPop: !_forced,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Harbor'),
          actions: <Widget>[
            IconButton(
              tooltip: 'Memory',
              onPressed: _forced
                  ? null
                  : () => unawaited(
                        Navigator.of(context).push(
                          MaterialPageRoute<void>(
                            builder: (BuildContext context) => MemoryScreen(
                              controller: services.chat,
                            ),
                          ),
                        ),
                      ),
              icon: const Icon(Icons.psychology_outlined),
            ),
            IconButton(
              tooltip: 'Engine diagnostics',
              onPressed: _forced ? null : () => _openEngineTab(services),
              icon: const Icon(Icons.memory_outlined),
            ),
            IconButton(
              tooltip: 'Check for updates',
              onPressed: _forced ? null : () => unawaited(_manualUpdateCheck()),
              icon: const Icon(Icons.system_update_alt),
            ),
            IconButton(
              tooltip: 'Refresh feature flags',
              onPressed: _forced ? null : () => unawaited(services.flags.refresh()),
              icon: const Icon(Icons.cloud_sync_outlined),
            ),
            IconButton(
              tooltip: 'Toggle theme',
              onPressed: widget.onToggleTheme,
              icon: const Icon(Icons.brightness_6_outlined),
            ),
          ],
        ),
        body: Column(
          children: <Widget>[
            if (services.flags.lastError != null && !_forced)
              MaterialBanner(
                content: Text(
                  'Feature flags are running from ${services.flags.source}: '
                  '${services.flags.lastError}',
                ),
                leading: const Icon(Icons.cloud_off_outlined),
                actions: <Widget>[
                  TextButton(
                    onPressed: () => unawaited(services.flags.refresh()),
                    child: const Text('Retry'),
                  ),
                ],
              ),
            if (!services.modelRuntime.ready)
              MaterialBanner(
                content: Text(
                  'Harbor has no working model: '
                  '${services.modelRuntime.error ?? services.modelRuntime.status}. '
                  'Chat will answer from an untrained network until a corpus is '
                  'gathered and training succeeds.',
                ),
                leading: const Icon(Icons.psychology_outlined),
                actions: <Widget>[
                  TextButton(
                    onPressed: () => setState(() => _tabIndex = 2),
                    child: const Text('Open Learn'),
                  ),
                ],
              ),
            if (_bootstrapError != null)
              MaterialBanner(
                content: Text(_bootstrapError!),
                leading: const Icon(Icons.error_outline),
                actions: <Widget>[
                  TextButton(
                    onPressed: () => setState(() => _bootstrapError = null),
                    child: const Text('Dismiss'),
                  ),
                ],
              ),
            Expanded(
              child: IndexedStack(
                index: _tabIndex,
                children: <Widget>[
                  _WorkspaceTab(services: services),
                  ChatScreen(controller: services.chat),
                  _LearnTab(services: services),
                  ToolsScreen(controller: services.tools),
                  _SettingsTab(services: services),
                ],
              ),
            ),
          ],
        ),
        bottomNavigationBar: NavigationBar(
          selectedIndex: _tabIndex,
          onDestinationSelected: _forced
              ? null
              : (int index) => setState(() => _tabIndex = index),
          destinations: const <NavigationDestination>[
            NavigationDestination(
              icon: Icon(Icons.dashboard_outlined),
              selectedIcon: Icon(Icons.dashboard),
              label: 'Workspace',
            ),
            NavigationDestination(
              icon: Icon(Icons.chat_bubble_outline),
              selectedIcon: Icon(Icons.chat_bubble),
              label: 'Chat',
            ),
            NavigationDestination(
              icon: Icon(Icons.school_outlined),
              selectedIcon: Icon(Icons.school),
              label: 'Learn',
            ),
            NavigationDestination(
              icon: Icon(Icons.handyman_outlined),
              selectedIcon: Icon(Icons.handyman),
              label: 'Tools',
            ),
            NavigationDestination(
              icon: Icon(Icons.settings_outlined),
              selectedIcon: Icon(Icons.settings),
              label: 'Settings',
            ),
          ],
        ),
        bottomSheet: _buildSoftUpdateBanner(services),
      ),
    );
  }

  /// Opens the engine diagnostics on top of the tabs.
  ///
  /// It is a route rather than a sixth destination because a `NavigationBar`
  /// with six items either shrinks the labels until they are unreadable or
  /// scrolls, and the diagnostics are something you visit deliberately rather
  /// than live in.
  void _openEngineTab(HarborServices services) {
    unawaited(
      Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (BuildContext context) => Scaffold(
            appBar: AppBar(title: const Text('Engine diagnostics')),
            body: _EngineTab(services: services),
          ),
        ),
      ),
    );
  }

  /// The dismissible soft-update banner, or null when nothing should show.
  Widget? _buildSoftUpdateBanner(HarborServices services) {
    final UpdateCheckResult? result = services.lastUpdateCheck;
    if (result == null || !result.isSoft || result.release == null) {
      return null;
    }
    return SoftUpdateBanner(
      release: result.release!,
      installedVersion: result.installedVersion.toString(),
      onUpdate: () {
        unawaited(
          DownloadProgressSheet.show(
            context,
            release: result.release!,
            controller: services.flow,
          ),
        );
        unawaited(services.flow.start(result.release!));
      },
      onDismiss: () async {
        await services.updateService.dismiss(result.release!.latestVersion);
        if (mounted) {
          setState(() {});
        }
      },
    );
  }
}

/// Training and the corpus it learns from, as two sub-tabs.
///
/// They belong together because the order is not a preference: training reads
/// whatever the corpus holds, so a learner that cannot find the corpus tab
/// reports "the corpus is empty" without explaining where to fix it.
class _LearnTab extends StatelessWidget {
  const _LearnTab({required this.services});

  final HarborServices services;

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Column(
        children: <Widget>[
          const TabBar(
            tabs: <Widget>[
              Tab(text: 'Training'),
              Tab(text: 'Corpus'),
            ],
          ),
          Expanded(
            child: TabBarView(
              children: <Widget>[
                TrainingScreen(
                  controller: services.training,
                  runtime: services.modelRuntime,
                ),
                CorpusScreen(controller: services.corpus),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// The server-driven workspace: renders whatever layout the flags allow.
class _WorkspaceTab extends StatelessWidget {
  const _WorkspaceTab({required this.services});

  final HarborServices services;

  @override
  Widget build(BuildContext context) {
    // The setup card is deliberately outside the server-driven layout: a
    // feature-flag matrix must never be able to hide the one control that makes
    // the app usable from scratch, and a remote flag that fails to load would
    // otherwise leave a new install with no entry point at all.
    return Column(
      children: <Widget>[
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
          child: OneTapSetupCard(autoPilot: services.autoPilot),
        ),
        Expanded(
          child: DynamicModuleContextBuilder(
            services: services,
            builder: (DynamicModuleContext moduleContext) =>
                services.registry.buildLayout(
              context,
              services.flags.layout,
              moduleContext,
            ),
          ),
        ),
      ],
    );
  }
}

/// Bridges [HarborServices] into a [DynamicModuleContext], rebuilding when the
/// update check result changes.
class DynamicModuleContextBuilder extends StatelessWidget {
  const DynamicModuleContextBuilder({
    super.key,
    required this.services,
    required this.builder,
  });

  final HarborServices services;
  final Widget Function(DynamicModuleContext context) builder;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: services.flags,
      builder: (BuildContext context, Widget? _) {
        return builder(
          DynamicModuleContext(
            flags: services.flags,
            reader: services.reader,
            agent: services.agent,
            updateService: services.updateService,
            currentUpdateResult: services.lastUpdateCheck,
            chat: services.chat,
            tools: services.tools,
            training: services.training,
            corpus: services.corpus,
            modelRuntime: services.modelRuntime,
          ),
        );
      },
    );
  }
}

/// Engine diagnostics and manual training controls.
class _EngineTab extends StatefulWidget {
  const _EngineTab({required this.services});

  final HarborServices services;

  @override
  State<_EngineTab> createState() => _EngineTabState();
}

class _EngineTabState extends State<_EngineTab> {
  String? _status;

  Future<void> _runTrainingRound() async {
    final HarborServices services = widget.services;
    services.scheduler.updateDeviceState(
      services.deviceStateSource.state.copyWith(
        isIdle: true,
        isCharging: true,
      ),
    );
    try {
      final TrainingRoundReport report = await services.scheduler.runRound(
        reason: 'manual round from the engine screen',
      );
      services.lastTrainingRound = report;
      await services.adapterStore.save('harbor_default', services.trainer.adapter);
      if (mounted) {
        setState(() {
          _status = 'Round complete: ${report.steps} step(s), loss '
              '${report.firstLoss.toStringAsFixed(4)} -> '
              '${report.lastLoss.toStringAsFixed(4)}';
        });
      }
    } on Object catch (e) {
      if (mounted) {
        setState(() => _status = 'Training failed: $e');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final HarborServices services = widget.services;
    final MlaCacheStats cacheStats = services.attention.cacheStats;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: <Widget>[
        _DiagnosticsCard(
          title: 'Multi-Head Latent Attention',
          subtitle: 'KV-cache compression',
          report: services.attention.describe(),
          highlight: <String, Object?>{
            'compression_ratio': '${cacheStats.compressionRatio.toStringAsFixed(2)}×',
            'bytes_per_token': cacheStats.mlaBytesPerToken,
            'vanilla_bytes_per_token': cacheStats.vanillaBytesPerToken,
          },
        ),
        const SizedBox(height: 12),
        _DiagnosticsCard(
          title: 'Fine-Grained Mixture of Experts',
          subtitle: 'Shared + top-K routed experts',
          report: services.moe.describe(),
        ),
        const SizedBox(height: 12),
        _DiagnosticsCard(
          title: 'On-Device LoRA',
          subtitle: 'Continuous background self-learning',
          report: services.trainer.describe(),
          highlight: <String, Object?>{
            'parameter_ratio':
                '${(services.adapter.config.parameterRatio * 100).toStringAsFixed(2)}% of a dense update',
            'adapter_version': services.adapter.version,
            'parameters': services.adapter.config.trainableParameters,
          },
        ),
        const SizedBox(height: 12),
        _DiagnosticsCard(
          title: 'Idle Training Scheduler',
          subtitle: 'Gated on idle + charging + thermal + budget',
          report: services.scheduler.describe(),
        ),
        const SizedBox(height: 12),
        _DiagnosticsCard(
          title: 'Subagent Runner',
          subtitle: 'Concurrent local task execution',
          report: services.runner.describe(),
        ),
        const SizedBox(height: 16),
        Card(
          elevation: 0,
          color: Theme.of(context).colorScheme.surfaceContainerHigh,
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Text(
                  'Manual training round',
                  style: Theme.of(context).textTheme.titleSmall,
                ),
                const SizedBox(height: 4),
                Text(
                  'Demonstrates the real optimiser: it runs AdamW steps over the '
                  'replay buffer seeded with the frozen engine\'s own outputs, '
                  'then persists the adapter.',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
                const SizedBox(height: 12),
                FilledButton.icon(
                  onPressed: () => unawaited(_runTrainingRound()),
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Run a round now'),
                ),
                if (_status != null) ...<Widget>[
                  const SizedBox(height: 12),
                  Text(_status!, style: Theme.of(context).textTheme.bodySmall),
                ],
                if (services.lastTrainingRound != null) ...<Widget>[
                  const SizedBox(height: 8),
                  Text(
                    'Last scheduler round: '
                    '${services.lastTrainingRound!.steps} steps, '
                    'improvement '
                    '${services.lastTrainingRound!.improvement.toStringAsFixed(5)}',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ],
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class _DiagnosticsCard extends StatelessWidget {
  const _DiagnosticsCard({
    required this.title,
    required this.subtitle,
    required this.report,
    this.highlight,
  });

  final String title;
  final String subtitle;
  final Map<String, Object?> report;
  final Map<String, Object?>? highlight;

  @override
  Widget build(BuildContext context) {
    final TextTheme text = Theme.of(context).textTheme;
    final ColorScheme colors = Theme.of(context).colorScheme;

    return Card(
      elevation: 0,
      color: colors.surfaceContainerHigh,
      child: ExpansionTile(
        shape: const Border(),
        title: Text(title, style: text.titleSmall),
        subtitle: Text(subtitle, style: text.bodySmall),
        childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
        children: <Widget>[
          if (highlight != null) ...<Widget>[
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: colors.primaryContainer,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: highlight!.entries
                    .map(
                      (MapEntry<String, Object?> e) => Text(
                        '${e.key}: ${e.value}',
                        style: text.bodySmall?.copyWith(
                          color: colors.onPrimaryContainer,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    )
                    .toList(growable: false),
              ),
            ),
            const SizedBox(height: 12),
          ],
          for (final MapEntry<String, Object?> e in report.entries)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 2),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  SizedBox(
                    width: 170,
                    child: Text(
                      e.key,
                      style: text.bodySmall?.copyWith(fontWeight: FontWeight.w600),
                    ),
                  ),
                  Expanded(
                    child: Text(_renderValue(e.value), style: text.bodySmall),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  static String _renderValue(Object? value) {
    if (value == null) {
      return '—';
    }
    if (value is List) {
      return value.isEmpty ? '[]' : value.join(', ');
    }
    if (value is Map) {
      return value.entries
          .map((MapEntry<Object?, Object?> e) => '${e.key}=${e.value}')
          .join(', ');
    }
    return value.toString();
  }
}

/// Settings: configuration, flags, subagent safety and update controls.
class _SettingsTab extends StatefulWidget {
  const _SettingsTab({required this.services});

  final HarborServices services;

  @override
  State<_SettingsTab> createState() => _SettingsTabState();
}

class _SettingsTabState extends State<_SettingsTab> {
  Future<void> _toggleDevice(DeviceState Function(DeviceState) mutate) async {
    final HarborServices services = widget.services;
    services.deviceStateSource.update(mutate(services.deviceStateSource.state));
    services.scheduler.updateDeviceState(services.deviceStateSource.state);
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final HarborServices services = widget.services;
    final DeviceState device = services.deviceStateSource.state;
    final TrainingGateDecision gate = services.scheduler.evaluateGate();

    return ListView(
      padding: const EdgeInsets.all(16),
      children: <Widget>[
        _SettingsSection(
          title: 'Build configuration',
          children: <Widget>[
            _SettingsRow('API base URL', services.config.apiBaseUrl),
            _SettingsRow('App version', services.config.appVersion),
            _SettingsRow('Platform', services.config.platform),
            _SettingsRow('Verbose logging', '${services.config.verboseLogging}'),
          ],
        ),
        const SizedBox(height: 12),
        _AssistantSection(services: services),
        const SizedBox(height: 12),
        _SettingsSection(
          title: 'Simulated device state',
          description:
              'The idle-training scheduler only runs when the device is idle '
              'and charging. Production builds feed these from platform battery '
              'and thermal channels; here they are toggled manually.',
          children: <Widget>[
            SwitchListTile(
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: const Text('Device idle'),
              value: device.isIdle,
              onChanged: (bool v) => unawaited(
                _toggleDevice((DeviceState s) => s.copyWith(isIdle: v)),
              ),
            ),
            SwitchListTile(
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: const Text('Charging'),
              value: device.isCharging,
              onChanged: (bool v) => unawaited(
                _toggleDevice((DeviceState s) => s.copyWith(isCharging: v)),
              ),
            ),
            SliderListTile(
              label: 'Battery '
                  '${(device.batteryLevel * 100).toStringAsFixed(0)}%',
              value: device.batteryLevel,
              onChanged: (double v) => unawaited(
                _toggleDevice((DeviceState s) => s.copyWith(batteryLevel: v)),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                'Thermal state: ${device.thermalState.name}',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ),
            Wrap(
              spacing: 8,
              children: ThermalState.values
                  .map(
                    (ThermalState t) => ChoiceChip(
                      label: Text(t.name),
                      selected: device.thermalState == t,
                      onSelected: (_) => unawaited(
                        _toggleDevice(
                          (DeviceState s) => s.copyWith(thermalState: t),
                        ),
                      ),
                    ),
                  )
                  .toList(growable: false),
            ),
            const SizedBox(height: 12),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: gate.allowed
                    ? Theme.of(context).colorScheme.primaryContainer
                    : Theme.of(context).colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Text(
                'Training gate: ${gate.allowed ? 'OPEN' : 'closed'} — ${gate.reason}',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Compute used today: ${services.scheduler.computeUsedToday.inSeconds}s '
              'of ${services.scheduler.guards.dailyComputeBudget.inMinutes}m',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
        const SizedBox(height: 12),
        _SettingsSection(
          title: 'Feature flags',
          description:
              'Resolved from the server matrix, overridable locally for testing.',
          children: <Widget>[
            _SettingsRow('Matrix version', '${services.flags.version}'),
            _SettingsRow('Source', services.flags.source),
            _SettingsRow('Sections', '${services.flags.layout.sections.length}'),
            _SettingsRow(
              'Visible modules',
              '${services.flags.visibleSections.fold<int>(0, (int a, DynamicSection s) => a + s.modules.length)}',
            ),
            _SettingsRow('Override count', '${services.flags.overrides.length}'),
            if (services.flags.lastError != null)
              _SettingsRow('Last error', services.flags.lastError!),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              children: <Widget>[
                OutlinedButton(
                  onPressed: () => unawaited(services.flags.refresh(force: true)),
                  child: const Text('Force refresh'),
                ),
                OutlinedButton(
                  onPressed: services.flags.overrides.isEmpty
                      ? null
                      : () => unawaited(services.flags.clearAllOverrides()),
                  child: const Text('Clear overrides'),
                ),
              ],
            ),
          ],
        ),
        const SizedBox(height: 12),
        _SettingsSection(
          title: 'Subagent safety',
          description:
              'Commands run without a shell against an explicit allow-list; any '
              'argument containing shell metacharacters is rejected before the '
              'process is spawned.',
          children: <Widget>[
            Text(
              'Allowed executables '
              '(${SystemCommandSubagent.defaultAllowedExecutables.length})',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 4),
            Text(
              (SystemCommandSubagent.defaultAllowedExecutables.toList()..sort())
                  .join(', '),
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
        const SizedBox(height: 12),
        _SettingsSection(
          title: 'Updates',
          description: 'OTA status and manual controls.',
          children: <Widget>[
            _SettingsRow('Strategy', services.updateService.strategy.name),
            _SettingsRow(
              'Last decision',
              services.lastUpdateCheck?.decision.name ?? 'not checked',
            ),
            _SettingsRow(
              'Latest available',
              services.lastUpdateCheck?.release?.latestVersion ?? '—',
            ),
            if (services.lastUpdateCheck?.error != null)
              _SettingsRow('Last error', services.lastUpdateCheck!.error!),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              children: <Widget>[
                OutlinedButton(
                  onPressed: () => unawaited(
                    services.updateService.checkForUpdate(silent: false),
                  ),
                  child: const Text('Check now'),
                ),
                OutlinedButton(
                  onPressed: () => unawaited(
                    services.updateService.clearDismissal(),
                  ),
                  child: const Text('Clear dismissed version'),
                ),
              ],
            ),
          ],
        ),
        const SizedBox(height: 24),
      ],
    );
  }
}

/// The assistant's own switches: memory, learning and reasoning.
///
/// These three are the difference between an assistant that just works and one
/// that needs a manual, so they live at the top of Settings rather than buried
/// under Labs, and they take effect on the very next turn because changing them
/// rebuilds the chat engine (see [AssistantSettings.onChange]).
class _AssistantSection extends StatelessWidget {
  const _AssistantSection({required this.services});

  final HarborServices services;

  @override
  Widget build(BuildContext context) {
    final AssistantSettings settings = services.assistantSettings;
    final AutoPilotController autoPilot = services.autoPilot;
    return ListenableBuilder(
      listenable: Listenable.merge(<Listenable>[settings, autoPilot]),
      builder: (BuildContext context, Widget? child) {
        return _SettingsSection(
          title: 'Assistant',
          description: 'On by default. Everything below is stored on this '
              'device and takes effect on the next message.',
          children: <Widget>[
            SwitchListTile(
              key: const ValueKey<String>('settings.memory'),
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: const Text('Remember what you tell it'),
              subtitle: Text(
                settings.memoryEnabled
                    ? '${services.memory.length} memories stored; relevant '
                        'ones are read back into every answer.'
                    : 'Answers ignore everything from earlier conversations.',
              ),
              value: settings.memoryEnabled,
              onChanged: settings.setMemoryEnabled,
            ),
            SwitchListTile(
              key: const ValueKey<String>('settings.learning'),
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: const Text('Learn from conversation'),
              subtitle: const Text(
                'Turns statements like "my name is" or "I prefer" into '
                'memories automatically.',
              ),
              value: settings.learningEnabled,
              onChanged: settings.setLearningEnabled,
            ),
            SwitchListTile(
              key: const ValueKey<String>('settings.auto_thinking'),
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: const Text('Think before answering'),
              subtitle: Text(
                settings.autoThinking
                    ? 'Chooses per question: a greeting answers directly, a '
                        'hard one gets a plan and a check.'
                    : 'Pinned to ${settings.thinkingMode.label}.',
              ),
              value: settings.thinkingEnabled,
              onChanged: (bool value) {
                if (value) {
                  settings.setFullyAutomatic();
                } else {
                  settings.setThinkingMode(hc.ThinkingStrategy.none);
                }
              },
            ),
            if (settings.thinkingEnabled && !settings.autoThinking)
              Padding(
                padding: const EdgeInsets.only(left: 8),
                child: Wrap(
                  spacing: 8,
                  children: <Widget>[
                    for (final hc.ThinkingStrategy strategy
                        in <hc.ThinkingStrategy>[
                      hc.ThinkingStrategy.concise,
                      hc.ThinkingStrategy.thorough,
                    ])
                      ChoiceChip(
                        key: ValueKey<String>(
                          'settings.thinking.${strategy.wire}',
                        ),
                        label: Text(strategy.label),
                        selected: settings.thinkingMode == strategy,
                        onSelected: (_) => settings.setThinkingMode(strategy),
                      ),
                  ],
                ),
              ),
            const Divider(height: 24),
            Row(
              children: <Widget>[
                Expanded(
                  child: Text(
                    autoPilot.status.message,
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ),
                const SizedBox(width: 8),
                OutlinedButton.icon(
                  key: const ValueKey<String>('settings.setup'),
                  onPressed: autoPilot.busy
                      ? null
                      : () => unawaited(autoPilot.runSetup()),
                  icon: const Icon(Icons.bolt),
                  label: const Text('Set up now'),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: <Widget>[
                Expanded(
                  child: Text(
                    'Keep learning while charging',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ),
                Switch(
                  key: const ValueKey<String>('settings.auto_learn'),
                  value: autoPilot.autoLearn,
                  onChanged: autoPilot.setAutoLearn,
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              'Gated on the real battery and thermal sensors: nothing runs '
              'below 35% or when the phone is hot.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        );
      },
    );
  }
}

class _SettingsSection extends StatelessWidget {
  const _SettingsSection({
    required this.title,
    required this.children,
    this.description,
  });

  final String title;
  final String? description;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    final TextTheme text = Theme.of(context).textTheme;
    return Card(
      elevation: 0,
      color: Theme.of(context).colorScheme.surfaceContainerHigh,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              title,
              style: text.titleSmall?.copyWith(fontWeight: FontWeight.w600),
            ),
            if (description != null) ...<Widget>[
              const SizedBox(height: 4),
              Text(description!, style: text.bodySmall),
            ],
            const SizedBox(height: 12),
            ...children,
          ],
        ),
      ),
    );
  }
}

class _SettingsRow extends StatelessWidget {
  const _SettingsRow(this.label, this.value);

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final TextTheme text = Theme.of(context).textTheme;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          SizedBox(
            width: 150,
            child: Text(
              label,
              style: text.bodySmall?.copyWith(fontWeight: FontWeight.w600),
            ),
          ),
          Expanded(child: Text(value, style: text.bodySmall)),
        ],
      ),
    );
  }
}

/// A labelled slider, since Material 3 has no built-in one.
class SliderListTile extends StatelessWidget {
  const SliderListTile({
    super.key,
    required this.label,
    required this.value,
    required this.onChanged,
  });

  final String label;
  final double value;
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: <Widget>[
        SizedBox(
          width: 120,
          child: Text(label, style: Theme.of(context).textTheme.bodySmall),
        ),
        Expanded(
          child: Slider(
            value: value.clamp(0.0, 1.0),
            divisions: 20,
            label: '${(value * 100).round()}%',
            onChanged: onChanged,
          ),
        ),
      ],
    );
  }
}