# Harbor

A production-grade, dual-app Flutter ecosystem in a single monorepo:

| Component | Path | What it is |
| --- | --- | --- |
| **Harbor** (client) | `apps/main_app` | A Flutter Android app with an on-device inference engine (MLA + fine-grained MoE + LoRA self-learning), an extended-thinking autonomous agent, a universal file reader, and a self-updating OTA engine. |
| **Harbor Update Server** | `apps/update_server` | A Dart `shelf` service that serves release metadata, the dynamic feature-flag matrix, and a Flutter Web admin dashboard for publishing releases and flipping flags live. |
| **Release pipeline** | `.github/workflows` | Tag-driven APK build → SHA256 → GitHub Release → `latest_version.json` → GitHub Pages, plus a server verify/deploy pipeline. Both pipelines are green on CI and have published `v1.0.0`; see [§7](#7-verification-status). |

The frozen interface between the two apps — the HTTP wire contract, the feature-flag matrix schema and the release-metadata schema — is specified in [`MASTER_PROMPT.txt`](MASTER_PROMPT.txt) §4. Treat it as normative: the client parses exactly those shapes and the server emits exactly those shapes.

---

## 1. Repository layout

```
.
├── .github/
│   ├── scripts/build_pages_site.sh     # assembles the complete Pages site (both pipelines)
│   └── workflows/
│       ├── build_and_release.yml       # v*.*.* tag -> signed APK -> Release -> Pages
│       └── deploy_update_server.yml    # CI for the server + admin web dashboard
├── apps/
│   ├── main_app/                      # the Flutter client ("Harbor")
│   │   ├── android/                   # platform scaffolding (Gradle 8.4 / AGP 8.1 / Kotlin 1.9.22)
│   │   ├── lib/
│   │   │   ├── core/
│   │   │   │   ├── engine/
│   │   │   │   │   ├── mla_attention.dart          # Multi-head Latent Attention + KV cache
│   │   │   │   │   ├── fine_grained_moe.dart       # shared + top-K routed experts, aux-loss-free
│   │   │   │   │   └── on_device_lora.dart         # LoRA adapters, AdamW trainer, idle scheduler
│   │   │   │   ├── agent/
│   │   │   │   │   ├── extended_thinking.dart      # Plan -> Execute -> Verify -> Replan
│   │   │   │   │   ├── subagent_runner.dart        # bounded worker pool + 3 built-in subagents
│   │   │   │   │   └── file_readers.dart           # universal file/archive/structured reader
│   │   │   │   └── update_engine/
│   │   │   │       ├── update_service.dart         # policy, OTA download, SHA256, install
│   │   │   │       ├── dynamic_feature_flag_provider.dart
│   │   │   │       └── widgets/update_dialog.dart  # force / soft / progress UI
│   │   │   ├── features/dynamic_module_registry.dart
│   │   │   └── main.dart
│   │   ├── test/                      # unit + widget tests
│   │   ├── analysis_options.yaml
│   │   └── pubspec.yaml
│   └── update_server/
│       ├── bin/server.dart            # entrypoint (plain Dart VM, no Flutter)
│       ├── lib/
│       │   ├── main.dart              # Flutter shell for the Web admin dashboard
│       │   └── src/                   # models, release_store, api_router, lifecycle (no Flutter)
│       ├── test/
│       ├── web/
│       └── pubspec.yaml
├── MASTER_PROMPT.txt                  # normative spec + frozen wire contract (§4)
└── README.md
```

---

## 2. Component A — the client engine

### 2.1 `mla_attention.dart` — Multi-Head Latent Attention

Implements the DeepSeek-V2/V3-style low-rank KV compression:

- A token's KV state is **not** `2 · numHeads · headDim` floats. Instead only a latent
  `c_kv = W_dkv · x` of width `kvLoraRank` plus a decoupled RoPE key `k_pe` of width
  `ropeHeadDim` are cached. `k_nope` and `v` are **regenerated on the fly** from `c_kv`.
- Cached bytes per token: `(kvLoraRank + ropeHeadDim) · 4`, versus
  `2 · numHeads · headDim · 4` for a vanilla cache. `MlaCacheStats` reports both and the
  resulting compression ratio, so the saving is measurable at runtime rather than asserted.
- Queries are compressed the same way (`qLoraRank`) and de-compressed per head.
- `RotaryEmbedding` precomputes a cos/sin table and **asserts** on an odd `ropeHeadDim` or an
  out-of-range position instead of producing silently wrong attention.

### 2.2 `fine_grained_moe.dart` — Fine-grained Mixture-of-Experts

- **Always-on shared experts** plus **Top-K routed experts**, both with a fine-grained
  intermediate width instead of a few wide experts.
- **Aux-loss-free load balancing**: a per-expert selection *bias* is nudged up for
  under-used experts and down for over-used ones. The bias affects **selection only** and
  never touches the gate's mixing weights, so the routing distribution is never distorted.
- Capacity factor + token dropping, normalised Top-K probabilities, and a real
  `forwardOne`/`forward` pair (batch == single-token result is asserted in tests).
- `ExpertLoadReport` exposes per-expert assignment counts, dropped tokens, balance entropy
  (normalised against `ln(n)`) and a `toJson` matching the documented contract. A perfectly
  uniform load scores exactly `1.0`.

### 2.3 `on_device_lora.dart` — background self-learning

- `y = W₀·x + (α/rank)·B·(A·x)` with `A ~ N(0, σ²)` and **`B = 0`**, so a freshly created
  adapter is a provable identity — merging it changes nothing (asserted bit-for-bit).
- `mergeInto` / `unmergeFrom` fold the adapter into a weight copy and back out again.
- Analytical backprop through the LoRA branch only (the frozen base weights are never
  updated), AdamW with decoupled weight decay and global-norm clipping.
- A bounded replay buffer with reservoir replacement, plus an `IdleTrainingScheduler` that
  trains **only** when every guard passes: idle ∧ charging ∧ battery ≥ 0.35 ∧ thermal ≤ fair
  ∧ daily compute budget remaining ∧ non-empty replay. Consecutive failures disable the
  scheduler rather than looping on a broken adapter.
- `FileLoraAdapterStore` persists adapters as JSON; `toJson`/`fromJson` round-trip `A` and
  `B` exactly and reject truncated tensors.

### 2.4 `extended_thinking.dart` — the agent loop

An explicit, inspectable **Plan → Execute → Verify → Repair** loop:

- `Plan` is a DAG of `ThoughtStep`s with `dependsOn` edges. `executionWaves()` groups
  independent steps so they run concurrently; `validate()` rejects dangling dependencies,
  self-dependencies and cycles (`PlanValidationException`) before anything executes.
- `HeuristicPlanningBackend` is a real, deterministic planner (no model weights required):
  it extracts quoted paths, bare paths with recognised extensions, directories, command
  intents and verification/listing intents from the goal text and turns them into steps.
- Verification is genuine, not a rubber stamp: a failed subagent, a non-zero exit code or a
  `CodeVerificationSubagent` reporting `error_count > 0` all mark a step unverified.
- Bounded repair replanning (`maxReplanCycles`) — the agent does not retry forever.
- The whole run is captured in an immutable `ThinkingTrace` with `toMarkdown()`, so the
  reasoning is auditable after the fact.

### 2.5 `subagent_runner.dart` — tool execution

- A bounded worker pool with a priority-ordered queue, per-task timeouts, cooperative
  cancellation (`CancellationToken`) and **failures-as-values**: a throwing subagent becomes
  a failed `SubagentResult`, never an unhandled exception.
- Three real subagents:
  - `FileParsingSubagent` — delegates to the universal file reader.
  - `SystemCommandSubagent` — the security boundary. `runInShell: false`, an explicit
    executable **allow-list**, and argument rejection for shell metacharacters
    (`; & | \` $ > < newline backslash`, `$(...)`, `&&`, `||`). No string is ever handed to
    a shell.
  - `CodeVerificationSubagent` — delimiter balance (string-literal aware), merge-conflict
    markers, line length, trailing whitespace, unresolved `TODO`/`FIXME` markers, missing
    final newline.

### 2.6 `file_readers.dart` — universal file reading

- Magic-number sniffing (not extension guessing) for ELF, Mach-O, PE, DEX, AXML, SQLite,
  WASM, Java class, ZIP, GZIP, BZIP2, XZ, Zstd, TAR (offset 257), PNG, JPEG, GIF, BMP, WebP,
  PDF and OLE2, plus UTF-8/16/32 BOM detection.
- Archive walking for zip / tar / tar.gz with explicit zip-bomb caps; entries are listed with
  names, sizes and a bounded text preview.
- Structured parsers for JSON, JSONL, YAML, CSV/TSV (RFC-4180-ish quoting), INI, properties,
  XML and a minimal TOML — with a documented, bounded `FileReadLimits` and warnings recorded
  whenever a limit truncates output rather than silently dropping data.
- Hex + ASCII dump for anything unrecognised.
- All structured accessors are `strict-casts` clean: JSON is validated with `is Map` /
  `is List` checks before any cast.

### 2.7 `update_engine/` — OTA self-update

The decision logic is a **pure function**, which is why it is exhaustively unit-testable:

```dart
updateAvailable = installed < latest
updateRequired  = (installed < minSupported) || forceUpdate
```

- **Silent check on launch** — a failure is captured as a value (`UpdateCheckResult.error`),
  never thrown at the app's startup path.
- **Force update** — a non-dismissible modal (`PopScope(canPop: false)` +
  `barrierDismissible: false`) when `installed < min_supported_version` or the server sets
  `force_update`. A dismissal can **never** suppress a forced update.
- **Soft update** — a dismissible banner, with the dismissed version persisted so it does not
  nag on every launch (but reappears for a newer release).
- **Direct APK OTA**: streamed download with an incrementally-computed SHA256
  (`sha256.startChunkedConversion` + `AccumulatorSink<Digest>`), compared against the
  server's digest. A mismatch throws `ChecksumMismatchException` and deletes the partial
  file — a corrupt APK is never handed to the installer. On success it installs natively via
  `ota_update` (`InstallStrategy.nativeStreaming`), or via the verify-then-install path.
- **Dynamic feature flags** gate the UI: modules and whole sections can be shown or hidden
  without shipping a new APK. Resolution order is
  `local override > remote flags > remote defaults > caller default`. An unknown module type
  renders a visible "unsupported module" placeholder rather than vanishing.
- All configuration comes from `--dart-define`, so the same source builds for emulator and
  production:

| Define | Default | Purpose |
| --- | --- | --- |
| `HARBOR_API_BASE_URL` | `http://10.0.2.2:8080` | Update server base URL |
| `HARBOR_APP_VERSION` | `1.0.0` | Installed version used by the update policy |
| `HARBOR_PLATFORM` | `android` | Platform sent to the update check |
| `HARBOR_VERBOSE_LOGGING` | `false` | Verbose engine/agent logging |
| `HARBOR_AUTO_CHECK_UPDATES` | `true` | Silently check for updates on launch |

---

## 3. Component B — the update server

A plain-Dart `shelf` service. `bin/server.dart` and everything under `lib/src/` are
**Flutter-free** so the server runs on the Dart VM; only `lib/main.dart` imports Flutter (it
is the Web admin dashboard shell).

### 3.1 Read endpoints

```
GET /api/v1/update-check?version={version}&platform={platform}
GET /api/v1/flags
```

`update-check` returns the latest version, download URL, SHA256, minimum supported version and
the changelog. Contract guarantees:

- `update_required = (installed < min_supported) OR force_update`
- `update_available = installed < latest`
- With **no published release** the server returns HTTP 200 and a zero payload
  (`latest_version == min_supported_version == "0.0.0"`) — the client must not crash on a
  freshly deployed server.
- A malformed `version` query parameter returns HTTP **400** with `{"error": ...}`.
- Semver comparison uses `pub_semver`, so `1.9.0 < 1.10.0` and build metadata/pre-release
  ordering are handled correctly (a naive string compare would get both wrong).

`flags` returns the dynamic key/value matrix plus the remote defaults and the UI layout
(sections → modules → gating flag).

Both GET endpoints carry an `ETag`, and CORS is permissive for reads
(`Access-Control-Allow-Origin: *`, `X-Admin-Token` and `Content-Type` allowed, `OPTIONS` → 204).

### 3.2 Admin endpoints

All mutations require the `X-Admin-Token` header:

- mismatch → **401**
- no token configured on the server **and** `HARBOR_ALLOW_DEV_TOKEN != true` → **503**
  `{"error":"admin token not configured"}`

Admins can adjust the minimum supported version, publish release metadata, trigger a forced
update, and toggle flags live. Mutations are what the admin dashboard drives.

### 3.3 The admin dashboard

`lib/main.dart` is a Flutter Web app that talks to the admin endpoints: edit the minimum
supported version, publish release metadata, trigger a forced update and flip feature flags —
all taking effect for clients on their next check, with no app release.

### 3.4 `latest_version.json`

Both the server route `GET /latest_version.json` and the release pipeline's Pages artifact use
this name, and both carry exactly the §4 release fields. The server adds one extra field,
`generated_at`, because it regenerates the document per request and a consumer wants to know how
fresh it is; the Pages artifact is a static file for a specific release, so it is timestamped by
its `published_at` instead. `generated_at` is additive — the frozen contract is unchanged, and
nothing requires the field to be present.

---

## 4. CI/CD

### `build_and_release.yml` — triggered by a `v*.*.*` tag

1. Java 17 (temurin) + Flutter **3.24.x** stable.
2. **Gate:** `flutter analyze --fatal-infos --fatal-warnings`, then `flutter test`. Both run
   *before* the APK is built, so a broken build cannot be published as a release.
3. `flutter build apk --release` with the version derived from the tag.
4. Compute the APK's SHA256 and emit `SHA256SUMS`, `CHANGELOG.md`, `version.txt`, `tag.txt`.
5. Create the GitHub Release and upload the APK plus checksums.
6. Assemble `latest_version.json` and deploy it to **GitHub Pages**.

The `update-check` response points at that release, so the whole loop is: tag → APK → metadata
→ every installed client sees the update on next launch.

`actions/deploy-pages` replaces the **whole** site with the uploaded artifact, and the spec has
two pipelines deploying to it: this one owns `latest_version.json`, `deploy_update_server.yml`
owns the admin dashboard. Publishing only its own half would mean whichever ran last deleted the
other's files — a dashboard deploy 404s the manifest, a release 404s the dashboard. Both
workflows therefore call `.github/scripts/build_pages_site.sh` and each publishes the complete
site: the dashboard plus the manifest, with the manifest built from this workflow's build
outputs or, in the server workflow, reconstructed from the published GitHub Release.

The dashboard's absolute base href is derived from the Pages project path reported by
`actions/configure-pages`, because a project site is served from `/<repo>/`. Hard-coding
`/update-admin/` produced a dashboard whose `index.html` loaded but whose every asset was
requested from the domain root and 404'd.

The workflow also has a `workflow_dispatch` trigger taking `version` and a boolean `dry_run`.
`dry_run: true` runs steps 1–4 only and skips the Release and the Pages deploy, which is how
the APK build and the widget tests get verified on CI *without* cutting a tag. This matters
because `flutter analyze` treats info-level lints as fatal by default
(`flutter_tools/lib/src/commands/analyze.dart` defaults `--fatal-infos` to `true`), so the
analysis gate is stricter than it may look.

`apps/main_app/android/build.gradle` pins NDK `23.1.7779620` (the Flutter 3.24.5 default), but
the `ubuntu-latest` image ships only NDK 27/28/29, so the workflow installs the pinned revision
with `sdkmanager` before building rather than depending on AGP's implicit download.

Release-shape inputs come from repository **variables** (not secrets):
`HARBOR_API_BASE_URL`, `HARBOR_MIN_SUPPORTED_VERSION`, `HARBOR_PUBLISH_IMAGE`. Only
`secrets.GITHUB_TOKEN` is used.

> Pages is enabled on this repository with `build_type: workflow`, which is the setting the
> `configure-pages` / `deploy-pages` actions require. If it is ever turned off, those steps fail
> with `Get Pages site failed`.

### `deploy_update_server.yml`

Verifies the server, publishes the admin dashboard to Pages, and optionally builds a server
binary artifact (gated on `workflow_dispatch` or `vars.HARBOR_PUBLISH_IMAGE == 'true'`).

The verify job installs Flutter and resolves with `flutter pub get`, then runs
`dart analyze --fatal-infos --fatal-warnings` and `dart test` using the Dart CLI from that same
Flutter SDK. Using one SDK for both keeps the analyzer version identical to the one that builds
the web app, and running the server suite on the plain Dart VM via `dart test` (rather than
`flutter test`) is what demonstrates the headless claim.

The publish job compiles `bin/server.dart` to a native executable, starts it, polls `/health`,
exercises `/api/v1/update-check`, and tears it down. The teardown is bounded: `wait` has no
timeout of its own, so a server that ignores `SIGTERM` would hold the runner until the job
timeout — see bug 17 below.

---

## 5. Android release setup

The `ota_update` plugin requires a `FileProvider` whose authority matches what the Dart side
passes to it. If `androidProviderAuthority` is omitted, the plugin derives it as
`<applicationId>.ota_update_provider` — this project's manifest declares exactly
`com.harbor.main_app.ota_update_provider`, so no Dart-side override is needed.

**The provider root matters.** The plugin downloads to
`context.getApplicationInfo().dataDir + "/files/ota_update"`, i.e. `<filesDir>/ota_update/`.
A FileProvider root that does not cover that directory makes `getUriForFile` throw
`IllegalArgumentException: Failed to find configured root`, and the install silently never
starts. `res/xml/provider_paths.xml` therefore declares:

```xml
<paths xmlns:android="http://schemas.android.com/apk/res/android">
    <external-files-path name="external_files" path="." />
    <cache-path name="cache" path="." />
    <files-path name="files" path="." />
</paths>
```

The APK is already SHA256-verified in Dart before install, so a corrupt payload never reaches
the installer. `REQUEST_INSTALL_PACKAGES` is declared in the manifest (and merged in by the
plugin); on Android 8+ the user must still grant "install unknown apps" for Harbor.

Release builds must be signed. Provide a keystore via `android/key.properties`
(git-ignored) or let CI inject one.

---

## 6. Local development

### Client

```bash
cd apps/main_app
flutter pub get
flutter analyze
flutter test
flutter run --dart-define=HARBOR_API_BASE_URL=http://10.0.2.2:8080
```

`10.0.2.2` is the host loopback as seen from the Android emulator.

### Server

`apps/update_server` declares `flutter: sdk: flutter`, because `lib/main.dart` is the Flutter
Web admin dashboard. That single dependency means **`dart pub get` cannot resolve the package
at all**:

```
Because update_server requires the Flutter SDK, version solving failed.
Flutter users should use `flutter pub` instead of `dart pub`.
```

So dependencies are always resolved with the Flutter tool, and the Dart CLI bundled with that
same Flutter SDK is then used for everything else:

```bash
cd apps/update_server
flutter pub get
HARBOR_ADMIN_TOKEN=dev-token dart run bin/server.dart

# The suite covers only Flutter-free code and runs on the plain Dart VM.
dart test
dart analyze --fatal-infos --fatal-warnings

# The dashboard, and a headless binary of the server.
flutter build web --release --base-href /update-admin/
dart compile exe bin/server.dart -o build/harbor-update-server
```

Everything under `bin/` and `lib/src/` is Flutter-free by design, which is what keeps
`dart compile exe` and `dart test` working; `deploy_update_server.yml` asserts that invariant
with a `grep` gate so it cannot rot.

The admin dashboard is served from `build/web` when it exists, and otherwise the server
answers with a JSON landing page (so a bare server is still self-describing).

Shutdown on `SIGINT`/`SIGTERM` is bounded and guaranteed to terminate: the listener stops, the
`InFlightTracker` middleware gives requests already being served up to five seconds to finish,
whatever is left is force-closed, state is flushed, and the process then calls `exit(0)`. That
last step is not optional — the signal subscriptions keep the Dart event loop alive after
`main()` returns, so without it the process lingers (bug 17 below).

---

## 7. Verification status

Being explicit about the boundary here matters, because the Flutter *engine* could not run in
the environment this monorepo was assembled in (the Linux Flutter SDK ships x86-64 binaries
only, and the host is aarch64 with no emulation available). The Flutter framework's **Dart
sources**, however, were available and were used directly — see below.

### Verified by real execution

- **Dart 3.5.4**, the exact SDK Flutter 3.24.5 bundles, was installed and used for every check
  below, so nothing rests on a newer SDK accepting syntax that 3.5.4 would reject.
- **204 tests executed locally**: **170** covering the engine and agent layers — MLA (cache
  geometry, compression ratio, RoPE assertions), the MoE router (top-K selection, shared
  experts, capacity/dropping, aux-loss-free bias, balance entropy), LoRA (identity at init,
  merge/unmerge, learning a synthetic mapping, gradient clipping, replay bounds, JSON
  round-trip, every `IdleTrainingScheduler` gate), the universal file reader (magic-number
  sniffing, archives, structured formats, limits), the subagent runner (allow-list rejection,
  concurrency bounds, priority order, timeouts, cancellation, failures-as-values) and the
  plan/execute/verify agent (DAG waves, verification verdicts, replanning) — plus **34**
  covering the update server, driven through real HTTP over loopback: the zero payload,
  update-decision math (including pre-release ordering), validation, admin auth (401/503), flag
  merging, `latest_version.json`, and the shutdown drain tracker.
- On CI the full client suite runs: **260 tests**, widget tests included.
- The **whole client tree — `lib/` *and* `test/`** — analyzes clean with
  `dart analyze --fatal-infos --fatal-warnings` under the repository's real
  `analysis_options.yaml`, resolving `package:flutter/material.dart` against the **real Flutter
  3.24.5 framework sources** plus `sky_engine`, and `package:flutter_test` against the **real
  `flutter_test` package** (including its `leak_tracker_flutter_testing` dependent). The widget
  tests are therefore compile-verified, not merely syntax-checked.
- The **update server layer** analyzes clean under its own stricter config, which adds
  `strict-raw-types` and `avoid_dynamic_calls`.
- Both workflow YAML files parse cleanly and all `run:` scripts pass `bash -n`.
- Every dependency constraint was checked against the pub.dev index for **Dart 3.5.4 /
  Flutter 3.24.5** — all resolve, including `ota_update ^6.0.0` (the last release compatible
  with Dart < 3.7; 7.x requires Dart ≥ 3.7).
- `update_service.dart` and `dynamic_feature_flag_provider.dart` were **executed** against a
  minimal stand-in for the two Flutter types they use (`@immutable`, `ChangeNotifier`).

### Proven on GitHub Actions

The first runs were not a formality — they failed, and fixing them is what the tables below
record. The pipeline now runs green end to end, including a real, published release. Every claim
in this section is an observed result, not a prediction:

| Workflow / job | Result |
| --- | --- |
| `build_and_release.yml` → *Build release APK* | **passes** — `flutter analyze --fatal-infos --fatal-warnings`, **260 tests**, `flutter build apk --release` (24.1 MB / 24,079,567 bytes), SHA256 + `SHA256SUMS`, artifact upload |
| `build_and_release.yml` → *Publish GitHub Release* | **passes** — created release `v1.0.0` with `app-release.apk` and `SHA256SUMS` |
| `build_and_release.yml` → *Publish update manifest to GitHub Pages* | **passes** — assembled the complete site and deployed it |
| `deploy_update_server.yml` → *Analyze and test update server* | **passes** — `flutter pub get`, the Flutter-free gate, `dart analyze --fatal-infos --fatal-warnings`, **34 tests** via `dart test` |
| `deploy_update_server.yml` → *Build and deploy admin dashboard* | **passes** — `flutter build web --release` and the Pages deploy |
| `deploy_update_server.yml` → *Compile and smoke-test shelf server* | **passes** — `dart compile exe`, `/health` + `/api/v1/update-check` over real HTTP, clean SIGTERM teardown |

The published artifacts are live and were verified over HTTP after the fact:

| Artifact | Check |
| --- | --- |
| `https://c4rl-daniel.github.io/Harbor/latest_version.json` | returns the frozen §4 payload: `latest_version 1.0.0`, `update_available true`, `update_required false`, `size_bytes 24079567` |
| `…/Harbor/SHA256SUMS` | `bffab2bf…c44bf  app-release.apk` |
| Release asset `app-release.apk` | downloaded and hashed locally → `bffab2bf…c44bf`, identical to both the checksum file and the manifest, so the integrity chain the client walks is real |
| `…/Harbor/update-admin/` | `base href="/Harbor/update-admin/"`, and `main.dart.js` / `flutter_bootstrap.js` return 200 |
| Landing page | links all three artifacts, and the dashboard is *still* served after a release deploy rewrote the site |

### Bugs this process found and fixed

Static analysis against the real Flutter 3.24.5 framework caught five compile errors before the
first push that would have failed `flutter build apk` on the first tag:

| File | Error | Fix |
| --- | --- | --- |
| `main.dart` | `CardThemeData` does not exist in Flutter 3.24.5 (only `CardTheme`) | use `CardTheme` |
| `update_service.dart` | `Digest.fromHex` does not exist in `package:crypto` | decode the 64-char hex explicitly |
| `update_service.dart` | `PlatformException` is in `services.dart`, not `foundation.dart` | add the `services.dart` import |
| `update_dialog.dart` | unused `ota_update` import (an error under this config) | remove it |
| `file_readers.dart` | archive 3.x `ArchiveFile` has no `compressedSize`; `GZipDecoder()` is not const; `loadYaml` was never imported | report `crc32` instead; drop `const`; add the import |

The first real CI runs then exposed a second class of failure — a mismatch between what was
*assumed* about the toolchain and what it actually does:

| # | Where | Problem | Fix |
| --- | --- | --- | --- |
| 1 | `deploy_update_server.yml` | both server jobs used `dart-lang/setup-dart`, but the package declares `flutter: sdk: flutter` (for `lib/main.dart`), so `dart pub get` **can never resolve it** — the run died at *Resolve Dart dependencies* | install Flutter 3.24.5 and resolve with `flutter pub get`, then use that SDK's bundled Dart CLI |
| 2 | `update_server/lib/main.dart` | `Version` used without importing `pub_semver` — an outright compile error that broke `flutter build web` | add the import |
| 3 | `update_server/lib/main.dart` | `dart:html` is deprecated and trips `avoid_web_libraries_in_flutter` under `--fatal-infos` | migrate the two `localStorage` call sites to `package:web` |
| 4 | `update_server/test/server_test.dart` | imported `flutter_test` for tests that only exercise Flutter-free code, which makes `dart test` impossible and drags `dart:ui` into a server suite | import `package:test`; drop the unused `flutter_test` dev-dependency |
| 5 | `android/app/build.gradle` | the NDK fallback literal (`26.1.10909125`) contradicted both its own comment and `FlutterExtension`, which pins `23.1.7779620` | align the fallback with the real pin |
| 6 | `test/*.dart` | 35 info-level lints. `flutter analyze` defaults `--fatal-infos` to **true**, so these blocked the release gate | applied the `dart fix` `const`/super-parameter fixes and made the flags explicit in the workflow |
| 7 | `test/update_widgets_test.dart` | the fake's `strategy` parameter was never supplied (`unused_element`). The obvious fix is to delete it — but `UpdateFlowController.start` branches on that strategy, so deleting it would have removed the only way to reach the `verifying` stage | keep it as `super.strategy` and add a test that exercises the branch |
| 8 | `README.md` | the `provider_paths.xml` snippet did not match the file, and the server instructions said `dart pub get` — the exact command that fails | correct both |

A note on #1, because it is the instructive one: local validation had passed against a mirror in
which the `flutter` dependency was *stripped* so that `dart pub get` would work. The mirror
therefore verified a package that was not the one being shipped. The fix was to make the mirror
faithful — the real Flutter framework as a `path` dependency, with the SDK-sourced dependents
(`flutter_test`, `leak_tracker_flutter_testing`) repointed the same way — and to re-run. It then
reproduced CI's findings exactly, including the same 35 issues at the same line and column,
which is what made them safe to fix locally.

Finally, the first *executed* runs — the first `flutter test` and the first `assembleRelease` —
found a third class: bugs that no amount of static analysis would have surfaced.

| # | Where | Problem | Fix |
| --- | --- | --- | --- |
| 9 | `dynamic_feature_flag_provider.dart` | the documented public `changes` broadcast stream **never emitted**: `_notify()` only called `notifyListeners()`, nothing ever added to the controller, yet `dispose()` closed it | add `_emitChange()` and call it from the three sites that actually replace the matrix (cache restore, seed install, network refresh), so a subscriber sees one event per change |
| 10 | `test/update_widgets_test.dart` | the test asserted `Navigator.maybePop()` returns `false` for a vetoed pop. It returns **`true`** — `RoutePopDisposition.doNotPop` is still a *handled* request — so the assertion tested the wrong thing while the app was correct | assert that the dialog survived instead of asserting the return value |
| 11 | `test/feature_flags_test.dart` | the `describe()` test called only `initialize()`, which by design never fetches, so `matrix_version` stayed `0` | call `refresh()` too, which is what the app does at startup |
| 12 | `android/build.gradle` | `assembleRelease` died in `:ota_update:verifyReleaseResources` with `AAPT: error: resource android:attr/lStar not found`. Flutter plugins are independent Gradle modules: `ota_update` 6.0.0 still declares `compileSdkVersion 28` while depending on a modern `androidx.core` whose resources reference an API-31 attribute, and AAPT resolves resources against the *module's* SDK | align every Android module's `compileSdk` with the app's, read back from `:app` so no second SDK number is hard-coded |
| 13 | `android/build.gradle` | the first version of that alignment registered its `afterEvaluate` hook **below** `subprojects { evaluationDependsOn(":app") }`, which had already forced `:app` to evaluate — Gradle refuses that with *"Cannot run Project.afterEvaluate(Closure) when the project is already evaluated"* | register the hook at the top of the script, which is also what makes the ordering correct (before AGP reads the DSL to create variants) |
| 14 | `build_and_release.yml` | the runner image ships NDK 27/28/29, but the app pins `23.1.7779620`, so the build depended on AGP's implicit SDK download and would have failed late in `:app:stripReleaseDebugSymbols` | install the pinned NDK with `sdkmanager` first, as a no-op when it is already there |
| 15 | both workflows | `actions/deploy-pages` replaces the **entire** site, so the two pipelines deleted each other's files: a dashboard deploy 404'd `latest_version.json`, a release 404'd the dashboard | one shared builder (`.github/scripts/build_pages_site.sh`) that both workflows call; each now publishes the complete site, reconstructing the manifest from the GitHub Release when it is the server workflow's turn |
| 16 | both workflows | the dashboard was built with `--base-href /update-admin/`, but a project Pages site is served from `/<repo>/`, so `index.html` loaded and **every asset 404'd**: `/update-admin/flutter_bootstrap.js` → 404 while `/Harbor/update-admin/flutter_bootstrap.js` → 200 | derive the absolute href from `actions/configure-pages`'s `base_path`, and stage the dashboard into only the last path segment so the artifact does not gain a second `/<repo>/` level |
| 17 | `update_server/bin/server.dart` | the compiled server **never exited on SIGTERM**. The signal handlers ran and reported `state flushed`, but the `ProcessSignal` subscriptions keep the Dart event loop alive after `main()` returns, so the smoke test's `wait` blocked — the job sat *in progress* for 20+ minutes. Reproduced locally: the same probe returns from `wait` in 0.25 s with the fix and never returns without it | call `exit(0)` once the shutdown work has finished and stdout is flushed, and make the shutdown itself bounded (stop the listener → drain in-flight requests through the new `InFlightTracker` → force-close), so no step can stall the exit |
| 18 | `deploy_update_server.yml` | the smoke test's teardown used an unbounded `wait`, which is what turned #17 into a hung runner | poll for up to 10 s, comparing `/proc` state (because `kill -0` also succeeds for a zombie), then escalate to `SIGKILL`; every job in both workflows also gets a `timeout-minutes` ceiling |

### Limits of this verification

- The client's 260 tests and the APK build execute **only on GitHub's x86-64 runners**. The
  environment this repository was assembled in has an aarch64 host and an x86-64-only Flutter
  engine, so `flutter test` and `flutter build apk` cannot run locally; the local mirrors are
  used for analysis and for executing the Flutter-free code.
- The `flutter_test`-based widget tests are executed on CI (they need `dart:ui`). Locally they
  are compile-verified against the real `flutter_test` package, not run.
- The admin dashboard is built and served, and its assets were fetched over HTTP, but no browser
  has rendered it here; nothing verifies its visual layout or its WebSocket-free polling loop.
- The release build falls back to **debug signing** when `android/key.properties` is absent,
  which is what CI produced. That APK installs, but it is not a Play-Store-ready artifact — see
  §5.
- The smoke test exercises `/health` and `/api/v1/update-check`; the admin mutations are covered
  by the 34-test suite over real HTTP, not by the smoke test.

---

## 8. Security notes

- The `SystemCommandSubagent` is the only path that executes anything, and it does so with
  `runInShell: false`, an executable allow-list and shell-metacharacter rejection. Adding an
  executable to that list is a deliberate, reviewable act.
- Update artifacts are SHA256-verified before the installer is invoked, and a mismatch
  deletes the partial download.
- Admin endpoints are token-gated; the server refuses admin mutations outright when no token
  is configured (unless explicitly opted into dev mode).
- Keep the admin token, keystore and `key.properties` out of the repository — `.gitignore`
  covers `.env*`, `*.jks`, `*.keystore` and `key.properties`.
- Rotate any credential that has been shared in plaintext.