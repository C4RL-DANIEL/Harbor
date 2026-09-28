# Harbor

A production-grade, dual-app Flutter ecosystem in a single monorepo:

| Component | Path | What it is |
| --- | --- | --- |
| **Harbor** (client) | `apps/main_app` | A Flutter Android app with an on-device inference engine (MLA + fine-grained MoE + LoRA self-learning), an extended-thinking autonomous agent, a universal file reader, and a self-updating OTA engine. |
| **Harbor Update Server** | `apps/update_server` | A Dart `shelf` service that serves release metadata, the dynamic feature-flag matrix, and a Flutter Web admin dashboard for publishing releases and flipping flags live. |
| **Release pipeline** | `.github/workflows` | Tag-driven APK build → SHA256 → GitHub Release → `latest_version.json` → GitHub Pages, plus a server deploy pipeline. |

The frozen interface between the two apps — the HTTP wire contract, the feature-flag matrix schema and the release-metadata schema — is specified in [`MASTER_PROMPT.txt`](MASTER_PROMPT.txt) §4. Treat it as normative: the client parses exactly those shapes and the server emits exactly those shapes.

---

## 1. Repository layout

```
.
├── .github/workflows/
│   ├── build_and_release.yml          # v*.*.* tag -> signed APK -> Release -> Pages
│   └── deploy_update_server.yml       # CI for the server + admin web dashboard
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
│       │   └── src/                   # models, release_store, api_router (no Flutter imports)
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

The release pipeline publishes `latest_version.json` to GitHub Pages containing every
`update-check` field plus `generated_at`, so the server (or a static host) can be seeded
directly from the release artifact.

---

## 4. CI/CD

### `build_and_release.yml` — triggered by a `v*.*.*` tag

1. Java 17 (temurin) + Flutter **3.24.x** stable.
2. **Gate:** `flutter analyze --fatal-infos --fatal-warnings`, then `flutter test`. Both run
   *before* the APK is built, so a broken build cannot be published as a release.
3. `flutter build apk --release` with the version derived from the tag.
4. Compute the APK's SHA256 and emit `SHA256SUMS`, `CHANGELOG.md`, `version.txt`, `tag.txt`.
5. Create the GitHub Release and upload the APK plus checksums.
6. Generate `latest_version.json` and deploy it to **GitHub Pages**.

The `update-check` response points at that release, so the whole loop is: tag → APK → metadata
→ every installed client sees the update on next launch.

The workflow also has a `workflow_dispatch` trigger taking `version` and a boolean `dry_run`.
`dry_run: true` runs steps 1–4 only and skips the Release and the Pages deploy, which is how
the APK build and the widget tests get verified on CI *without* cutting a tag. This matters
because `flutter analyze` treats info-level lints as fatal by default
(`flutter_tools/lib/src/commands/analyze.dart` defaults `--fatal-infos` to `true`), so the
analysis gate is stricter than it may look.

Release-shape inputs come from repository **variables** (not secrets):
`HARBOR_API_BASE_URL`, `HARBOR_MIN_SUPPORTED_VERSION`, `HARBOR_PUBLISH_IMAGE`. Only
`secrets.GITHUB_TOKEN` is used.

> Pages is enabled on this repository with `build_type: workflow`, which is the setting the
> `configure-pages` / `deploy-pages` actions require. If it is ever turned off, those steps fail
> with `Get Pages site failed`.

### `deploy_update_server.yml`

Verifies the server, builds the Flutter Web admin dashboard to Pages
(`base-href /update-admin/`), and optionally publishes a server binary artifact (gated on
`workflow_dispatch` or `vars.HARBOR_PUBLISH_IMAGE == 'true'`).

The verify job installs Flutter and resolves with `flutter pub get`, then runs
`dart analyze --fatal-infos --fatal-warnings` and `dart test` using the Dart CLI from that same
Flutter SDK. Using one SDK for both keeps the analyzer version identical to the one that builds
the web app, and running the server suite on the plain Dart VM via `dart test` (rather than
`flutter test`) is what demonstrates the headless claim.

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
answers with a JSON landing page (so a bare server is still self-describing). Shutdown is
graceful on `SIGINT`/`SIGTERM`.

---

## 7. Verification status

Being explicit about the boundary here matters, because the Flutter *engine* could not run in
the environment this monorepo was assembled in (the Linux Flutter SDK ships x86-64 binaries
only, and the host is aarch64 with no emulation available). The Flutter framework's **Dart
sources**, however, were available and were used directly — see below.

### Verified by real execution

- **Dart 3.5.4**, the exact SDK Flutter 3.24.5 bundles, was installed and used for every check
  below, so nothing rests on a newer SDK accepting syntax that 3.5.4 would reject.
- **200 tests pass**:
  - **170** covering the engine and agent layers — MLA (cache geometry, compression ratio,
    RoPE assertions), the MoE router (top-K selection, shared experts, capacity/dropping,
    aux-loss-free bias, balance entropy), LoRA (identity at init, merge/unmerge, learning a
    synthetic mapping, gradient clipping, replay bounds, JSON round-trip, every
    `IdleTrainingScheduler` gate), the universal file reader (magic-number sniffing, archives,
    structured formats, limits), the subagent runner (allow-list rejection, concurrency
    bounds, priority order, timeouts, cancellation, failures-as-values) and the
    plan/execute/verify agent (DAG waves, verification verdicts, replanning).
  - **30** covering the update server, driven through real HTTP over loopback: the zero
    payload, update-decision math (including pre-release ordering), validation, admin auth
    (401/503), flag merging, and `latest_version.json`.
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

The first runs were not a formality — they failed, and fixing them is what the second table
below records. Current state:

| Workflow / job | Result |
| --- | --- |
| `deploy_update_server.yml` → *Analyze and test update server* | **passes** — `flutter pub get`, the Flutter-free gate, `dart analyze --fatal-infos --fatal-warnings`, and `dart test` (30 tests) |
| `deploy_update_server.yml` → *Build and deploy admin dashboard* | `flutter build web --release` **passes**; the Pages publish step failed only because Pages was not enabled on the repository |
| `build_and_release.yml` → *Build release APK* | reaches `flutter pub get` successfully; the analysis gate then failed on 35 info-level lints, now fixed |
| `build_and_release.yml` → *Release* / *Pages* | correctly **skipped** under `dry_run` |

GitHub Pages is now enabled on the repository with `build_type: workflow`, so neither deploy
depends on a manual settings change any more.

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

### Still not verified here — the next CI run is authoritative

- `flutter test` (executing the 11 widget tests) and `flutter build apk --release` have not yet
  run to completion. The analysis gate that previously blocked them now passes, so the next
  dispatch or tag reaches them.
- The widget tests are compile-verified and were audited for the usual runtime traps: they pump
  `MaterialApp` wrappers rather than the real app, use hand-written fakes and a `MockClient`
  (no plugins, no network, no `Platform`/`dart:io` branching), and their two `pumpAndSettle`
  calls wrap a dialog route transition over static content, so they should settle.
- The APK build is the remaining unknown: R8 with `shrinkResources` under
  `android/app/proguard-rules.pro`, and whether the runner provides NDK `23.1.7779620`.
- `apps/update_server/lib/main.dart` compiles (`flutter build web` passed on CI) but has never
  been *run*.

A first-run failure in the widget tests or the APK build blocks the release rather than shipping
it, which is the intended behaviour.

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