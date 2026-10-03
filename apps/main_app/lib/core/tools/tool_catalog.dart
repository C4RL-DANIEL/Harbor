// The tool catalogue the assistant is given.
//
// Extracted from `main.dart` because the engine is now rebuilt whenever a
// capability switch changes, and rebuilding must not re-register a tool whose
// closure captured a stale host. One function owns "what tools exist", so the
// composition root and the settings-driven rebuild path produce the identical
// registry and cannot drift apart.
//
// The registry is constructed from a [PlatformBridge] and an HTTP client, both
// of which outlive any single engine, which is what makes a rebuild cheap and
// safe.

import 'package:harbor_core/harbor_core.dart';
import 'package:http/http.dart' as http;

import '../platform/platform_bridge.dart';
import 'tools_controller.dart';

/// Builds every tool this build of Harbor can offer.
///
/// Device tools reach the phone through the platform channel; web tools run in
/// process. HTTPS-only and a byte ceiling are enforced here rather than inside
/// the widgets so a screen cannot accidentally widen the policy.
ToolRegistry buildToolRegistry({
  required PlatformBridge bridge,
  required http.Client client,
}) {
  return ToolRegistry(<Tool>[
    ...deviceTools(hostCallerFrom(bridge)),
    ...webTools(
      client: client,
      // The collector policy already restricts the corpus pipeline; the chat
      // tools stay on https so a fetched page cannot downgrade mid-conversation.
      allow: (Uri uri) => uri.scheme == 'https',
      maxBytes: 256 * 1024,
    ),
  ]);
}
