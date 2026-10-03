// The one-tap setup card.
//
// Harbor's pipeline has an order a new user cannot guess: text must be gathered
// before a tokenizer can be learned, the tokenizer sizes the model, and the
// model has to be trained before answers mean anything. Each of those lives on
// its own tab, so the honest default experience was four tabs and a blank chat.
//
// This card collapses that into one button. It shows the *real* state of each
// step (how much text exists, whether a model is built, how many memories have
// been learned) rather than a checklist that guesses, and it stays on screen
// after the run so the background-learn switch is discoverable from the same
// place. When the pipeline cannot proceed — no text on the device — it says so
// and names the fix instead of spinning forever.

import 'dart:async';

import 'package:flutter/material.dart';

import '../../core/auto/auto_pilot.dart';

/// A card that runs [AutoPilotController.runSetup] and reports its state.
class OneTapSetupCard extends StatelessWidget {
  /// Creates the card over [autoPilot].
  const OneTapSetupCard({super.key, required this.autoPilot});

  /// The pipeline being driven.
  final AutoPilotController autoPilot;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: autoPilot,
      builder: (BuildContext context, Widget? child) {
        final AutoPilotStatus status = autoPilot.status;
        final ColorScheme colors = Theme.of(context).colorScheme;
        final TextTheme text = Theme.of(context).textTheme;
        final bool failed = status.stage == AutoStage.failed;

        return Card(
          key: const ValueKey<String>('setup.card'),
          elevation: 0,
          color: failed
              ? colors.errorContainer
              : colors.primaryContainer.withOpacity(0.45),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Row(
                  children: <Widget>[
                    Icon(
                      failed
                          ? Icons.error_outline
                          : status.stage == AutoStage.ready
                              ? Icons.verified_outlined
                              : Icons.auto_awesome,
                      color: failed ? colors.onErrorContainer : colors.primary,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        status.stage == AutoStage.ready
                            ? 'Harbor is running on this device'
                            : 'Get Harbor working',
                        style: text.titleSmall,
                      ),
                    ),
                    if (status.running)
                      const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                  ],
                ),
                const SizedBox(height: 6),
                Text(
                  status.message,
                  style: text.bodySmall?.copyWith(
                    color: failed ? colors.onErrorContainer : null,
                  ),
                ),
                if (status.running) ...<Widget>[
                  const SizedBox(height: 10),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(value: status.progress),
                  ),
                ],
                const SizedBox(height: 10),
                Wrap(
                  spacing: 12,
                  runSpacing: 4,
                  children: <Widget>[
                    _Fact(
                      label: 'corpus',
                      value: '${(status.corpusChars / 1000).toStringAsFixed(1)}k chars',
                    ),
                    _Fact(
                      label: 'model',
                      value: status.modelReady ? 'built' : 'not built',
                    ),
                    _Fact(label: 'memories', value: '${status.memories}'),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: <Widget>[
                    FilledButton.icon(
                      key: const ValueKey<String>('setup.run'),
                      onPressed:
                          status.running ? null : () => unawaited(autoPilot.runSetup()),
                      icon: status.running
                          ? const SizedBox(
                              width: 14,
                              height: 14,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.play_arrow),
                      label: Text(
                        status.running
                            ? 'Setting up…'
                            : status.stage == AutoStage.ready
                                ? 'Run again'
                                : 'Set everything up',
                      ),
                    ),
                    const SizedBox(width: 12),
                    if (status.stage == AutoStage.ready && !status.running)
                      Row(
                        mainAxisSize: MainAxisSize.min,
                        children: <Widget>[
                          Switch(
                            key: const ValueKey<String>('setup.autolearn'),
                            value: autoPilot.autoLearn,
                            onChanged: (bool value) =>
                                autoPilot.setAutoLearn(value),
                          ),
                          Text('Keep learning', style: text.bodySmall),
                        ],
                      ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _Fact extends StatelessWidget {
  const _Fact({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final TextTheme text = Theme.of(context).textTheme;
    return Semantics(
      label: '$label: $value',
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          Text('$label ', style: text.bodySmall?.copyWith(fontWeight: FontWeight.w600)),
          Text(value, style: text.bodySmall),
        ],
      ),
    );
  }
}
