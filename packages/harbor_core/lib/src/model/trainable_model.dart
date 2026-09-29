// The training-side contract.
//
// `LanguageModelRuntime` (in `interfaces.dart`) is what a *consumer* needs to
// generate text. This file adds what a *trainer* needs: parameter enumeration,
// gradient control, adapter management, and a differentiable window step. The
// split exists so the chat engine and the web dashboard can depend on the small
// read-only surface and never accidentally hold a gradient buffer alive.

import 'dart:math' as math;

import 'interfaces.dart';
import 'linalg.dart';

/// What one differentiable window produced.
///
/// Loss and accuracy are reported unscaled — they describe the window as given,
/// independent of any gradient-accumulation scale factor — so a training curve
/// is comparable across different batch sizes.
class TrainingOutcome {
  /// Creates an outcome.
  const TrainingOutcome({
    required this.loss,
    required this.correct,
    required this.predicted,
    required this.tokens,
    required this.gradNorm,
  });

  /// Mean cross-entropy over the window, in nats.
  final double loss;

  /// Number of positions whose arg-max prediction matched the target.
  final int correct;

  /// Number of scored positions, i.e. `tokens - 1`.
  final int predicted;

  /// Tokens consumed, including the final one that is only ever a target.
  final int tokens;

  /// Global gradient norm measured after this window was accumulated.
  final double gradNorm;

  /// Fraction of positions predicted correctly.
  double get accuracy => predicted == 0 ? 0 : correct / predicted;

  /// `exp(loss)`, clamped so a diverging run reports a large finite number
  /// rather than infinity.
  double get perplexity => loss > 30 ? math.exp(30) : math.exp(loss);

  /// JSON form for the live progress stream.
  Map<String, Object?> toJson() => <String, Object?>{
        'loss': loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'correct': correct,
        'predicted': predicted,
        'tokens': tokens,
        'grad_norm': gradNorm,
      };

  @override
  String toString() => 'TrainingOutcome(loss: ${loss.toStringAsFixed(4)}, '
      'ppl: ${perplexity.toStringAsFixed(2)}, '
      'acc: ${(accuracy * 100).toStringAsFixed(1)}%)';
}

/// A causal language model that can be trained in place.
///
/// Implementations own their parameters outright: [parameters] returns live
/// buffers, so an optimiser mutates the model directly rather than through a
/// copy. That is what makes on-device training possible without a second full
/// copy of the weights in memory.
abstract class TrainableLanguageModel implements LanguageModelRuntime {
  /// Every trainable tensor, in a deterministic order.
  ///
  /// When an adapter is attached this excludes the frozen base weights, which
  /// is exactly the set the optimiser should see.
  List<ParamRef> get parameters;

  /// Zeroes every gradient buffer in the model.
  void zeroGrad();

  /// The global L2 norm of every gradient concatenated.
  double gradNorm();

  /// Multiplies every gradient by [factor].
  ///
  /// Used by gradient accumulation: each window's gradient is scaled by
  /// `1/batchSize` before the optimiser steps, so the accumulated gradient is
  /// the mean over the batch.
  void scaleGradients(double factor);

  /// Trainable scalars, honouring any attached adapter.
  int get trainableParameterCount;

  /// Scalars that exist but are frozen while an adapter is attached.
  int get frozenParameterCount;

  /// Total scalars in the model, trainable or not.
  int get totalParameterCount;

  /// True when a LoRA adapter is attached and receiving gradients.
  bool get hasAdapter;

  /// Rank of the attached adapter, or zero.
  int get adapterRank;

  /// Trainable scalars inside the attached adapter.
  int get adapterParameterCount;

  /// The adapter's tensors, empty when no adapter is attached.
  List<ParamRef> get adapterParameters;

  /// Attaches a fresh adapter to every linear projection.
  ///
  /// The base weights stop receiving gradients, so training switches from
  /// "fit the model" to "fit a small correction to a fixed model" — the
  /// continual-learning mode. [seed] makes the adapter initialisation
  /// reproducible.
  void attachAdapter(int rank, {double alpha, int? seed});

  /// Removes the adapter and restores full fine-tuning.
  void detachAdapter();

  /// Folds the adapter into the base weights and detaches it.
  ///
  /// Afterwards [logitsFor] produces exactly what it produced while the adapter
  /// was attached, at no extra cost.
  void mergeAdapter();

  /// Runs one window forward and backward, accumulating gradients.
  ///
  /// [tokens] must be a contiguous window: position `i` is scored against
  /// `tokens[i + 1]`, so a window of `n` tokens yields `n - 1` predictions.
  /// Gradients are added to whatever is already in the buffers; call
  /// [zeroGrad] first for a fresh batch. [lossScale] multiplies the gradient
  /// (not the reported loss).
  TrainingOutcome accumulateWindow(List<int> tokens, {double lossScale = 1.0});

  /// Mean cross-entropy of [tokens] without touching any gradient buffer.
  double evaluateWindow(List<int> tokens);

  /// Samples a continuation of [prompt].
  ///
  /// Stops after [maxNewTokens], when [stopToken] is sampled, or when the
  /// context window is full. [onToken] is called with each produced id so a UI
  /// can stream the result.
  List<int> generate(
    List<int> prompt, {
    int maxNewTokens,
    Sampler? sampler,
    int? stopToken,
    void Function(int token)? onToken,
  });

  /// Serialises every weight and the configuration.
  Map<String, Object?> toJson();

  /// Restores weights written by [toJson], validating every shape.
  void loadJson(Map<String, Object?> json);
}