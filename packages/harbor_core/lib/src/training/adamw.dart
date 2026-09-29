// AdamW.
//
// The optimiser operates on an explicit `List<ParamRef>` rather than reaching
// into the model, for the same reason the forward and backward passes thread
// their caches explicitly: a model that owns its optimiser state cannot share
// parameters between two training runs, and the adapter experiment (train a
// frozen base plus a fresh low-rank branch) needs exactly that.
//
// "Decoupled" weight decay means the decay is applied to the parameter directly
// after the adaptive step, not added to the gradient. Adding it to the gradient
// makes the effective decay proportional to the adaptive denominator, which
// silently weakens regularisation on rarely-updated parameters.

import 'dart:math' as math;
import 'dart:typed_data';

import '../model/linalg.dart';

/// Adam with decoupled weight decay and bias correction.
class AdamW {
  /// Creates an optimiser over [parameters].
  ///
  /// The parameter list is copied so later mutation of the caller's list cannot
  /// desynchronise it from the moment buffers allocated here.
  AdamW({
    required List<ParamRef> parameters,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.epsilon = 1e-8,
    this.weightDecay = 0.01,
  })  : _parameters = List<ParamRef>.of(parameters),
        _firstMoment = <Float64List>[
          for (final ParamRef parameter in parameters)
            Float64List(parameter.values.length),
        ],
        _secondMoment = <Float64List>[
          for (final ParamRef parameter in parameters)
            Float64List(parameter.values.length),
        ] {
    if (_parameters.isEmpty) {
      throw ArgumentError.value(parameters, 'parameters', 'must not be empty');
    }
    if (beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1) {
      throw ArgumentError('beta1 and beta2 must lie in [0, 1)');
    }
    if (epsilon <= 0 || weightDecay < 0) {
      throw ArgumentError('epsilon must be positive and weightDecay non-negative');
    }
  }

  /// Exponential decay rate of the first moment.
  final double beta1;

  /// Exponential decay rate of the second moment.
  final double beta2;

  /// Numerical floor in the denominator.
  final double epsilon;

  /// Decoupled weight-decay coefficient.
  final double weightDecay;

  final List<ParamRef> _parameters;
  final List<Float64List> _firstMoment;
  final List<Float64List> _secondMoment;
  int _step = 0;

  /// Number of [step] calls applied so far, used for bias correction.
  int get stepCount => _step;

  /// The global L2 norm of every gradient in the parameter list.
  double globalGradientNorm() {
    double sum = 0;
    for (final ParamRef parameter in _parameters) {
      final Float64List grads = parameter.grads;
      for (int i = 0; i < grads.length; i++) {
        sum += grads[i] * grads[i];
      }
    }
    return math.sqrt(sum);
  }

  /// Scales every gradient down so the global norm is at most [maxNorm].
  ///
  /// Returns the norm measured *before* scaling, which is the number worth
  /// showing in a diagnostics panel: clipping hides a spike, and the spike is
  /// the signal that something went wrong.
  double clipByGlobalNorm(double maxNorm) {
    if (maxNorm <= 0) {
      throw ArgumentError.value(maxNorm, 'maxNorm', 'must be positive');
    }
    final double norm = globalGradientNorm();
    if (norm <= maxNorm || norm == 0) {
      return norm;
    }
    final double scale = maxNorm / norm;
    for (final ParamRef parameter in _parameters) {
      final Float64List grads = parameter.grads;
      for (int i = 0; i < grads.length; i++) {
        grads[i] *= scale;
      }
    }
    return norm;
  }

  /// Applies one update at [learningRate].
  ///
  /// Returns the global gradient norm measured before the update, so a caller
  /// can log gradient health without a second pass. Note that this is the norm
  /// *after* any clipping the caller already applied.
  double step({required double learningRate}) {
    if (learningRate <= 0) {
      throw ArgumentError.value(learningRate, 'learningRate', 'must be positive');
    }
    _step++;
    final double norm = globalGradientNorm();
    final double correction1 = 1 - math.pow(beta1, _step).toDouble();
    final double correction2 = 1 - math.pow(beta2, _step).toDouble();
    for (int p = 0; p < _parameters.length; p++) {
      final ParamRef parameter = _parameters[p];
      final Float64List values = parameter.values;
      final Float64List grads = parameter.grads;
      final Float64List moment1 = _firstMoment[p];
      final Float64List moment2 = _secondMoment[p];
      for (int i = 0; i < values.length; i++) {
        final double grad = grads[i];
        final double m = moment1[i] = beta1 * moment1[i] + (1 - beta1) * grad;
        final double v =
            moment2[i] = beta2 * moment2[i] + (1 - beta2) * grad * grad;
        final double mHat = m / correction1;
        final double vHat = v / correction2;
        final double update = mHat / (math.sqrt(vHat) + epsilon);
        values[i] -= learningRate * (update + weightDecay * values[i]);
      }
    }
    return norm;
  }

  /// Zeroes every gradient in the parameter list.
  void zeroGrad() {
    for (final ParamRef parameter in _parameters) {
      parameter.grads.fillRange(0, parameter.grads.length, 0);
    }
  }

  /// Diagnostic snapshot: step count and the norms of the moment buffers.
  ///
  /// A second-moment norm that is orders of magnitude below the first-moment
  /// norm means the gradients are almost constant, which is the signature of a
  /// model that has collapsed to predicting one token.
  Map<String, Object?> stats() {
    double first = 0;
    double second = 0;
    for (int p = 0; p < _parameters.length; p++) {
      final Float64List m = _firstMoment[p];
      final Float64List v = _secondMoment[p];
      for (int i = 0; i < m.length; i++) {
        first += m[i] * m[i];
        second += v[i] * v[i];
      }
    }
    return <String, Object?>{
      'step': _step,
      'first_moment_norm': math.sqrt(first),
      'second_moment_norm': math.sqrt(second),
      'gradient_norm': globalGradientNorm(),
      'parameters': _parameters.length,
    };
  }

  @override
  String toString() =>
      'AdamW(step $_step, ${_parameters.length} tensors, wd $weightDecay)';
}