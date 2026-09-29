// Dense numerical primitives for the on-device model runtime.
//
// Everything here is hand-rolled on top of `Float64List` for three reasons:
//
//  * the package must compile to JavaScript for the Flutter Web dashboard, so a
//    native BLAS binding is not an option;
//  * the forward *and* backward passes have to be readable side by side, which
//    an opaque third-party tensor library would make harder, not easier;
//  * `double` (not `float`) is used throughout so that numerical gradient
//    checking in the test suite is meaningful — float32 rounding noise is
//    roughly the same size as the finite-difference error we want to measure.
//
// Layout convention: every [Matrix] is row-major with `rows` rows of `cols`
// contiguous doubles, and `values[r * cols + c]` is element (r, c). A "vector"
// is a plain `Float64List` of length `n`.

import 'dart:math' as math;
import 'dart:typed_data';

/// A dense row-major matrix that owns both its values and their gradients.
///
/// Parameters and gradients live in the same object, in parallel `Float64List`s
/// of identical length. That keeps the optimiser trivial (it walks a flat list
/// of tensors) and makes it impossible to accidentally apply an update to a
/// gradient whose shape drifted from its parameter.
class Matrix {
  /// Allocates a zero-filled [rows] x [cols] matrix.
  Matrix(this.rows, this.cols)
      : values = Float64List(rows * cols),
        grads = Float64List(rows * cols) {
    if (rows <= 0 || cols <= 0) {
      throw ArgumentError('matrix must be at least 1x1, got ${rows}x$cols');
    }
  }

  /// Wraps an existing [values] buffer without copying it.
  ///
  /// Used by the checkpoint loader so a restored model shares storage with the
  /// decoded snapshot instead of doubling peak memory.
  Matrix.fromBuffer(this.rows, this.cols, this.values)
      : grads = Float64List(rows * cols) {
    if (values.length != rows * cols) {
      throw ArgumentError(
        'buffer has ${values.length} values, expected ${rows * cols}',
      );
    }
  }

  /// Number of rows.
  final int rows;

  /// Number of columns.
  final int cols;

  /// Row-major parameter storage.
  final Float64List values;

  /// Row-major gradient storage, same length as [values].
  final Float64List grads;

  /// Number of scalars in this tensor.
  int get length => values.length;

  /// Reads element (r, c).
  double at(int r, int c) => values[r * cols + c];

  /// Writes element (r, c).
  void set(int r, int c, double value) => values[r * cols + c] = value;

  /// Accumulates `grads[r * cols + c] += delta`.
  void addGrad(int r, int c, double delta) => grads[r * cols + c] += delta;

  /// Zeroes every gradient.
  void zeroGrad() => grads.fillRange(0, grads.length, 0);

  /// A deep copy of the parameters (no gradients).
  Matrix clone() => Matrix.fromBuffer(rows, cols, Float64List.fromList(values));

  /// A uniform Xavier/Glorot initialiser scaled by the fan-in and fan-out.
  ///
  /// The bound is `sqrt(6 / (fanIn + fanOut))`, which keeps the variance of a
  /// linear layer's output approximately independent of its width. [random] is
  /// injected rather than created so every test can pin the initialisation with
  /// a fixed seed.
  static Matrix xavier(
    int rows,
    int cols,
    math.Random random,
  ) {
    final Matrix matrix = Matrix(rows, cols);
    final double bound = math.sqrt(6.0 / (rows + cols));
    for (int i = 0; i < matrix.values.length; i++) {
      matrix.values[i] = (random.nextDouble() * 2.0 - 1.0) * bound;
    }
    return matrix;
  }

  /// A normal initialiser with mean 0 and the given standard deviation.
  static Matrix normal(
    int rows,
    int cols,
    math.Random random, {
    double std = 0.02,
  }) {
    final Matrix matrix = Matrix(rows, cols);
    for (int i = 0; i < matrix.values.length; i++) {
      matrix.values[i] = _gaussian(random) * std;
    }
    return matrix;
  }

  /// `y = W x` for a length-[cols] input.
  Float64List matVec(Float64List x) {
    if (x.length != cols) {
      throw ArgumentError('expected input of length $cols, got ${x.length}');
    }
    final Float64List out = Float64List(rows);
    for (int r = 0; r < rows; r++) {
      final int base = r * cols;
      double sum = 0;
      for (int c = 0; c < cols; c++) {
        sum += values[base + c] * x[c];
      }
      out[r] = sum;
    }
    return out;
  }

  /// `y = Wᵀ g` for a length-[rows] input — the vector-Jacobian product of
  /// [matVec] with respect to its input.
  Float64List matVecTranspose(Float64List g) {
    if (g.length != rows) {
      throw ArgumentError('expected input of length $rows, got ${g.length}');
    }
    final Float64List out = Float64List(cols);
    for (int r = 0; r < rows; r++) {
      final double scale = g[r];
      if (scale == 0) {
        continue;
      }
      final int base = r * cols;
      for (int c = 0; c < cols; c++) {
        out[c] += values[base + c] * scale;
      }
    }
    return out;
  }

  /// Accumulates the rank-1 outer product `grads += scale * g xᵀ`.
  ///
  /// This is the weight gradient of `matVec`: `dW = g xᵀ`. [scale] carries the
  /// `1/batchSize` factor of a mean-reduced loss.
  void addOuter(Float64List g, Float64List x, {double scale = 1.0}) {
    if (g.length != rows || x.length != cols) {
      throw ArgumentError(
        'outer product shape mismatch: g=${g.length} (want $rows), '
        'x=${x.length} (want $cols)',
      );
    }
    for (int r = 0; r < rows; r++) {
      final double gv = g[r] * scale;
      if (gv == 0) {
        continue;
      }
      final int base = r * cols;
      for (int c = 0; c < cols; c++) {
        grads[base + c] += gv * x[c];
      }
    }
  }

  /// The L2 norm of every gradient concatenated.
  ///
  /// Used for global-norm gradient clipping, which stops a single unlucky batch
  /// from blowing the model apart.
  double gradNorm() {
    double sum = 0;
    for (int i = 0; i < grads.length; i++) {
      sum += grads[i] * grads[i];
    }
    return math.sqrt(sum);
  }
}

/// A named view over a parameter buffer and its matching gradient buffer.
///
/// The optimiser only ever sees these, which is what lets `AdamW` stay agnostic
/// about whether it is updating a matrix, a bias vector, or a LoRA factor.
class ParamRef {
  /// Creates a reference to [values]/[grads], which must be the same length.
  ParamRef(this.name, this.values, this.grads) {
    if (values.length != grads.length) {
      throw ArgumentError(
        'parameter "$name" has ${values.length} values but '
        '${grads.length} gradients',
      );
    }
  }

  /// Human-readable name, used by checkpoints and diagnostics.
  final String name;

  /// The live parameter buffer.
  final Float64List values;

  /// The gradient buffer written by the backward pass.
  final Float64List grads;

  /// Number of scalars.
  int get length => values.length;

  /// Stable JSON key: the name with non-alphanumerics collapsed to `_`.
  String get key => name.replaceAll(RegExp(r'[^A-Za-z0-9]+'), '_');
}

/// Numerically stable softmax over a fresh list.
Float64List softmax(Float64List logits) {
  if (logits.isEmpty) {
    throw ArgumentError.value(logits, 'logits', 'must not be empty');
  }
  double max = logits[0];
  for (int i = 1; i < logits.length; i++) {
    if (logits[i] > max) {
      max = logits[i];
    }
  }
  final Float64List out = Float64List(logits.length);
  double total = 0;
  for (int i = 0; i < logits.length; i++) {
    final double value = math.exp(logits[i] - max);
    out[i] = value;
    total += value;
  }
  if (total <= 0 || !total.isFinite) {
    // Degenerate logits (all -inf, or a NaN slipped in): fall back to a uniform
    // distribution rather than propagating NaN through the whole network.
    final double uniform = 1.0 / logits.length;
    out.fillRange(0, out.length, uniform);
    return out;
  }
  final double inverse = 1.0 / total;
  for (int i = 0; i < out.length; i++) {
    out[i] *= inverse;
  }
  return out;
}

/// Maps the logits on the simplex back to logit space: `g = p ⊙ (gp - ⟨gp,p⟩)`.
///
/// This is the Jacobian-vector product of [softmax]. [gradProbs] must have the
/// same length as [probs].
Float64List softmaxBackward(Float64List gradProbs, Float64List probs) {
  double dot = 0;
  for (int i = 0; i < probs.length; i++) {
    dot += gradProbs[i] * probs[i];
  }
  final Float64List out = Float64List(probs.length);
  for (int i = 0; i < probs.length; i++) {
    out[i] = probs[i] * (gradProbs[i] - dot);
  }
  return out;
}

/// The tanh approximation of GELU, `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`.
///
/// Chosen over the exact `erf` form because `math.erf` is unavailable in
/// dart2js; the approximation matches the exact function to well under 1e-4.
double gelu(double x) {
  const double kBeta = 0.7978845608028654; // sqrt(2 / pi)
  final double inner = kBeta * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1.0 + _tanh(inner));
}

/// The derivative of [gelu] with respect to its input.
double geluGrad(double x) {
  const double kBeta = 0.7978845608028654;
  final double x3 = x * x * x;
  final double inner = kBeta * (x + 0.044715 * x3);
  final double t = _tanh(inner);
  final double sech2 = 1.0 - t * t;
  final double innerGrad = kBeta * (1.0 + 3.0 * 0.044715 * x * x);
  return 0.5 * (1.0 + t) + 0.5 * x * sech2 * innerGrad;
}

/// Hyperbolic tangent, defined locally so the computation is identical on the
/// VM and in JavaScript (where `math` is a thin wrapper over `Math`).
double _tanh(double x) {
  if (x > 20) {
    return 1.0;
  }
  if (x < -20) {
    return -1.0;
  }
  final double e2 = math.exp(2 * x);
  return (e2 - 1) / (e2 + 1);
}

/// The result of a layer-normalisation forward pass: the output plus the two
/// statistics the backward pass needs.
class LayerNormCache {
  /// Creates a cache for a normalised vector of length [length].
  LayerNormCache(this.length);

  /// Length of the normalised vector.
  final int length;

  /// Mean of the input.
  double mean = 0;

  /// Reciprocal of the input's standard deviation.
  double invStd = 1;

  /// The normalised (pre-scale) values, `(x - mean) * invStd`.
  late final Float64List normalised = Float64List(length);

  /// The input, retained for the parameter gradients.
  late final Float64List input = Float64List(length);
}

/// Layer normalisation with learnable scale and shift.
///
/// `y = gamma ⊙ (x - mean) / sqrt(var + eps) + beta`, where the variance is the
/// biased (divide-by-n) estimator.
Float64List layerNormForward(
  Float64List x,
  Float64List gamma,
  Float64List beta,
  LayerNormCache cache, {
  double eps = 1e-5,
}) {
  final int n = x.length;
  double sum = 0;
  for (int i = 0; i < n; i++) {
    sum += x[i];
    cache.input[i] = x[i];
  }
  final double mean = sum / n;
  double variance = 0;
  for (int i = 0; i < n; i++) {
    final double d = x[i] - mean;
    variance += d * d;
  }
  variance /= n;
  final double invStd = 1.0 / math.sqrt(variance + eps);
  cache
    ..mean = mean
    ..invStd = invStd;

  final Float64List out = Float64List(n);
  for (int i = 0; i < n; i++) {
    final double normalised = (x[i] - mean) * invStd;
    cache.normalised[i] = normalised;
    out[i] = normalised * gamma[i] + beta[i];
  }
  return out;
}

/// The vector-Jacobian product of [layerNormForward].
///
/// Accumulates into [gradGamma] and [gradBeta] and returns the gradient with
/// respect to the input. [gradOut] must be the same length as [cache].
Float64List layerNormBackward(
  Float64List gradOut,
  LayerNormCache cache,
  Float64List gamma,
  Float64List gradGamma,
  Float64List gradBeta,
) {
  final int n = cache.length;
  final Float64List gradNormalised = Float64List(n);
  for (int i = 0; i < n; i++) {
    gradNormalised[i] = gradOut[i] * gamma[i];
    gradGamma[i] += gradOut[i] * cache.normalised[i];
    gradBeta[i] += gradOut[i];
  }

  double sumGrad = 0;
  double sumGradNorm = 0;
  for (int i = 0; i < n; i++) {
    sumGrad += gradNormalised[i];
    sumGradNorm += gradNormalised[i] * cache.normalised[i];
  }
  final double meanGrad = sumGrad / n;
  final double invStd = cache.invStd;
  final Float64List out = Float64List(n);
  for (int i = 0; i < n; i++) {
    out[i] = invStd *
        (gradNormalised[i] -
            meanGrad -
            cache.normalised[i] * sumGradNorm / n);
  }
  return out;
}

/// Rotary position embeddings applied to a single head vector in place.
///
/// Pairs of adjacent channels are rotated by an angle proportional to the
/// position, so attention scores depend on *relative* offset. The same table is
/// precomputed once per block and reused for every position.
class RotaryEmbedding {
  /// Precomputes the cosine/sine tables for [dim] channels up to [maxPositions].
  ///
  /// [dim] must be even; the base frequency is 10000, matching the original
  /// RoPE paper.
  RotaryEmbedding({
    required this.dim,
    required this.maxPositions,
    double theta = 10000.0,
  }) {
    if (dim.isOdd) {
      throw ArgumentError.value(dim, 'dim', 'must be even for rotary pairs');
    }
    final int half = dim ~/ 2;
    cos = Float64List(maxPositions * half);
    sin = Float64List(maxPositions * half);
    for (int position = 0; position < maxPositions; position++) {
      for (int i = 0; i < half; i++) {
        final double frequency =
            position / math.pow(theta, (2 * i) / dim).toDouble();
        cos[position * half + i] = math.cos(frequency);
        sin[position * half + i] = math.sin(frequency);
      }
    }
  }

  /// Number of channels rotated (always even).
  final int dim;

  /// Number of positions the tables cover.
  final int maxPositions;

  /// Cosine table, `[position * (dim/2) + i]`.
  late final Float64List cos;

  /// Sine table, `[position * (dim/2) + i]`.
  late final Float64List sin;

  /// Rotates [vector] (length [dim]) in place by [position].
  void apply(Float64List vector, int position) {
    if (position >= maxPositions) {
      throw ArgumentError(
        'position $position exceeds the rotary table of $maxPositions',
      );
    }
    final int half = dim ~/ 2;
    final int base = position * half;
    for (int i = 0; i < half; i++) {
      final double c = cos[base + i];
      final double s = sin[base + i];
      final double a = vector[2 * i];
      final double b = vector[2 * i + 1];
      vector[2 * i] = a * c - b * s;
      vector[2 * i + 1] = a * s + b * c;
    }
  }

  /// The inverse rotation, used by the backward pass.
  ///
  /// Rotation is orthogonal, so the transpose simply negates the sine term. The
  /// backward pass therefore needs no extra table.
  void applyInverse(Float64List vector, int position) {
    final int half = dim ~/ 2;
    final int base = position * half;
    for (int i = 0; i < half; i++) {
      final double c = cos[base + i];
      final double s = sin[base + i];
      final double a = vector[2 * i];
      final double b = vector[2 * i + 1];
      vector[2 * i] = a * c + b * s;
      vector[2 * i + 1] = -a * s + b * c;
    }
  }
}

/// Standard normal sample via the Box-Muller transform.
double _gaussian(math.Random random) {
  double u1 = random.nextDouble();
  while (u1 <= 1e-12) {
    u1 = random.nextDouble();
  }
  final double u2 = random.nextDouble();
  return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2);
}