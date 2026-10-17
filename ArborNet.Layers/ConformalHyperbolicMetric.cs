using System;

namespace ArborNet.Layers
{
    /// <summary>
    /// Implements metric operations and projections for a Conformal Hyperbolic Scale Bundle (H^n x R_+).
    /// Hyperbolic coordinates are represented in the Lorentz hyperboloid model with signature (-1, 1, ..., 1).
    /// </summary>
    public sealed class ConformalHyperbolicMetric
    {
        private readonly double _hausdorffDimension;
        private readonly double _curvature;
        private const double Epsilon = 1e-7;

        /// <summary>
        /// Initializes a new instance of the <see cref="ConformalHyperbolicMetric"/> class.
        /// </summary>
        /// <param name="hausdorffDimension">The non-integer or effective Hausdorff dimension weight for the scale fiber.</param>
        /// <param name="curvature">Hyperbolic curvature parameter (c > 0).</param>
        public ConformalHyperbolicMetric(double hausdorffDimension = 1.0, double curvature = 1.0)
        {
            if (hausdorffDimension <= 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(hausdorffDimension), "Hausdorff dimension weight must be positive.");
            }

            if (curvature <= 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(curvature), "Curvature must be positive.");
            }

            _hausdorffDimension = hausdorffDimension;
            _curvature = curvature;
        }

        /// <summary>
        /// Computes the Minkowski pseudo-inner product &lt;u, v&gt; = -u_0 * v_0 + \sum_{i=1}^n u_i * v_i.
        /// </summary>
        /// <param name="u">First hyperboloid coordinate array (u_0 is the time-like coordinate).</param>
        /// <param name="v">Second hyperboloid coordinate array (v_0 is the time-like coordinate).</param>
        /// <returns>Minkowski inner product value.</returns>
        public static double MinkowskiInnerProduct(ReadOnlySpan<double> u, ReadOnlySpan<double> v)
        {
            if (u.Length != v.Length || u.Length < 2)
            {
                throw new ArgumentException("Input vectors must have identical lengths of at least 2.");
            }

            double product = -u[0] * v[0];
            for (int i = 1; i < u.Length; i++)
            {
                product += u[i] * v[i];
            }

            return product;
        }

        /// <summary>
        /// Computes the combined geodesic and conformal scale distance on H^n x R_+.
        /// Formula: ds^2 = d_Lorentz(u, v)^2 + d_H^2 * (log(tau_u) - log(tau_v))^2.
        /// </summary>
        /// <param name="hyperbolicU">Hyperbolic coordinates of point u.</param>
        /// <param name="tauU">Scale coordinate of point u (tau > 0).</param>
        /// <param name="hyperbolicV">Hyperbolic coordinates of point v.</param>
        /// <param name="tauV">Scale coordinate of point v (tau > 0).</param>
        /// <returns>Total bundle metric distance.</returns>
        public double ComputeDistance(
            ReadOnlySpan<double> hyperbolicU,
            double tauU,
            ReadOnlySpan<double> hyperbolicV,
            double tauV)
        {
            if (tauU <= 0.0 || tauV <= 0.0)
            {
                throw new ArgumentOutOfRangeException("Scale parameters tau must be strictly positive.");
            }

            double inner = MinkowskiInnerProduct(hyperbolicU, hyperbolicV);

            // In hyperboloid space with curvature c, <u, v> <= -1/c. Clamp for numerical stability.
            double minInner = -1.0 / _curvature;
            if (inner > minInner)
            {
                inner = minInner;
            }

            double argument = -_curvature * inner;
            if (argument < 1.0)
            {
                argument = 1.0;
            }

            // Arcosh(x) = log(x + sqrt(x^2 - 1))
            double hyperbolicDist = Math.Log(argument + Math.Sqrt(Math.Max(0.0, (argument * argument) - 1.0))) / Math.Sqrt(_curvature);

            // Log-conformal scale fiber metric: d_H * |log(tau_u) - log(tau_v)|
            double logScaleDiff = Math.Log(tauU) - Math.Log(tauV);
            double scaleDist = _hausdorffDimension * Math.Abs(logScaleDiff);

            return Math.Sqrt((hyperbolicDist * hyperbolicDist) + (scaleDist * scaleDist));
        }

        /// <summary>
        /// Projects an unconstrained spatial vector into the upper sheet of the Lorentz hyperboloid.
        /// </summary>
        /// <param name="spatialVector">Input coordinate array representing space-like components.</param>
        /// <param name="destination">Output buffer receiving the projected point [x0, x1, ..., xn].</param>
        public void ProjectToHyperboloid(ReadOnlySpan<double> spatialVector, Span<double> destination)
        {
            if (destination.Length != spatialVector.Length + 1)
            {
                throw new ArgumentException("Destination length must be spatialVector.Length + 1.");
            }

            double sumSquares = 0.0;
            for (int i = 0; i < spatialVector.Length; i++)
            {
                destination[i + 1] = spatialVector[i];
                sumSquares += spatialVector[i] * spatialVector[i];
            }

            // For curvature c: x0 = sqrt(1/c + sum(x_i^2))
            destination[0] = Math.Sqrt((1.0 / _curvature) + sumSquares);
        }

        /// <summary>
        /// Computes parallel transport of a tangent vector along the geodesic connecting two hyperboloid points.
        /// </summary>
        /// <param name="sourcePoint">Starting point on H^n.</param>
        /// <param name="targetPoint">Destination point on H^n.</param>
        /// <param name="tangentVector">Tangent vector at sourcePoint to be transported.</param>
        /// <param name="result">Output buffer for the transported tangent vector at targetPoint.</param>
        public void ParallelTransport(
            ReadOnlySpan<double> sourcePoint,
            ReadOnlySpan<double> targetPoint,
            ReadOnlySpan<double> tangentVector,
            Span<double> result)
        {
            if (sourcePoint.Length != targetPoint.Length || sourcePoint.Length != tangentVector.Length || sourcePoint.Length != result.Length)
            {
                throw new ArgumentException("All vectors must have identical dimensions.");
            }

            // TODO: Complete the Cartan connection parallel transport implementation:
            // 1. Calculate the Riemannian distance d(sourcePoint, targetPoint).
            // 2. Derive the unit velocity tangent vector pointing from sourcePoint toward targetPoint.
            // 3. Project tangentVector along the velocity vector and apply Lorentz boost transformation.
            for (int i = 0; i < sourcePoint.Length; i++)
            {
                result[i] = tangentVector[i];
            }
        }
    }
}
