// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Layers
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Implements continuous geometric projections of multi-scale fractal spectra and
    /// prime-gap dynamics into the Lorentz hyperboloid manifold H^d = {x in R^{d+1} : &lt;x,x&gt;_L = -1/c, x_0 &gt; 0}.
    /// </summary>
    public class LorentzianFractalEmbedding
    {
        private const double Epsilon = 1e-7;
        private readonly int _spatialDimension;
        private readonly double _hausdorffDimension;
        private readonly double _curvature;

        /// <summary>
        /// Gets the total manifold coordinate dimension (spatial dimension d + 1 time-like dimension).
        /// </summary>
        public int AmbientDimension => _spatialDimension + 1;

        /// <summary>
        /// Gets the spatial dimension d.
        /// </summary>
        public int SpatialDimension => _spatialDimension;

        /// <summary>
        /// Gets the effective Hausdorff dimension scale exponent d_H.
        /// </summary>
        public double HausdorffDimension => _hausdorffDimension;

        /// <summary>
        /// Gets the Gaussian curvature scale magnitude c (c &gt; 0, where sectional curvature K = -c).
        /// </summary>
        public double Curvature => _curvature;

        /// <summary>
        /// Initializes a new instance of the <see cref="LorentzianFractalEmbedding"/> class.
        /// </summary>
        /// <param name="spatialDimension">The spatial coordinate dimension d (ambient dimension is d + 1).</param>
        /// <param name="hausdorffDimension">
        /// The fractal scaling dimension d_H used to scale multi-frequency spectra (defaults to 1.2618595 ~ Koch snowflake).
        /// </param>
        /// <param name="curvature">Negative curvature constant scale c &gt; 0 (defaults to 1.0).</param>
        public LorentzianFractalEmbedding(
            int spatialDimension,
            double hausdorffDimension = 1.2618595071429,
            double curvature = 1.0)
        {
            if (spatialDimension <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(spatialDimension), "Spatial dimension must be strictly positive.");
            }

            if (hausdorffDimension <= 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(hausdorffDimension), "Hausdorff dimension must be strictly positive.");
            }

            if (curvature <= 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(curvature), "Curvature magnitude c must be strictly positive.");
            }

            _spatialDimension = spatialDimension;
            _hausdorffDimension = hausdorffDimension;
            _curvature = curvature;
        }

        /// <summary>
        /// Computes the Lorentzian (Minkowski) inner product &lt;u, v&gt;_L between two vectors in R^{d+1}.
        /// Formula: &lt;u, v&gt;_L = -u_0 * v_0 + \sum_{i=1}^d u_i * v_i.
        /// </summary>
        /// <param name="u">First vector in R^{d+1}.</param>
        /// <param name="v">Second vector in R^{d+1}.</param>
        /// <returns>The scalar Lorentzian inner product.</returns>
        public double ComputeMinkowskiInnerProduct(ReadOnlySpan<double> u, ReadOnlySpan<double> v)
        {
            if (u.Length != AmbientDimension || v.Length != AmbientDimension)
            {
                throw new ArgumentException($"Vectors must match the ambient dimension ({AmbientDimension}).");
            }

            double result = -u[0] * v[0];
            for (int i = 1; i < AmbientDimension; i++)
            {
                result += u[i] * v[i];
            }

            return result;
        }

        /// <summary>
        /// Computes the geodesic distance d_L(u, v) between two points on the Lorentz hyperboloid H^d.
        /// Formula: d_L(u, v) = (1 / sqrt(c)) * arcosh(-c * &lt;u, v&gt;_L).
        /// </summary>
        /// <param name="u">Point u on the hyperboloid.</param>
        /// <param name="v">Point v on the hyperboloid.</param>
        /// <returns>The geodesic distance along the manifold.</returns>
        public double ComputeLorentzDistance(ReadOnlySpan<double> u, ReadOnlySpan<double> v)
        {
            double inner = ComputeMinkowskiInnerProduct(u, v);
            double alpha = -_curvature * inner;

            // Numerical safety clamp: arcosh is defined for alpha >= 1.0
            if (alpha < 1.0)
            {
                alpha = 1.0;
            }

            return Math.Acosh(alpha) / Math.Sqrt(_curvature);
        }

        /// <summary>
        /// Projects a Euclidean spatial vector in R^d onto the Lorentz hyperboloid manifold
        /// via the hyperbolic exponential map at the canonical origin point mu = (1/sqrt(c), 0, ..., 0)^T.
        /// </summary>
        /// <param name="euclideanSpatial">The input spatial coordinates in R^d.</param>
        /// <param name="lorentzOutput">Destination pre-allocated span of length d + 1.</param>
        public void ExpMapOrigin(ReadOnlySpan<double> euclideanSpatial, Span<double> lorentzOutput)
        {
            if (euclideanSpatial.Length != _spatialDimension)
            {
                throw new ArgumentException($"Input spatial vector length must be {_spatialDimension}.", nameof(euclideanSpatial));
            }

            if (lorentzOutput.Length != AmbientDimension)
            {
                throw new ArgumentException($"Output span must have length equal to ambient dimension ({AmbientDimension}).", nameof(lorentzOutput));
            }

            double normSq = 0.0;
            for (int i = 0; i < euclideanSpatial.Length; i++)
            {
                double val = euclideanSpatial[i];
                normSq += val * val;
            }

            double norm = Math.Sqrt(normSq);
            double sqrtC = Math.Sqrt(_curvature);
            double invSqrtC = 1.0 / sqrtC;

            if (norm < Epsilon)
            {
                lorentzOutput[0] = invSqrtC;
                lorentzOutput[1..].Clear();
                return;
            }

            double scaledNorm = sqrtC * norm;
            lorentzOutput[0] = Math.Cosh(scaledNorm) * invSqrtC;

            double spatialScale = Math.Sinh(scaledNorm) / (sqrtC * norm);
            for (int i = 0; i < euclideanSpatial.Length; i++)
            {
                lorentzOutput[i + 1] = euclideanSpatial[i] * spatialScale;
            }
        }

        /// <summary>
        /// Maps a point on the Lorentz hyperboloid back into the tangent space at the origin
        /// (spatial coordinates in R^d) via the logarithmic map Log_mu(x).
        /// </summary>
        /// <param name="lorentzPoint">Point on the hyperboloid in R^{d+1}.</param>
        /// <param name="tangentSpatialOutput">Pre-allocated destination span in R^d.</param>
        public void LogMapOrigin(ReadOnlySpan<double> lorentzPoint, Span<double> tangentSpatialOutput)
        {
            if (lorentzPoint.Length != AmbientDimension)
            {
                throw new ArgumentException($"Lorentzian point must have dimension {AmbientDimension}.", nameof(lorentzPoint));
            }

            if (tangentSpatialOutput.Length != _spatialDimension)
            {
                throw new ArgumentException($"Tangent output must have dimension {_spatialDimension}.", nameof(tangentSpatialOutput));
            }

            double sqrtC = Math.Sqrt(_curvature);
            double inner = -lorentzPoint[0] / sqrtC; // <mu, x>_L with mu = (1/sqrt(c), 0, ...)
            double alpha = -_curvature * inner;

            if (alpha < 1.0)
            {
                alpha = 1.0;
            }

            double dist = Math.Acosh(alpha) / sqrtC;
            double spatialNormSq = 0.0;

            for (int i = 1; i < AmbientDimension; i++)
            {
                spatialNormSq += lorentzPoint[i] * lorentzPoint[i];
            }

            double spatialNorm = Math.Sqrt(spatialNormSq);
            if (spatialNorm < Epsilon || dist < Epsilon)
            {
                tangentSpatialOutput.Clear();
                return;
            }

            double scale = dist / spatialNorm;
            for (int i = 0; i < _spatialDimension; i++)
            {
                tangentSpatialOutput[i] = lorentzPoint[i + 1] * scale;
            }
        }

        /// <summary>
        /// Embeds a multi-scale fractal frequency spectrum into a spatial tangent vector,
        /// modulated by the configured Hausdorff dimension d_H, and projects it onto the Lorentz hyperboloid.
        /// </summary>
        /// <param name="spectralAmplitudes">Multi-scale spectral features or wavelet coefficients.</param>
        /// <param name="lorentzOutput">Pre-allocated destination span in R^{d+1}.</param>
        public void ProjectFractalSpectrum(ReadOnlySpan<double> spectralAmplitudes, Span<double> lorentzOutput)
        {
            Span<double> spatial = stackalloc double[_spatialDimension];
            spatial.Clear();

            int len = Math.Min(spectralAmplitudes.Length, _spatialDimension);
            for (int i = 0; i < len; i++)
            {
                // Multi-scale harmonic weight modulated by d_H: w_k = (k + 1)^(-d_H)
                double harmonicWeight = Math.Pow(i + 1.0, -_hausdorffDimension);
                spatial[i] = spectralAmplitudes[i] * harmonicWeight;
            }

            ExpMapOrigin(spatial, lorentzOutput);
        }

        /// <summary>
        /// Projects discrete prime gap trajectories into continuous Lorentz coordinates.
        /// Combines the logarithmic prime-gap density with fractal scaling dynamics.
        /// </summary>
        /// <param name="primeGap">The prime gap value g_n = p_{n+1} - p_n.</param>
        /// <param name="primeIndex">Index n of the prime in the sequence.</param>
        /// <param name="lorentzOutput">Pre-allocated destination span in R^{d+1}.</param>
        public void ProjectPrimeGapDynamics(long primeGap, long primeIndex, Span<double> lorentzOutput)
        {
            Span<double> spatial = stackalloc double[_spatialDimension];
            spatial.Clear();

            double normalizedGap = Math.Log(Math.Max(1.0, (double)primeGap));
            double indexScale = Math.Log(Math.Max(2.0, (double)primeIndex));

            for (int i = 0; i < _spatialDimension; i++)
            {
                // Dispersive quasi-periodic phase encoding across spatial dimensions
                double frequency = Math.Pow(i + 1, _hausdorffDimension / 2.0);
                double phase = (primeIndex * frequency) % (2.0 * Math.PI);

                spatial[i] = (normalizedGap / indexScale) * Math.Cos(phase);
            }

            ExpMapOrigin(spatial, lorentzOutput);
        }

        /// <summary>
        /// Converts a point in the Poincaré ball B^d into equivalent Lorentz hyperboloid coordinates in H^d.
        /// </summary>
        /// <param name="poincarePoint">Point inside the unit ball (norm &lt; 1 / sqrt(c)).</param>
        /// <param name="lorentzOutput">Destination coordinates in R^{d+1}.</param>
        public void ConvertPoincareToLorentz(ReadOnlySpan<double> poincarePoint, Span<double> lorentzOutput)
        {
            if (poincarePoint.Length != _spatialDimension)
            {
                throw new ArgumentException($"Poincaré point must have dimension {_spatialDimension}.", nameof(poincarePoint));
            }

            if (lorentzOutput.Length != AmbientDimension)
            {
                throw new ArgumentException($"Lorentzian output must have dimension {AmbientDimension}.", nameof(lorentzOutput));
            }

            double normSq = 0.0;
            for (int i = 0; i < poincarePoint.Length; i++)
            {
                normSq += poincarePoint[i] * poincarePoint[i];
            }

            double c = _curvature;
            double denominator = 1.0 - (c * normSq);
            if (denominator < Epsilon)
            {
                denominator = Epsilon;
            }

            double invSqrtC = 1.0 / Math.Sqrt(c);
            lorentzOutput[0] = invSqrtC * ((1.0 + (c * normSq)) / denominator);

            double spatialFactor = 2.0 / denominator;
            for (int i = 0; i < _spatialDimension; i++)
            {
                lorentzOutput[i + 1] = poincarePoint[i] * spatialFactor;
            }
        }

        /// <summary>
        /// Converts a point on the Lorentz hyperboloid H^d to the Poincaré unit ball B^d.
        /// </summary>
        /// <param name="lorentzPoint">Point in R^{d+1} satisfying &lt;x, x&gt;_L = -1/c.</param>
        /// <param name="poincareOutput">Destination coordinates in R^d.</param>
        public void ConvertLorentzToPoincare(ReadOnlySpan<double> lorentzPoint, Span<double> poincareOutput)
        {
            if (lorentzPoint.Length != AmbientDimension)
            {
                throw new ArgumentException($"Lorentzian point must have dimension {AmbientDimension}.", nameof(lorentzPoint));
            }

            if (poincareOutput.Length != _spatialDimension)
            {
                throw new ArgumentException($"Poincaré output must have dimension {_spatialDimension}.", nameof(poincareOutput));
            }

            double sqrtC = Math.Sqrt(_curvature);
            double denominator = 1.0 + (sqrtC * lorentzPoint[0]);
            if (Math.Abs(denominator) < Epsilon)
            {
                denominator = Epsilon;
            }

            for (int i = 0; i < _spatialDimension; i++)
            {
                poincareOutput[i] = lorentzPoint[i + 1] / denominator;
            }
        }
    }
}