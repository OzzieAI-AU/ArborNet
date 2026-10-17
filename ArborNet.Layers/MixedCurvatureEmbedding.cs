using System;

namespace ArborNet.Layers
{
    /// <summary>
    /// Configuration parameter set defining the dimension split and sectional curvatures
    /// for a Cartesian product manifold M = H^{d_h} x S^{d_s} x R^{d_e}.
    /// </summary>
    public readonly struct ProductManifoldConfig
    {
        public readonly int HyperbolicDim;
        public readonly int SphericalDim;
        public readonly int EuclideanDim;
        public readonly float HyperbolicCurvature;
        public readonly float SphericalCurvature;
        public readonly float Epsilon;

        public int TotalDim => HyperbolicDim + SphericalDim + EuclideanDim;

        public ProductManifoldConfig(
            int hyperbolicDim,
            int sphericalDim,
            int euclideanDim,
            float hyperbolicCurvature = 1.0f,
            float sphericalCurvature = 1.0f,
            float epsilon = 1e-5f)
        {
            if (hyperbolicDim < 0 || sphericalDim < 0 || euclideanDim < 0)
            {
                throw new ArgumentOutOfRangeException("Component dimensions must be non-negative.");
            }

            if (hyperbolicDim + sphericalDim + euclideanDim == 0)
            {
                throw new ArgumentException("Total manifold dimension must be strictly positive.");
            }

            if (hyperbolicCurvature <= 0f || sphericalCurvature <= 0f)
            {
                throw new ArgumentOutOfRangeException("Curvature parameters must be strictly positive.");
            }

            HyperbolicDim = hyperbolicDim;
            SphericalDim = sphericalDim;
            EuclideanDim = euclideanDim;
            HyperbolicCurvature = hyperbolicCurvature;
            SphericalCurvature = sphericalCurvature;
            Epsilon = epsilon;
        }
    }

    /// <summary>
    /// Core geometric operations and projection layer for factorized product spaces
    /// combining constant negative curvature (Poincare ball), constant positive curvature
    /// (hypersphere), and zero-curvature Euclidean spaces.
    /// </summary>
    public class MixedCurvatureEmbedding
    {
        private readonly ProductManifoldConfig _config;

        public ProductManifoldConfig Config => _config;

        public MixedCurvatureEmbedding(ProductManifoldConfig config)
        {
            _config = config;
        }

        /// <summary>
        /// Projects an unconstrained continuous latent vector into the product manifold.
        /// Hyperbolic slice is clamped within the Poincare open unit ball.
        /// Spherical slice is normalized onto the hypersphere.
        /// Euclidean slice remains unconstrained.
        /// </summary>
        /// <param name="rawVector">Input vector of length equal to TotalDim.</param>
        /// <param name="projectedVector">Destination buffer of length equal to TotalDim.</param>
        public void ProjectToManifold(ReadOnlySpan<float> rawVector, Span<float> projectedVector)
        {
            if (rawVector.Length != _config.TotalDim)
            {
                throw new ArgumentException($"Source vector length {rawVector.Length} does not match manifold dimension {_config.TotalDim}.");
            }

            if (projectedVector.Length != _config.TotalDim)
            {
                throw new ArgumentException($"Destination buffer length {projectedVector.Length} does not match manifold dimension {_config.TotalDim}.");
            }

            int offset = 0;

            // 1. Hyperbolic projection (Poincare Ball: ||x|| < 1 / sqrt(c))
            if (_config.HyperbolicDim > 0)
            {
                float c = _config.HyperbolicCurvature;
                float maxNorm = (1.0f / MathF.Sqrt(c)) - _config.Epsilon;
                float sqNorm = 0f;

                for (int i = 0; i < _config.HyperbolicDim; i++)
                {
                    float val = rawVector[offset + i];
                    sqNorm += val * val;
                }

                float norm = MathF.Sqrt(sqNorm);
                float scale = 1.0f;
                if (norm > maxNorm && norm > _config.Epsilon)
                {
                    scale = maxNorm / norm;
                }

                for (int i = 0; i < _config.HyperbolicDim; i++)
                {
                    projectedVector[offset + i] = rawVector[offset + i] * scale;
                }

                offset += _config.HyperbolicDim;
            }

            // 2. Spherical projection (Hypersphere: ||x|| = 1 / sqrt(c))
            if (_config.SphericalDim > 0)
            {
                float c = _config.SphericalCurvature;
                float targetRadius = 1.0f / MathF.Sqrt(c);
                float sqNorm = 0f;

                for (int i = 0; i < _config.SphericalDim; i++)
                {
                    float val = rawVector[offset + i];
                    sqNorm += val * val;
                }

                float norm = MathF.Sqrt(sqNorm);
                if (norm < _config.Epsilon)
                {
                    // Degenerate origin point: map deterministically to north pole
                    for (int i = 0; i < _config.SphericalDim; i++)
                    {
                        projectedVector[offset + i] = 0f;
                    }
                    projectedVector[offset] = targetRadius;
                }
                else
                {
                    float scale = targetRadius / norm;
                    for (int i = 0; i < _config.SphericalDim; i++)
                    {
                        projectedVector[offset + i] = rawVector[offset + i] * scale;
                    }
                }

                offset += _config.SphericalDim;
            }

            // 3. Euclidean identity projection
            if (_config.EuclideanDim > 0)
            {
                for (int i = 0; i < _config.EuclideanDim; i++)
                {
                    projectedVector[offset + i] = rawVector[offset + i];
                }
            }
        }

        /// <summary>
        /// Computes the squared product Riemannian geodesic distance d_M^2(u, v) = d_H^2 + d_S^2 + d_E^2.
        /// </summary>
        public float ComputeGeodesicDistanceSquared(ReadOnlySpan<float> u, ReadOnlySpan<float> v)
        {
            if (u.Length != _config.TotalDim || v.Length != _config.TotalDim)
            {
                throw new ArgumentException("Input vector dimension mismatch.");
            }

            float totalDistanceSquared = 0f;
            int offset = 0;

            // Hyperbolic Poincare Geodesic Distance
            if (_config.HyperbolicDim > 0)
            {
                float c = _config.HyperbolicCurvature;
                float sqNormU = 0f;
                float sqNormV = 0f;
                float sqDiff = 0f;

                for (int i = 0; i < _config.HyperbolicDim; i++)
                {
                    float ui = u[offset + i];
                    float vi = v[offset + i];
                    sqNormU += ui * ui;
                    sqNormV += vi * vi;
                    float diff = ui - vi;
                    sqDiff += diff * diff;
                }

                float denom = (1.0f - c * sqNormU) * (1.0f - c * sqNormV);
                denom = MathF.Max(denom, _config.Epsilon);

                float alpha = 1.0f + 2.0f * c * sqDiff / denom;
                alpha = MathF.Max(alpha, 1.0f); // Numerical guard for acosh domain [1, inf)

                // acosh(alpha) = ln(alpha + sqrt(alpha^2 - 1))
                float acoshVal = MathF.Log(alpha + MathF.Sqrt(MathF.Max(alpha * alpha - 1.0f, 0f)));
                float dH = (1.0f / MathF.Sqrt(c)) * acoshVal;
                totalDistanceSquared += dH * dH;

                offset += _config.HyperbolicDim;
            }

            // Spherical Geodesic Distance
            if (_config.SphericalDim > 0)
            {
                float c = _config.SphericalCurvature;
                float targetRadius = 1.0f / MathF.Sqrt(c);
                float dot = 0f;

                for (int i = 0; i < _config.SphericalDim; i++)
                {
                    dot += u[offset + i] * v[offset + i];
                }

                // Cosine angle normalized by radius squared
                float cosTheta = dot / (targetRadius * targetRadius);
                cosTheta = Math.Clamp(cosTheta, -1.0f, 1.0f);

                float angle = MathF.Acos(cosTheta);
                float dS = targetRadius * angle;
                totalDistanceSquared += dS * dS;

                offset += _config.SphericalDim;
            }

            // Euclidean Geodesic Distance
            if (_config.EuclideanDim > 0)
            {
                float dE2 = 0f;
                for (int i = 0; i < _config.EuclideanDim; i++)
                {
                    float diff = u[offset + i] - v[offset + i];
                    dE2 += diff * diff;
                }
                totalDistanceSquared += dE2;
            }

            return totalDistanceSquared;
        }

        /// <summary>
        /// Computes parallel transport and log-map tangent space representations for
        /// downstream continuous queries and loss matching.
        /// </summary>
        /// <param name="basePoint">The tangent base coordinate.</param>
        /// <param name="targetPoint">The target manifold point.</param>
        /// <param name="tangentVector">Output tangent vector log_{basePoint}(targetPoint).</param>
        public void TangentLogMap(
            ReadOnlySpan<float> basePoint,
            ReadOnlySpan<float> targetPoint,
            Span<float> tangentVector)
        {
            // TODO: Implement Mobius logarithmic map on Poincare component and spherical Riemannian log map.
            // Prototype handles Euclidean fallback copy across all dimensions.
            for (int i = 0; i < _config.TotalDim; i++)
            {
                tangentVector[i] = targetPoint[i] - basePoint[i];
            }
        }
    }
}
