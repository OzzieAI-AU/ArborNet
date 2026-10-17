using System;

namespace ArborNet.Layers
{
    /// <summary>
    /// Provides foundational mathematical primitives for embedding multi-scale fractal dimensions
    /// and discrete arithmetic trajectories into an adelic-hyperbolic product space.
    /// Combines continuous Poincaré ball hyperbolic geometry with ultrametric p-adic metrics.
    /// </summary>
    public sealed class AdelicHyperbolicEmbedding
    {
        private const float Epsilon = 1e-6f;
        private const float MaxNormBound = 1.0f - 1e-5f;

        /// <summary>
        /// Gets the manifold curvature magnitude c (|c| = 1.0 for standard Poincaré ball).
        /// </summary>
        public float Curvature { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="AdelicHyperbolicEmbedding"/> class.
        /// </summary>
        /// <param name="curvature">Negative curvature constant scale (defaults to 1.0f).</param>
        public AdelicHyperbolicEmbedding(float curvature = 1.0f)
        {
            if (curvature <= 0.0f)
            {
                throw new ArgumentOutOfRangeException(nameof(curvature), "Curvature magnitude must be positive.");
            }

            Curvature = curvature;
        }

        /// <summary>
        /// Computes the geodesic distance between two vectors inside the Poincaré ball.
        /// Formula: d_H(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2) * (1 - ||v||^2))).
        /// </summary>
        /// <param name="u">Point u within the unit ball.</param>
        /// <param name="v">Point v within the unit ball.</param>
        /// <returns>The non-Euclidean geodesic hyperbolic distance.</returns>
        public float ComputePoincareDistance(ReadOnlySpan<float> u, ReadOnlySpan<float> v)
        {
            if (u.Length != v.Length)
            {
                throw new ArgumentException("Vectors u and v must have identical dimensionality.");
            }

            float normSqU = 0.0f;
            float normSqV = 0.0f;
            float diffNormSq = 0.0f;

            for (int i = 0; i < u.Length; i++)
            {
                float ui = u[i];
                float vi = v[i];
                float delta = ui - vi;

                normSqU += ui * ui;
                normSqV += vi * vi;
                diffNormSq += delta * delta;
            }

            normSqU = MathF.Min(normSqU, MaxNormBound);
            normSqV = MathF.Min(normSqV, MaxNormBound);

            float denominator = (1.0f - normSqU) * (1.0f - normSqV);
            if (denominator < Epsilon)
            {
                denominator = Epsilon;
            }

            float alpha = 1.0f + (2.0f * Curvature * diffNormSq) / denominator;
            if (alpha < 1.0f)
            {
                alpha = 1.0f;
            }

            return MathF.Acosh(alpha) / MathF.Sqrt(Curvature);
        }

        /// <summary>
        /// Computes the p-adic ultrametric valuation distance between two integer coordinates.
        /// Formula: d_p(x, y) = p^(-v_p(x - y)), where v_p(n) is the highest power of p dividing n.
        /// </summary>
        /// <param name="x">First scalar integer coordinate.</param>
        /// <param name="y">Second scalar integer coordinate.</param>
        /// <param name="prime">The prime base p for the ultrametric field.</param>
        /// <returns>The non-Archimedean distance between x and y.</returns>
        public float ComputePAdicDistance(long x, long y, int prime)
        {
            if (prime < 2)
            {
                throw new ArgumentException("Prime base must be >= 2.", nameof(prime));
            }

            long diff = x - y;
            if (diff == 0)
            {
                return 0.0f;
            }

            if (diff < 0)
            {
                diff = -diff;
            }

            int valuation = 0;
            while (diff % prime == 0)
            {
                valuation++;
                diff /= prime;
            }

            return MathF.Pow(prime, -valuation);
        }

        /// <summary>
        /// Combines hyperbolic distance with a discrete ultrametric component into an adelic product metric.
        /// </summary>
        /// <param name="uContinuous">Hyperbolic latent vector for continuous fractal dimension features.</param>
        /// <param name="vContinuous">Hyperbolic latent vector for target fractal features.</param>
        /// <param name="xDiscrete">Discrete arithmetic index (e.g., prime gap accumulator or IFS branch index).</param>
        /// <param name="yDiscrete">Discrete arithmetic index.</param>
        /// <param name="primeBase">Prime base for the ultrametric evaluation.</param>
        /// <param name="hyperbolicWeight">Weight assigned to the continuous hyperbolic component.</param>
        /// <returns>Combined product metric distance.</returns>
        public float ComputeAdelicProductDistance(
            ReadOnlySpan<float> uContinuous,
            ReadOnlySpan<float> vContinuous,
            long xDiscrete,
            long yDiscrete,
            int primeBase,
            float hyperbolicWeight = 0.5f)
        {
            float dH = ComputePoincareDistance(uContinuous, vContinuous);
            float dP = ComputePAdicDistance(xDiscrete, yDiscrete, primeBase);

            return MathF.Sqrt((hyperbolicWeight * dH * dH) + ((1.0f - hyperbolicWeight) * dP * dP));
        }

        /// <summary>
        /// Projects a Euclidean vector onto the Poincaré ball via the hyperbolic exponential map at the origin.
        /// Exp_0(v) = tanh(sqrt(c) * ||v||) * (v / (sqrt(c) * ||v||)).
        /// </summary>
        /// <param name="euclideanVector">Input Euclidean feature vector.</param>
        /// <param name="poincareOutput">Pre-allocated destination span for the projected vector.</param>
        public void ProjectToPoincareBall(ReadOnlySpan<float> euclideanVector, Span<float> poincareOutput)
        {
            if (euclideanVector.Length != poincareOutput.Length)
            {
                throw new ArgumentException("Input and output spans must match in length.");
            }

            float normSq = 0.0f;
            for (int i = 0; i < euclideanVector.Length; i++)
            {
                normSq += euclideanVector[i] * euclideanVector[i];
            }

            float norm = MathF.Sqrt(normSq);
            if (norm < Epsilon)
            {
                euclideanVector.CopyTo(poincareOutput);
                return;
            }

            float sqrtC = MathF.Sqrt(Curvature);
            float scale = MathF.Tanh(sqrtC * norm) / (sqrtC * norm);

            for (int i = 0; i < euclideanVector.Length; i++)
            {
                poincareOutput[i] = euclideanVector[i] * scale;
            }
        }

        /// <summary>
        /// Executes a Möbius gyrovector addition of two points in the Poincaré ball: u ⊕ v.
        /// </summary>
        /// <param name="u">Left gyrovector operand.</param>
        /// <param name="v">Right gyrovector operand.</param>
        /// <param name="result">Pre-allocated span for storing the gyrovector sum.</param>
        public void MobiusAddition(ReadOnlySpan<float> u, ReadOnlySpan<float> v, Span<float> result)
        {
            if (u.Length != v.Length || u.Length != result.Length)
            {
                throw new ArgumentException("Operand and result vectors must have identical dimensions.");
            }

            float normSqU = 0.0f;
            float normSqV = 0.0f;
            float dotUV = 0.0f;

            for (int i = 0; i < u.Length; i++)
            {
                normSqU += u[i] * u[i];
                normSqV += v[i] * v[i];
                dotUV += u[i] * v[i];
            }

            float c = Curvature;
            float denominator = 1.0f + (2.0f * c * dotUV) + (c * c * normSqU * normSqV);
            if (MathF.Abs(denominator) < Epsilon)
            {
                denominator = Epsilon;
            }

            float termU = 1.0f + (2.0f * c * dotUV) + (c * normSqV);
            float termV = 1.0f - (c * normSqU);

            for (int i = 0; i < u.Length; i++)
            {
                result[i] = ((termU * u[i]) + (termV * v[i])) / denominator;
            }
        }
    }
}
