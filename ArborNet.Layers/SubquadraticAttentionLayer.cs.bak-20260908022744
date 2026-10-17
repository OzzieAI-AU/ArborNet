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

    #region Using Statements:

    using ArborNet.Core.Interfaces;
    using ArborNet.Layers;
    using System;
    using System.Buffers;
    using System.Collections;
    using System.Collections.Generic;
    using System.Data;
    using System.Threading.Tasks;
    /// <summary>
    /// Highly optimized Subquadratic Sparse Attention (SSA) Layer.
    /// Reduces memory and computational complexity from O(N^2) to O(N) using feature-map linear attention.
    /// Ideal for processing extremely long context sequences.
    /// </summary>

    #endregion

    public sealed class SubquadraticAttentionLayer
    {
        private readonly int _headCount;
        private readonly int _headDim;

        /// <summary>
        /// Initializes a new instance of the <see cref="SubquadraticAttentionLayer"/> class.
        /// </summary>
        /// <param name="headCount">The number of parallel attention heads.</param>
        /// <param name="headDim">The dimension size of each head.</param>
        public SubquadraticAttentionLayer(int headCount, int headDim)
        {
            if (headCount <= 0) throw new ArgumentOutOfRangeException(nameof(headCount));
            if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
            _headCount = headCount;
            _headDim = headDim;
        }
        /// <summary>
        /// Applies the ELU + 1 feature map to ensure strictly positive values,
        /// which allows bypassing the Softmax denominator bottleneck.
        /// </summary>
        /// <param name="x">The input value to be mapped.</param>
        /// <returns>The mapped value, guaranteed to be strictly positive.</returns>

        private static float FeatureMap(float x)
        {
            return x > 0f ? x + 1f : MathF.Exp(x);
        }
        /// <summary>
        /// Derivative of the ELU + 1 feature map.
        /// </summary>
        /// <param name="x">The input value at which to evaluate the derivative.</param>
        /// <returns>The derivative value of the feature map at <paramref name="x"/>.</returns>

        private static float FeatureMapGrad(float x)
        {
            return x > 0f ? 1f : MathF.Exp(x);
        }

        /// <summary>
        /// High-performance forward pass of the Subquadratic Attention.
        /// Input arrays represent contiguous flattened dimensions of shape [Batch, SeqLen, Heads, HeadDim].
        /// </summary>
        public void Forward(
            float[] Q, float[] K, float[] V, float[] Output,
            int batchSize, int seqLen)
        {
            int batchStride = seqLen * _headCount * _headDim;
            int seqStride = _headCount * _headDim;
            int kvSize = _headDim * _headDim;

            // Process each batch and head independently in parallel
            Parallel.For(0, batchSize * _headCount, index =>
            {
                int b = index / _headCount;
                int h = index % _headCount;

                // Safe stack allocation threshold (16KB) to avoid stack overflows
                bool useStack = kvSize <= 4096;
                float[]? rentedKv = null;
                float[]? rentedKSum = null;
                float[]? rentedQMapped = null;

                Span<float> kvContext = useStack ? stackalloc float[kvSize] : (rentedKv = ArrayPool<float>.Shared.Rent(kvSize));
                Span<float> kSum = useStack ? stackalloc float[_headDim] : (rentedKSum = ArrayPool<float>.Shared.Rent(_headDim));
                Span<float> qMapped = useStack ? stackalloc float[_headDim] : (rentedQMapped = ArrayPool<float>.Shared.Rent(_headDim));

                kvContext.Clear();
                kSum.Clear();

                // 1. Build the global KV-Context Matrix and K-Sum Vector: O(N) Complexity
                for (int t = 0; t < seqLen; t++)
                {
                    int tokenOffset = (b * batchStride) + (t * seqStride) + (h * _headDim);

                    for (int d1 = 0; d1 < _headDim; d1++)
                    {
                        float kVal = FeatureMap(K[tokenOffset + d1]);
                        kSum[d1] += kVal;

                        for (int d2 = 0; d2 < _headDim; d2++)
                        {
                            kvContext[d1 * _headDim + d2] += kVal * V[tokenOffset + d2];
                        }
                    }
                }

                // 2. Perform query multiplication and normalizer division: O(N) Complexity
                for (int t = 0; t < seqLen; t++)
                {
                    int tokenOffset = (b * batchStride) + (t * seqStride) + (h * _headDim);
                    float normalizer = 0f;

                    for (int d1 = 0; d1 < _headDim; d1++)
                    {
                        float qVal = FeatureMap(Q[tokenOffset + d1]);
                        qMapped[d1] = qVal;
                        normalizer += qVal * kSum[d1];
                    }

                    normalizer = normalizer > 1e-6f ? normalizer : 1e-6f;
                    float invNorm = 1f / normalizer;

                    for (int d2 = 0; d2 < _headDim; d2++)
                    {
                        float outVal = 0f;
                        for (int d1 = 0; d1 < _headDim; d1++)
                        {
                            outVal += qMapped[d1] * kvContext[d1 * _headDim + d2];
                        }
                        Output[tokenOffset + d2] = outVal * invNorm;
                    }
                }

                // Cleanup rented pool buffers if used
                if (rentedKv != null) ArrayPool<float>.Shared.Return(rentedKv);
                if (rentedKSum != null) ArrayPool<float>.Shared.Return(rentedKSum);
                if (rentedQMapped != null) ArrayPool<float>.Shared.Return(rentedQMapped);
            });
        }

        /// <summary>
        /// Exact mathematical backward pass of the Subquadratic Attention.
        /// Computes and returns the analytical gradients for Q, K, and V on the CPU.
        /// </summary>
        public void Backward(
            float[] Q, float[] K, float[] V, float[] GradOut,
            float[] GradQ, float[] GradK, float[] GradV,
            int batchSize, int seqLen)
        {
            int batchStride = seqLen * _headCount * _headDim;
            int seqStride = _headCount * _headDim;
            int kvSize = _headDim * _headDim;

            Parallel.For(0, batchSize * _headCount, index =>
            {
                int b = index / _headCount;
                int h = index % _headCount;

                bool useStack = kvSize <= 4096;
                float[]? rentedKv = null, rentedKSum = null, rentedGradKv = null, rentedGradKSum = null;

                Span<float> kvContext = useStack ? stackalloc float[kvSize] : (rentedKv = ArrayPool<float>.Shared.Rent(kvSize));
                Span<float> kSum = useStack ? stackalloc float[_headDim] : (rentedKSum = ArrayPool<float>.Shared.Rent(_headDim));
                Span<float> gradKv = useStack ? stackalloc float[kvSize] : (rentedGradKv = ArrayPool<float>.Shared.Rent(kvSize));
                Span<float> gradKSum = useStack ? stackalloc float[_headDim] : (rentedGradKSum = ArrayPool<float>.Shared.Rent(_headDim));

                kvContext.Clear();
                kSum.Clear();
                gradKv.Clear();
                gradKSum.Clear();

                // Recompute forward context matrices
                for (int t = 0; t < seqLen; t++)
                {
                    int offset = (b * batchStride) + (t * seqStride) + (h * _headDim);
                    for (int d1 = 0; d1 < _headDim; d1++)
                    {
                        float kVal = FeatureMap(K[offset + d1]);
                        kSum[d1] += kVal;
                        for (int d2 = 0; d2 < _headDim; d2++)
                        {
                            kvContext[d1 * _headDim + d2] += kVal * V[offset + d2];
                        }
                    }
                }

                // First backward pass: Compute gradients for Q, global context, and normalizer
                for (int t = 0; t < seqLen; t++)
                {
                    int offset = (b * batchStride) + (t * seqStride) + (h * _headDim);
                    float normalizer = 0f;

                    for (int d1 = 0; d1 < _headDim; d1++)
                    {
                        normalizer += FeatureMap(Q[offset + d1]) * kSum[d1];
                    }
                    normalizer = normalizer > 1e-6f ? normalizer : 1e-6f;
                    float invNorm = 1f / normalizer;

                    float oTGradOut = 0f;
                    for (int d2 = 0; d2 < _headDim; d2++)
                    {
                        float outVal = 0f;
                        for (int d1 = 0; d1 < _headDim; d1++)
                        {
                            outVal += FeatureMap(Q[offset + d1]) * kvContext[d1 * _headDim + d2];
                        }
                        oTGradOut += (outVal * invNorm) * GradOut[offset + d2];
                    }

                    // Compute gradient for Q
                    for (int d1 = 0; d1 < _headDim; d1++)
                    {
                        float term1 = 0f;
                        for (int d2 = 0; d2 < _headDim; d2++)
                        {
                            term1 += kvContext[d1 * _headDim + d2] * GradOut[offset + d2];
                        }
                        float term2 = oTGradOut * kSum[d1];
                        GradQ[offset + d1] = (term1 - term2) * invNorm * FeatureMapGrad(Q[offset + d1]);

                        // Accumulate gradients for K and V dependencies
                        gradKSum[d1] -= oTGradOut * invNorm * FeatureMap(Q[offset + d1]);
                        for (int d2 = 0; d2 < _headDim; d2++)
                        {
                            gradKv[d1 * _headDim + d2] += FeatureMap(Q[offset + d1]) * GradOut[offset + d2] * invNorm;
                        }
                    }
                }

                // Second backward pass: Compute gradients for K and V using accumulated context gradients
                for (int t = 0; t < seqLen; t++)
                {
                    int offset = (b * batchStride) + (t * seqStride) + (h * _headDim);

                    for (int d1 = 0; d1 < _headDim; d1++)
                    {
                        float kVal = FeatureMap(K[offset + d1]);
                        float kValGrad = FeatureMapGrad(K[offset + d1]);

                        float gradKAccum = gradKSum[d1];
                        for (int d2 = 0; d2 < _headDim; d2++)
                        {
                            // dL/dV_t = (dL/dS)^T * phi(K_t)
                            GradV[offset + d2] = gradKv[d1 * _headDim + d2] * kVal;

                            // Accumulate dL/dphi(K_t)
                            gradKAccum += gradKv[d1 * _headDim + d2] * V[offset + d2];
                        }

                        // Apply the chain rule through the feature map derivative
                        GradK[offset + d1] = gradKAccum * kValGrad;
                    }
                }

                if (rentedKv != null) ArrayPool<float>.Shared.Return(rentedKv);
                if (rentedKSum != null) ArrayPool<float>.Shared.Return(rentedKSum);
                if (rentedGradKv != null) ArrayPool<float>.Shared.Return(rentedGradKv);
                if (rentedGradKSum != null) ArrayPool<float>.Shared.Return(rentedGradKSum);
            });
        }
    }
}