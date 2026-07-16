using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace ArborNet.ML.Layers
{
    /// <summary>
    /// Mathematically rigorous, autograd-compliant O(N) Subquadratic Linear Attention Operation.
    /// Eliminates the memory footprint of standard attention while maintaining complete backpropagation capabilities.
    /// </summary>
    public sealed class SubquadraticAttentionOp : IAutogradOperation
    {
        private readonly int _headCount;
        private readonly int _headDim;
        private readonly int _dModel;

        // Cached forward states required for backward pass
        private float[]? _cachedQ;
        private float[]? _cachedK;
        private float[]? _cachedV;
        private TensorShape? _inputShape;

        public SubquadraticAttentionOp(int headCount, int headDim, int dModel)
        {
            _headCount = headCount;
            _headDim = headDim;
            _dModel = dModel;
        }

        private static float FeatureMap(float x) => x > 0f ? x + 1f : MathF.Exp(x);
        private static float FeatureMapGrad(float x) => x > 0f ? 1f : MathF.Exp(x);

        public ITensor Forward(params ITensor[] inputs)
        {
            var Q_tensor = inputs[0];
            var K_tensor = inputs[1];
            var V_tensor = inputs[2];

            _inputShape = Q_tensor.Shape;
            _cachedQ = Q_tensor.ToArray();
            _cachedK = K_tensor.ToArray();
            _cachedV = V_tensor.ToArray();

            int batchSize = Q_tensor.Shape[0];
            int seqLen = Q_tensor.Shape[1];
            float[] Output = new float[_cachedQ.Length];

            int batchStride = seqLen * _headCount * _headDim;
            int seqStride = _headCount * _headDim;
            int kvContextSize = _headDim * _headDim;

            Parallel.For(0, batchSize * _headCount, index =>
            {
                int b = index / _headCount;
                int h = index % _headCount;

                Span<float> kvContext = stackalloc float[kvContextSize];
                Span<float> kSum = stackalloc float[_headDim];
                Span<float> qMapped = stackalloc float[_headDim];

                kvContext.Clear();
                kSum.Clear();

                for (int t = 0; t < seqLen; t++)
                {
                    int offset = (b * batchStride) + (t * seqStride) + (h * _headDim);
                    for (int d1 = 0; d1 < _headDim; d1++)
                    {
                        float kVal = FeatureMap(_cachedK[offset + d1]);
                        kSum[d1] += kVal;
                        for (int d2 = 0; d2 < _headDim; d2++)
                        {
                            kvContext[d1 * _headDim + d2] += kVal * _cachedV[offset + d2];
                        }
                    }
                }

                for (int t = 0; t < seqLen; t++)
                {
                    int offset = (b * batchStride) + (t * seqStride) + (h * _headDim);
                    float normalizer = 0f;

                    for (int d1 = 0; d1 < _headDim; d1++)
                    {
                        float qVal = FeatureMap(_cachedQ[offset + d1]);
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
                        Output[offset + d2] = outVal * invNorm;
                    }
                }
            });

            return Tensor.FromArray(Output, Q_tensor.Shape, Q_tensor.Device);
        }

        public IList<ITensor?> Backward(ITensor gradOutput)
        {
            if (_cachedQ == null || _cachedK == null || _cachedV == null || _inputShape == null)
                throw new InvalidOperationException("Backward pass called without matching forward pass caching.");

            int batchSize = _inputShape[0];
            int seqLen = _inputShape[1];

            float[] GradOut = gradOutput.ToArray();
            float[] GradQ = new float[_cachedQ.Length];
            float[] GradK = new float[_cachedK.Length];
            float[] GradV = new float[_cachedV.Length];

            int batchStride = seqLen * _headCount * _headDim;
            int seqStride = _headCount * _headDim;
            int kvContextSize = _headDim * _headDim;

            Parallel.For(0, batchSize * _headCount, index =>
            {
                int b = index / _headCount;
                int h = index % _headCount;

                Span<float> kvContext = stackalloc float[kvContextSize];
                Span<float> kSum = stackalloc float[_headDim];
                Span<float> gradKv = stackalloc float[kvContextSize];
                Span<float> gradKSum = stackalloc float[_headDim];

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
                        float kVal = FeatureMap(_cachedK[offset + d1]);
                        kSum[d1] += kVal;
                        for (int d2 = 0; d2 < _headDim; d2++)
                        {
                            kvContext[d1 * _headDim + d2] += kVal * _cachedV[offset + d2];
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
                        normalizer += FeatureMap(_cachedQ[offset + d1]) * kSum[d1];
                    }
                    normalizer = normalizer > 1e-6f ? normalizer : 1e-6f;
                    float invNorm = 1f / normalizer;

                    float oTGradOut = 0f;
                    for (int d2 = 0; d2 < _headDim; d2++)
                    {
                        float outVal = 0f;
                        for (int d1 = 0; d1 < _headDim; d1++)
                        {
                            outVal += FeatureMap(_cachedQ[offset + d1]) * kvContext[d1 * _headDim + d2];
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
                        GradQ[offset + d1] = (term1 - term2) * invNorm * FeatureMapGrad(_cachedQ[offset + d1]);

                        // Accumulate gradients for K and V dependencies
                        gradKSum[d1] -= oTGradOut * invNorm * FeatureMap(_cachedQ[offset + d1]);
                        for (int d2 = 0; d2 < _headDim; d2++)
                        {
                            gradKv[d1 * _headDim + d2] += FeatureMap(_cachedQ[offset + d1]) * GradOut[offset + d2] * invNorm;
                        }
                    }
                }

                // Second backward pass: Compute gradients for K and V using accumulated context gradients
                for (int t = 0; t < seqLen; t++)
                {
                    int offset = (b * batchStride) + (t * seqStride) + (h * _headDim);

                    for (int d1 = 0; d1 < _headDim; d1++)
                    {
                        float kVal = FeatureMap(_cachedK[offset + d1]);
                        float kValGrad = FeatureMapGrad(_cachedK[offset + d1]);

                        float gradKAccum = gradKSum[d1];
                        for (int d2 = 0; d2 < _headDim; d2++)
                        {
                            GradV[offset + d2] = gradKv[d1 * _headDim + d2] * kVal;
                            gradKAccum += gradKv[d1 * _headDim + d2] * _cachedV[offset + d2];
                        }

                        GradK[offset + d1] = gradKAccum * kValGrad;
                    }
                }
            });

            var device = gradOutput.Device;
            return new List<ITensor?>
            {
                Tensor.FromArray(GradQ, _inputShape, device),
                Tensor.FromArray(GradK, _inputShape, device),
                Tensor.FromArray(GradV, _inputShape, device)
            };
        }
    }
}
