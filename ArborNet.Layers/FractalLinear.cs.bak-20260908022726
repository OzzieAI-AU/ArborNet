// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Layers.Fractal
{

    #region Using Statements:

    using ArborNet.Activations;
    using ArborNet.Core;
    using ArborNet.Core.Initializers;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;
    using System;
    using System.Collections.Generic;
    using System.Runtime.CompilerServices;
    using System.Threading.Tasks;
    /// <summary>
    /// Represents a fully connected (linear) layer initialized using fractal patterns.
    /// Supports optional bias tensor addition.
    /// </summary>

    #endregion
    public class FractalLinear : ILayer
    {
        /// <summary>
        /// Gets the weight tensor of the linear layer, initialized fractally.
        /// </summary>
        public ITensor Weights { get; }
        /// <summary>
        /// Gets the bias tensor of the linear layer, initialized to zeros.
        /// </summary>
        public ITensor Bias { get; }
        private readonly bool _useBias;

        public FractalLinear(int inFeatures, int outFeatures, FractalType initType, bool useBias = true)
        {
            Weights = FractalInitializers.Generate(inFeatures, outFeatures, initType);
            _useBias = useBias;

            if (useBias)
            {
                Bias = Tensor.Zeros(new TensorShape(outFeatures));
            }
        }
        /// <summary>
        /// Performs the forward pass of the linear layer.
        /// Computes matrix multiplication of input and weights, adding bias if enabled.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The resulting tensor after projection and optional bias addition.</returns>

        public ITensor Forward(ITensor input)
        {
            var output = input.MatMul(Weights);
            if (_useBias)
            {
                output = output.Add(Bias);
            }
            return output;
        }
        /// <summary>
        /// Gets the learnable parameters of the layer.
        /// </summary>
        /// <returns>An enumerable collection containing the weights and optional bias tensors.</returns>

        public IEnumerable<ITensor> Parameters()
        {
            yield return Weights;
            if (_useBias) yield return Bias;
        }
    }
    /// <summary>
    /// Represents a subquadratic attention layer that computes self-attention 
    /// with linear complexity relative to the sequence length.
    /// </summary>

    public class SubquadraticAttention : ILayer
    {
        private readonly FractalLinear _qProj, _kProj, _vProj, _oProj;
        private readonly int _headCount;
        private readonly int _headDim;
        private readonly int _dModel;

        public SubquadraticAttention(int dModel, int headCount, FractalType initType)
        {
            _dModel = dModel;
            _headCount = headCount;
            _headDim = dModel / headCount;

            _qProj = new FractalLinear(dModel, dModel, initType, false);
            _kProj = new FractalLinear(dModel, dModel, initType, false);
            _vProj = new FractalLinear(dModel, dModel, initType, false);
            _oProj = new FractalLinear(dModel, dModel, initType, false);
        }
        /// <summary>
        /// Feature map applied to Queries and Keys to approximate the softmax attention matrix.
        /// Ensures non-negative values for linear attention calculations.
        /// </summary>
        /// <param name="x">The raw element value.</param>
        /// <returns>A non-negative projected value.</returns>

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float FeatureMap(float x) => x > 0f ? x + 1f : MathF.Exp(x);
        /// <summary>
        /// Performs the forward pass of the linear layer.
        /// Computes matrix multiplication of input and weights, adding bias if enabled.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The resulting tensor after projection and optional bias addition.</returns>

        public ITensor Forward(ITensor input)
        {
            var shape = input.Shape;
            int batchSize = shape[0];
            int seqLen = shape[1];

            var Q_tensor = _qProj.Forward(input);
            var K_tensor = _kProj.Forward(input);
            var V_tensor = _vProj.Forward(input);

            float[] Q = Q_tensor.ToArray();
            float[] K = K_tensor.ToArray();
            float[] V = V_tensor.ToArray();
            float[] Output = new float[Q.Length];

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
                        float kVal = FeatureMap(K[offset + d1]);
                        kSum[d1] += kVal;
                        for (int d2 = 0; d2 < _headDim; d2++)
                        {
                            kvContext[d1 * _headDim + d2] += kVal * V[offset + d2];
                        }
                    }
                }

                for (int t = 0; t < seqLen; t++)
                {
                    int offset = (b * batchStride) + (t * seqStride) + (h * _headDim);
                    float normalizer = 0f;

                    for (int d1 = 0; d1 < _headDim; d1++)
                    {
                        float qVal = FeatureMap(Q[offset + d1]);
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

            var rawOutputTensor = Tensor.FromArray(Output, new TensorShape(batchSize, seqLen, _dModel));
            return _oProj.Forward(rawOutputTensor);
        }
        /// <summary>
        /// Gets the learnable parameters of the layer.
        /// </summary>
        /// <returns>An enumerable collection containing the weights and optional bias tensors.</returns>

        public IEnumerable<ITensor> Parameters()
        {
            foreach (var p in _qProj.Parameters()) yield return p;
            foreach (var p in _kProj.Parameters()) yield return p;
            foreach (var p in _vProj.Parameters()) yield return p;
            foreach (var p in _oProj.Parameters()) yield return p;
        }
    }
    /// <summary>
    /// Represents a complete Transformer block constructed using fractal layers,
    /// subquadratic attention, layer normalization, and a feed-forward network.
    /// </summary>

    public class FractalTransformerBlock : ILayer
    {
        private readonly LayerNorm _norm1;
        private readonly SubquadraticAttention _attention;

        private readonly LayerNorm _norm2;
        private readonly FractalLinear _ffn1;
        private readonly Gelu _gelu;
        private readonly FractalLinear _ffn2;

        public FractalTransformerBlock(int dModel, int dFF, int headCount, FractalType initType)
        {
            _norm1 = new LayerNorm(new[] { dModel });
            _attention = new SubquadraticAttention(dModel, headCount, initType);

            _norm2 = new LayerNorm(new[] { dModel });
            _ffn1 = new FractalLinear(dModel, dFF, initType);
            _gelu = new Gelu();
            _ffn2 = new FractalLinear(dFF, dModel, initType);
        }
        /// <summary>
        /// Performs the forward pass of the linear layer.
        /// Computes matrix multiplication of input and weights, adding bias if enabled.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The resulting tensor after projection and optional bias addition.</returns>

        public ITensor Forward(ITensor input)
        {
            var norm1Out = _norm1.Forward(input);
            var attnOut = _attention.Forward(norm1Out);
            var x = input.Add(attnOut);

            var norm2Out = _norm2.Forward(x);
            var ff1Out = _ffn1.Forward(norm2Out);
            var geluOut = _gelu.Forward(ff1Out);
            var ff2Out = _ffn2.Forward(geluOut);
            x = x.Add(ff2Out);

            return x;
        }
        /// <summary>
        /// Gets the learnable parameters of the layer.
        /// </summary>
        /// <returns>An enumerable collection containing the weights and optional bias tensors.</returns>

        public IEnumerable<ITensor> Parameters()
        {
            foreach (var p in _norm1.Parameters()) yield return p;
            foreach (var p in _attention.Parameters()) yield return p;
            foreach (var p in _norm2.Parameters()) yield return p;
            foreach (var p in _ffn1.Parameters()) yield return p;
            foreach (var p in _ffn2.Parameters()) yield return p;
        }
    }
}
