namespace ArborNet.Layers.Fractal
{


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
    /// A dense linear projection initialized with Fractal Weights.
    /// Uses native ArborNet Core operations without Fluent wrappers.
    /// </summary>
    public class FractalLinear : ILayer
    {
        public ITensor Weights { get; }
        public ITensor Bias { get; }
        private readonly bool _useBias;

        public FractalLinear(int inFeatures, int outFeatures, FractalType initType, bool useBias = true)
        {
            Weights = FractalInitializers.Generate(inFeatures, outFeatures, initType);
            _useBias = useBias;

            if (useBias)
            {
                // Replaced X.Zeros with native Tensor.Zeros
                Bias = Tensor.Zeros(new TensorShape(outFeatures));
            }
        }

        public ITensor Forward(ITensor input)
        {
            // Standard core tensor operations
            var output = input.MatMul(Weights);
            if (_useBias)
            {
                output = output.Add(Bias);
            }
            return output;
        }

        public IEnumerable<ITensor> Parameters()
        {
            yield return Weights;
            if (_useBias) yield return Bias;
        }
    }

    /// <summary>
    /// ArborNet ILayer wrapper for Subquadratic Sparse Attention (O(N) Complexity).
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float FeatureMap(float x) => x > 0f ? x + 1f : MathF.Exp(x);

        public ITensor Forward(ITensor input)
        {
            var shape = input.Shape;
            int batchSize = shape[0];
            int seqLen = shape[1];

            var Q_tensor = _qProj.Forward(input);
            var K_tensor = _kProj.Forward(input);
            var V_tensor = _vProj.Forward(input);

            // Replaced Fluent X.Of() with direct casting to Tensor to access ToArray()
            float[] Q = ((Tensor)Q_tensor).ToArray();
            float[] K = ((Tensor)K_tensor).ToArray();
            float[] V = ((Tensor)V_tensor).ToArray();
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

            // Replaced Fluent X.FromArray with native Tensor.FromArray
            var rawOutputTensor = Tensor.FromArray(Output, new TensorShape(batchSize, seqLen, _dModel));

            return _oProj.Forward(rawOutputTensor);
        }

        public IEnumerable<ITensor> Parameters()
        {
            foreach (var p in _qProj.Parameters()) yield return p;
            foreach (var p in _kProj.Parameters()) yield return p;
            foreach (var p in _vProj.Parameters()) yield return p;
            foreach (var p in _oProj.Parameters()) yield return p;
        }
    }

    /// <summary>
    /// A single block combining Subquadratic Attention and FFN.
    /// Completely rewritten to use explicit core layer instantiations instead of Fluent chains.
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
            // Explicitly instantiate core layers to avoid Fluent dependencies
            _norm1 = new LayerNorm(new[] { dModel });
            _attention = new SubquadraticAttention(dModel, headCount, initType);

            _norm2 = new LayerNorm(new[] { dModel });
            _ffn1 = new FractalLinear(dModel, dFF, initType);
            _gelu = new Gelu();
            _ffn2 = new FractalLinear(dFF, dModel, initType);
        }

        public ITensor Forward(ITensor input)
        {
            // 1. Attention Block with residual connection
            var norm1Out = _norm1.Forward(input);
            var attnOut = _attention.Forward(norm1Out);
            var x = input.Add(attnOut);

            // 2. Feed-Forward Block with residual connection
            var norm2Out = _norm2.Forward(x);
            var ff1Out = _ffn1.Forward(norm2Out);
            var geluOut = _gelu.Forward(ff1Out);
            var ff2Out = _ffn2.Forward(geluOut);
            x = x.Add(ff2Out);

            return x;
        }

        public IEnumerable<ITensor> Parameters()
        {
            // We must also yield the parameters from the LayerNorms now that they are explicit fields
            foreach (var p in _norm1.Parameters()) yield return p;
            foreach (var p in _attention.Parameters()) yield return p;

            foreach (var p in _norm2.Parameters()) yield return p;
            foreach (var p in _ffn1.Parameters()) yield return p;
            foreach (var p in _ffn2.Parameters()) yield return p;
        }
    }
}