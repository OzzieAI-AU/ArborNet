using ArborNet.Core.Functional;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Layers;
using ArborNet.Core.Tensors;
using System;
using System.Collections.Generic;

namespace ArborNet.Layers
{
    /// <summary>
    /// Thread-safe categorical embedding table layer.
    /// </summary>
    public class Embedding : BaseLayer
    {
        private ITensor _weights;
        public int NumEmbeddings { get; }
        public int EmbeddingDim { get; }

        public Embedding(int numEmbeddings, int embeddingDim)
        {
            NumEmbeddings = numEmbeddings;
            EmbeddingDim = embeddingDim;
            _weights = Initializers.Normal(new TensorShape(numEmbeddings, embeddingDim));
            _weights.RequiresGrad = true;
        }

        public override ITensor Forward(ITensor indices)
        {
            if (indices == null) throw new ArgumentNullException(nameof(indices));

            float[] idxData = indices.ToArray();
            float[] wData = _weights.ToArray();
            float[] outData = new float[idxData.Length * EmbeddingDim];

            for (int i = 0; i < idxData.Length; i++)
            {
                int tokenIdx = (int)idxData[i];
                if (tokenIdx < 0 || tokenIdx >= NumEmbeddings)
                    throw new IndexOutOfRangeException($"Token index {tokenIdx} is out of bounds for vocab size {NumEmbeddings}.");

                int wOffset = tokenIdx * EmbeddingDim;
                int outOffset = i * EmbeddingDim;

                for (int d = 0; d < EmbeddingDim; d++)
                {
                    outData[outOffset + d] = wData[wOffset + d];
                }
            }

            var outputShapeList = new List<int>(indices.Shape.Dimensions) { EmbeddingDim };
            var result = Tensor.FromArray(outData, new TensorShape(outputShapeList.ToArray()), indices.Device);

            if (_weights.RequiresGrad)
            {
                var capturedIndices = indices;
                var capturedWeights = _weights;

                result.GradFn = gradOut =>
                {
                    float[] goData = gradOut.ToArray();
                    float[] indicesArray = capturedIndices.ToArray();
                    float[] gradWeightsData = new float[capturedWeights.Shape.TotalElements];

                    for (int i = 0; i < indicesArray.Length; i++)
                    {
                        int tokenIdx = (int)indicesArray[i];
                        if (tokenIdx >= 0 && tokenIdx < NumEmbeddings)
                        {
                            int wOffset = tokenIdx * EmbeddingDim;
                            int outOffset = i * EmbeddingDim;

                            for (int d = 0; d < EmbeddingDim; d++)
                            {
                                gradWeightsData[wOffset + d] += goData[outOffset + d];
                            }
                        }
                    }

                    var gradWeights = Tensor.FromArray(gradWeightsData, capturedWeights.Shape, gradOut.Device);

                    // Thread-Safe Atomic gradient accumulation
                    capturedWeights.AccumulateGrad(gradWeights);

                    return Tensor.Zeros(capturedIndices.Shape, capturedIndices.Device);
                };
            }

            return result;
        }

        public override IEnumerable<ITensor> Parameters()
        {
            yield return _weights;
        }
    }
}