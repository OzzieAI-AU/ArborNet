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

    using ArborNet.Core.Functional;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    /// <summary>
    /// Represents a thread-safe categorical embedding table layer that maps integer indices to dense vectors.
    /// </summary>

    #endregion

    public class Embedding : BaseLayer
    {
        private ITensor _weights;
        /// <summary>
        /// Gets the size of the dictionary of embeddings (vocabulary size).
        /// </summary>
        public int NumEmbeddings { get; }
        /// <summary>
        /// Gets the size of each embedding vector.
        /// </summary>
        public int EmbeddingDim { get; }

        public Embedding(int numEmbeddings, int embeddingDim)
        {
            NumEmbeddings = numEmbeddings;
            EmbeddingDim = embeddingDim;
            _weights = Initializers.Normal(new TensorShape(numEmbeddings, embeddingDim));
            _weights.RequiresGrad = true;
        }
        /// <summary>
        /// Performs the forward pass of the embedding layer, mapping input indices to their corresponding embedding vectors.
        /// </summary>
        /// <param name="indices">A tensor containing index values to retrieve from the embedding table.</param>
        /// <returns>A tensor containing the retrieved embedding vectors with the additional embedding dimension appended.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="indices"/> is null.</exception>
        /// <exception cref="IndexOutOfRangeException">Thrown when an index in <paramref name="indices"/> is out of bounds for the vocabulary size.</exception>

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
        /// <summary>
        /// Retrieves the learnable parameters of this layer.
        /// </summary>
        /// <returns>An enumerable collection containing the weight tensor of the embedding layer.</returns>

        public override IEnumerable<ITensor> Parameters()
        {
            yield return _weights;
        }
    }
}