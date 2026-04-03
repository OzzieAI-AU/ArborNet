using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Functional;

namespace ArborNet.Layers
{
    /// <summary>
    /// Implements an embedding layer that maps integer indices (token IDs) to dense vector representations.
    /// </summary>
    /// <remarks>
    /// This layer maintains a learnable lookup table of shape <c>(numEmbeddings, embeddingDim)</c>.
    /// It is a core component in NLP models, transformers, and sequence processing architectures.
    /// </remarks>
    public class Embedding : BaseLayer
    {
        /// <summary>
        /// The learnable embedding weight matrix of shape <c>(NumEmbeddings, EmbeddingDim)</c>.
        /// </summary>
        private ITensor weights;

        /// <summary>
        /// Gets the number of embeddings (vocabulary size).
        /// </summary>
        public int NumEmbeddings { get; }

        /// <summary>
        /// Gets the dimensionality of each embedding vector.
        /// </summary>
        public int EmbeddingDim { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Embedding"/> class.
        /// </summary>
        /// <param name="numEmbeddings">The size of the embedding dictionary (number of unique tokens the layer can embed).</param>
        /// <param name="embeddingDim">The size of each embedding vector.</param>
        public Embedding(int numEmbeddings, int embeddingDim)
        {
            NumEmbeddings = numEmbeddings;
            EmbeddingDim = embeddingDim;
            weights = Initializers.Normal(new TensorShape(numEmbeddings, embeddingDim));
        }

        /// <summary>
        /// Performs the forward pass of the embedding layer.
        /// </summary>
        /// <param name="indices">The input tensor containing integer indices to retrieve embeddings for.</param>
        /// <returns>A tensor containing the corresponding embedding vectors.</returns>
        /// <remarks>
        /// Current implementation is a temporary stub. It should be replaced with a proper gather/scatter
        /// operation that selects rows from the <see cref="weights"/> matrix based on the provided indices.
        /// </remarks>
        public override ITensor Forward(ITensor indices)
        {
            // Simple gather - replace with proper implementation when Gather is added
            return weights; // stub
        }

        /// <summary>
        /// Returns all trainable parameters of this layer.
        /// </summary>
        /// <returns>An enumerable containing the embedding weights tensor.</returns>
        public override IEnumerable<ITensor> Parameters()
        {
            yield return weights;
        }
    }
}