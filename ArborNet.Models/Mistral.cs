using ArborNet.Core.Interfaces;
using ArborNet.Core.Models;
using ArborNet.Layers;
using System.Collections.Generic;

namespace ArborNet.Models
{
    /// <summary>
    /// Implements the Mistral decoder-only transformer model architecture.
    /// </summary>
    /// <remarks>
    /// Mistral is an efficient large language model featuring grouped-query attention (GQA),
    /// sliding window attention, and RMSNorm. This implementation follows the original
    /// architecture with an embedding layer, stacked MistralBlocks, final normalization,
    /// and a language modeling head.
    /// </remarks>
    public class Mistral : BaseModel
    {
        /// <summary>
        /// Token embedding layer that maps vocabulary indices to dense hidden representations.
        /// </summary>
        private readonly Embedding embedding;

        /// <summary>
        /// Collection of Mistral transformer blocks (layers) that process the hidden states.
        /// </summary>
        private readonly List<MistralBlock> layers;

        /// <summary>
        /// Final layer normalization applied to the output of the last transformer block.
        /// </summary>
        private readonly LayerNorm norm;

        /// <summary>
        /// Language modeling head that projects the final hidden states to vocabulary logits.
        /// </summary>
        private readonly Linear head;

        /// <summary>
        /// Returns all trainable parameters of the Mistral model.
        /// </summary>
        /// <returns>An enumerable collection of all parameter tensors in the model.</returns>
        public override IEnumerable<ITensor> Parameters() => parameters;

        /// <summary>
        /// Initializes a new instance of the <see cref="Mistral"/> class.
        /// </summary>
        /// <param name="vocabSize">Size of the vocabulary (number of tokens).</param>
        /// <param name="hiddenDim">Dimensionality of the hidden representations (model dimension).</param>
        /// <param name="numLayers">Number of transformer layers (depth of the model).</param>
        /// <param name="numHeads">Number of attention heads.</param>
        /// <param name="kvHeads">Number of key-value heads for grouped-query attention.</param>
        /// <param name="slidingWindow">Size of the sliding window for attention mechanism.</param>
        public Mistral(int vocabSize, int hiddenDim, int numLayers, int numHeads, int kvHeads, int slidingWindow)
        {
            embedding = new Embedding(vocabSize, hiddenDim);
            layers = new List<MistralBlock>();
            for (int i = 0; i < numLayers; i++)
                layers.Add(new MistralBlock(hiddenDim, numHeads, kvHeads, slidingWindow));

            norm = new LayerNorm(new[] { hiddenDim });
            head = new Linear(hiddenDim, vocabSize);

            parameters.AddRange(embedding.Parameters());
            foreach (var l in layers) parameters.AddRange(l.Parameters());
            parameters.AddRange(norm.Parameters());
            parameters.AddRange(head.Parameters());
        }

        /// <summary>
        /// Performs a forward pass through the complete Mistral model.
        /// </summary>
        /// <param name="input">Input tensor containing token indices. Expected shape: (batchSize, sequenceLength).</param>
        /// <returns>
        /// Output logits tensor of shape (batchSize, sequenceLength, vocabSize) 
        /// representing unnormalized scores for each token in the vocabulary.
        /// </returns>
        public override ITensor Forward(ITensor input)
        {
            var x = embedding.Forward(input);
            foreach (var layer in layers)
                x = layer.Forward(x);
            x = norm.Forward(x);
            return head.Forward(x);
        }
    }
}