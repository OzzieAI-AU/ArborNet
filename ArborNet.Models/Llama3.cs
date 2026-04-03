using ArborNet.Core.Interfaces;
using ArborNet.Core.Models;
using ArborNet.Layers;
using System.Collections.Generic;

namespace ArborNet.Models
{
    /// <summary>
    /// Implements the Llama 3 decoder-only transformer language model.
    /// </summary>
    /// <remarks>
    /// This model consists of token embeddings, a stack of transformer blocks,
    /// final layer normalization, and a linear language modeling head.
    /// It inherits parameter management and base functionality from <see cref="BaseModel"/>.
    /// </remarks>
    public class Llama3 : BaseModel
    {
        /// <summary>
        /// Token embedding layer mapping vocabulary indices to dense vectors.
        /// </summary>
        private readonly Embedding tokenEmbedding;

        /// <summary>
        /// Collection of transformer decoder blocks forming the core of the model.
        /// </summary>
        private readonly List<TransformerBlock> layers;

        /// <summary>
        /// Final layer normalization applied to the output of the last transformer layer.
        /// </summary>
        private readonly LayerNorm norm;

        /// <summary>
        /// Linear projection layer that converts hidden states to vocabulary-sized logits.
        /// </summary>
        private readonly Linear head;

        /// <summary>
        /// Initializes a new instance of the <see cref="Llama3"/> class.
        /// </summary>
        /// <param name="vocabSize">The size of the vocabulary (number of distinct tokens).</param>
        /// <param name="hiddenSize">The dimensionality of the model embeddings and hidden states.</param>
        /// <param name="numLayers">The number of transformer layers in the model.</param>
        /// <param name="numHeads">The number of attention heads in each transformer layer.</param>
        /// <param name="maxSeqLen">The maximum sequence length supported by the model.</param>
        public Llama3(int vocabSize, int hiddenSize, int numLayers, int numHeads, int maxSeqLen)
        {
            tokenEmbedding = new Embedding(vocabSize, hiddenSize);
            layers = new List<TransformerBlock>();
            for (int i = 0; i < numLayers; i++)
                layers.Add(new TransformerBlock(hiddenSize, numHeads));

            norm = new LayerNorm(new[] { hiddenSize });
            head = new Linear(hiddenSize, vocabSize);

            parameters.AddRange(tokenEmbedding.Parameters());
            foreach (var l in layers) parameters.AddRange(l.Parameters());
            parameters.AddRange(norm.Parameters());
            parameters.AddRange(head.Parameters());
        }

        /// <summary>
        /// Performs a forward pass through the Llama 3 model.
        /// </summary>
        /// <param name="input">The input tensor containing token indices. Expected shape is (batchSize, sequenceLength).</param>
        /// <returns>The output logits tensor with shape (batchSize, sequenceLength, vocabSize).</returns>
        public override ITensor Forward(ITensor input)
        {
            var x = tokenEmbedding.Forward(input);
            foreach (var layer in layers)
                x = layer.Forward(x);
            x = norm.Forward(x);
            return head.Forward(x);
        }
    }
}