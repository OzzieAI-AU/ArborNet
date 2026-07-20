// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Models
{

    #region Using Statements:

    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Layers;
    using System.Collections.Generic;
    /// <summary>
    /// Implements the Llama 3 decoder-only transformer language model architecture.
    /// </summary>
    /// <remarks>
    /// This model consists of token embeddings, a stack of decoder-only transformer blocks,
    /// a final layer normalization, and a linear language modeling head.
    /// It inherits parameter management, weight tracking, and serialization functionality from <see cref="BaseModel"/>.
    /// </remarks>
    /// <seealso cref="BaseModel"/>

    #endregion

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
        /// Executes the forward pass of the Llama 3 model, processing input token IDs to compute output logits.
        /// </summary>
        /// <param name="input">The input tensor containing token IDs. Expected shape is <c>(batchSize, sequenceLength)</c>.</param>
        /// <returns>A tensor containing the unnormalized prediction scores (logits) for each token in the vocabulary. Shape is <c>(batchSize, sequenceLength, vocabSize)</c>.</returns>
        /// <remarks>
        /// The tensor transformations flow as follows:
        /// <list type="number">
        /// <item>
        /// <description>Tokens are converted to dense representations: <c>(batchSize, sequenceLength) -> (batchSize, sequenceLength, hiddenSize)</c>.</description>
        /// </item>
        /// <item>
        /// <description>Representations pass sequentially through all <paramref name="layers"/>: <c>(batchSize, sequenceLength, hiddenSize)</c>.</description>
        /// </item>
        /// <item>
        /// <description>The sequence output is normalized: <c>(batchSize, sequenceLength, hiddenSize)</c>.</description>
        /// </item>
        /// <item>
        /// <description>The linear head projects representations to logits: <c>(batchSize, sequenceLength, vocabSize)</c>.</description>
        /// </item>
        /// </list>
        /// </remarks>

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