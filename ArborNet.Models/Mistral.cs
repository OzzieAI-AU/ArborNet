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
    /// Implements the Mistral decoder-only transformer model architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Mistral is an efficient large language model featuring grouped-query attention (GQA),
    /// sliding window attention (SWA), and RMSNorm/LayerNorm. This implementation follows the original
    /// architecture utilizing an embedding layer, stacked <see cref="MistralBlock"/> transformer layers,
    /// final normalization, and a language modeling head.
    /// </para>
    /// <para>
    /// This model handles the orchestration of raw token tensor processing through the complete transformer stack, 
    /// transforming input indices into vocabulary-space probability distributions (logits).
    /// </para>
    /// </remarks>
    /// <seealso cref="BaseModel"/>
    /// <seealso cref="MistralBlock"/>

    #endregion

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
        /// Returns all trainable parameters of the complete Mistral model.
        /// </summary>
        /// <returns>
        /// An <see cref="IEnumerable{ITensor}"/> containing all model parameter tensors, including the embedding, 
        /// transformer layers, normalization parameters, and projection head weights.
        /// </returns>
        /// <remarks>
        /// Parameters are accumulated sequentially during model construction.
        /// </remarks>

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
        /// <param name="input">Input tensor containing token indices. Expected shape: <c>(batchSize, sequenceLength)</c>.</param>
        /// <returns>
        /// An <see cref="ITensor"/> representing unnormalized prediction logits of shape <c>(batchSize, sequenceLength, vocabSize)</c>.
        /// </returns>
        /// <remarks>
        /// The execution flow is as follows:
        /// <list type="number">
        /// <item><description>Convert input tokens to dense embeddings via the <see cref="embedding"/> layer.</description></item>
        /// <item><description>Sequentially process the embeddings through all stacked <see cref="layers"/>.</description></item>
        /// <item><description>Apply final normalization using the <see cref="norm"/> layer.</description></item>
        /// <item><description>Project normalized features to vocabulary dimension via the <see cref="head"/> linear layer.</description></item>
        /// </list>
        /// </remarks>

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