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
    /// Represents the GPT-NeoX decoder-only autoregressive transformer architecture.
    /// </summary>
    /// <remarks>
    /// GPT-NeoX is a large-scale autoregressive language model featuring 
    /// parallel attention and feed-forward sublayers, rotary positional embeddings,
    /// and a final language modeling head. This implementation composes 
    /// token embeddings, positional encodings, stacked transformer blocks,
    /// and an output projection layer.
    /// </remarks>
    /// <seealso cref="BaseModel" />

    #endregion

    public class GPTNeoX : BaseModel
    {
        /// <summary>
        /// Token embedding layer mapping vocabulary indices to dense vectors.
        /// </summary>
        private readonly Embedding embedding;

        /// <summary>
        /// Positional encoding layer that injects position information into token embeddings.
        /// </summary>
        private readonly PositionalEncoding posEncoding;

        /// <summary>
        /// Collection of transformer blocks forming the core of the model.
        /// </summary>
        private readonly List<TransformerBlock> layers;

        /// <summary>
        /// Final linear projection layer that converts hidden states to vocabulary logits.
        /// </summary>
        private readonly Linear output;

        /// <summary>
        /// Initializes a new instance of the <see cref="GPTNeoX"/> class.
        /// </summary>
        /// <param name="vocabSize">Size of the vocabulary (number of distinct tokens).</param>
        /// <param name="hiddenSize">Dimensionality of the embeddings and hidden states.</param>
        /// <param name="numLayers">Number of transformer layers (depth of the model).</param>
        /// <param name="numHeads">Number of attention heads in each transformer layer.</param>
        /// <param name="maxSeqLen">Maximum sequence length for which positional encodings are computed.</param>
        public GPTNeoX(int vocabSize, int hiddenSize, int numLayers, int numHeads, int maxSeqLen)
        {
            embedding = new Embedding(vocabSize, hiddenSize);
            posEncoding = new PositionalEncoding(hiddenSize, maxSeqLen);
            layers = new List<TransformerBlock>();
            for (int i = 0; i < numLayers; i++)
                layers.Add(new TransformerBlock(hiddenSize, numHeads));

            output = new Linear(hiddenSize, vocabSize);

            parameters.AddRange(embedding.Parameters());
            parameters.AddRange(posEncoding.Parameters());
            foreach (var l in layers) parameters.AddRange(l.Parameters());
            parameters.AddRange(output.Parameters());
        }
        /// <summary>
        /// Performs a forward pass of the GPT-NeoX model, processing the input token tensor through 
        /// embedding, positional encoding, stacked transformer blocks, and output projection.
        /// </summary>
        /// <param name="input">The input tensor containing token indices of shape <c>(batchSize, sequenceLength)</c>.</param>
        /// <returns>
        /// An <see cref="ITensor"/> containing the calculated logits for each vocabulary token, 
        /// with shape <c>(batchSize, sequenceLength, vocabSize)</c>.
        /// </returns>

        public override ITensor Forward(ITensor input)
        {
            var x = embedding.Forward(input);
            x = posEncoding.Forward(x);
            foreach (var layer in layers)
                x = layer.Forward(x);
            return output.Forward(x);
        }
    }
}