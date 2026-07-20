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

    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Layers;
    using System;
    using System.Collections.Generic;
    /// <summary>
    /// Represents a production-grade Generative Pre-trained Transformer (GPT) model architecture.
    /// </summary>
    /// <remarks>
    /// This model features causal self-attention, layer normalization, and full autograd support,
    /// designed for sequence-to-sequence autoregressive tasks using stable ArborNet primitives.
    /// </remarks>

    #endregion

    public class GPT : BaseModel
    {
        /// <summary>
        /// Token embedding layer mapping vocabulary indices to dense vectors.
        /// </summary>
        private readonly Embedding tokenEmbedding;

        /// <summary>
        /// Positional encoding layer that adds position-based information to token embeddings.
        /// </summary>
        private readonly PositionalEncoding posEmbedding;

        /// <summary>
        /// Stack of transformer blocks implementing the core model computation.
        /// </summary>
        private readonly List<TransformerBlock> layers;

        /// <summary>
        /// Final layer normalization applied before the output projection head.
        /// </summary>
        private readonly LayerNorm finalNorm;

        /// <summary>
        /// Linear output head projecting final hidden states to vocabulary logits.
        /// </summary>
        private readonly Linear outputHead;

        /// <summary>
        /// Maximum sequence length supported by the positional encoding.
        /// </summary>
        private readonly int _maxSeqLen;

        /// <summary>
        /// Initializes a new instance of the <see cref="GPT"/> class with the specified architecture.
        /// </summary>
        /// <param name="vocabSize">The size of the vocabulary.</param>
        /// <param name="nLayers">The number of transformer layers.</param>
        /// <param name="nHeads">The number of attention heads in each transformer block.</param>
        /// <param name="dModel">The model dimension (embedding dimension and hidden size).</param>
        /// <param name="maxSeqLen">The maximum sequence length this model can process.</param>
        /// <param name="device">The compute device to use for parameter allocation. Defaults to CPU if null.</param>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown when <paramref name="vocabSize"/> or <paramref name="nLayers"/> is less than or equal to zero.
        /// </exception>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="dModel"/> is not divisible by <paramref name="nHeads"/>.
        /// </exception>
        public GPT(int vocabSize, int nLayers, int nHeads, int dModel, int maxSeqLen, Device device = null)
        {
            if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
            if (nLayers <= 0) throw new ArgumentOutOfRangeException(nameof(nLayers));
            if (dModel % nHeads != 0) throw new ArgumentException("dModel must be divisible by nHeads");

            device ??= Device.CPU;
            _maxSeqLen = maxSeqLen;

            tokenEmbedding = new Embedding(vocabSize, dModel);
            posEmbedding = new PositionalEncoding(dModel, maxSeqLen, device);
            layers = new List<TransformerBlock>();
            for (int i = 0; i < nLayers; i++)
                layers.Add(new TransformerBlock(dModel, nHeads));

            finalNorm = new LayerNorm(new[] { dModel });
            outputHead = new Linear(dModel, vocabSize, device);

            parameters.AddRange(tokenEmbedding.Parameters());
            parameters.AddRange(posEmbedding.Parameters());
            foreach (var l in layers)
                parameters.AddRange(l.Parameters());
            parameters.AddRange(finalNorm.Parameters());
            parameters.AddRange(outputHead.Parameters());
        }
        /// <summary>
        /// Performs a forward pass through the GPT model.
        /// </summary>
        /// <param name="input">The input tensor containing token IDs with shape <c>[batch, sequenceLength]</c>.</param>
        /// <returns>A logit tensor with shape <c>[batch, sequenceLength, vocabSize]</c>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>
        /// <exception cref="ArgumentException">
        /// Thrown when the rank of <paramref name="input"/> is less than 2, or when the input sequence length exceeds <see cref="_maxSeqLen"/>.
        /// </exception>

        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input);

            if (input.Shape[input.Shape.Rank - 1] > _maxSeqLen)
                throw new ArgumentException($"Sequence length exceeds maximum {_maxSeqLen}");

            var x = tokenEmbedding.Forward(input);
            x = posEmbedding.Forward(x);

            foreach (var layer in layers)
                x = layer.Forward(x);

            x = finalNorm.Forward(x);
            return outputHead.Forward(x);
        }
        /// <summary>
        /// Validates that the input tensor conforms to expected shape requirements.
        /// </summary>
        /// <param name="input">The tensor to validate.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the rank of <paramref name="input"/> is less than 2.</exception>

        private void ValidateInput(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (input.Shape.Rank < 2) throw new ArgumentException("Input must have at least 2 dimensions [batch, seqLen]");
        }
        /// <summary>
        /// Retrieves all trainable parameters associated with this GPT model instance.
        /// </summary>
        /// <returns>An enumerable collection of <see cref="ITensor"/> parameters, ordered from input to output layers.</returns>

        public override IEnumerable<ITensor> Parameters() => parameters;
    }
}