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

    using System.Collections.Generic;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;
    /// <summary>
    /// Implements a single Mistral transformer block consisting of pre-norm self-attention
    /// and a feed-forward network with residual connections.
    /// </summary>
    /// <remarks>
    /// This block follows a standard pre-norm transformer architecture:
    /// <list type="bullet">
    ///   <item>
    ///     <description>LayerNorm → Multi-Head Attention → Residual addition of the block input.</description>
    ///   </item>
    ///   <item>
    ///     <description>LayerNorm → Feed-Forward (Linear + ReLU + Linear) → Residual addition of the pre-FFN state.</description>
    ///   </item>
    /// </list>
    /// Pre-layer normalization (Pre-LN) provides superior gradient stability during backpropagation, 
    /// enabling the training of deeper model architectures.
    /// </remarks>

    #endregion

    public class MistralBlock : BaseLayer
    {
        /// <summary>
        /// Layer normalization applied before the self-attention sub-layer.
        /// </summary>
        private readonly LayerNorm norm1;

        /// <summary>
        /// Multi-head self-attention mechanism.
        /// </summary>
        private readonly MultiHeadAttention attention;

        /// <summary>
        /// Layer normalization applied before the feed-forward sub-layer.
        /// </summary>
        private readonly LayerNorm norm2;

        /// <summary>
        /// First linear projection in the feed-forward network (expands to 4× hidden dimension).
        /// </summary>
        private readonly Linear ff1;

        /// <summary>
        /// Second linear projection in the feed-forward network (projects back to hidden dimension).
        /// </summary>
        private readonly Linear ff2;

        /// <summary>
        /// Aggregated list of all trainable parameters from the sub-layers.
        /// </summary>
        private readonly List<ITensor> _parameters = new();

        /// <summary>
        /// Initializes a new instance of the <see cref="MistralBlock"/> class.
        /// </summary>
        /// <param name="hiddenDim">The hidden dimension size of the model.</param>
        /// <param name="numHeads">The number of attention heads.</param>
        /// <param name="kvHeads">The number of key-value heads (for grouped-query attention).</param>
        /// <param name="slidingWindow">The sliding window size for attention masking.</param>
        public MistralBlock(int hiddenDim, int numHeads, int kvHeads, int slidingWindow)
        {
            norm1 = new LayerNorm(new[] { hiddenDim });
            attention = new MultiHeadAttention(hiddenDim, numHeads);
            norm2 = new LayerNorm(new[] { hiddenDim });
            ff1 = new Linear(hiddenDim, hiddenDim * 4);
            ff2 = new Linear(hiddenDim * 4, hiddenDim);

            _parameters.AddRange(norm1.Parameters());
            _parameters.AddRange(attention.Parameters());
            _parameters.AddRange(norm2.Parameters());
            _parameters.AddRange(ff1.Parameters());
            _parameters.AddRange(ff2.Parameters());
        }
        /// <summary>
        /// Performs the forward pass computation for the Mistral transformer block.
        /// </summary>
        /// <param name="x">The input activation tensor from the preceding layer or embeddings stage.</param>
        /// <returns>
        /// A new <see cref="ITensor"/> containing the block output after executing the normalized attention sequence,
        /// feed-forward sub-layer sequence, and their respective residual connections.
        /// </returns>

        public override ITensor Forward(ITensor x)
        {
            var residual = x;
            x = norm1.Forward(x);
            x = attention.Forward(x);
            x = x.Add(residual);

            residual = x;
            x = norm2.Forward(x);
            x = ff2.Forward(ff1.Forward(x).Relu());
            return x.Add(residual);
        }
        /// <summary>
        /// Retrieves all trainable parameter tensors contained within this block's sub-layers.
        /// </summary>
        /// <returns>An enumerable collection of trainable <see cref="ITensor"/> parameters, including weights and biases.</returns>

        public override IEnumerable<ITensor> Parameters() => _parameters;
    }
}