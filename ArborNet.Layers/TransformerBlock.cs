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

    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;
    using System.Collections.Generic;
    /// <summary>
    /// Implements a single Transformer encoder block consisting of multi-head self-attention
    /// and a position-wise feed-forward network, with residual connections around each sub-layer.
    /// </summary>
    /// <remarks>
    /// This follows the architecture from "Attention Is All You Need" (Vaswani et al., 2017).
    /// The forward pass applies attention, adds a residual connection, then applies a 
    /// two-layer feed-forward network with ReLU activation and a second residual connection.
    /// </remarks>

    #endregion

    public class TransformerBlock : BaseLayer
    {
        private readonly MultiHeadAttention attention;
        private readonly Linear ff1;
        private readonly Linear ff2;

        /// <summary>
        /// Initializes a new instance of the <see cref="TransformerBlock"/> class.
        /// </summary>
        /// <param name="dModel">The model dimension (embedding size).</param>
        /// <param name="numHeads">The number of attention heads.</param>
        /// <param name="ffDim">
        /// The inner dimension of the feed-forward network. 
        /// If <c>0</c> (default), it is automatically set to <c>dModel * 4</c>.
        /// </param>
        public TransformerBlock(int dModel, int numHeads, int ffDim = 0)
        {
            ffDim = ffDim == 0 ? dModel * 4 : ffDim;
            attention = new MultiHeadAttention(dModel, numHeads);
            ff1 = new Linear(dModel, ffDim);
            ff2 = new Linear(ffDim, dModel);
        }
        /// <summary>
        /// Performs a forward pass through the transformer block.
        /// </summary>
        /// <param name="input">The input tensor of shape (batchSize, sequenceLength, dModel).</param>
        /// <returns>
        /// The output tensor of the same shape as <paramref name="input"/>, 
        /// after applying self-attention and the feed-forward network with residual connections.
        /// </returns>
        /// <remarks>
        /// The computation flow is defined as:
        /// <list type="number">
        /// <item>
        /// <description>Calculate attention: <c>attn = Attention(input)</c></description>
        /// </item>
        /// <item>
        /// <description>First residual connection: <c>x = input + attn</c></description>
        /// </item>
        /// <item>
        /// <description>Feed-forward projection: <c>ff = Linear2(ReLU(Linear1(x)))</c></description>
        /// </item>
        /// <item>
        /// <description>Second residual connection: <c>output = x + ff</c></description>
        /// </item>
        /// </list>
        /// </remarks>

        public override ITensor Forward(ITensor input)
        {
            var attn = attention.Forward(input);
            var x = input.Add(attn);
            var ff = ff2.Forward(ff1.Forward(x).Relu());
            return x.Add(ff);
        }
        /// <summary>
        /// Returns all trainable parameters used by this transformer block.
        /// </summary>
        /// <returns>
        /// An enumerable containing all parameters from the multi-head attention module 
        /// and both linear layers of the feed-forward network.
        /// </returns>

        public override IEnumerable<ITensor> Parameters()
        {
            foreach (var p in attention.Parameters()) yield return p;
            foreach (var p in ff1.Parameters()) yield return p;
            foreach (var p in ff2.Parameters()) yield return p;
        }
    }
}