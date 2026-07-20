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

    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Implements the Multi-Head Self-Attention (MHA) mechanism as proposed in the seminal paper 
    /// "Attention Is All You Need" (Vaswani et al., 2017).
    /// </summary>
    /// <remarks>
    /// Multi-Head Attention allows the model to jointly attend to information from different representation 
    /// subspaces at different positions. This is achieved by projecting the queries, keys, and values 
    /// multiple times with different, learnable linear projections, performing scaled dot-product attention 
    /// in parallel, concatenating the outputs, and projecting them once more.
    /// </remarks>

    #endregion

    public class MultiHeadAttention : BaseLayer
    {
        /// <summary>
        /// The dimensionality of the model (embedding dimension).
        /// </summary>
        private readonly int dModel;

        /// <summary>
        /// The number of parallel attention heads.
        /// </summary>
        private readonly int numHeads;

        /// <summary>
        /// The dimension of each attention head (dModel / numHeads).
        /// </summary>
        private readonly int dHead;

        /// <summary>
        /// The learnable weight matrices for the query (Wq), key (Wk), value (Wv), and output (Wo) projections.
        /// </summary>
        private readonly ITensor Wq, Wk, Wv, Wo;

        /// <summary>
        /// Initializes a new instance of the <see cref="MultiHeadAttention"/> class.
        /// </summary>
        /// <param name="dModel">The dimensionality of the model (embedding size).</param>
        /// <param name="numHeads">The number of attention heads to use.</param>
        /// <param name="useBias">Whether to use bias terms in the linear projections (currently unused in this implementation).</param>
        /// <exception cref="ArgumentException">Thrown when <paramref name="dModel"/> is not divisible by <paramref name="numHeads"/>.</exception>
        public MultiHeadAttention(int dModel, int numHeads, bool useBias = true)
        {
            if (dModel % numHeads != 0)
                throw new ArgumentException("dModel must be divisible by numHeads");

            this.dModel = dModel;
            this.numHeads = numHeads;
            this.dHead = dModel / numHeads;

            Wq = Tensor.Randn(new TensorShape(dModel, dModel));
            Wk = Tensor.Randn(new TensorShape(dModel, dModel));
            Wv = Tensor.Randn(new TensorShape(dModel, dModel));
            Wo = Tensor.Randn(new TensorShape(dModel, dModel));

            Wq.RequiresGrad = Wk.RequiresGrad = Wv.RequiresGrad = Wo.RequiresGrad = true;
        }
        /// <summary>
        /// Executes the forward pass of the Multi-Head Self-Attention mechanism.
        /// </summary>
        /// <param name="input">The input tensor of shape (batch_size, sequence_length, <see cref="dModel"/>).</param>
        /// <returns>An <see cref="ITensor"/> containing the pooled context representation, with shape (batch_size, sequence_length, <see cref="dModel"/>).</returns>
        /// <remarks>
        /// This method executes the following steps:
        /// <list type="number">
        /// <item><description>Linearly projects the input to generate Queries, Keys, and Values.</description></item>
        /// <item><description>Reshapes and transposes projections to isolate the individual attention heads.</description></item>
        /// <item><description>Calculates scaled dot-product attention scores: Softmax((Q * K^T) / sqrt(<see cref="dHead"/>)).</description></item>
        /// <item><description>Computes the weighted sum of Values based on the attention scores.</description></item>
        /// <item><description>Transposes, reshapes, and concatenates all head outputs back into <see cref="dModel"/> dimensions.</description></item>
        /// <item><description>Applies the final linear projection <see cref="Wo"/>.</description></item>
        /// </list>
        /// </remarks>

        public override ITensor Forward(ITensor input)
        {
            var batch = input.Shape[0];
            var seq = input.Shape[1];

            var Q = input.MatMul(Wq).Reshape(batch, seq, numHeads, dHead).Transpose(new[] { 0, 2, 1, 3 });
            var K = input.MatMul(Wk).Reshape(batch, seq, numHeads, dHead).Transpose(new[] { 0, 2, 1, 3 });
            var V = input.MatMul(Wv).Reshape(batch, seq, numHeads, dHead).Transpose(new[] { 0, 2, 1, 3 });

            var scale = MathF.Sqrt(dHead);
            var scores = Q.MatMul(K.Transpose(new[] { 0, 1, 3, 2 })).Divide(scale);
            var attn = scores.Softmax(-1);
            var context = attn.MatMul(V);

            context = context.Transpose(new[] { 0, 2, 1, 3 }).Reshape(batch, seq, dModel);
            return context.MatMul(Wo);
        }
        /// <summary>
        /// Retrieves the collection of all trainable weight parameters associated with this layer.
        /// </summary>
        /// <returns>An enumerable sequence of <see cref="ITensor"/> containing the weight matrices for the query, key, value, and output projections.</returns>

        public override IEnumerable<ITensor> Parameters()
        {
            yield return Wq; yield return Wk; yield return Wv; yield return Wo;
        }
    }
}