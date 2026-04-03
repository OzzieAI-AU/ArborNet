using System;
using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Layers
{
    /// <summary>
    /// Implements the Multi-Head Self-Attention mechanism as described in "Attention Is All You Need".
    /// </summary>
    /// <remarks>
    /// This layer linearly projects the input into queries, keys, and values, splits them into multiple heads,
    /// performs scaled dot-product attention in parallel, concatenates the results, and applies a final output
    /// projection. All projection matrices are trainable.
    /// </remarks>
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
        /// Performs the forward pass of the multi-head attention mechanism.
        /// </summary>
        /// <param name="input">The input tensor of shape (batch_size, sequence_length, dModel).</param>
        /// <returns>The output tensor of shape (batch_size, sequence_length, dModel).</returns>
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
        /// Returns all trainable parameters of this layer.
        /// </summary>
        /// <returns>An enumerable containing the query, key, value, and output projection weight tensors.</returns>
        public override IEnumerable<ITensor> Parameters()
        {
            yield return Wq; yield return Wk; yield return Wv; yield return Wo;
        }
    }
}