using System;
using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Layers
{
    /// <summary>
    /// Implements multi-head scaled dot-product self-attention as described in
    /// "Attention Is All You Need" (Vaswani et al., 2017).
    /// </summary>
    /// <remarks>
    /// This layer projects the input into query, key, and value representations,
    /// computes scaled dot-product attention across multiple heads in parallel,
    /// and applies a final output projection. Supports optional bias terms
    /// in all linear transformations.
    /// </remarks>
    public class Attention : BaseLayer
    {
        private readonly int embedDim;
        private readonly int numHeads;
        private readonly int headDim;
        private readonly bool useBias;

        private ITensor wq, wk, wv, wo;
        private ITensor? bq, bk, bv, bo;

        /// <summary>
        /// Initializes a new instance of the <see cref="Attention"/> class.
        /// </summary>
        /// <param name="embedDim">The embedding dimension (d_model).</param>
        /// <param name="numHeads">The number of attention heads.</param>
        /// <param name="useBias">
        /// Whether to include bias terms in the query, key, value, and output projections.
        /// Default is <see langword="true"/>.
        /// </param>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="embedDim"/> is not evenly divisible by <paramref name="numHeads"/>.
        /// </exception>
        public Attention(int embedDim, int numHeads, bool useBias = true)
        {
            if (embedDim % numHeads != 0)
                throw new ArgumentException("embedDim must be divisible by numHeads");

            this.embedDim = embedDim;
            this.numHeads = numHeads;
            this.headDim = embedDim / numHeads;
            this.useBias = useBias;

            wq = Tensor.Randn(new TensorShape(embedDim, embedDim));
            wk = Tensor.Randn(new TensorShape(embedDim, embedDim));
            wv = Tensor.Randn(new TensorShape(embedDim, embedDim));
            wo = Tensor.Randn(new TensorShape(embedDim, embedDim));

            wq.RequiresGrad = wk.RequiresGrad = wv.RequiresGrad = wo.RequiresGrad = true;

            if (useBias)
            {
                bq = Tensor.Zeros(new TensorShape(embedDim));
                bk = Tensor.Zeros(new TensorShape(embedDim));
                bv = Tensor.Zeros(new TensorShape(embedDim));
                bo = Tensor.Zeros(new TensorShape(embedDim));
                bq.RequiresGrad = bk.RequiresGrad = bv.RequiresGrad = bo.RequiresGrad = true;
            }
        }

        /// <summary>
        /// Performs the forward pass of the multi-head attention mechanism.
        /// </summary>
        /// <param name="input">Input tensor of shape (batch, sequence_length, embed_dim).</param>
        /// <returns>
        /// Output tensor of the same shape as <paramref name="input"/> 
        /// containing the attention output.
        /// </returns>
        /// <remarks>
        /// The computation consists of:
        /// <list type="bullet">
        ///   <item>Linear projections to obtain Q, K, V</item>
        ///   <item>Reshape and transpose to separate attention heads</item>
        ///   <item>Scaled dot-product attention with optional biases</item>
        ///   <item>Softmax over the key dimension</item>
        ///   <item>Weighted sum of values followed by output projection</item>
        /// </list>
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            int batch = input.Shape[0];
            int seq = input.Shape[1];

            var q = input.MatMul(wq).Reshape(batch, seq, numHeads, headDim).Transpose(new[] { 0, 2, 1, 3 });
            var k = input.MatMul(wk).Reshape(batch, seq, numHeads, headDim).Transpose(new[] { 0, 2, 1, 3 });
            var v = input.MatMul(wv).Reshape(batch, seq, numHeads, headDim).Transpose(new[] { 0, 2, 1, 3 });

            if (useBias)
            {
                q = q.Add(bq.ReshapeWithBroadcast(q.Shape, -1));
                k = k.Add(bk.ReshapeWithBroadcast(k.Shape, -1));
                v = v.Add(bv.ReshapeWithBroadcast(v.Shape, -1));
            }

            var scale = (float)Math.Sqrt(headDim);
            var scores = q.MatMul(k.Transpose(new[] { 0, 1, 3, 2 })).Divide(scale);
            var attn = scores.Softmax(-1);
            var context = attn.MatMul(v);

            context = context.Transpose(new[] { 0, 2, 1, 3 }).Reshape(batch, seq, embedDim);
            var output = context.MatMul(wo);

            if (useBias) output = output.Add(bo.ReshapeWithBroadcast(output.Shape, -1));

            return output;
        }

        /// <summary>
        /// Returns all learnable parameters of this attention layer.
        /// </summary>
        /// <returns>
        /// An enumerable containing the weight matrices (and bias vectors if enabled)
        /// that require gradient computation.
        /// </returns>
        public override IEnumerable<ITensor> Parameters()
        {
            yield return wq; yield return wk; yield return wv; yield return wo;
            if (useBias)
            {
                yield return bq!; yield return bk!; yield return bv!; yield return bo!;
            }
        }
    }
}