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
    /// Represents a Multi-Head Attention (MHA) layer, a key building block in Transformer architectures.
    /// This layer computes scaled dot-product attention dynamically across multiple representations.
    /// </summary>

    #endregion

    public class Attention : BaseLayer
    {
        private readonly int embedDim;
        private readonly int numHeads;
        private readonly int headDim;
        private readonly bool useBias;

        private ITensor wq, wk, wv, wo;
        private ITensor? bq, bk, bv, bo;

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
        /// Performs the forward pass of the Multi-Head Attention layer.
        /// </summary>
        /// <param name="input">The input tensor with shape <c>[Batch, Sequence, EmbedDim]</c>.</param>
        /// <returns>The calculated output tensor with shape <c>[Batch, Sequence, EmbedDim]</c>.</returns>

        public override ITensor Forward(ITensor input)
        {
            int batch = input.Shape[0];
            int seq = input.Shape[1];

            var projQ = input.MatMul(wq);
            var projK = input.MatMul(wk);
            var projV = input.MatMul(wv);

            if (useBias)
            {
                projQ = projQ.Add(bq);
                projK = projK.Add(bk);
                projV = projV.Add(bv);
            }

            var q = projQ.Reshape(batch, seq, numHeads, headDim).Transpose(new[] { 0, 2, 1, 3 });
            var k = projK.Reshape(batch, seq, numHeads, headDim).Transpose(new[] { 0, 2, 1, 3 });
            var v = projV.Reshape(batch, seq, numHeads, headDim).Transpose(new[] { 0, 2, 1, 3 });

            var scale = (float)Math.Sqrt(headDim);
            var scores = q.MatMul(k.Transpose(new[] { 0, 1, 3, 2 })).Divide(scale);
            var attn = scores.Softmax(-1);
            var context = attn.MatMul(v);

            context = context.Transpose(new[] { 0, 2, 1, 3 }).Reshape(batch, seq, embedDim);
            var output = context.MatMul(wo);

            if (useBias) output = output.Add(bo);

            return output;
        }
        /// <summary>
        /// Returns all learnable parameter tensors (weights and biases) of the Attention layer.
        /// </summary>
        /// <returns>An enumerable collection of <see cref="ITensor"/> parameters requiring optimization.</returns>

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