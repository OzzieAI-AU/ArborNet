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
    using ArborNet.Layers.Normalization;
    using System;
    using System.Collections.Generic;
    /// <summary>
    /// Implements Layer Normalization (LayerNorm) as introduced in the paper 
    /// "Layer Normalization" by Jimmy Lei Ba et al. (2016).
    /// </summary>
    /// <remarks>
    /// Layer Normalization normalizes the activations across the feature dimensions 
    /// for each individual sample independently. It is widely used in transformer 
    /// architectures and recurrent neural networks because it does not depend on 
    /// batch statistics.
    /// 
    /// The forward pass computes:
    /// <code>
    /// normalized = (input - mean) / sqrt(variance + eps)
    /// output = gamma * normalized + beta
    /// </code>
    /// where mean and variance are calculated over the dimensions specified by normalizedShape.
    /// </remarks>

    #endregion

    public class LayerNorm : BaseNormalization
    {
        public LayerNorm(int[] normalizedShape, float eps = 1e-5f, bool useAffine = true)
            : base(new TensorShape(normalizedShape).TotalElements, eps, useAffine) { }
        /// <summary>
        /// Performs the forward pass normalization of the input tensor across the designated feature dimensions.
        /// </summary>
        /// <param name="input">The input tensor to be normalized.</param>
        /// <returns>A new <see cref="ITensor"/> containing the normalized elements.</returns>
        /// <remarks>
        /// This method calculates the mean and variance along the last dimension of the input tensor,
        /// then standardizes the tensor using these statistics adjusted by the epsilon value.
        /// </remarks>

        protected override ITensor Normalize(ITensor input)
        {
            // FIXED: Added keepDims: true to avoid shape mismatch crashes during broadcasting
            var mean = input.Mean(-1, keepDims: true);
            var var_ = input.Subtract(mean).Pow(2).Mean(-1, keepDims: true);
            var std = var_.Add(Eps).Sqrt();
            return input.Subtract(mean).Divide(std);
        }
        /// <summary>
        /// Computes the gradient of the loss with respect to the input tensor during the backward pass.
        /// </summary>
        /// <param name="input">The original input tensor from the forward pass.</param>
        /// <param name="gradOutput">The gradient of the loss with respect to the output of this layer.</param>
        /// <returns>A new <see cref="ITensor"/> representing the gradient of the loss with respect to the input.</returns>
        /// <remarks>
        /// This method implements the standard analytical backward pass derivative for Layer Normalization,
        /// propagating the incoming gradients back through the mean and variance computations.
        /// </remarks>

        protected override ITensor ComputeGradInput(ITensor input, ITensor gradOutput)
        {
            // FIXED: Standard stable analytical LayerNorm gradient calculation
            var mean = input.Mean(-1, keepDims: true);
            var var_ = input.Subtract(mean).Pow(2).Mean(-1, keepDims: true);
            var std = var_.Add(Eps).Sqrt();
            var normalized = input.Subtract(mean).Divide(std);

            var N = Tensor.FromScalar((float)input.Shape[input.Shape.Rank - 1], input.Device);
            var ivar = std.Pow(-1);

            var gradNorm = gradOutput.Multiply(UseAffine ? Gamma : Tensor.Ones(input.Shape, input.Device));

            var sum_gradNorm = gradNorm.Sum(-1, keepDims: true);
            var sum_gradNorm_norm = gradNorm.Multiply(normalized).Sum(-1, keepDims: true);

            return gradNorm.Subtract(sum_gradNorm.Divide(N))
                           .Subtract(normalized.Multiply(sum_gradNorm_norm.Divide(N)))
                           .Multiply(ivar);
        }
    }
}