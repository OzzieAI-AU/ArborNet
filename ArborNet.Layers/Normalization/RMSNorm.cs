// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Layers.Normalization
{

    #region Using Statements:

    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    using System.Text;
    /// <summary>
    /// Implements Root Mean Square Normalization (RMSNorm).
    /// RMSNorm is a computationally efficient variant of Layer Normalization that normalizes 
    /// the inputs by their root mean square (RMS) rather than both mean and variance. 
    /// This design is scale-invariant, faster to compute, and widely utilized in modern 
    /// large language model architectures such as Llama and Mistral.
    /// </summary>

    #endregion


    public class RMSNorm : BaseNormalization
    {
        public RMSNorm(int numFeatures, float eps = 1e-6f, bool useAffine = true)
            : base(numFeatures, eps, useAffine) { }
        /// <summary>
        /// Performs the forward pass normalization on the input tensor.
        /// </summary>
        /// <param name="input">The input tensor to be normalized.</param>
        /// <returns>The normalized tensor calculated via the root mean square.</returns>
        /// <remarks>
        /// The normalization scales the input using the formula: 
        /// <c>y = x / sqrt(mean(x^2) + eps)</c>.
        /// </remarks>

        protected override ITensor Normalize(ITensor input)
        {
            // FIXED: Resolved double Sqrt bug. Standard formula: RMS = sqrt(mean(x^2) + eps)
            var rms = input.Pow(2).Mean(-1, keepDims: true).Add(Eps).Sqrt();
            return input.Divide(rms);
        }
        /// <summary>
        /// Computes the gradient of the loss with respect to the input tensor during the backward pass.
        /// </summary>
        /// <param name="input">The original input tensor from the forward pass.</param>
        /// <param name="gradOutput">The gradient of the loss with respect to the output of this layer.</param>
        /// <returns>The gradient of the loss with respect to the input tensor.</returns>
        /// <remarks>
        /// Implements a stable analytical derivative of the RMSNorm operation, accounting for 
        /// scaling by affine parameters if enabled.
        /// </remarks>

        protected override ITensor ComputeGradInput(ITensor input, ITensor gradOutput)
        {
            // FIXED: Stable analytical RMSNorm gradient
            var rms = input.Pow(2).Mean(-1, keepDims: true).Add(Eps).Sqrt();
            var normalized = input.Divide(rms);

            var gradNorm = gradOutput.Multiply(UseAffine ? Gamma : Tensor.Ones(input.Shape, input.Device));
            var meanGrad = gradNorm.Multiply(normalized).Mean(-1, keepDims: true);

            return gradNorm.Subtract(normalized.Multiply(meanGrad)).Divide(rms);
        }
    }
}