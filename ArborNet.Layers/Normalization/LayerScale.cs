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
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    using System.Text;
    /// <summary>
    /// Represents a LayerScale layer that performs affine scaling (gamma * input) on an input tensor.
    /// This layer is commonly used in modern neural network architectures, such as deep transformers,
    /// to stabilize training by scaling the outputs of residual branches.
    /// </summary>

    #endregion


    public class LayerScale : BaseLayer
    {
        private readonly ITensor gamma;

        public LayerScale(int numFeatures, float initScale = 1e-2f)
        {
            gamma = Tensor.FromScalar(initScale); // , new TensorShape(new[] { numFeatures })
            gamma.RequiresGrad = true;
        }
        /// <summary>
        /// Performs the forward pass by multiplying the input tensor by the learnable scaling parameter.
        /// </summary>
        /// <param name="input">The input tensor to be scaled.</param>
        /// <returns>The scaled tensor resulting from the element-wise multiplication of the input and gamma.</returns>

        public override ITensor Forward(ITensor input)
        {
            return input.Multiply(gamma);
        }
        /// <summary>
        /// Retrieves the learnable parameters of this layer.
        /// </summary>
        /// <returns>An enumerable collection containing the scaling parameter <see cref="gamma"/>.</returns>

        public override IEnumerable<ITensor> Parameters()
        {
            yield return gamma;
        }
    }
}