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

    using ArborNet.Activations;
    using ArborNet.Core;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Layers;
    using System.Collections.Generic;
    /// <summary>
    /// Implements a ConvNeXt block as described in the ConvNeXt architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A ConvNeXt block modernizes standard convolutional networks by incorporating design principles
    /// from Vision Transformers (ViTs). It consists of a depthwise 7x7 convolution followed by an 
    /// inverted-bottleneck Multi-Layer Perceptron (MLP) containing pointwise linear layers and 
    /// a GELU activation function.
    /// </para>
    /// <para>
    /// Layer Normalization is applied before each major component, and a residual connection is maintained
    /// around the entire block to facilitate training convergence in deep architectures.
    /// </para>
    /// </remarks>
    /// <seealso cref="BaseModel" />
    /// <seealso cref="LayerNorm" />
    /// <seealso cref="Conv2D" />
    /// <seealso cref="Linear" />

    #endregion

    public class ConvNeXtBlock : BaseModel
    {
        /// <summary>
        /// First LayerNorm applied to the input features before the depthwise convolution.
        /// </summary>
        private readonly LayerNorm norm1;

        /// <summary>
        /// Depthwise 7x7 convolution operating on the normalized input.
        /// </summary>
        private readonly Conv2D dwConv;

        /// <summary>
        /// Second LayerNorm applied after the depthwise convolution and before the MLP.
        /// </summary>
        private readonly LayerNorm norm2;

        /// <summary>
        /// First pointwise linear layer that expands the channel dimension by a factor of 4.
        /// </summary>
        private readonly Linear pw1;

        /// <summary>
        /// Second pointwise linear layer that projects the features back to the original dimension.
        /// </summary>
        private readonly Linear pw2;

        /// <summary>
        /// Initializes a new instance of the <see cref="ConvNeXtBlock"/> class.
        /// </summary>
        /// <param name="dim">The number of input and output channels (feature dimension) for the block.</param>
        public ConvNeXtBlock(int dim)
        {
            norm1 = new LayerNorm(new[] { dim });
            dwConv = new Conv2D(dim, dim, 7, 1, 3, true); // depthwise-style
            norm2 = new LayerNorm(new[] { dim });
            pw1 = new Linear(dim, dim * 4);
            pw2 = new Linear(dim * 4, dim);

            parameters.AddRange(norm1.Parameters());
            parameters.AddRange(dwConv.Parameters());
            parameters.AddRange(norm2.Parameters());
            parameters.AddRange(pw1.Parameters());
            parameters.AddRange(pw2.Parameters());
        }
        /// <summary>
        /// Performs a forward pass through the ConvNeXt block.
        /// </summary>
        /// <param name="x">The input tensor containing the feature maps.</param>
        /// <returns>A new <see cref="ITensor"/> containing the residual sum of the input and block outputs.</returns>
        /// <remarks>
        /// The forward computation follows the structured sequence:
        /// <code>
        /// Output = x + pw2(GELU(pw1(LN2(DWConv(LN1(x))))))
        /// </code>
        /// and preserves the input spatial and channel dimensions.
        /// </remarks>

        public override ITensor Forward(ITensor x)
        {
            var residual = x;
            x = norm1.Forward(x);
            x = dwConv.Forward(x);
            x = norm2.Forward(x);
            x = pw1.Forward(x).Gelu();           // uses extension from Activations
            x = pw2.Forward(x);
            return x.Add(residual);
        }
        /// <summary>
        /// Retrieves all trainable parameter tensors associated with this ConvNeXt block's sub-layers.
        /// </summary>
        /// <returns>An enumerable collection of <see cref="ITensor"/> objects representing the model's parameters.</returns>
        /// <remarks>
        /// The returned parameters include the weights and biases of the LayerNorm layers, 
        /// depthwise convolutions, and pointwise projections.
        /// </remarks>

        public override IEnumerable<ITensor> Parameters() => parameters;
    }
}