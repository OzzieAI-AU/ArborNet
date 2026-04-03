using ArborNet.Activations;
using ArborNet.Core;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Models;
using ArborNet.Layers;
using System.Collections.Generic;

namespace ArborNet.Models
{
    /// <summary>
    /// Implements a ConvNeXt block as described in the ConvNeXt architecture.
    /// </summary>
    /// <remarks>
    /// A ConvNeXt block consists of a depthwise 7x7 convolution followed by an inverted-bottleneck
    /// MLP (pointwise linear layers with GELU activation), with LayerNorm applied before each
    /// major component and a residual connection around the entire block.
    /// </remarks>
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
        /// <param name="x">The input tensor.</param>
        /// <returns>The output tensor after applying the block with residual connection.</returns>
        /// <remarks>
        /// The forward computation follows the structure:
        /// x = x + pw2(GELU(pw1(LN2(DWConv(LN1(x))))))
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
        /// Returns all trainable parameters used within this ConvNeXt block.
        /// </summary>
        /// <returns>A collection containing all parameters from the contained layers.</returns>
        public override IEnumerable<ITensor> Parameters() => parameters;
    }
}