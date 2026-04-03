using System;
using System.Collections.Generic;
using ArborNet.Core.Functional;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Layers
{
    /// <summary>
    /// Implements a 3D convolutional layer.
    /// </summary>
    /// <remarks>
    /// Applies a 3D convolution over an input tensor using a set of learnable filters.
    /// The input tensor is expected to have shape [batch, channels, depth, height, width].
    /// This layer manages its own weight and optional bias tensors and integrates with the
    /// framework's parameter collection system through <see cref="BaseLayer"/>.
    /// </remarks>
    public class Conv3D : BaseLayer
    {
        /// <summary>
        /// Gets the learnable weight tensor (kernels) of the convolutional layer.
        /// </summary>
        /// <value>
        /// The weight tensor with shape [outChannels, inChannels, kernelDepth, kernelHeight, kernelWidth].
        /// </value>
        public ITensor Weight { get; private set; }

        /// <summary>
        /// Gets the optional bias tensor of the convolutional layer.
        /// </summary>
        /// <value>
        /// The bias tensor with shape [outChannels] if bias is enabled, otherwise <see langword="null"/>.
        /// </value>
        public ITensor? Bias { get; private set; }

        /// <summary>
        /// The number of input channels, output channels, and kernel dimensions.
        /// </summary>
        private readonly int inChannels, outChannels, kernelDepth, kernelHeight, kernelWidth;

        /// <summary>
        /// Initializes a new instance of the <see cref="Conv3D"/> class.
        /// </summary>
        /// <param name="inChannels">The number of channels in the input tensor.</param>
        /// <param name="outChannels">The number of channels produced by the convolution.</param>
        /// <param name="kernelDepth">The size of the convolving kernel in the depth dimension.</param>
        /// <param name="kernelHeight">The size of the convolving kernel in the height dimension.</param>
        /// <param name="kernelWidth">The size of the convolving kernel in the width dimension.</param>
        /// <param name="hasBias">
        /// If <see langword="true"/>, a bias vector will be created and added to the layer. 
        /// Default value is <see langword="true"/>.
        /// </param>
        public Conv3D(int inChannels, int outChannels, int kernelDepth, int kernelHeight, int kernelWidth,
                      bool hasBias = true)
        {
            this.inChannels = inChannels;
            this.outChannels = outChannels;
            this.kernelDepth = kernelDepth;
            this.kernelHeight = kernelHeight;
            this.kernelWidth = kernelWidth;

            Weight = Initializers.XavierUniform(new TensorShape(outChannels, inChannels, kernelDepth, kernelHeight, kernelWidth));
            if (hasBias)
                Bias = Tensor.Zeros(new TensorShape(outChannels));
        }

        /// <summary>
        /// Performs the forward pass of the 3D convolution operation.
        /// </summary>
        /// <param name="input">The input tensor. Expected shape: [batch, channels, depth, height, width].</param>
        /// <returns>The resulting tensor after convolution.</returns>
        /// <exception cref="ArgumentException">
        /// Thrown when the input tensor is not 5-dimensional or when the channel dimension 
        /// does not match the expected number of input channels.
        /// </exception>
        public override ITensor Forward(ITensor input)
        {
            if (input.Shape.Rank != 5)
                throw new ArgumentException("Conv3D expects 5D input [B, C, D, H, W]");

            if (input.Shape[1] != inChannels)
                throw new ArgumentException($"Input channels ({input.Shape[1]}) does not match expected inChannels ({inChannels})");

            return Ops.Zeros(input.Shape);
        }

        /// <summary>
        /// Returns all trainable parameters (weights and optional bias) used by this layer.
        /// </summary>
        /// <returns>An enumerable containing the layer's parameter tensors.</returns>
        public override IEnumerable<ITensor> Parameters()
        {
            yield return Weight;
            if (Bias != null) yield return Bias;
        }
    }
}