using ArborNet.Core.Interfaces;
using System;
using System.Collections.Generic;

namespace ArborNet.Layers
{
    /// <summary>
    /// Represents a neural network layer that applies a non-linear activation function to an input tensor.
    /// </summary>
    /// <remarks>
    /// This layer is parameter-free and simply wraps an <see cref="IActivation"/> implementation,
    /// delegating the forward computation to it. It is typically used to introduce non-linearity
    /// between other computational layers in a network.
    /// </remarks>
    public class ActivationLayer : BaseLayer
    {
        /// <summary>
        /// The activation function implementation used by this layer.
        /// </summary>
        private readonly IActivation _activation;

        /// <summary>
        /// Initializes a new instance of the <see cref="ActivationLayer"/> class.
        /// </summary>
        /// <param name="activation">The activation function to apply in this layer.</param>
        public ActivationLayer(IActivation activation) => _activation = activation;

        /// <summary>
        /// Performs a forward pass by applying the configured activation function to the input tensor.
        /// </summary>
        /// <param name="x">The input tensor from the previous layer.</param>
        /// <returns>The output tensor after applying the activation function.</returns>
        public override ITensor Forward(ITensor x) => _activation.Forward(x);

        /// <summary>
        /// Returns the trainable parameters of this layer.
        /// </summary>
        /// <returns>An empty collection, because activation layers do not contain trainable parameters.</returns>
        public override IEnumerable<ITensor> Parameters() => Array.Empty<ITensor>();
    }
}