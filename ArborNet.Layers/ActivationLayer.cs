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
    using System;
    using System.Collections.Generic;
    /// <summary>
    /// Represents a neural network layer that applies a non-linear activation function to an input tensor.
    /// </summary>
    /// <remarks>
    /// This layer is parameter-free and acts as a wrapper around an <see cref="IActivation"/> implementation.
    /// It delegates the forward propagation pass directly to the encapsulated activation function.
    /// Typically used to introduce non-linearity into the network model without introducing extra learnable weights.
    /// </remarks>

    #endregion

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
        /// Executes the forward propagation step of this layer, applying the activation function to the provided input tensor.
        /// </summary>
        /// <param name="x">The input tensor (<see cref="ITensor"/>) containing the pre-activation values from the preceding layer.</param>
        /// <returns>An <see cref="ITensor"/> containing the post-activation values.</returns>

        public override ITensor Forward(ITensor x) => _activation.Forward(x);
        /// <summary>
        /// Retrieves the collection of trainable parameters (weights and biases) associated with this layer.
        /// </summary>
        /// <returns>
        /// An empty enumerable of <see cref="ITensor"/>, as this activation layer does not maintain any trainable parameters.
        /// </returns>

        public override IEnumerable<ITensor> Parameters() => Array.Empty<ITensor>();
    }
}