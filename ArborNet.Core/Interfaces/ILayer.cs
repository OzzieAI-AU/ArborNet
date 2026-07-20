// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Interfaces
{

    #region Using Statements:

    using System.Collections.Generic;
    using ArborNet.Core.Tensors;

    #endregion

    /// <summary>
    /// Defines the fundamental contract for all neural network layers within the ArborNet framework.
    /// </summary>
    /// <remarks>
    /// This interface serves as the primary abstraction for building modular neural network architectures.
    /// Any component that transforms data in the forward pass and, optionally, maintains learnable parameters
    /// (such as dense, convolutional, recurrent, or normalization layers) must implement this interface.
    /// Standardizing this contract ensures seamless integration with network containers, training loops,
    /// and optimization algorithms.
    /// </remarks>
    public interface ILayer
    {
        /// <summary>
        /// Performs the forward propagation pass of the layer, transforming the input tensor into an output tensor.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> containing the activation values from the preceding layer or the raw input data.</param>
        /// <returns>An <see cref="ITensor"/> representing the computed activations output by this layer.</returns>
        /// <exception cref="System.ArgumentNullException">Thrown when the <paramref name="input"/> tensor is <see langword="null"/>.</exception>
        /// <exception cref="System.ArgumentException">Thrown when the shape, dimensionality, layout, or data type of the <paramref name="input"/> tensor is incompatible with the layer's configuration.</exception>
        /// <exception cref="System.InvalidOperationException">Thrown if the execution context or device assignment of the input tensor is incompatible with the layer's internal parameters.</exception>
        /// <remarks>
        /// <para>
        /// During training, implementations of this method typically cache intermediate states, execution graphs, or the input tensor itself
        /// to facilitate gradient computation during the subsequent backward pass. For evaluation or inference,
        /// implementations should optimize memory and execution speed by avoiding caching mechanisms.
        /// </para>
        /// <para>
        /// The output tensor should be allocated on the same execution device (e.g., CPU or GPU via CUDA) as the <paramref name="input"/> tensor.
        /// </para>
        /// </remarks>
        ITensor Forward(ITensor input);
        /// <summary>
        /// Retrieves an enumerable collection of all learnable and trainable parameters managed by this layer.
        /// </summary>
        /// <returns>An <see cref="IEnumerable{T}"/> of <see cref="ITensor"/> objects representing the trainable parameters (such as weights and biases) of this layer. Returns an empty collection if the layer has no trainable parameters.</returns>
        /// <remarks>
        /// This method is queried by optimization algorithms (such as SGD, Adam, or RMSprop) to register, track, and update
        /// the layer's parameters during backpropagation. Implementations must return direct references to
        /// the underlying parameter tensors so that gradient updates are applied in-place to the layer's actual state.
        /// </remarks>

        IEnumerable<ITensor> Parameters();
    }
}