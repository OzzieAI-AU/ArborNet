using System.Collections.Generic;
using ArborNet.Core.Tensors;

namespace ArborNet.Core.Interfaces
{
    /// <summary>
    /// Core interface for all neural network layers in ArborNet.
    /// </summary>
    /// <remarks>
    /// This interface defines the fundamental contract that all neural network layers must implement.
    /// It ensures consistent behavior for forward propagation and parameter management across
    /// the entire framework, enabling seamless composition of complex network architectures
    /// and integration with optimizers and training loops.
    /// </remarks>
    public interface ILayer
    {
        /// <summary>Performs the forward pass.</summary>
        /// <param name="input">The input tensor to the layer.</param>
        /// <returns>The output tensor after applying the layer's transformation.</returns>
        /// <remarks>
        /// This method performs the forward computation for the layer. The input tensor's shape
        /// must be compatible with the layer's expected input dimensions.
        /// </remarks>
        ITensor Forward(ITensor input);

        /// <summary>Returns all learnable parameters.</summary>
        /// <returns>An enumerable collection of tensors representing the learnable parameters of this layer.</returns>
        /// <remarks>
        /// Returns all tensors that should be updated during the training process (weights, biases, etc.).
        /// This collection is used by optimizers to apply gradient updates.
        /// </remarks>
        IEnumerable<ITensor> Parameters();
    }
}