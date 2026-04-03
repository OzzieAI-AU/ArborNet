using System;
using ArborNet.Core.Interfaces;

namespace ArborNet.Core.Interfaces
{
    /// <summary>
    /// Interface for neural network models in ArborNet.
    /// Defines the contract for models that can perform forward passes on input tensors,
    /// similar to layers but potentially encompassing more complex structures like composed layers.
    /// </summary>
    public interface IModel
    {
        /// <summary>
        /// Performs the forward pass of the model on the given input tensor.
        /// </summary>
        /// <param name="input">The input tensor to the model.</param>
        /// <returns>The output tensor after applying the model's transformations.</returns>
        ITensor Forward(ITensor input);

        /// <summary>
        /// Gets the parameters of the model (e.g., weights and biases).
        /// </summary>
        /// <returns>An enumerable collection of the model's parameters.</returns>
        System.Collections.Generic.IEnumerable<ITensor> Parameters();

        /// <summary>
        /// Sets the model to training mode.
        /// </summary>
        void Train();

        /// <summary>
        /// Sets the model to evaluation mode.
        /// </summary>
        void Eval();
    }
}