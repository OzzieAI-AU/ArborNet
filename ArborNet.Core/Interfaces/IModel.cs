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

    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Devices;

    #endregion

    /// <summary>
    /// Defines the contract for all neural network models within the ArborNet framework.
    /// </summary>
    /// <remarks>
    /// Implementing classes represent computational graphs that can perform forward propagation,
    /// manage learnable parameters, toggle execution states between training and evaluation, 
    /// and migrate execution and storage across different hardware devices.
    /// </remarks>
    /// <seealso cref="ITensor"/>
    /// <seealso cref="Device"/>
    public interface IModel
    {
        /// <summary>
        /// Performs the forward pass computation of the neural network model using the specified input tensor.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> containing the feature data to be processed by the model.</param>
        /// <returns>An <see cref="ITensor"/> representing the generated output predictions or computed activation state of the network.</returns>
        /// <exception cref="ArgumentNullException">Thrown when the <paramref name="input"/> tensor is <see langword="null"/>.</exception>
        /// <exception cref="InvalidOperationException">Thrown when the input dimensions, shape, or data type do not match the expected specifications of the model's architecture.</exception>
        ITensor Forward(ITensor input);
        /// <summary>
        /// Retrieves an enumerable collection of all trackable and learnable parameters (such as weights and biases) 
        /// associated with this model and its nested sub-layers.
        /// </summary>
        /// <returns>An <see cref="IEnumerable{T}"/> of <see cref="ITensor"/> containing the model's parameter tensors.</returns>

        IEnumerable<ITensor> Parameters();
        /// <summary>
        /// Sets the model and all of its constituent sub-modules to training mode.
        /// </summary>
        /// <remarks>
        /// This state change activates training-specific behaviors in layers that require different execution paths 
        /// during training than during inference, such as enabling Dropout and allowing Batch Normalization layers 
        /// to update their running statistics.
        /// </remarks>

        void Train();
        /// <summary>
        /// Sets the model and all of its constituent sub-modules to evaluation (inference) mode.
        /// </summary>
        /// <remarks>
        /// This state change disables training-specific behaviors to ensure deterministic predictions. 
        /// For example, Dropout layers are deactivated, and Batch Normalization layers use frozen running statistics 
        /// instead of computing current batch statistics.
        /// </remarks>

        void Eval();
        /// <summary>
        /// Recursively migrates the model, its layers, and all underlying parameters to the specified target execution device.
        /// </summary>
        /// <param name="device">The target <see cref="Device"/> (such as CPU or a specific GPU) to transfer the model and its parameters to.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="device"/> is <see langword="null"/>.</exception>
        /// <exception cref="PlatformNotSupportedException">Thrown if the target <paramref name="device"/> is not supported, recognized, or available in the current runtime environment.</exception>

        void To(Device device);
    }
}