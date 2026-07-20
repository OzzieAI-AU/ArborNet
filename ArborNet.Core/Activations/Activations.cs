// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Activations
{

    #region Using Statements:

    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using System;
    /// <summary>
    /// Provides a foundational abstract base class for all neural network activation functions within the ArborNet framework.
    /// This class implements core validation rules, device tracking capability, and satisfies the contract 
    /// defined by the <see cref="IActivation"/> interface.
    /// </summary>
    /// <remarks>
    /// All custom activation functions (such as ReLU, Sigmoid, Tanh, etc.) should inherit from this base class to ensure
    /// consistent validation behavior, unified device allocation handling, and seamless integration with both CPU and CUDA execution pipelines.
    /// </remarks>

    #endregion


    public abstract class BaseActivation : IActivation
    {
        /// <summary>
        /// Gets or sets the target hardware device used for performing the activation function's computations.
        /// </summary>
        /// <value>
        /// The <see cref="Device"/> context where computations are executed. The default is <see cref="Device.CPU"/>.
        /// </value>
        /// <remarks>
        /// Derived classes must monitor and adhere to this property, ensuring that any internal tensor allocations,
        /// mathematical operations, or state transitions are dispatched to the appropriate execution context.
        /// </remarks>
        protected Device Device { get; set; } = Device.CPU;
        /// <summary>
        /// Performs the mathematical forward pass of the activation function on the specified input tensor.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> containing the pre-activation values.</param>
        /// <returns>An <see cref="ITensor"/> containing the computed post-activation values.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="input"/> does not meet the necessary validation criteria (e.g., null shape, empty shape, or missing device configuration).
        /// </exception>
        /// <remarks>
        /// Implementations of this method should call <see cref="ValidateInput(ITensor)"/> before performing mathematical computations
        /// to guarantee the structural and contextual integrity of the incoming tensor.
        /// </remarks>

        public abstract ITensor Forward(ITensor input);
        /// <summary>
        /// Validates that the specified input tensor meets the minimum structural and contextual requirements for processing.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> to be evaluated.</param>
        /// <exception cref="ArgumentNullException">Thrown if the <paramref name="input"/> reference is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">
        /// Thrown if <paramref name="input"/> lacks a valid shape description, contains zero total elements, 
        /// or does not specify an execution device context.
        /// </exception>

        protected void ValidateInput(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (input.Shape == null) throw new ArgumentException("Input tensor must have a valid shape.", nameof(input));
            if (input.Shape.TotalElements == 0) throw new ArgumentException("Input tensor cannot be empty.", nameof(input));
            if (input.Device == null) throw new ArgumentException("Input tensor must specify a device.", nameof(input));
        }
        /// <summary>
        /// Configures the activation function to operate on the specified target hardware device.
        /// </summary>
        /// <param name="device">The hardware <see cref="Device"/> to transition to. If <see langword="null"/>, defaults to <see cref="Device.CPU"/>.</param>
        /// <remarks>
        /// Derived implementations that allocate device-bound state, maintain local parameters, or manage internal
        /// caches should override this virtual method to guarantee that all associated resources are properly migrated 
        /// to the new hardware context.
        /// </remarks>

        public virtual void To(Device device)
        {
            Device = device ?? Device.CPU;
        }
    }
}