using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using System;

namespace ArborNet.Activations
{
    /// <summary>
    /// Base class for activation functions with validation, device support, and common utilities.
    /// </summary>
    public abstract class BaseActivation : IActivation
    {
        /// <summary>
        /// The computation device (CPU or accelerator) on which this activation operates.
        /// </summary>
        /// <remarks>
        /// Defaults to <see cref="Device.CPU"/>. Derived classes should respect this value
        /// when implementing device-specific operations.
        /// </remarks>
        protected Device Device { get; set; } = Device.CPU;

        /// <summary>
        /// Computes the forward pass of the activation function on the provided input.
        /// </summary>
        /// <param name="input">The input tensor to apply the activation to.</param>
        /// <returns>A tensor containing the result of the activation function.</returns>
        public abstract ITensor Forward(ITensor input);

        /// <summary>
        /// Validates the input tensor for common error conditions before processing.
        /// </summary>
        /// <param name="input">The tensor to validate.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>
        /// <exception cref="ArgumentException">
        /// Thrown when the tensor has no shape, contains zero elements, or does not specify a device.
        /// </exception>
        protected void ValidateInput(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (input.Shape == null) throw new ArgumentException("Input tensor must have a valid shape.", nameof(input));
            if (input.Shape.TotalElements == 0) throw new ArgumentException("Input tensor cannot be empty.", nameof(input));
            if (input.Device == null) throw new ArgumentException("Input tensor must specify a device.", nameof(input));
        }

        /// <summary>
        /// Moves the activation function (and any associated resources) to the specified device.
        /// </summary>
        /// <param name="device">The target device. If <see langword="null"/>, defaults to <see cref="Device.CPU"/>.</param>
        /// <remarks>
        /// This is a virtual method. Derived classes that allocate device-specific resources
        /// should override this method to ensure proper device migration.
        /// </remarks>
        public virtual void To(Device device)
        {
            Device = device ?? Device.CPU;
        }
    }
}