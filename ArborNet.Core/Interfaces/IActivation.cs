using ArborNet.Core.Devices;
using ArborNet.Core.Tensors;

namespace ArborNet.Core.Interfaces
{
    /// <summary>
    /// Core interface for all activation functions in ArborNet.
    /// Guarantees device-aware, numerically-stable forward passes with full autograd support.
    /// </summary>
    /// <remarks>
    /// All implementations must be thread-safe and support both CPU and GPU devices through the
    /// <see cref="Device"/> abstraction. The forward pass must be fully differentiable to integrate
    /// correctly with the autograd system.
    /// </remarks>
    public interface IActivation
    {
        /// <summary>
        /// Computes the forward pass of the activation function.
        /// </summary>
        /// <param name="input">The input tensor (must not be null and must have valid shape).</param>
        /// <returns>The output tensor after applying the activation.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the input tensor has an invalid shape or device configuration.</exception>
        ITensor Forward(ITensor input);
    }

    /// <summary>
    /// Base implementation providing common functionality for activations.
    /// </summary>
    /// <remarks>
    /// This abstract class provides input validation, device management, and a foundation
    /// for all concrete activation implementations. It ensures consistent error handling
    /// and device-aware behavior across the library.
    /// </remarks>
    public abstract class BaseActivation : IActivation
    {
        /// <summary>
        /// The device this activation operates on (defaults to CPU).
        /// </summary>
        /// <value>The current computation device.</value>
        protected Device Device { get; set; } = Device.CPU;

        /// <inheritdoc />
        public abstract ITensor Forward(ITensor input);

        /// <summary>
        /// Validates the input tensor and ensures device consistency.
        /// </summary>
        /// <param name="input">The tensor to validate.</param>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="input"/> is null.</exception>
        /// <exception cref="ArgumentException">
        /// Thrown if the tensor has a null shape, zero elements, or no device specified.
        /// </exception>
        protected void ValidateInput(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (input.Shape == null) throw new ArgumentException("Input tensor must have a valid shape.", nameof(input));
            if (input.Shape.TotalElements == 0) throw new ArgumentException("Input tensor cannot be empty.", nameof(input));
            if (input.Device == null) throw new ArgumentException("Input tensor must specify a device.", nameof(input));
        }

        /// <summary>
        /// Sets the device for this activation and all future operations.
        /// </summary>
        /// <param name="device">The target device. If null, defaults to CPU.</param>
        /// <remarks>
        /// This method enables seamless movement of the activation between different hardware
        /// backends (CPU, CUDA, etc.) without changing the activation's internal logic.
        /// </remarks>
        public virtual void To(Device device)
        {
            Device = device ?? Device.CPU;
        }
    }
}