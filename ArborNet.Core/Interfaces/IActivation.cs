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
}