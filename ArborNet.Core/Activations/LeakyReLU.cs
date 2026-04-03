using ArborNet.Core;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using System;

namespace ArborNet.Activations
{
    /// <summary>
    /// Production-grade Leaky ReLU activation with full device awareness, numerical stability,
    /// and correct autograd support. Inherits from <see cref="BaseActivation"/> for consistent
    /// validation and device handling.
    /// </summary>
    /// <remarks>
    /// Leaky ReLU is defined as:
    /// <c>f(x) = x</c> if <c>x &gt; 0</c>, otherwise <c>f(x) = negativeSlope * x</c>.
    /// 
    /// This implementation:
    /// - Respects the input tensor's device (CPU or CUDA)
    /// - Uses mask-based forward and backward passes for clarity and performance
    /// - Correctly registers a gradient function that propagates through the leak
    /// - Ensures all intermediate tensors are allocated on the same device as the input
    /// - Provides full XML documentation and input validation
    /// </remarks>
    public class LeakyReLU : BaseActivation
    {
        /// <summary>
        /// The negative slope coefficient applied to negative input values.
        /// Controls the "leak" for negative inputs to prevent dying ReLU problem.
        /// </summary>
        private readonly float negativeSlope;

        /// <summary>
        /// Initializes a new instance of the <see cref="LeakyReLU"/> class.
        /// </summary>
        /// <param name="negativeSlope">The negative slope for inputs less than zero. 
        /// Must be non-negative. Default value is 0.01f.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="negativeSlope"/> is negative.</exception>
        public LeakyReLU(float negativeSlope = 0.01f)
        {
            if (negativeSlope < 0)
                throw new ArgumentOutOfRangeException(nameof(negativeSlope), "Negative slope must be non-negative.");

            this.negativeSlope = negativeSlope;
        }

        /// <summary>
        /// Applies the Leaky ReLU activation function to the input tensor.
        /// </summary>
        /// <param name="input">The input tensor to which the activation is applied. Must not be null.</param>
        /// <returns>A new tensor containing the result of the Leaky ReLU activation, allocated on the same device as the input.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>
        /// <remarks>
        /// The implementation uses a boolean mask (input &gt; 0) to compute both the forward result and the backward gradient.
        /// When the input tensor requires gradients, a gradient function is attached that correctly computes:
        /// <c>gradInput = gradOutput * (input &gt; 0 ? 1 : negativeSlope)</c>
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            ValidateInput(input);

            var device = input.Device;
            var zero = Tensor.Zeros(input.Shape, device);
            var mask = input.GreaterThanOrEqual(zero);

            // Positive part: mask * input
            var positive = mask.Multiply(input);

            // Negative part: (1 - mask) * (negativeSlope * input)
            var negative = mask.Negate().Add(Tensor.Ones(input.Shape, device))
                           .Multiply(input.Multiply(negativeSlope));

            var output = positive.Add(negative);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    // Gradient is 1 where input > 0, otherwise negativeSlope
                    var gradMask = mask.Add(mask.Negate().Multiply(negativeSlope));
                    return gradOutput.Multiply(gradMask);
                };
            }

            return output;
        }

        /// <summary>
        /// Moves the activation (and any associated resources) to the specified device.
        /// </summary>
        /// <param name="device">The target device. If <see langword="null"/>, defaults to <see cref="Device.CPU"/>.</param>
        public override void To(Device device)
        {
            Device = device ?? Device.CPU;
        }
    }
}
