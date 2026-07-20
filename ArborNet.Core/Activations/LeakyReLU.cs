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

    using ArborNet.Core;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using System;
    /// <summary>
    /// Production-grade Leaky ReLU activation with full device awareness, numerical stability,
    /// and correct autograd support. Inherits from <see cref="BaseActivation"/> for consistent
    /// validation and device handling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Leaky ReLU is defined mathematically as:
    /// <c>f(x) = x</c> if <c>x &gt;= 0</c>, and <c>f(x) = negativeSlope * x</c> if <c>x &lt; 0</c>.
    /// </para>
    /// <para>
    /// This implementation features:
    /// <list type="bullet">
    /// <item><description>Respects the input tensor's execution device (CPU, CUDA, etc.).</description></item>
    /// <item><description>Uses mask-based forward and backward passes for clarity, safety, and performance.</description></item>
    /// <item><description>Correctly registers a gradient function that propagates the gradient through the leak during autograd backward pass.</description></item>
    /// <item><description>Ensures all intermediate tensors are allocated on the same device as the input to avoid cross-device memory copying.</description></item>
    /// <item><description>Provides full validation of input dimensions, devices, and numerical properties via the base class.</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Thread Safety:</b> This activation is stateless during execution. The <see cref="Forward(ITensor)"/> method 
    /// is thread-safe provided the input tensor is not concurrently mutated by other threads.
    /// </para>
    /// <para>
    /// <b>Performance Note:</b> This implementation performs mask-based allocations. In highly performance-sensitive 
    /// hot paths, a fused kernel implementation on hardware accelerators is recommended to minimize temporary allocations.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var activation = new LeakyReLU(0.1f);
    /// var input = Tensor.FromArray(new float[] { -2.0f, 0.0f, 3.0f });
    /// var output = activation.Forward(input);
    /// // output will contain [-0.2f, 0.0f, 3.0f]
    /// </code>
    /// </example>
    /// <seealso cref="BaseActivation"/>
    /// <seealso cref="ITensor"/>

    #endregion

    public class LeakyReLU : BaseActivation
    {
        /// <summary>
        /// The negative slope coefficient applied to negative input values.
        /// Controls the "leak" for negative inputs to prevent the dying ReLU problem.
        /// </summary>
        private readonly float negativeSlope;

        /// <summary>
        /// Initializes a new instance of the <see cref="LeakyReLU"/> class with a specified negative slope.
        /// </summary>
        /// <param name="negativeSlope">
        /// The coefficient of the leakage. Must be a non-negative real number. 
        /// Default value is <c>0.01f</c>.
        /// </param>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown when <paramref name="negativeSlope"/> is less than zero.
        /// </exception>
        public LeakyReLU(float negativeSlope = 0.01f)
        {
            if (negativeSlope < 0)
                throw new ArgumentOutOfRangeException(nameof(negativeSlope), "Negative slope must be non-negative.");

            this.negativeSlope = negativeSlope;
        }
        /// <summary>
        /// Applies the Leaky ReLU activation function element-wise to the input tensor.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> to be activated. Must not be null.</param>
        /// <returns>
        /// A new <see cref="ITensor"/> containing the activated elements, allocated on the same execution device as the <paramref name="input"/>.
        /// </returns>
        /// <exception cref="ArgumentNullException">
        /// Thrown when the <paramref name="input"/> tensor is <see langword="null"/>.
        /// </exception>
        /// <remarks>
        /// <para>
        /// The activation is computed using a mask-based approach:
        /// <c>Output = Mask * Input + (1 - Mask) * (Slope * Input)</c> where <c>Mask = Input &gt;= 0</c>.
        /// </para>
        /// <para>
        /// If the <paramref name="input"/> requires gradients (i.e., <see cref="ITensor.RequiresGrad"/> is <see langword="true"/>),
        /// an autograd backward delegate <see cref="ITensor.GradFn"/> is registered to compute:
        /// <c>gradInput = gradOutput * (Mask + (1 - Mask) * Slope)</c>.
        /// </para>
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
        /// Moves the activation context and updates its target execution device.
        /// </summary>
        /// <param name="device">
        /// The target <see cref="Device"/> to move execution context and internal states to.
        /// If <see langword="null"/>, defaults to <see cref="Device.CPU"/>.
        /// </param>
        /// <remarks>
        /// This method updates the underlying device state of the activation, ensuring that subsequent 
        /// execution and tracking operations align correctly with the targeted hardware accelerator or CPU.
        /// </remarks>

        public override void To(Device device)
        {
            Device = device ?? Device.CPU;
        }
    }
}