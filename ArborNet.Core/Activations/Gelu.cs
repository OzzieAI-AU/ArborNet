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

    using System;
    using ArborNet.Core;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Represents a production-grade Gaussian Error Linear Unit (GELU) activation function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// GELU is a high-performance activation function used extensively in modern neural network architectures
    /// such as BERT, GPT, and other Transformer-based models. It scales inputs by the cumulative distribution
    /// function of the standard normal distribution.
    /// </para>
    /// <para>
    /// This implementation utilizes a numerically stable approximation using the hyperbolic tangent function:
    /// <c>GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))</c>
    /// </para>
    /// <para>
    /// This class is device-aware, allocating all intermediary tensors on the same device as the input tensor,
    /// and provides full autograd support through automated gradient tracking during the backward pass.
    /// </para>
    /// </remarks>
    /// <seealso cref="ArborNet.Activations.BaseActivation" />

    #endregion

    public class Gelu : BaseActivation
    {
        /// <summary>
        /// The coefficient constant (0.044715) applied to the cubic term within the GELU approximation formula.
        /// </summary>
        private const float COEFF = 0.044715f;

        /// <summary>
        /// The pre-computed constant value of <c>sqrt(2 / π)</c> (approximately 0.79788456)
        /// used to scale the inner polynomial of the GELU approximation.
        /// </summary>
        private const float SQRT_2_OVER_PI = 0.7978845608028654f;
        /// <summary>
        /// Computes the forward pass of the GELU activation function.
        /// </summary>
        /// <param name="input">The input tensor containing the values to be activated. This tensor must not be null and must reside on a valid computational device.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise GELU activation values, allocated on the same device as the input.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// <para>
        /// All intermediate tensor allocations are bound to the device of the <paramref name="input"/> tensor to prevent cross-device memory transfers and maintain high performance.
        /// </para>
        /// <para>
        /// If <see cref="ITensor.RequiresGrad"/> is enabled on the input tensor, a custom backward gradient 
        /// function (<see cref="ITensor.GradFn"/>) is registered on the resulting output tensor to enable 
        /// automatic differentiation during backpropagation.
        /// </para>
        /// <para>
        /// The backward gradient pass uses the chain rule to propagate gradients through the approximation:
        /// <c>dOut/dx = gradOutput * (0.5 * (1 + tanh(y)) + 0.5 * x * (1 - tanh²(y)) * sqrt(2/π))</c>
        /// where <c>y = sqrt(2/π) * (x + 0.044715 * x³)</c>.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code>
        /// var gelu = new Gelu();
        /// var output = gelu.Forward(inputTensor);
        /// </code>
        /// </example>

        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var device = input.Device;
            var one = Tensor.Ones(input.Shape, device);
            var half = Tensor.FromScalar(0.5f, device);
            var coeff = Tensor.FromScalar(COEFF, device);
            var sqrt2pi = Tensor.FromScalar(SQRT_2_OVER_PI, device);

            var x3 = input.Multiply(input).Multiply(input);
            var inner = input.Add(coeff.Multiply(x3));
            var tanhArg = sqrt2pi.Multiply(inner);
            var tanh = new Tanh().Forward(tanhArg);
            var factor = half.Multiply(one.Add(tanh));
            var output = input.Multiply(factor);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var tanhSq = tanh.Multiply(tanh);
                    var sech2 = one.Subtract(tanhSq);
                    var phi = half.Multiply(one.Add(tanh))
                              .Add(input.Multiply(sech2).Multiply(sqrt2pi).Multiply(half));
                    return gradOutput.Multiply(phi);
                };
            }

            return output;
        }
    }
}