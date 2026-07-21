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
    using ArborNet.Core.Autograd;
    /// <summary>
    /// Represents the Exponential Linear Unit (ELU) activation function layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Exponential Linear Unit (ELU) activation function is defined mathematically as:
    /// <c>f(x) = x</c> if <c>x &gt;= 0</c>, and <c>f(x) = alpha * (exp(x) - 1)</c> if <c>x &lt; 0</c>.
    /// </para>
    /// <para>
    /// ELU alleviates the vanishing gradient problem by using the identity function for positive values. 
    /// Unlike standard Rectified Linear Units (ReLUs), ELUs produce negative values, which allows them 
    /// to push mean unit activations closer to zero. Mean activations closer to zero enable faster learning 
    /// as they bring the gradient closer to the natural gradient.
    /// </para>
    /// <para>
    /// This class inherits from <see cref="BaseActivation"/> and integrates with the autograd engine 
    /// by constructing a computational graph during the forward pass when the input tracks gradients.
    /// </para>
    /// </remarks>
    /// <example>
    /// The following example demonstrates how to initialize and use the ELU activation layer within a neural network pipeline:
    /// <code>
    /// var elu = new ELU(alpha: 1.0f);
    /// ITensor input = Tensor.FromArray(new float[] { -1.5f, 0.0f, 2.0f });
    /// ITensor output = elu.Forward(input);
    /// </code>
    /// </example>

    #endregion

    public class ELU : BaseActivation
    {
        /// <summary>
        /// The scaling factor (<c>α</c>) for the negative saturation regime of the ELU function.
        /// </summary>
        /// <remarks>
        /// This parameter controls the value to which negative inputs saturate. It must be non-negative.
        /// </remarks>
        private readonly float alpha;

        /// <summary>
        /// A small constant used for numerical stability in tensor calculations.
        /// </summary>
        private const float EPS = 1e-8f;

        /// <summary>
        /// Initializes a new instance of the <see cref="ELU"/> class with the specified alpha value.
        /// </summary>
        /// <param name="alpha">The scale factor for negative values. Must be non-negative. Default is 1.0f.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="alpha"/> is less than 0.</exception>
        public ELU(float alpha = 1.0f)
        {
            if (alpha < 0) throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be non-negative");
            this.alpha = alpha;
        }
        /// <summary>
        /// Computes the forward pass of the ELU activation function on the given input tensor.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> containing the values to be activated.</param>
        /// <returns>A new <see cref="ITensor"/> containing the activated output values.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>
        /// <remarks>
        /// If the <paramref name="input"/> tensor tracks gradients (<see cref="ITensor.RequiresGrad"/> is <c>true</c>), 
        /// this method associates a backward gradient function with the resulting output tensor to enable backpropagation.
        /// </remarks>

        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            ValidateInput(input);

            var device = input.Device;
            var zero = Tensor.Zeros(input.Shape, device);
            var one = Tensor.Ones(input.Shape, device);
            var epsTensor = Tensor.FromScalar(EPS, device);

            var exp_x = input.Exp();
            var exp_x_minus_1 = exp_x.Subtract(one);

            var negative_part = exp_x_minus_1.Multiply(alpha);
            var mask = input.GreaterThanOrEqual(zero);

            var output = mask.Multiply(input).Add(mask.Negate().Multiply(negative_part));

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var gradInput = ComputeGrad(input, gradOutput);
                    input.AccumulateGrad(gradInput);
                    return gradInput;
                };
            }
            return output;
        }
        /// <summary>
        /// Computes the backward gradient (Jacobian-vector product) for the ELU activation function during backpropagation.
        /// </summary>
        /// <param name="input">The original input <see cref="ITensor"/> from the forward pass.</param>
        /// <param name="gradOutput">The upstream gradient <see cref="ITensor"/> backpropagated from the subsequent layer.</param>
        /// <returns>An <see cref="ITensor"/> representing the computed gradient with respect to the input.</returns>
        /// <remarks>
        /// The derivative of the ELU function is computed as:
        /// <para>
        /// <c>f'(x) = 1</c> if <c>x &gt;= 0</c>, and <c>f'(x) = alpha * exp(x)</c> if <c>x &lt; 0</c>.
        /// </para>
        /// This derivative is multiplied element-wise by the upstream gradient (<paramref name="gradOutput"/>) using the Chain Rule.
        /// </remarks>

        private ITensor ComputeGrad(ITensor input, ITensor gradOutput)
        {
            var device = input.Device;
            var zero = Tensor.Zeros(input.Shape, device);
            var exp = input.Exp();
            var mask = input.GreaterThanOrEqual(zero);
            var grad = mask.Add(mask.Negate().Multiply(alpha).Multiply(exp));
            return grad.Multiply(gradOutput);
        }

    }
}