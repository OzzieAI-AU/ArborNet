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
    /// Implements the Mish activation function, a smooth, self-regularized, non-monotonic activation function.
    /// Mish is mathematically defined as: <c>Mish(x) = x * tanh(softplus(x))</c>, where <c>softplus(x) = ln(1 + e^x)</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Mish has been shown to match or exceed the performance of other popular activation functions such as ReLU and Swish 
    /// across various deep learning architectures. It features:
    /// <list type="bullet">
    /// <item>
    /// <term>Unbounded above</term>
    /// <description>Eliminates saturation of gradients during training, preventing slow convergence near the positive extreme.</description>
    /// </item>
    /// <item>
    /// <term>Bounded below</term>
    /// <description>Provides strong regularization effects by capping the influence of extreme negative inputs.</description>
    /// </item>
    /// <item>
    /// <term>Non-monotonicity</term>
    /// <description>Preserves small negative gradients, enhancing gradient flow and information propagation through deep architectures.</description>
    /// </item>
    /// <item>
    /// <term>Smoothness</term>
    /// <description>Continuously differentiable (C1 smooth), avoiding optimization difficulties associated with sharp transitions (like those in ReLU) and ensuring smooth loss landscapes.</description>
    /// </item>
    /// </list>
    /// </para>
    /// <para>
    /// This implementation integrates natively with the framework's tensor system and utilizes autograd 
    /// backpropagation mechanics via computational graph registration.
    /// </para>
    /// </remarks>
    /// <seealso cref="BaseActivation" />
    /// <seealso cref="ITensor" />

    #endregion

    public class Mish : BaseActivation
    {
        private const float STABILITY_THRESHOLD = 20.0f;
        /// <summary>
        /// Performs the forward pass of the Mish activation function.
        /// Computes <c>Mish(x) = x * tanh(softplus(x))</c> in a numerically stable manner.
        /// </summary>
        /// <param name="input">The input tensor <c>x</c> to be activated.</param>
        /// <returns>A new tensor containing the activated values.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>

        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input);
            var device = input.Device;
            var one = Tensor.Ones(input.Shape, device);
            var zero = Tensor.Zeros(input.Shape, device);

            // Condition mask: Is input > 20.0f?
            var thresholdTensor = Tensor.FromScalar(STABILITY_THRESHOLD, device);
            var isLarge = input.GreaterThan(thresholdTensor);

            // 1. Stable Softplus:
            // If x > 20: softplus(x) ≈ x
            // Else: log(1 + exp(x))
            var exp_x = input.Exp();
            var softplus = isLarge.Where(isLarge, input, exp_x.Add(one).Log());

            // 2. Stable Tanh(Softplus):
            // If x > 20: tanh(softplus(x)) ≈ 1.0
            var exp_2_softplus = softplus.Multiply(2.0f).Exp();
            var numerator = exp_2_softplus.Subtract(one);
            var denominator = exp_2_softplus.Add(one);
            var tanh_softplus_normal = numerator.Divide(denominator);

            var tanh_softplus = isLarge.Where(isLarge, one, tanh_softplus_normal);

            // Mish(x) = x * tanh(softplus(x))
            var result = input.Multiply(tanh_softplus);

            if (input.RequiresGrad)
            {
                result.GradFn = gradOutput =>
                {
                    var gradInput = ComputeGrad(input, tanh_softplus, isLarge, gradOutput);
                    input.AccumulateGrad(gradInput);
                    return gradInput;
                };
            }

            return result;
        }
        /// <summary>
        /// Computes the gradient of the Mish activation function during the backward pass (backpropagation).
        /// Uses the precomputed forward values and applies the chain rule.
        /// </summary>
        /// <param name="input">The original input tensor from the forward pass.</param>
        /// <param name="tanh_softplus">The precomputed <c>tanh(softplus(input))</c> tensor from the forward pass.</param>
        /// <param name="isLarge">A boolean mask tensor indicating which input elements exceeded the stability threshold.</param>
        /// <param name="gradOutput">The incoming gradient tensor propagated from the next layer.</param>
        /// <returns>The gradient of the loss with respect to the input tensor.</returns>

        private ITensor ComputeGrad(ITensor input, ITensor tanh_softplus, ITensor isLarge, ITensor gradOutput)
        {
            var device = input.Device;
            var one = Tensor.Ones(input.Shape, device);
            var zero = Tensor.Zeros(input.Shape, device);

            // 1. Stable Sigmoid(x):
            // If x > 20: sigmoid(x) ≈ 1.0
            // If x < -20: sigmoid(x) ≈ 0.0
            // Else: 1.0 / (1.0 + exp(-x))
            var negThresholdTensor = Tensor.FromScalar(-STABILITY_THRESHOLD, device);
            var isExtremelyNegative = input.LessEqual(negThresholdTensor);

            var exp_neg_x = input.Multiply(-1.0f).Exp();
            var sigmoid_normal = one.Divide(one.Add(exp_neg_x));

            var sigmoid_x = isLarge.Where(isLarge, one,
                            isExtremelyNegative.Where(isExtremelyNegative, zero, sigmoid_normal));

            // 2. Stable Sech^2(Softplus):
            // sech^2(x) = 1.0 - tanh^2(x)
            // If x > 20: sech^2(softplus(x)) ≈ 0.0 (since tanh ≈ 1.0)
            var tanh_squared = tanh_softplus.Multiply(tanh_softplus);
            var sech_squared_normal = one.Subtract(tanh_squared);
            var sech_squared = isLarge.Where(isLarge, zero, sech_squared_normal);

            // 3. Compute Mish derivative:
            // Mish'(x) = tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x)
            var second_term = input.Multiply(sech_squared).Multiply(sigmoid_x);
            var localGrad = tanh_softplus.Add(second_term);

            // Chain Rule
            return localGrad.Multiply(gradOutput);
        }
    }
}