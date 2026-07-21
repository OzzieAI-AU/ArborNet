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
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Represents a production-grade Hard Sigmoid activation function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Hard Sigmoid function is a piecewise linear approximation of the standard sigmoid function.
    /// It is mathematically defined as:
    /// </para>
    /// <para>
    /// <c>HardSigmoid(x) = max(0, min(1, (x + 3) / 6))</c>
    /// </para>
    /// <para>
    /// This approximation is computationally more efficient than the standard sigmoid activation function
    /// because it avoids expensive exponential operations, making it highly suitable for deep learning models
    /// optimized for resource-constrained environments, mobile hardware, or edge devices.
    /// </para>
    /// <para>
    /// During the backward pass, the gradient is computed using the subgradient of the piecewise function:
    /// <c>d/dx HardSigmoid(x) = 1/6</c> if <c>-3 &lt; x &lt;= 3</c>, and <c>0</c> otherwise.
    /// </para>
    /// </remarks>
    /// <example>
    /// This example shows how to instantiate and use the <see cref="HardSigmoid"/> activation function.
    /// <code>
    /// var activation = new HardSigmoid();
    /// ITensor input = Tensor.FromArray(new float[] { -4.0f, -3.0f, 0.0f, 3.0f, 4.0f });
    /// ITensor output = activation.Forward(input);
    /// // Output will contain values: [0.0f, 0.0f, 0.5f, 1.0f, 1.0f]
    /// </code>
    /// </example>
    /// <seealso cref="BaseActivation" />
    /// <seealso cref="ITensor" />

    #endregion

    public class HardSigmoid : BaseActivation
    {
        /// <summary>
        /// Computes the forward pass of the Hard Sigmoid activation function.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> containing the values to be activated. This tensor can be of any shape and must reside on an active compute device.</param>
        /// <returns>A new <see cref="ITensor"/> containing the activated values, restricted to the range [0.0, 1.0].</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// <para>
        /// The forward operation performs element-wise computation:
        /// <c>y = clamp((x + 3) / 6, 0, 1)</c>.
        /// </para>
        /// <para>
        /// If the <paramref name="input"/> tensor has its <c>RequiresGrad</c> property set to <see langword="true"/>,
        /// a backward gradient function (<c>GradFn</c>) is registered on the resulting output tensor to enable auto-differentiation.
        /// The backward pass calculates the gradient of the loss with respect to the input as:
        /// <c>grad_input = grad_output * (1 / 6)</c> for elements where <c>-3 &lt; input &lt;= 3</c>, and <c>0</c> otherwise.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code>
        /// var hardSigmoid = new HardSigmoid();
        /// var input = Tensor.FromScalar(1.5f);
        /// var output = hardSigmoid.Forward(input);
        /// </code>
        /// </example>
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var output = input.Add(3f).Divide(6f).Clip(0f, 1f);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var mask = input.GreaterThan(Tensor.FromScalar(-3f, input.Device))
                                   .Multiply(input.LessEqual(Tensor.FromScalar(3f, input.Device)));
                    var gradInput = gradOutput.Multiply(mask.Divide(6f));
                    input.AccumulateGrad(gradInput);
                    return gradInput;
                };
            }

            return output;
        }
    }
}