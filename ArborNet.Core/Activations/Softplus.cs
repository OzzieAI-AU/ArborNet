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
    /// Implements the Softplus activation function with numerical stability.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Softplus activation function is a smooth, continuously differentiable approximation of the rectifier (ReLU) function.
    /// It is mathematically defined as:
    /// <c>f(x) = ln(1 + e^x)</c>
    /// </para>
    /// <para>
    /// For large positive values of <c>x</c>, evaluating <c>e^x</c> directly can cause floating-point overflow.
    /// To ensure numerical stability, this implementation approximates the function as the identity function 
    /// (<c>f(x) = x</c>) when the input exceeds <see cref="STABILITY_THRESHOLD"/>.
    /// </para>
    /// <para>
    /// This implementation provides full autograd support. The derivative of the Softplus function
    /// is the logistic sigmoid function:
    /// <c>f'(x) = 1 / (1 + e^(-x))</c>
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var softplus = new Softplus();
    /// ITensor input = Tensor.FromArray(new float[] { -1.0f, 0.0f, 25.0f });
    /// ITensor output = softplus.Forward(input);
    /// </code>
    /// </example>
    /// <seealso cref="BaseActivation"/>

    #endregion

    public class Softplus : BaseActivation
    {
        /// <summary>
        /// The numerical stability threshold above which the Softplus function is approximated by the identity function.
        /// </summary>
        /// <remarks>
        /// This threshold prevents floating-point overflow during the computation of the exponential function.
        /// For values where <c>x &gt; STABILITY_THRESHOLD</c>, the value of <c>ln(1 + e^x)</c> is mathematically 
        /// indistinguishable from <c>x</c> in standard single-precision floating-point representation.
        /// </remarks>
        private const float STABILITY_THRESHOLD = 20.0f;
        /// <summary>
        /// Computes the forward pass of the Softplus activation function on the input tensor.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> to apply the activation function to.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise Softplus activation values.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// <para>
        /// The calculation is split based on the <see cref="STABILITY_THRESHOLD"/>:
        /// <list type="bullet">
        /// <item>
        /// <description>For elements where <c>x &gt; STABILITY_THRESHOLD</c>: <c>y = x</c></description>
        /// </item>
        /// <item>
        /// <description>For elements where <c>x &lt;= STABILITY_THRESHOLD</c>: <c>y = ln(1 + e^x)</c></description>
        /// </item>
        /// </list>
        /// </para>
        /// <para>
        /// If the input tensor's <see cref="ITensor.RequiresGrad"/> property is set to <see langword="true"/>,
        /// a custom gradient function is attached to the output tensor's <see cref="ITensor.GradFn"/> property. 
        /// The backward pass computes the gradient as the element-wise product of the incoming gradient (<c>gradOutput</c>) 
        /// and the derivative of the Softplus function (which is the element-wise <see cref="Sigmoid"/> of the input tensor).
        /// </para>
        /// </remarks>

        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var device = input.Device;
            var one = Tensor.Ones(input.Shape, device);

            // Stable computation: for large x, softplus(x) ≈ x
            // Otherwise use log(1 + exp(x)) with log1p for small values
            var isLarge = input.GreaterThan(Tensor.FromScalar(STABILITY_THRESHOLD, device));
            var stableSoftplus = input.Where(isLarge, input, input.Exp().Add(one).Log());

            var output = stableSoftplus;

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    ITensor sigmoid = new Sigmoid().Forward(input);
                    var gradInput = gradOutput.Multiply(sigmoid);
                    input.AccumulateGrad(gradInput);
                    return gradInput;
                };
            }

            return output;
        }
    }
}