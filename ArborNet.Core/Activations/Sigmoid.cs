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
    /// Implements the Sigmoid activation function with numerical stability (eps) and full autograd support.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Sigmoid function is defined mathematically as:
    /// <c>Sigmoid(x) = 1 / (1 + exp(-x))</c>.
    /// Its derivative is computed as:
    /// <c>d/dx [Sigmoid(x)] = Sigmoid(x) * (1 - Sigmoid(x))</c>.
    /// </para>
    /// <para>
    /// This activation function maps any real-valued input tensor to a range between 0 and 1.
    /// It is designed with numerical stability in mind to prevent overflow and underflow errors 
    /// associated with computing exponentiation on extreme positive or negative values.
    /// </para>
    /// <para>
    /// The implementation utilizes a domain-split approach to handle positive and negative inputs:
    /// <list type="bullet">
    /// <item>
    /// <description>For <c>x &gt;= 0</c>: <c>1 / (1 + exp(-x))</c></description>
    /// </item>
    /// <item>
    /// <description>For <c>x &lt; 0</c>: <c>exp(x) / (1 + exp(x))</c></description>
    /// </item>
    /// </list>
    /// </para>
    /// <para>
    /// When the input tensor's <see cref="ITensor.RequiresGrad"/> is set to <see langword="true"/>,
    /// a custom backward gradient callback is registered on the output tensor to support automatic differentiation.
    /// </para>
    /// </remarks>
    /// <seealso cref="BaseActivation" />
    /// <seealso cref="ITensor" />

    #endregion

    public class Sigmoid : BaseActivation
    {
        /// <summary>
        /// A small epsilon value (<c>1e-8</c>) added to the denominator for numerical stability.
        /// </summary>
        /// <remarks>
        /// Prevents division-by-zero errors or floating-point instability when <c>exp(-x)</c> or <c>exp(x)</c> 
        /// is evaluated near extreme boundary values.
        /// </remarks>
        private const float EPS = 1e-8f;
        /// <summary>
        /// Computes the forward pass of the Sigmoid activation function element-wise on the input tensor.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> to which the sigmoid function is applied.</param>
        /// <returns>A new <see cref="ITensor"/> containing the computed sigmoid activation values.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// <para>
        /// This method validates the input and performs the sigmoid operation element-wise using a stable, split-domain formula.
        /// </para>
        /// <para>
        /// If the <paramref name="input"/> tensor's <see cref="ITensor.RequiresGrad"/> property is set to <see langword="true"/>,
        /// a custom backpropagation function (<c>GradFn</c>) is attached to the returned tensor. The local gradient 
        /// is computed as <c>output * (1 - output)</c>, which is then combined with the incoming <c>gradOutput</c> 
        /// via element-wise multiplication.
        /// </para>
        /// </remarks>

        public override ITensor Forward(ITensor input)
        {

            if (input == null) throw new ArgumentNullException(nameof(input));
            ValidateInput(input);

            var device = input.Device;
            var zero = Tensor.Zeros(input.Shape, device);

            // Stable sigmoid: 
            // For x >= 0: 1 / (1 + exp(-x))
            // For x < 0: exp(x) / (1 + exp(x))
            var isPositive = input.GreaterThanOrEqual(zero);

            var expNeg = input.Multiply(-1.0f).Exp();
            var posPart = Tensor.Ones(input.Shape, device).Divide(Tensor.Ones(input.Shape, device).Add(expNeg).Add(EPS));

            var expPos = input.Exp();
            var negPart = expPos.Divide(Tensor.Ones(input.Shape, device).Add(expPos).Add(EPS));

            var output = isPositive.Where(isPositive, posPart, negPart);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var localGrad = output.Multiply(Tensor.Ones(input.Shape, device).Subtract(output));
                    return localGrad.Multiply(gradOutput);
                };
            }

            return output;
        }
    }
}