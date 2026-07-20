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
    /// Represents the Tanh (Hyperbolic Tangent) activation function layer with full autograd support.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Tanh activation function maps real-valued inputs to the range (-1, 1). 
    /// This S-shaped curve is zero-centered, which often makes it preferable to the sigmoid activation 
    /// function in neural network hidden layers as it helps prevent training dynamics from getting stuck.
    /// </para>
    /// <para>
    /// Mathematically, the function is formulated as:
    /// </para>
    /// <para>
    /// <c>Tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)</c>
    /// </para>
    /// <para>
    /// The derivative (gradient) of the Tanh function is given by:
    /// </para>
    /// <para>
    /// <c>d/dx Tanh(x) = 1 - Tanh(x)^2</c>
    /// </para>
    /// <para>
    /// This implementation is device-aware, numerically stable (utilizing an epsilon offset to prevent division by zero), 
    /// and automatically registers gradient computations for backpropagation when the input tensor requires gradients.
    /// </para>
    /// </remarks>
    /// <seealso cref="BaseActivation" />
    /// <seealso cref="ITensor" />

    #endregion

    public class Tanh : BaseActivation
    {
        /// <summary>
        /// A small epsilon value (<c>1e-8</c>) added to the denominator to prevent division-by-zero 
        /// and maintain numerical stability during the forward pass.
        /// </summary>
        private const float EPS = 1e-8f;
        /// <summary>
        /// Computes the forward pass of the Tanh activation function element-wise on the input tensor.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> containing the values to be activated.</param>
        /// <returns>A new <see cref="ITensor"/> containing the computed Tanh activation values mapped to the open interval (-1, 1).</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// <para>
        /// If the <paramref name="input"/> tensor's <see cref="ITensor.RequiresGrad"/> property is set to <see langword="true"/>,
        /// a backward gradient function (<see cref="ITensor.GradFn"/>) is registered on the returned output tensor.
        /// </para>
        /// <para>
        /// The gradient calculation during backpropagation uses the chain rule:
        /// </para>
        /// <para>
        /// <c>dL/dx = dL/dy * (1 - y^2)</c>
        /// </para>
        /// <para>
        /// where <c>y = Tanh(x)</c> is the output of the forward pass, and <c>dL/dy</c> is the incoming gradient (<paramref name="gradOutput"/>).
        /// </para>
        /// </remarks>

        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            ValidateInput(input);

            var device = input.Device;
            var two = Tensor.FromScalar(2.0f, device);
            var one = Tensor.FromScalar(1.0f, device);
            var exp2x = input.Multiply(two).Exp();
            var numerator = exp2x.Subtract(one);
            var denominator = exp2x.Add(one).Add(EPS);
            var output = numerator.Divide(denominator);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var tanhSq = output.Multiply(output);
                    var oneMinusTanhSq = one.Subtract(tanhSq);
                    return gradOutput.Multiply(oneMinusTanhSq);
                };
            }

            return output;
        }
    }
}