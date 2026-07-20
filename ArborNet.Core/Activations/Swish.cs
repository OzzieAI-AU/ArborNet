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
    /// Represents the Swish activation function, a smooth, non-monotonic activation function.
    /// Mathematically defined as: <c>Swish(x) = x * sigmoid(beta * x)</c>, where beta typically defaults to 1.0.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Swish was proposed by Ramachandran et al. at Google Brain and has been empirically shown to outperform 
    /// traditional activation functions like ReLU and ELU on deep neural network architectures across various tasks.
    /// Its key properties include being bounded below, unbounded above, smooth (infinitely differentiable), and non-monotonic.
    /// </para>
    /// <para>
    /// This implementation ensures numerical stability during exponentiation and registers a custom 
    /// autograd backward pass using the exact analytical gradient: 
    /// <c>f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))</c>.
    /// </para>
    /// </remarks>
    /// <example>
    /// This example shows how to instantiate the Swish activation function and apply it to an input tensor:
    /// <code>
    /// var swish = new Swish();
    /// ITensor input = Tensor.RandomNormal(new long[] { 2, 3 });
    /// ITensor output = swish.Forward(input);
    /// </code>
    /// </example>
    /// <seealso cref="BaseActivation"/>
    /// <seealso cref="ITensor"/>

    #endregion

    public class Swish : BaseActivation
    {
        /// <summary>
        /// A small constant offset added to negative inputs prior to exponentiation.
        /// This helps prevent floating-point underflow/overflow and ensures numerical stability.
        /// </summary>
        private const float STABILITY_EPS = 1e-8f;
        /// <summary>
        /// Computes the forward pass of the Swish activation function on the provided input tensor.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> containing the pre-activation values.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise Swish activation values.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// <para>
        /// The computation utilizes a stable sigmoid implementation:
        /// <c>f(x) = x * (1 / (1 + exp(-x + epsilon)))</c> where epsilon is <see cref="STABILITY_EPS"/>.
        /// </para>
        /// <para>
        /// If <see cref="ITensor.RequiresGrad"/> is enabled on the input, a gradient function (<see cref="ITensor.GradFn"/>) 
        /// is registered. This delegate automatically computes exact local gradients during backpropagation:
        /// <c>dOut/dIn = gradOutput * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))</c>.
        /// </para>
        /// <para>
        /// This method allocates intermediate tensors for the computation graph. Users should manage 
        /// resource disposal or context scopes if working in memory-constrained environments.
        /// </para>
        /// </remarks>

        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var device = input.Device;
            var one = Tensor.Ones(input.Shape, device);
            var negInput = input.Multiply(-1.0f).Add(STABILITY_EPS);
            var expNeg = negInput.Exp();
            var denom = one.Add(expNeg);
            var sigmoid = one.Divide(denom);
            var output = input.Multiply(sigmoid);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var sig = new Sigmoid().Forward(input);
                    var oneMinusSig = one.Subtract(sig);
                    var localGrad = sig.Add(input.Multiply(sig.Multiply(oneMinusSig)));
                    return localGrad.Multiply(gradOutput);
                };
            }

            return output;
        }
    }
}