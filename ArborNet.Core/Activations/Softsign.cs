// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Activations
{

    #region Using Statements:

    using ArborNet.Activations;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    using System.Text;
    /// <summary>
    /// Represents the Softsign activation function layer in a neural network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Softsign activation function is a smooth, continuously differentiable alternative 
    /// to the hyperbolic tangent (<c>tanh</c>) function. It is defined mathematically as:
    /// </para>
    /// <para>
    /// <c>Softsign(x) = x / (1 + |x|)</c>
    /// </para>
    /// <para>
    /// It maps arbitrary real-valued input tensor elements to the open range (-1, 1). 
    /// Compared to the <c>tanh</c> activation, Softsign approaches its asymptotic limits 
    /// more slowly (exhibiting a flatter tail). This milder saturation can help mitigate 
    /// the vanishing gradient problem during backpropagation in deep neural network architectures.
    /// </para>
    /// </remarks>
    /// <example>
    /// The following example demonstrates how to instantiate the Softsign activation layer and compute its forward pass:
    /// <code>
    /// var softsign = new Softsign();
    /// ITensor output = softsign.Forward(inputTensor);
    /// </code>
    /// </example>
    /// <seealso cref="BaseActivation"/>
    /// <seealso cref="ITensor"/>

    #endregion

    public class Softsign : BaseActivation
    {
        /// <summary>
        /// Computes the forward pass of the Softsign activation function.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> containing the values to be activated.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise activated values, mapped to the range (-1, 1).</returns>
        /// <exception cref="ArgumentNullException">Thrown when the <paramref name="input"/> tensor is <see langword="null"/>.</exception>
        /// <remarks>
        /// <para>
        /// This method performs an element-wise operation using the following formula:
        /// <c>output = input / (1 + |input|)</c>
        /// </para>
        /// <para>
        /// If the <paramref name="input"/> tensor has gradient tracking enabled (<see cref="ITensor.RequiresGrad"/> is <see langword="true"/>),
        /// this method will construct and register a custom backward gradient function (<see cref="ITensor.GradFn"/>) on the output tensor.
        /// The analytical derivative used for the backpropagation step is:
        /// </para>
        /// <para>
        /// <c>d/dx (Softsign(x)) = 1 / (1 + |x|)^2</c>
        /// </para>
        /// <para>
        /// Consequently, the incoming gradient (<c>gradOutput</c>) is backpropagated as:
        /// <c>gradInput = gradOutput * (1 / (1 + |input|)^2)</c>
        /// </para>
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var output = input.Divide(Tensor.Ones(input.Shape, input.Device).Add(input.Abs()));

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var denom = Tensor.Ones(input.Shape, input.Device).Add(input.Abs());
                    var gradInput = gradOutput.Multiply(Tensor.Ones(input.Shape, input.Device).Divide(denom.Multiply(denom)));
                    input.AccumulateGrad(gradInput);
                    return gradInput;
                };
            }

            return output;
        }
    }
}