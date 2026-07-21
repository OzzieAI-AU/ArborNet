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
    /// Represents the production-grade TanhShrink activation function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The TanhShrink activation function is mathematically defined as:
    /// <br/>
    /// <c>TanhShrink(x) = x - tanh(x)</c>
    /// </para>
    /// <para>
    /// This activation function is typically used in neural network architectures to shrink values near zero 
    /// while keeping larger values linear, acting as a soft-shrinkage operator. This class inherits from 
    /// <see cref="BaseActivation"/> to integrate seamlessly into the ArborNet activation and autograd pipeline.
    /// </para>
    /// <para>
    /// During the backward pass, if the input tensor requires gradients, the backpropagation engine will 
    /// utilize the dynamically assigned gradient function to propagate errors backward through the computational graph.
    /// </para>
    /// </remarks>
    /// <example>
    /// The following example demonstrates how to instantiate and apply the <see cref="TanhShrink"/> activation function:
    /// <code>
    /// // Assuming an execution context with an active ITensor implementation
    /// ITensor input = Tensor.FromArray(new float[] { -2.0f, -0.5f, 0.0f, 0.5f, 2.0f });
    /// TanhShrink activation = new TanhShrink();
    /// ITensor output = activation.Forward(input);
    /// </code>
    /// </example>
    /// <seealso cref="BaseActivation"/>
    /// <seealso cref="ITensor"/>
    /// <seealso cref="Tanh"/>

    #endregion

    public class TanhShrink : BaseActivation
    {
        /// <summary>
        /// Performs the forward pass computation of the TanhShrink activation function.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> to apply the activation function to.</param>
        /// <returns>
        /// A new <see cref="ITensor"/> containing the activated values calculated element-wise as <c>input - tanh(input)</c>.
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown when the <paramref name="input"/> tensor is <see langword="null"/>.</exception>
        /// <remarks>
        /// <para>
        /// If the <paramref name="input"/> tensor's <see cref="ITensor.RequiresGrad"/> property is set to <see langword="true"/>,
        /// this method dynamically assigns a gradient calculation delegate to the output tensor's <see cref="ITensor.GradFn"/>
        /// property. This enables automatic differentiation during the backward propagation pass.
        /// </para>
        /// <para>
        /// The assigned backward gradient function computes the derivative of the activation function, scaled by the incoming gradient
        /// <c>gradOutput</c>, using the formula:
        /// <br/>
        /// <c>gradOutput * (1 - tanh(input)^2)</c>
        /// </para>
        /// </remarks>
        /// <seealso cref="ITensor.RequiresGrad"/>
        /// <seealso cref="ITensor.GradFn"/>
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var output = input.Subtract(new Tanh().Forward(input));

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var tanh = new Tanh().Forward(input);
                    var gradInput = gradOutput.Multiply(Tensor.Ones(input.Shape, input.Device).Subtract(tanh.Multiply(tanh)));
                    input.AccumulateGrad(gradInput);
                    return gradInput;
                };
            }

            return output;
        }
    }
}