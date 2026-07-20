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
    /// Implements the Rectified Linear Unit (ReLU) activation function with full autograd support.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Rectified Linear Unit (ReLU) is a non-linear activation function defined element-wise as:
    /// <c>ReLU(x) = max(0, x)</c>
    /// </para>
    /// <para>
    /// ReLU is one of the most commonly used activation functions in deep neural networks.
    /// It introduces non-linearity while preserving the gradient for positive inputs,
    /// helping to mitigate the vanishing gradient problem. This implementation fully supports
    /// the autograd system by registering a gradient function that applies the same
    /// positivity mask during backpropagation.
    /// </para>
    /// <para>
    /// The derivative (subgradient) of the ReLU function is:
    /// <c>d/dx ReLU(x) = 1</c> if <c>x &gt; 0</c> else <c>0</c>.
    /// </para>
    /// </remarks>
    /// <example>
    /// The following example demonstrates how to instantiate the ReLU activation function and perform a forward pass:
    /// <code>
    /// var relu = new ReLU();
    /// var input = Tensor.FromArray(new float[] { -2.0f, -0.5f, 0.0f, 1.5f, 3.0f });
    /// var output = relu.Forward(input);
    /// // output contains: [0.0f, 0.0f, 0.0f, 1.5f, 3.0f]
    /// </code>
    /// </example>
    /// <seealso cref="BaseActivation"/>
    /// <seealso cref="ITensor"/>

    #endregion

    public class ReLU : BaseActivation
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ReLU"/> class.
        /// </summary>
        public ReLU()
        {
        }
        /// <summary>
        /// Computes the forward pass of the ReLU activation.
        /// </summary>
        /// <param name="input">The input tensor to which the ReLU activation is applied.</param>
        /// <returns>A new <see cref="ITensor"/> containing the activated values, where each element is <c>max(0, x)</c>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// <para>
        /// The implementation creates a binary mask of the same shape and device as the input tensor,
        /// where elements are <c>1</c> if <c>input &gt; 0</c> and <c>0</c> otherwise.
        /// The resulting output is computed via element-wise multiplication: <c>output = input * mask</c>.
        /// </para>
        /// <para>
        /// When <see cref="ITensor.RequiresGrad"/> is enabled on the <paramref name="input"/> tensor,
        /// a backward gradient function is attached to the output tensor's <see cref="ITensor.GradFn"/>.
        /// During backpropagation, this function propagates gradients only through the elements that
        /// were positive in the forward pass by multiplying the incoming gradient by the pre-computed mask:
        /// <c>gradInput = grad * mask</c>.
        /// </para>
        /// </remarks>

        public override ITensor Forward(ITensor input)
        {
            // Create mask: 1 where x > 0, 0 otherwise
            var zero = Tensor.Zeros(input.Shape, input.Device);
            var mask = input.GreaterThan(zero);        // This returns Tensor

            // ReLU = x * mask
            var output = input.Multiply(mask);

            // Backward: gradient * mask (subgradient of ReLU)
            if (input.RequiresGrad)
            {
                output.GradFn = grad =>
                {
                    // dy/dx = 1 if x > 0 else 0
                    var gradInput = grad.Multiply(mask);
                    if (input.Grad == null)
                    {
                        input.Grad = gradInput;
                    }
                    else
                    {
                        input.Grad = input.Grad.Add(gradInput);
                    }
                    return gradInput;
                };
            }

            return output;
        }
    }
}