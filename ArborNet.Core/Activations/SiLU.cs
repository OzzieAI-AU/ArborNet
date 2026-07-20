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

    using ArborNet.Core;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Implements the SiLU (Sigmoid Linear Unit) activation function, also commonly referred to as the Swish activation function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The SiLU activation function is mathematically defined as:
    /// <c>f(x) = x * sigmoid(x) = x / (1 + e^(-x))</c>.
    /// </para>
    /// <para>
    /// Key characteristics of the SiLU activation function include:
    /// <list type="bullet">
    /// <item>
    /// <description><b>Smoothness:</b> Unlike traditional Rectified Linear Units (ReLU), SiLU is smooth and continuously differentiable everywhere, which aids optimization during gradient descent.</description>
    /// </item>
    /// <item>
    /// <description><b>Non-monotonicity:</b> It exhibits a small, negative dip for negative input values, allowing small negative gradients to propagate rather than being completely zeroed out.</description>
    /// </item>
    /// <item>
    /// <description><b>Self-Gating:</b> The function utilizes the input itself as a gate to scale its activation output dynamically.</description>
    /// </item>
    /// </list>
    /// </para>
    /// </remarks>
    /// <example>
    /// The following example demonstrates how to instantiate and use the SiLU activation function:
    /// <code>
    /// // Create an input tensor (e.g., shape of 2x3)
    /// ITensor input = Tensor.Random(new long[] { 2, 3 });
    /// 
    /// // Initialize the activation layer
    /// SiLU silu = new SiLU();
    /// 
    /// // Execute the forward pass
    /// ITensor output = silu.Forward(input);
    /// </code>
    /// </example>
    /// <seealso cref="BaseActivation"/>
    /// <seealso cref="ITensor"/>

    #endregion

    public class SiLU : BaseActivation
    {
        /// <summary>
        /// Computes the forward pass of the SiLU activation function.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> containing the values to be activated.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise activated values.</returns>
        /// <remarks>
        /// <para>
        /// If the <paramref name="input"/> tensor's <see cref="ITensor.RequiresGrad"/> property is set to <c>true</c>,
        /// this method will register a custom backward gradient function (<see cref="ITensor.GradFn"/>) on the returned tensor.
        /// </para>
        /// <para>
        /// The derivative of the SiLU function used during the backward pass is calculated as:
        /// <c>f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))</c>.
        /// </para>
        /// <para>
        /// To prevent device mismatch errors in multi-hardware environments (such as CUDA/GPU mixed with CPU execution),
        /// the backward propagation step explicitly queries and utilizes the original input tensor's device (<see cref="ITensor.Device"/>).
        /// </para>
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            var sigmoid = new Sigmoid().Forward(input);
            var output = input.Multiply(sigmoid);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var sig = new Sigmoid().Forward(input);
                    // FIXED: Explicitly passed 'input.Device' to avoid CPU-GPU mismatch crashes
                    var ones = Tensor.Ones(input.Shape, input.Device);
                    var grad = sig.Add(input.Multiply(sig.Multiply(ones.Subtract(sig))));
                    return grad.Multiply(gradOutput);
                };
            }
            return output;
        }
    }
}