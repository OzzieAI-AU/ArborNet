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
    /// Represents a production-grade Softmax activation function featuring numerical stability,
    /// full automatic differentiation (autograd) support, and arbitrary axis resolution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Softmax function normalizes an N-dimensional input <see cref="ITensor"/> along a specified axis.
    /// The elements of the resulting output tensor lie in the range [0, 1] and sum to 1.0 along the designated dimension,
    /// making it suitable for representing probability distributions.
    /// </para>
    /// <para>
    /// Mathematical Definition:
    /// <code>
    /// Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    /// </code>
    /// </para>
    /// <para>
    /// To prevent numerical issues such as overflow or underflow (due to large exponentiation values), 
    /// this implementation employs the "max-subtraction trick" (subtracting the maximum value along the target axis 
    /// prior to exponentiation).
    /// </para>
    /// </remarks>
    /// <seealso cref="BaseActivation" />

    #endregion

    public class Softmax : BaseActivation
    {
        /// <summary>
        /// The axis along which the softmax operation is computed.
        /// </summary>
        /// <remarks>
        /// A negative value is interpreted as counting backwards from the last dimension.
        /// </remarks>
        private readonly int axis;

        /// <summary>
        /// Initializes a new instance of the <see cref="Softmax"/> class.
        /// </summary>
        /// <param name="axis">The axis to compute the softmax over. 
        /// Default is -1 (the last axis). Negative values are supported and resolved 
        /// relative to the tensor rank during the forward pass.</param>
        public Softmax(int axis = -1)
        {
            this.axis = axis;
        }
        /// <summary>
        /// Executes the forward pass computation of the Softmax function on the provided input tensor.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> containing the unnormalized log probabilities (logits).</param>
        /// <returns>A new <see cref="ITensor"/> containing the computed softmax probabilities, preserving the shape of the input.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the resolved <see cref="axis"/> is outside the valid bounds of the <paramref name="input"/> tensor's rank.</exception>
        /// <remarks>
        /// <para>
        /// This method is numerically stabilized. If the input tensor has <see cref="ITensor.RequiresGrad"/> set to <see langword="true"/>,
        /// an autograd backward gradient function (<c>GradFn</c>) is attached to the returned tensor.
        /// </para>
        /// <para>
        /// The backward pass calculates the vector-Jacobian product (VJP) defined by:
        /// <code>
        /// dL/dx_i = p_i * (dL/dy_i - sum(dL/dy_j * p_j))
        /// </code>
        /// where <c>p</c> is the output (probabilities) of the forward pass, and <c>dL/dy</c> is the incoming gradient (<c>gradOutput</c>).
        /// </para>
        /// </remarks>

        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            ValidateInput(input);

            int ax = axis < 0 ? input.Shape.Rank + axis : axis;
            if (ax < 0 || ax >= input.Shape.Rank)
                throw new ArgumentOutOfRangeException(nameof(axis));

            var device = input.Device;
            var maxVal = input.Max(ax);
            var shifted = input.Subtract(maxVal.ReshapeWithBroadcast(input.Shape, ax));
            var exp = shifted.Exp();
            var sumExp = exp.Sum(ax);
            var output = exp.Divide(sumExp.ReshapeWithBroadcast(input.Shape, ax));

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var weighted = output.Multiply(gradOutput);
                    var sumWeighted = weighted.Sum(ax);
                    var scaled = sumWeighted.ReshapeWithBroadcast(output.Shape, ax);
                    return output.Multiply(gradOutput.Subtract(scaled));
                };
            }

            return output;
        }
    }
}