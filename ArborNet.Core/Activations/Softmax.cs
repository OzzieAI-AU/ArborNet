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
        private readonly int axis;

        public Softmax(int axis = -1)
        {
            this.axis = axis;
        }

        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            ValidateInput(input);

            int ax = axis < 0 ? input.Shape.Rank + axis : axis;
            if (ax < 0 || ax >= input.Shape.Rank)
                throw new ArgumentOutOfRangeException(nameof(axis));

            var maxVal = input.Max(ax, keepDims: true);
            var shifted = input.Subtract(maxVal);
            var exp = shifted.Exp();
            var sumExp = exp.Sum(ax, keepDims: true);
            var output = exp.Divide(sumExp);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var weighted = output.Multiply(gradOutput);
                    var sumWeighted = weighted.Sum(ax, keepDims: true);
                    var gradInput = output.Multiply(gradOutput.Subtract(sumWeighted));

                    // FIX: Accumulate the gradient back to the input!
                    input.AccumulateGrad(gradInput);

                    return gradInput;
                };
            }

            return output;
        }
    }
}