using System;
using ArborNet.Core;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Activations
{
    /// <summary>
    /// Production-grade Softmax with numerical stability (max subtraction),
    /// full autograd support, and arbitrary axis handling.
    /// Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    /// </summary>
    public class Softmax : BaseActivation
    {
        /// <summary>
        /// The axis along which the softmax operation is computed.
        /// A negative value is interpreted as counting from the last dimension.
        /// </summary>
        private readonly int axis;

        /// <summary>
        /// Initializes a new instance of the <see cref="Softmax"/> class.
        /// </summary>
        /// <param name="axis">The axis to compute the softmax over. 
        /// Default is -1 (last axis). Negative values are supported and resolved 
        /// relative to the tensor rank during forward pass.</param>
        public Softmax(int axis = -1)
        {
            this.axis = axis;
        }

        /// <summary>
        /// Computes the softmax activation along the configured axis with numerical stability.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>A tensor of the same shape as <paramref name="input"/> containing 
        /// the softmax probabilities along the specified axis.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="input"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if the resolved axis is invalid 
        /// for the input tensor's rank.</exception>
        /// <remarks>
        /// This implementation subtracts the maximum value along the target axis before 
        /// exponentiation to ensure numerical stability. When the input requires gradients, 
        /// a custom gradient function is attached that implements the Jacobian-vector product 
        /// for softmax: output * (gradOutput - sum(output * gradOutput, axis)).
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