using System;
using ArborNet.Core;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Activations
{
    /// <summary>
    /// Implements the Sigmoid activation function with numerical stability (eps) and full autograd support.
    /// Sigmoid(x) = 1 / (1 + exp(-x)). Derivative = sigmoid(x) * (1 - sigmoid(x)).
    /// </summary>
    /// <remarks>
    /// This activation function is numerically stable and automatically registers
    /// a custom gradient function when the input tensor requires gradients.
    /// The implementation follows the mathematically equivalent but more stable form:
    /// sigmoid(x) = 1 / (1 + exp(-x)).
    /// </remarks>
    public class Sigmoid : BaseActivation
    {
        /// <summary>
        /// Small constant added to the denominator for numerical stability.
        /// </summary>
        /// <remarks>
        /// Prevents division-by-zero errors when <c>exp(-x)</c> becomes very large.
        /// </remarks>
        private const float EPS = 1e-8f;

        /// <summary>
        /// Computes the sigmoid activation element-wise on the input tensor.
        /// </summary>
        /// <param name="input">The input tensor to which the sigmoid function will be applied.</param>
        /// <returns>A new tensor containing the result of the sigmoid function applied element-wise.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// If the input tensor has <see cref="ITensor.RequiresGrad"/> set to <see langword="true"/>,
        /// a gradient function is registered to support automatic differentiation.
        /// The local gradient is computed as: <c>output * (1 - output)</c>.
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            ValidateInput(input);

            var device = input.Device;
            var one = Tensor.Ones(input.Shape, device);
            var negInput = input.Multiply(-1.0f);
            var expNeg = negInput.Exp();
            var denom = one.Add(expNeg).Add(EPS);
            var output = one.Divide(denom);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var oneMinusOut = one.Subtract(output);
                    var localGrad = output.Multiply(oneMinusOut);
                    return localGrad.Multiply(gradOutput);
                };
            }

            return output;
        }
    }
}