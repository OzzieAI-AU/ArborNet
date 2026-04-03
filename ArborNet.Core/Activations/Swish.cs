using System;
using ArborNet.Core;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Activations
{
    /// <summary>
    /// Production-grade Swish activation with numerical stability.
    /// Swish(x) = x * sigmoid(x).
    /// Full autograd with exact local gradient: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
    /// </summary>
    public class Swish : BaseActivation
    {
        /// <summary>
        /// Small constant added to the negative input before exponentiation 
        /// to improve numerical stability and prevent potential overflow or NaN values.
        /// </summary>
        private const float STABILITY_EPS = 1e-8f;

        /// <summary>
        /// Computes the Swish activation function on the provided input tensor.
        /// </summary>
        /// <param name="input">The input tensor to apply the Swish activation to.</param>
        /// <returns>A tensor containing the result of the Swish activation (x * sigmoid(x)).</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>
        /// <remarks>
        /// Implements a numerically stable sigmoid using an epsilon offset. 
        /// When the input tensor requires gradients, a custom gradient function is registered 
        /// that computes the exact local derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
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