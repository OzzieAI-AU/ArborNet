using System;
using ArborNet.Core;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Activations
{
    /// <summary>
    /// Implements the Softplus activation function with numerical stability.
    /// Softplus(x) = log(1 + exp(x)).
    /// For large x, directly returns x to avoid overflow.
    /// Full autograd support with correct derivative (sigmoid(x)).
    /// </summary>
    public class Softplus : BaseActivation
    {
        /// <summary>
        /// Threshold above which the Softplus function is approximated by the identity function
        /// to prevent numerical overflow in the exponential computation.
        /// </summary>
        private const float STABILITY_THRESHOLD = 20.0f;

        /// <summary>
        /// Computes the Softplus activation function on the input tensor using a numerically stable method.
        /// </summary>
        /// <param name="input">The input tensor to apply the activation function to.</param>
        /// <returns>A tensor containing the result of applying Softplus element-wise.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// For values greater than <see cref="STABILITY_THRESHOLD"/>, the function returns the input directly
        /// to avoid overflow. Otherwise it computes <c>log(1 + exp(x))</c>.
        /// When <see cref="ITensor.RequiresGrad"/> is <see langword="true"/>, a gradient function is attached
        /// that computes the derivative using the Sigmoid function.
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var device = input.Device;
            var one = Tensor.Ones(input.Shape, device);

            // Stable computation: for large x, softplus(x) ≈ x
            // Otherwise use log(1 + exp(x)) with log1p for small values
            var isLarge = input.GreaterThan(Tensor.FromScalar(STABILITY_THRESHOLD, device));
            var stableSoftplus = input.Where(isLarge, input, input.Exp().Add(one).Log());

            var output = stableSoftplus;

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    // Derivative of softplus is sigmoid(x)
                    ITensor sigmoid = new Sigmoid().Forward(input);
                    return gradOutput.Multiply(sigmoid);
                };
            }

            return output;
        }
    }
}