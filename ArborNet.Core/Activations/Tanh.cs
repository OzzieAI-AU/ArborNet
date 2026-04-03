using System;
using ArborNet.Core;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Activations
{
    /// <summary>
    /// Implements the Tanh (Hyperbolic Tangent) activation function with full autograd support.
    /// Tanh(x) = (exp(2*x) - 1) / (exp(2*x) + 1). Derivative = 1 - tanh(x)^2.
    /// Numerically stable, device-aware, production-ready.
    /// </summary>
    public class Tanh : BaseActivation
    {
        /// <summary>
        /// Small epsilon value added to the denominator for numerical stability to prevent division by zero.
        /// </summary>
        private const float EPS = 1e-8f;

        /// <summary>
        /// Computes the hyperbolic tangent (tanh) of the input tensor element-wise.
        /// </summary>
        /// <param name="input">The input tensor to apply the tanh activation to.</param>
        /// <returns>A new tensor containing the tanh activation applied element-wise, with the same shape and device as the input.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// This implementation uses the numerically stable formulation:
        /// <c>tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)</c>.
        /// <para>
        /// When <see cref="ITensor.RequiresGrad"/> is <see langword="true"/>, the backward function is automatically
        /// registered as <c>gradOutput * (1 - tanh(x)^2)</c>, enabling full autograd support.
        /// </para>
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            ValidateInput(input);

            var device = input.Device;
            var two = Tensor.FromScalar(2.0f, device);
            var one = Tensor.FromScalar(1.0f, device);
            var exp2x = input.Multiply(two).Exp();
            var numerator = exp2x.Subtract(one);
            var denominator = exp2x.Add(one).Add(EPS);
            var output = numerator.Divide(denominator);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var tanhSq = output.Multiply(output);
                    var oneMinusTanhSq = one.Subtract(tanhSq);
                    return gradOutput.Multiply(oneMinusTanhSq);
                };
            }

            return output;
        }
    }
}