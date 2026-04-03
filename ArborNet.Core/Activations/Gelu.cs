using System;
using ArborNet.Core;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Activations
{
    /// <summary>
    /// Production-grade GELU (Gaussian Error Linear Unit) activation.
    /// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    /// Numerically stable, device-aware, full autograd support.
    /// </summary>
    public class Gelu : BaseActivation
    {
        /// <summary>
        /// Coefficient for the cubic term (0.044715) in the GELU approximation.
        /// </summary>
        private const float COEFF = 0.044715f;

        /// <summary>
        /// Pre-computed value of √(2/π) used in the GELU tanh approximation.
        /// </summary>
        private const float SQRT_2_OVER_PI = 0.7978845608028654f;

        /// <summary>
        /// Computes the GELU activation function for the input tensor using a numerically stable tanh approximation.
        /// </summary>
        /// <param name="input">The input tensor. Must not be null.</param>
        /// <returns>A tensor of the same shape containing the GELU-activated values.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>
        /// <remarks>
        /// <para>
        /// The implementation follows the approximation:<br/>
        /// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        /// </para>
        /// <para>
        /// When <see cref="ITensor.RequiresGrad"/> is true, a custom gradient function is registered
        /// to enable full backpropagation through the activation.
        /// </para>
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var device = input.Device;
            var one = Tensor.Ones(input.Shape, device);
            var half = Tensor.FromScalar(0.5f, device);
            var coeff = Tensor.FromScalar(COEFF, device);
            var sqrt2pi = Tensor.FromScalar(SQRT_2_OVER_PI, device);

            var x3 = input.Multiply(input).Multiply(input);
            var inner = input.Add(coeff.Multiply(x3));
            var tanhArg = sqrt2pi.Multiply(inner);
            var tanh = new Tanh().Forward(tanhArg);
            var factor = half.Multiply(one.Add(tanh));
            var output = input.Multiply(factor);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var tanhSq = tanh.Multiply(tanh);
                    var sech2 = one.Subtract(tanhSq);
                    var phi = half.Multiply(one.Add(tanh))
                              .Add(input.Multiply(sech2).Multiply(sqrt2pi).Multiply(half));
                    return gradOutput.Multiply(phi);
                };
            }

            return output;
        }
    }
}