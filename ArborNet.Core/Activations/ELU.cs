using System;
using ArborNet.Core;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Autograd;

namespace ArborNet.Activations
{
    /// <summary>
    /// Implements the ELU (Exponential Linear Unit) activation function.
    /// ELU(x) = x if x >= 0, else alpha * (exp(x) - 1).
    /// Inherits from BaseActivation and uses autograd for backpropagation.
    /// </summary>
    public class ELU : BaseActivation
    {
        /// <summary>
        /// Scaling factor for the negative saturation regime of the ELU function.
        /// Controls the value to which negative inputs saturate.
        /// </summary>
        private readonly float alpha;

        /// <summary>
        /// Small constant used for numerical stability in tensor operations.
        /// </summary>
        private const float EPS = 1e-8f;

        /// <summary>
        /// Initializes a new instance of the ELU class with the specified alpha value.
        /// </summary>
        /// <param name="alpha">The alpha parameter for the ELU function. Default is 1.0f.</param>
        public ELU(float alpha = 1.0f)
        {
            if (alpha < 0) throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be non-negative");
            this.alpha = alpha;
        }

        /// <summary>
        /// Computes the forward pass of the ELU activation function.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor after applying ELU.</returns>
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            ValidateInput(input);

            var device = input.Device;
            var zero = Tensor.Zeros(input.Shape, device);
            var one = Tensor.Ones(input.Shape, device);
            var epsTensor = Tensor.FromScalar(EPS, device);

            var exp_x = input.Exp();
            var exp_x_minus_1 = exp_x.Subtract(one);

            var negative_part = exp_x_minus_1.Multiply(alpha);
            var mask = input.GreaterThanOrEqual(zero);

            var output = mask.Multiply(input).Add(mask.Negate().Multiply(negative_part));

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput => ComputeGrad(input, gradOutput);
            }
            return output;
        }

        /// <summary>
        /// Computes the gradient for the ELU activation during backpropagation using the Chain Rule.
        /// ELU'(x) = 1 if x >= 0 else alpha * exp(x)
        /// </summary>
        /// <param name="input">The original input tensor.</param>
        /// <param name="gradOutput">The upstream gradient (from the next layer / loss function).</param>
        /// <returns>The final gradient tensor with respect to the input.</returns>
        private ITensor ComputeGrad(ITensor input, ITensor gradOutput)
        {
            var device = input.Device;
            var zero = Tensor.Zeros(input.Shape, device);
            var exp = input.Exp();
            var mask = input.GreaterThanOrEqual(zero);
            var grad = mask.Add(mask.Negate().Multiply(alpha).Multiply(exp));
            return grad.Multiply(gradOutput);
        }

    }
}