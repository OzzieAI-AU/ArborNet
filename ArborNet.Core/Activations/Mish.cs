using System;
using ArborNet.Core;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Activations
{
    /// <summary>
    /// Implements the Mish activation function.
    /// Mish(x) = x * tanh(softplus(x)).
    /// Inherits from BaseActivation and uses autograd for backpropagation.
    /// </summary>
    /// <remarks>
    /// Mish is a smooth, non-monotonic activation function that has demonstrated superior performance
    /// compared to ReLU and Swish in many deep learning architectures. 
    /// It is mathematically defined as: Mish(x) = x × tanh(softplus(x)), where softplus(x) = ln(1 + e^x).
    /// This implementation is fully compatible with the framework's tensor abstraction and automatic
    /// differentiation system.
    /// </remarks>
    public class Mish : BaseActivation
    {
        /// <summary>
        /// Computes the forward pass of the Mish activation.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor after applying the Mish activation.</returns>
        /// <remarks>
        /// The forward pass is implemented using only primitive tensor operations (Exp, Add, Log, 
        /// Multiply, Subtract, Divide) to ensure correct gradient tracking within the autograd system.
        /// If the input tensor requires gradients, a gradient function is registered with the output tensor.
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            // Create a tensor of 1s matching the input shape to avoid broadcasting errors
            var one = Tensor.Ones(input.Shape);

            // softplus(x) = log(1 + exp(x))
            var exp_x = input.Exp();
            var softplus = exp_x.Add(one).Log();

            // tanh(softplus) = (exp(2 * softplus) - 1) / (exp(2 * softplus) + 1)
            var exp_2_softplus = softplus.Multiply(2.0).Exp(); // Utilizes Multiply(double)
            var numerator = exp_2_softplus.Subtract(one);
            var denominator = exp_2_softplus.Add(one);
            var tanh_softplus = numerator.Divide(denominator);

            // Mish(x) = x * tanh(softplus(x))
            var result = input.Multiply(tanh_softplus);

            // Record the operation for Autograd
            if (input.RequiresGrad)
            {
                // Assign a Func<ITensor, ITensor> to satisfy the interface
                result.GradFn = new Func<ITensor, ITensor>(gradOutput => ComputeGrad(input, result, gradOutput));
            }

            return result;
        }

        /// <summary>
        /// Computes the gradient for the Mish activation during backpropagation using the Chain Rule.
        /// </summary>
        /// <param name="input">The original input tensor.</param>
        /// <param name="output">The output tensor from the forward pass.</param>
        /// <param name="gradOutput">The upstream gradient (from the next layer / loss function).</param>
        /// <returns>The final gradient tensor with respect to the input.</returns>
        /// <remarks>
        /// This method implements the analytical derivative of the Mish function:
        /// Mish'(x) = tanh(softplus(x)) + x × sech²(softplus(x)) × sigmoid(x)
        /// All intermediate values are recomputed from the original input to avoid storing
        /// the full computation graph. The parameter 'output' is provided for API consistency
        /// with other activations but is not used in this implementation.
        /// </remarks>
        private ITensor ComputeGrad(ITensor input, ITensor output, ITensor gradOutput)
        {
            // Create a tensor of 1s matching the input shape
            var one = Tensor.Ones(input.Shape);

            // 1. Recompute softplus(x)
            var exp_x = input.Exp();
            var softplus = exp_x.Add(one).Log();

            // 2. Recompute tanh(softplus(x))
            var exp_2_softplus = softplus.Multiply(2.0).Exp();
            var tanh_softplus = exp_2_softplus.Subtract(one).Divide(exp_2_softplus.Add(one));

            // 3. Compute sigmoid(x) = 1 / (1 + exp(-x))
            var exp_neg_x = input.Multiply(-1.0).Exp();
            var sigmoid_x = one.Divide(one.Add(exp_neg_x));

            // 4. Compute sech^2(softplus(x)) = 1 - tanh^2(softplus(x))
            var tanh_squared = tanh_softplus.Multiply(tanh_softplus);
            var sech_squared = one.Subtract(tanh_squared);

            // 5. Compute second term: x * sech^2(softplus(x)) * sigmoid(x)
            var second_term = input.Multiply(sech_squared).Multiply(sigmoid_x);

            // 6. Local Derivative: Mish'(x) = tanh(softplus(x)) + [x * sech^2(softplus(x)) * sigmoid(x)]
            var localGrad = tanh_softplus.Add(second_term);

            // 7. CHAIN RULE: Final Gradient = Local Gradient * Upstream Gradient
            var finalGrad = localGrad.Multiply(gradOutput);

            return finalGrad;
        }
    }
}