using System;
using ArborNet.Core;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Activations
{
    /// <summary>
    /// Implements the ReLU (Rectified Linear Unit) activation function with full autograd support.
    /// ReLU(x) = max(0, x). Gradient is 1 for x > 0 and 0 otherwise.
    /// </summary>
    /// <remarks>
    /// ReLU is one of the most commonly used activation functions in deep neural networks.
    /// It introduces non-linearity while preserving the gradient for positive inputs,
    /// helping mitigate the vanishing gradient problem. This implementation fully supports
    /// the autograd system by recording a gradient function that applies the same
    /// positivity mask during backpropagation.
    /// </remarks>
    public class ReLU : BaseActivation
    {

        /// <summary>
        /// Computes the forward pass of the ReLU activation.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor after applying ReLU.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>
        /// <remarks>
        /// The implementation uses a boolean mask (input &gt; 0) to zero out negative values.
        /// When <see cref="ITensor.RequiresGrad"/> is true on the input, a gradient function
        /// is attached to the output that multiplies the incoming gradient by the same mask.
        /// This ensures correct gradient flow during the backward pass.
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            // Create mask: 1 where x > 0, 0 otherwise
            var zero = Tensor.Zeros(input.Shape, input.Device);
            var mask = input.GreaterThan(zero);        // This returns Tensor

            // ReLU = x * mask
            var output = input.Multiply(mask);

            // Backward: gradient * mask (subgradient of ReLU)
            if (input.RequiresGrad)
            {
                output.GradFn = grad =>
                {
                    // dy/dx = 1 if x > 0 else 0
                    var gradInput = grad.Multiply(mask);
                    if (input.Grad == null)
                    {
                        input.Grad = gradInput;
                    }
                    else
                    {
                        input.Grad = input.Grad.Add(gradInput);
                    }
                    return gradInput;
                };
            }

            return output;
        }
    }
}