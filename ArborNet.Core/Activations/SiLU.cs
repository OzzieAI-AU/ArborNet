using ArborNet.Core;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Activations
{
    /// <summary>
    /// Implements the SiLU (Sigmoid Linear Unit) activation function, also known as Swish.
    /// </summary>
    /// <remarks>
    /// The SiLU function is defined as <c>x * sigmoid(x)</c> and is a smooth, non-monotonic
    /// activation function that has been shown to outperform ReLU in many deep learning scenarios.
    /// </remarks>
    public class SiLU : BaseActivation
    {
        /// <summary>
        /// Performs the forward pass of the SiLU activation function.
        /// </summary>
        /// <param name="input">The input tensor to which the activation is applied.</param>
        /// <returns>A tensor containing the result of applying SiLU element-wise to the input.</returns>
        /// <remarks>
        /// Computes <c>output = input * sigmoid(input)</c>.
        /// When <see cref="ITensor.RequiresGrad"/> is <c>true</c>, a gradient function is attached
        /// that implements the derivative: <c>sigmoid(x) * (1 + x * (1 - sigmoid(x)))</c>.
        /// The current implementation recomputes the sigmoid in the gradient function for clarity;
        /// a production implementation should cache intermediate activations for efficiency.
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            var sigmoid = new Sigmoid().Forward(input);
            var output = input.Multiply(sigmoid);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var sig = new Sigmoid().Forward(input); // recompute or cache in real impl
                    var grad = sig.Add(input.Multiply(sig.Multiply(Tensor.Ones(input.Shape).Subtract(sig))));
                    return grad.Multiply(gradOutput);
                };
            }
            return output;
        }
    }
}