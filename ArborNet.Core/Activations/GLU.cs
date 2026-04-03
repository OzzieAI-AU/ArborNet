using System;
using ArborNet.Core;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Autograd;

namespace ArborNet.Activations
{
    /// <summary>
    /// Implements the GLU (Gated Linear Unit) activation function.
    /// Splits the input tensor along the last dimension into two halves, applies sigmoid to the second half,
    /// and multiplies the first half by it. Requires the last dimension of the input to be even.
    /// Inherits from BaseActivation and uses autograd for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Gated Linear Unit (GLU) was introduced in "Language Modeling with Gated Convolutional Networks"
    /// by Yann N. Dauphin, Angela Fan, Michael Auli, and David Grangier (2017).
    /// </para>
    /// <para>
    /// Mathematically: <c>GLU(x) = x₁ ⊙ σ(x₂)</c> where <c>x = [x₁, x₂]</c> is the split along the last dimension,
    /// <c>⊙</c> denotes element-wise multiplication, and <c>σ</c> is the sigmoid function.
    /// </para>
    /// <para>
    /// This implementation is fully integrated with the autograd system. When <see cref="ITensor.RequiresGrad"/> 
    /// is <see langword="true"/>, the computation graph is automatically tracked for backpropagation.
    /// </para>
    /// </remarks>
    public class GLU : BaseActivation
    {
        /// <summary>
        /// Computes the forward pass of the GLU activation.
        /// </summary>
        /// <param name="input">The input tensor, with last dimension even.</param>
        /// <returns>The output tensor after applying GLU.</returns>
        /// <exception cref="ArgumentException">Thrown when the last dimension of the input tensor is not even.</exception>
        /// <remarks>
        /// <para>
        /// The method performs a non-copying logical split of the input tensor along its final dimension,
        /// applies a sigmoid gate to the second half, and multiplies it with the first half.
        /// </para>
        /// <para>
        /// All tensor operations (<c>Slice</c>, <c>Sigmoid.Forward</c>, and <c>Multiply</c>) are autograd-aware.
        /// No manual gradient function registration is required.
        /// </para>
        /// <para>
        /// The output tensor has the same shape as the input tensor.
        /// </para>
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            if (input.Shape.Dimensions[input.Shape.Dimensions.Length - 1] % 2 != 0)
            {
                throw new ArgumentException("The last dimension of the input tensor must be even for GLU.");
            }

            int lastDimIndex = input.Shape.Dimensions.Length - 1;
            int halfSize = input.Shape.Dimensions[lastDimIndex] / 2;

            // Create slice specifications for the first half
            var slicesA = new (int start, int end, int step)[input.Shape.Dimensions.Length];
            for (int i = 0; i < lastDimIndex; i++)
            {
                slicesA[i] = (0, input.Shape.Dimensions[i], 1);
            }
            slicesA[lastDimIndex] = (0, halfSize, 1);
            ITensor a = input.Slice(slicesA);

            // Create slice specifications for the second half
            var slicesB = new (int start, int end, int step)[input.Shape.Dimensions.Length];
            for (int i = 0; i < lastDimIndex; i++)
            {
                slicesB[i] = (0, input.Shape.Dimensions[i], 1);
            }
            slicesB[lastDimIndex] = (halfSize, input.Shape.Dimensions[lastDimIndex], 1);
            ITensor b = input.Slice(slicesB);

            // Apply sigmoid to the second half
            ITensor gate = new Sigmoid().Forward(b);

            // Multiply the first half by the gate
            ITensor output = a.Multiply(gate);

            // Autograd integration: Since tensor operations like Slice, Sigmoid, and Multiply
            // are autograd-enabled (they set GradFn on the resulting tensor if input.RequiresGrad),
            // the output tensor will automatically have the correct gradient function.
            // No additional setup is needed here as the computation chain handles it.

            return output;
        }
    }
}