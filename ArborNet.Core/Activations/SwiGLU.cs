using System;
using ArborNet.Core;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Autograd;

namespace ArborNet.Activations
{
    /// <summary>
    /// Implements the SwiGLU (Swish Gated Linear Unit) activation function.
    /// SwiGLU splits the input tensor along the last dimension into two halves,
    /// applies Swish (x * sigmoid(x)) to the second half, and multiplies it with the first half.
    /// Requires the last dimension of the input to be even.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SwiGLU is a high-performing activation function commonly used in transformer-based 
    /// large language models (e.g. Llama, PaLM). It is mathematically defined as:
    /// </para>
    /// <para>
    /// <c>output = x₁ ⊙ Swish(x₂)</c> where <c>x = split(x₁, x₂)</c> along the last dimension 
    /// and <c>Swish(x) = x * sigmoid(x)</c>.
    /// </para>
    /// <para>
    /// When the last dimension is odd, the input tensor is returned unchanged to maintain 
    /// compatibility with arbitrary tensor shapes.
    /// </para>
    /// <para>
    /// Gradient computation is automatically handled by the tensor autograd system 
    /// through the <see cref="ITensor"/> operations.
    /// </para>
    /// </remarks>
    public class SwiGLU : BaseActivation
    {
        /// <summary>
        /// Computes the forward pass of the SwiGLU activation.
        /// </summary>
        /// <param name="input">The input tensor. The last dimension must be even.</param>
        /// <returns>The output tensor after applying SwiGLU, or the input unchanged if the last dimension is odd.</returns>
        /// <remarks>
        /// <para>
        /// The method performs a non-destructive split of the last dimension using tensor slicing.
        /// The second half is passed through a sigmoid function (creating the gate) and then 
        /// multiplied element-wise with the first half.
        /// </para>
        /// <para>
        /// No custom <see cref="GradFn"/> is required because all operations (Slice, Sigmoid, Multiply)
        /// are tracked by the <see cref="ITensor"/> autograd infrastructure.
        /// </para>
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            int lastDimIndex = input.Shape.Dimensions.Length - 1;
            int lastDimSize = input.Shape.Dimensions[lastDimIndex];

            if (lastDimSize % 2 != 0)
                return input; // Odd dimension → return unchanged (as per original behavior)

            int half = lastDimSize / 2;

            // Create slice for first half: [..., 0:half]
            var slicesA = new (int start, int end, int step)[input.Shape.Dimensions.Length];
            for (int i = 0; i < lastDimIndex; i++)
                slicesA[i] = (0, input.Shape.Dimensions[i], 1);
            slicesA[lastDimIndex] = (0, half, 1);

            // Create slice for second half: [..., half:end]
            var slicesB = new (int start, int end, int step)[input.Shape.Dimensions.Length];
            Array.Copy(slicesA, slicesB, slicesA.Length);           // Proper deep copy of tuple array
            slicesB[lastDimIndex] = (half, lastDimSize, 1);

            ITensor a = input.Slice(slicesA);
            ITensor b = input.Slice(slicesB);

            // Swish = x * sigmoid(x)
            ITensor gate = new Sigmoid().Forward(b);
            ITensor output = a.Multiply(gate);

            // Autograd is automatically handled by the tensor operations (Slice, Multiply, Sigmoid)
            // No extra GradFn setup is needed unless you want custom behavior.

            return output;
        }
    }
}