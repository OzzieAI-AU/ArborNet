// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Activations
{

    #region Using Statements:

    using System;
    using ArborNet.Core;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Core.Autograd;
    /// <summary>
    /// Implements the Gated Linear Unit (GLU) activation function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Gated Linear Unit (GLU) was introduced in "Language Modeling with Gated Convolutional Networks"
    /// by Yann N. Dauphin, Angela Fan, Michael Auli, and David Grangier (2017).
    /// </para>
    /// <para>
    /// Mathematically, GLU splits the input tensor <c>x</c> along its last dimension into two equal halves,
    /// <c>x₁</c> and <c>x₂</c>, and computes:
    /// <br/>
    /// <c>GLU(x) = x₁ ⊙ σ(x₂)</c>
    /// <br/>
    /// where <c>x = [x₁, x₂]</c>, <c>⊙</c> denotes the element-wise multiplication (Hadamard product), 
    /// and <c>σ</c> represents the standard logistic sigmoid function.
    /// </para>
    /// <para>
    /// Because the input is halved along the last dimension, the size of the last dimension of the input 
    /// tensor must be even. The resulting output tensor will have the same shape as the input tensor, 
    /// except that the final dimension's size is reduced by half.
    /// </para>
    /// <para>
    /// This implementation is fully integrated with ArborNet's autograd engine. When the input tensor 
    /// has its <see cref="ITensor.RequiresGrad"/> property set to <see langword="true"/>, the operations 
    /// performed within this activation are tracked in the computation graph, enabling seamless backpropagation.
    /// </para>
    /// </remarks>
    /// <example>
    /// The following example demonstrates how to initialize and apply the GLU activation function:
    /// <code>
    /// // Create an input tensor with an even size on the last dimension (e.g., shape [2, 4])
    /// ITensor input = Tensor.Create(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 }, new long[] { 2, 4 });
    /// input.RequiresGrad = true;
    /// 
    /// // Instantiate the GLU activation
    /// var glu = new GLU();
    /// 
    /// // Compute the forward pass (resulting shape will be [2, 2])
    /// ITensor output = glu.Forward(input);
    /// </code>
    /// </example>
    /// <seealso cref="BaseActivation"/>
    /// <seealso cref="ITensor"/>

    #endregion

    public class GLU : BaseActivation
    {
        /// <summary>
        /// Computes the forward pass of the Gated Linear Unit (GLU) activation function.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> to apply the activation to. The last dimension must be of even length.</param>
        /// <returns>A new <see cref="ITensor"/> representing the activated output, with the size of its last dimension halved.</returns>
        /// <exception cref="ArgumentException">Thrown when the last dimension of the <paramref name="input"/> tensor is not even.</exception>
        /// <exception cref="NullReferenceException">Thrown if the <paramref name="input"/> tensor or its shape is null.</exception>
        /// <remarks>
        /// <para>
        /// This method performs a non-copying, logical split of the input tensor along its final dimension. 
        /// It extracts the first half (<c>a</c>) and the second half (<c>b</c>), applies an element-wise 
        /// sigmoid gate to the second half, and returns the element-wise product of both halves:
        /// <c>output = a * sigmoid(b)</c>.
        /// </para>
        /// <para>
        /// All intermediate tensor operations (including <c>Slice</c>, <c>Sigmoid.Forward</c>, and <c>Multiply</c>) 
        /// are intrinsically autograd-aware. If the input tensor requires gradients, the generated output tensor 
        /// will contain the appropriate gradient function linking back to the parent operations in the computation graph.
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