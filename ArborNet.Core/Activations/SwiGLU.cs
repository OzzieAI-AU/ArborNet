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
    /// Represents the SwiGLU (Swish Gated Linear Unit) activation function layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SwiGLU is a high-performing gated activation function variant widely utilized in state-of-the-art 
    /// Large Language Models (LLMs) such as LLaMA and PaLM. It leverages a gating mechanism to control 
    /// the propagation of representation features through neural network pathways.
    /// </para>
    /// <para>
    /// Mathematically, a standard SwiGLU operation is defined as:
    /// <c>SwiGLU(x) = (x * W + b) ⊙ Swish(x * V + c)</c>
    /// where <c>⊙</c> represents the element-wise (Hadamard) product.
    /// </para>
    /// <para>
    /// In this implementation, the input tensor is divided along its last dimension into two equal-sized 
    /// sub-tensors: <c>x₁</c> and <c>x₂</c>. The activation is computed as:
    /// <c>output = x₁ ⊙ Swish(x₂)</c> where <c>Swish(x) = x * Sigmoid(x)</c>.
    /// </para>
    /// <para>
    /// Note: To partition the input tensor successfully, the size of its last dimension must be an even number. 
    /// If the last dimension is odd, the operation acts as a fallback bypass, returning the input tensor unchanged 
    /// to avoid dimension mismatch runtime errors and maintain compatibility.
    /// </para>
    /// <para>
    /// Gradient computation and backward propagation are managed dynamically by the autograd subsystem through the 
    /// tracked execution graph of the constituent operations (<c>Slice</c>, <c>Sigmoid</c>, and <c>Multiply</c>).
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// ITensor input = TensorFactory.CreateRandom(new long[] { 2, 8 }); // Last dimension is even (8)
    /// SwiGLU activation = new SwiGLU();
    /// ITensor output = activation.Forward(input);
    /// // The output shape is { 2, 4 } because the last dimension is split in half.
    /// </code>
    /// </example>
    /// <seealso cref="ArborNet.Activations.BaseActivation" />
    /// <seealso cref="ArborNet.Core.Interfaces.ITensor" />

    #endregion

    public class SwiGLU : BaseActivation
    {
        /// <summary>
        /// Computes the forward pass of the SwiGLU activation function.
        /// </summary>
        /// <param name="input">The input multidimensional tensor (<see cref="ITensor"/>) to activate. 
        /// The size of its last dimension must be an even number to execute the split gating logic.</param>
        /// <returns>
        /// An <see cref="ITensor"/> representing the gated activation with the last dimension size halved, 
        /// or the original <paramref name="input"/> tensor if the last dimension is odd.
        /// </returns>
        /// <exception cref="NullReferenceException">Thrown when the <paramref name="input"/> tensor is <see langword="null"/>.</exception>
        /// <remarks>
        /// <para>
        /// The forward execution pipeline operates as follows:
        /// <list type="number">
        /// <item>
        /// <description>Determines the index and length of the final dimension of the input tensor.</description>
        /// </item>
        /// <item>
        /// <description>If the size of the last dimension is odd, bypasses processing and returns the input tensor unmodified.</description>
        /// </item>
        /// <item>
        /// <description>Calculates the split midpoint of the final dimension.</description>
        /// </item>
        /// <item>
        /// <description>Constructs multi-dimensional slice indices for the first half <c>a</c> and the second half <c>b</c>.</description>
        /// </item>
        /// <item>
        /// <description>Slices the input tensor along the final dimension to obtain the linear path <c>a</c> and gate path <c>b</c>.</description>
        /// </item>
        /// <item>
        /// <description>Applies a Sigmoid function to the gate pathway <c>b</c>.</description>
        /// </item>
        /// <item>
        /// <description>Multiplies <c>a</c> element-wise by the active gate tensor to produce the final output.</description>
        /// </item>
        /// </list>
        /// </para>
        /// <para>
        /// No manual backward pass or derivative registration (<c>GradFn</c>) is required, as the internal operations 
        /// are automatically recorded to the dynamic execution tape of the autograd framework.
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