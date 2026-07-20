// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Activations
{

    #region Using Statements:

    using ArborNet.Activations;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    using System.Text;
    /// <summary>
    /// Represents a production-grade Hard Hyperbolic Tangent (HardTanh) activation function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The HardTanh activation function restricts input values to the range [-1, 1] using a piecewise linear approximation of the standard Tanh function.
    /// Mathematically, it is defined as:
    /// </para>
    /// <para>
    /// <c>HardTanh(x) = max(min_val, min(max_val, x))</c> where <c>min_val = -1</c> and <c>max_val = 1</c>.
    /// </para>
    /// <para>
    /// Compared to the standard hyperbolic tangent, HardTanh is computationally cheaper because it avoids transcendental function 
    /// evaluations (exponential calculations), replacing them with simple thresholding and clipping operations. This makes it highly 
    /// suitable for performance-critical deep learning models running on resource-constrained hardware or in high-throughput pipelines.
    /// </para>
    /// <para>
    /// During the backward pass (if automatic differentiation is enabled), the gradient is propagated only for inputs strictly within 
    /// the active range. Specifically, the derivative is:
    /// </para>
    /// <para>
    /// <c>d/dx (HardTanh(x)) = 1</c> if <c>-1 &lt; x &lt;= 1</c>, otherwise <c>0</c>.
    /// </para>
    /// <para>
    /// This implementation is device-agnostic, deferring execution to the underlying <see cref="ITensor"/> which handles CPU or CUDA dispatch seamlessly.
    /// </para>
    /// </remarks>
    /// <threadsafety static="true" instance="true"/>

    #endregion

    public class HardTanh : BaseActivation
    {
        /// <summary>
        /// Computes the forward pass of the HardTanh activation function on the specified input tensor.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> containing the source values to be activated. Must not be <see langword="null"/>.</param>
        /// <returns>
        /// A new <see cref="ITensor"/> containing the activated (clipped) elements within the range [-1, 1].
        /// The returned tensor has the same shape, data type, and hardware device assignment as the <paramref name="input"/> tensor.
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown when the <paramref name="input"/> tensor is <see langword="null"/>.</exception>
        /// <remarks>
        /// <para>
        /// This method performs an element-wise clipping operation. If the input tensor has its <see cref="ITensor.RequiresGrad"/> 
        /// property set to <see langword="true"/>, this method attaches a backward gradient computation function (<c>GradFn</c>) 
        /// to the output tensor to facilitate automatic differentiation.
        /// </para>
        /// <para>
        /// The gradient mask is calculated as the element-wise logical AND of two conditions:
        /// <list type="bullet">
        /// <item>
        /// <description>Input values strictly greater than -1.0f.</description>
        /// </item>
        /// <item>
        /// <description>Input values less than or equal to 1.0f.</description>
        /// </item>
        /// </list>
        /// The incoming gradient (<c>gradOutput</c>) is then multiplied by this mask to produce the accumulated gradient.
        /// </para>
        /// <para>
        /// <b>Performance Note:</b> All tensor operations (clipping, comparison, and element-wise multiplication) are executed 
        /// on the tensor's native device (e.g., host CPU memory or CUDA GPU VRAM) to prevent costly host-device synchronization barriers.
        /// </para>
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var output = input.Clip(-1f, 1f);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var mask = input.GreaterThan(Tensor.FromScalar(-1f, input.Device))
                                   .Multiply(input.LessEqual(Tensor.FromScalar(1f, input.Device)));
                    return gradOutput.Multiply(mask);
                };
            }

            return output;
        }
    }
}