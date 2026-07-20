// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Interfaces
{

    #region Using Statements:

    using ArborNet.Core.Devices;
    using ArborNet.Core.Tensors;

    #endregion

    /// <summary>
    /// Defines the contract for all activation functions within the ArborNet deep learning framework.
    /// Activation functions introduce non-linearity into neural network layers, enabling the model to learn complex patterns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// All implementations of this interface must be stateless, thread-safe, and capable of executing 
    /// on both CPU and accelerator hardware (such as GPUs) through the framework's device abstraction.
    /// </para>
    /// <para>
    /// For seamless integration with the automatic differentiation (autograd) engine, the forward pass 
    /// implementation must construct the backward execution graph when the input tensor requires gradients.
    /// </para>
    /// </remarks>
    /// <example>
    /// <para>
    /// The following example demonstrates a custom implementation of a Rectified Linear Unit (ReLU) activation:
    /// </para>
    /// <code language="csharp">
    /// public class ReluActivation : IActivation
    /// {
    ///     public ITensor Forward(ITensor input)
    ///     {
    ///         if (input == null)
    ///         {
    ///             throw new ArgumentNullException(nameof(input));
    ///         }
    ///         
    ///         // Performs element-wise maximum operation: max(0, x)
    ///         return input.Maximum(0.0f);
    ///     }
    /// }
    /// </code>
    /// </example>
    /// <seealso cref="ITensor"/>
    /// <seealso cref="IDevice"/>
    public interface IActivation
    {
        /// <summary>
        /// Computes the forward pass of the activation function applying the non-linear transformation element-wise.
        /// </summary>
        /// <param name="input">The input tensor containing pre-activation values. Must not be null and must reside on a valid device.</param>
        /// <returns>A new <see cref="ITensor"/> containing the activated values, maintaining the same shape, data type, and device placement as the input.</returns>
        /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="input"/> is <see langword="null"/>.</exception>
        /// <exception cref="System.ArgumentException">Thrown when the input tensor has an invalid shape, unsupported data type, or incompatible device configuration.</exception>
        /// <exception cref="System.InvalidOperationException">Thrown if the underlying hardware device execution fails during the computation.</exception>
        /// <remarks>
        /// Implementations must ensure that if <paramref name="input"/> has gradient tracking enabled 
        /// (e.g., <c>input.RequiresGrad == true</c>), the resulting output tensor is properly registered 
        /// with the autograd engine, linking it to the corresponding backward derivative operation.
        /// </remarks>
        /// <example>
        /// <para>
        /// The following example shows how to call the <see cref="Forward"/> method using an activation instance:
        /// </para>
        /// <code language="csharp">
        /// IActivation activation = new SigmoidActivation();
        /// ITensor inputTensor = DeviceContext.CreateTensor(new float[] { -1.0f, 0.0f, 1.0f });
        /// ITensor activatedTensor = activation.Forward(inputTensor);
        /// </code>
        /// </example>
        ITensor Forward(ITensor input);
    }
}