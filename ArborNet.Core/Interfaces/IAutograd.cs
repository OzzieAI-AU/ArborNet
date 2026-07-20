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

    using System;
    using System.Collections.Generic;

    #endregion

    /// <summary>
    /// Defines the contract for mathematical operations that support automatic differentiation (autograd).
    /// Implementations of this interface encapsulate both the forward computation logic and the backward
    /// gradient propagation rules.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Classes implementing this interface must be stateless, thread-safe, and device-aware (e.g., CPU, GPU)
    /// to support concurrent operations and heterogeneous execution environments.
    /// </para>
    /// <para>
    /// During the forward pass, the operation computes the output tensor. During the backward pass,
    /// it calculates the Vector-Jacobian Product (VJP) using the incoming gradient.
    /// </para>
    /// </remarks>
    public interface IAutogradOperation
    {
        /// <summary>
        /// Performs the forward pass computation of the operation.
        /// </summary>
        /// <param name="inputs">An array of input tensors on which the operation is performed.</param>
        /// <returns>A new <see cref="ITensor"/> containing the computed output of the operation.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="inputs"/> is null or contains null elements.</exception>
        /// <exception cref="ArgumentException">Thrown when the number of inputs or their dimensions are invalid for this operation.</exception>
        ITensor Forward(params ITensor[] inputs);
        /// <summary>
        /// Performs the backward pass computation, propagating the gradient from the output tensor back to the input tensors.
        /// </summary>
        /// <param name="gradOutput">The gradient of the loss function with respect to the output of the forward pass.</param>
        /// <returns>
        /// An <see cref="IList{T}"/> containing the computed gradients with respect to each input tensor,
        /// ordered identically to the inputs passed to the forward pass. An element may be <see langword="null"/>
        /// if the corresponding input does not require gradients or is non-differentiable.
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="gradOutput"/> is null.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the backward pass is executed before a forward pass, or if the internal state is invalid.</exception>

        IList<ITensor?> Backward(ITensor gradOutput);
    }

    /// <summary>
    /// Defines the contract for an autograd context (e.g., a gradient tape) that manages operation recording,
    /// dynamic computation graph construction, and backpropagation.
    /// </summary>
    /// <remarks>
    /// The autograd context tracks operations on tensors to construct a directed acyclic graph (DAG) of computations,
    /// which is subsequently traversed in reverse topological order during the backward pass to calculate gradients.
    /// </remarks>
    public interface IAutogradContext
    {
        /// <summary>
        /// Gets a value indicating whether the context is currently active and recording operations.
        /// </summary>
        /// <value>
        /// <see langword="true"/> if operations are being tracked and appended to the computation graph; otherwise, <see langword="false"/>.
        /// </value>
        bool IsRecording { get; }
        /// <summary>
        /// Starts or resumes recording operations. Tensor operations executed while recording is enabled
        /// will be added to the dynamic computation graph.
        /// </summary>
        /// <remarks>
        /// Calling this method when <see cref="IsRecording"/> is already <see langword="true"/> has no effect.
        /// </remarks>

        void StartRecording();
        /// <summary>
        /// Temporarily stops or pauses recording operations. Operations executed while recording is disabled
        /// will not be tracked, which is useful for inference, validation, or non-differentiable operations.
        /// </summary>
        /// <remarks>
        /// Calling this method when <see cref="IsRecording"/> is already <see langword="false"/> has no effect.
        /// </remarks>

        void StopRecording();
        /// <summary>
        /// Records an executed autograd operation, associating its inputs and resulting output within the computation graph.
        /// </summary>
        /// <param name="operation">The executed <see cref="IAutogradOperation"/> to register.</param>
        /// <param name="inputs">The input tensors consumed by the operation.</param>
        /// <param name="output">The output tensor produced by the operation.</param>
        /// <remarks>
        /// This method is typically invoked internally by tensor operations when <see cref="IsRecording"/> is <see langword="true"/>
        /// to build the dynamic execution tape.
        /// </remarks>
        /// <exception cref="ArgumentNullException">
        /// Thrown when <paramref name="operation"/>, <paramref name="inputs"/>, or <paramref name="output"/> is null.
        /// </exception>
        /// <exception cref="InvalidOperationException">Thrown if called while <see cref="IsRecording"/> is <see langword="false"/>.</exception>

        void Record(IAutogradOperation operation, ITensor[] inputs, ITensor output);
        /// <summary>
        /// Registers a custom gradient function (closure) to define backward propagation logic for a specific output tensor.
        /// </summary>
        /// <param name="output">The output tensor for which the custom backward step is defined.</param>
        /// <param name="gradFn">
        /// A delegate representing the custom gradient function. It receives the gradient with respect to
        /// the output and must return the corresponding gradient for the input.
        /// </param>
        /// <remarks>
        /// This provides a lightweight alternative to implementing <see cref="IAutogradOperation"/> for custom or one-off
        /// mathematical operations, allowing inline definition of the backward pass.
        /// </remarks>
        /// <exception cref="ArgumentNullException">
        /// Thrown when <paramref name="output"/> or <paramref name="gradFn"/> is null.
        /// </exception>

        void RecordClosure(ITensor output, Func<ITensor, ITensor> gradFn);
        /// <summary>
        /// Initiates the backpropagation process starting from the specified root tensor.
        /// </summary>
        /// <param name="root">The terminal (loss) tensor from which backpropagation begins.</param>
        /// <param name="initialGradient">
        /// The seed gradient for the root tensor. Typically a scalar value of 1.0 for loss tensors.
        /// If <see langword="null"/>, a default tensor of ones with the same shape as <paramref name="root"/> is used.
        /// </param>
        /// <remarks>
        /// This method performs a reverse topological sort on the recorded computation graph starting from the <paramref name="root"/>
        /// and sequentially executes the backward passes of all recorded operations to accumulate gradients.
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="root"/> is null.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the computation graph cannot be backpropagated (e.g., cyclic dependencies or missing tracking data).</exception>

        void Backward(ITensor root, ITensor? initialGradient = null);
        /// <summary>
        /// Initiates the backpropagation process using the context's default root or the most recently recorded terminal tensor.
        /// </summary>
        /// <remarks>
        /// This is a convenience method that automatically identifies the terminal node (usually the scalar loss tensor)
        /// in the active computation graph and begins backpropagation with an implicit seed gradient of 1.0.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the context contains no recorded operations or if the default root cannot be determined.</exception>

        void Backward();
        /// <summary>
        /// Initiates the backpropagation process to calculate gradients specifically with respect to the target tensor.
        /// </summary>
        /// <param name="tensor">The target tensor to propagate gradients back from.</param>
        /// <remarks>
        /// This overload allows initiating backward propagation from a specific intermediate tensor
        /// rather than the root/loss tensor, enabling partial Jacobian computations or targeted gradient evaluation.
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="tensor"/> is null.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the tensor is not registered or tracked in the current context.</exception>

        void Backward(ITensor tensor);
        /// <summary>
        /// Clears all recorded operations, custom closures, and intermediate computation graph states, resetting the context to its initial state.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This method should be called to free up memory associated with the dynamic computation graph after a backward pass completes,
        /// preventing memory leaks from retained intermediate tensors.
        /// </para>
        /// <para>
        /// Clearing the context invalidates any subsequent calls to <see cref="Backward()"/> until new operations are recorded.
        /// </para>
        /// </remarks>

        void Clear();
    }

    /// <summary>
    /// Defines the contract for an isolated gradient computation function associated with specific operations.
    /// </summary>
    public interface IGradientFunction
    {
        /// <summary>
        /// Computes the gradients of the operation with respect to its inputs, given the incoming gradient of the output.
        /// </summary>
        /// <param name="gradOutput">The gradient of the loss function with respect to the output of the associated operation.</param>
        /// <param name="inputs">The original input tensors passed to the forward pass of the operation.</param>
        /// <returns>
        /// An <see cref="IList{T}"/> containing computed gradients corresponding to each element in <paramref name="inputs"/>,
        /// or <see langword="null"/> for input elements that are not differentiable or do not require gradients.
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="gradOutput"/> or <paramref name="inputs"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the dimensions of <paramref name="gradOutput"/> or <paramref name="inputs"/> are incompatible.</exception>
        IList<ITensor?> ComputeGrad(ITensor gradOutput, params ITensor[] inputs);
    }
}