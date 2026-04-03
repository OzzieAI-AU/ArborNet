using System;
using System.Collections.Generic;

namespace ArborNet.Core.Interfaces
{
    /// <summary>
    /// Interface for autograd operations that support forward computation and backward gradient propagation.
    /// Implementations define how to compute outputs from inputs and how to propagate gradients back to inputs.
    /// Must be thread-safe and device-aware.
    /// </summary>
    public interface IAutogradOperation
    {
        /// <summary>
        /// Performs the forward pass of the operation.
        /// </summary>
        /// <param name="inputs">The input tensors to the operation.</param>
        /// <returns>The output tensor resulting from the operation.</returns>
        ITensor Forward(params ITensor[] inputs);

        /// <summary>
        /// Computes the backward pass, propagating the gradient from the output back to the inputs.
        /// Returns gradients in the same order as inputs.
        /// </summary>
        /// <param name="gradOutput">The gradient tensor with respect to the output.</param>
        /// <returns>List of gradient tensors, one for each input (may be null if no gradient).</returns>
        IList<ITensor?> Backward(ITensor gradOutput);
    }

    /// <summary>
    /// Interface for autograd contexts, such as gradient tapes, that manage the recording and execution of operations.
    /// Provides mechanisms to track computations and compute gradients via backpropagation.
    /// </summary>
    public interface IAutogradContext
    {
        /// <summary>
        /// Gets a value indicating whether the autograd context is currently recording operations
        /// to build the computation graph for automatic differentiation.
        /// </summary>
        bool IsRecording { get; }

        /// <summary>
        /// Begins recording operations to construct the computation graph used during backpropagation.
        /// </summary>
        void StartRecording();

        /// <summary>
        /// Ends the recording of operations. Subsequent tensor operations will not be tracked.
        /// </summary>
        void StopRecording();

        /// <summary>
        /// Records an autograd operation, its inputs, and produced output for later gradient computation.
        /// </summary>
        /// <param name="operation">The autograd operation that was executed.</param>
        /// <param name="inputs">The input tensors that were passed to the operation.</param>
        /// <param name="output">The output tensor produced by the forward pass.</param>
        void Record(IAutogradOperation operation, ITensor[] inputs, ITensor output);

        /// <summary>
        /// Records a custom gradient function (closure) for an output tensor.
        /// This allows attaching arbitrary backward logic to tensors that do not correspond
        /// to a standard <see cref="IAutogradOperation"/>.
        /// </summary>
        /// <param name="output">The output tensor to attach the gradient function to.</param>
        /// <param name="gradFn">Function that receives the incoming gradient and returns the gradient for the associated input.</param>
        void RecordClosure(ITensor output, Func<ITensor, ITensor> gradFn);

        /// <summary>
        /// Executes backpropagation starting from the specified root tensor.
        /// </summary>
        /// <param name="root">The root tensor from which to begin the backward pass.</param>
        /// <param name="initialGradient">
        /// The initial gradient with respect to the root tensor.
        /// If <see langword="null"/>, a tensor of ones with the same shape as the root is used.
        /// </param>
        void Backward(ITensor root, ITensor? initialGradient = null);

        /// <summary>
        /// Executes backpropagation using the context's default root or the most recently recorded tensor.
        /// </summary>
        void Backward();

        /// <summary>
        /// Executes backpropagation for the specified tensor.
        /// </summary>
        /// <param name="tensor">The tensor with respect to which gradients should be computed.</param>
        void Backward(ITensor tensor);

        /// <summary>
        /// Clears all recorded operations, closures, and intermediate data from the context.
        /// </summary>
        void Clear();
    }

    /// <summary>
    /// Interface for gradient functions that define how to compute gradients for specific operations.
    /// </summary>
    public interface IGradientFunction
    {
        /// <summary>
        /// Computes the gradients with respect to the operation inputs given the gradient with respect to its output.
        /// </summary>
        /// <param name="gradOutput">The gradient tensor with respect to the operation's output.</param>
        /// <param name="inputs">The original input tensors that were passed to the operation.</param>
        /// <returns>
        /// A list containing the gradient for each input (in the same order), or <see langword="null"/>
        /// for inputs that do not require a gradient.
        /// </returns>
        IList<ITensor?> ComputeGrad(ITensor gradOutput, params ITensor[] inputs);
    }
}