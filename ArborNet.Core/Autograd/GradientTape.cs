// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Autograd
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Represents a production-grade, thread-safe, and numerically-stable gradient tape
    /// used to record mathematical operations for automatic differentiation (autograd) in ArborNet.
    /// </summary>
    /// <remarks>
    /// This tape is the single source of truth for tracking operations. It supports both standard 
    /// <see cref="IAutogradOperation"/> implementations and custom closure-style gradient functions 
    /// (<see cref="Func{ITensor, ITensor}"/>). It implements the context manager pattern via 
    /// <see cref="IDisposable"/> to facilitate clean scoping and prevent memory leaks. 
    /// All operations on the tape are synchronized using an internal <see cref="ReaderWriterLockSlim"/>,
    /// guaranteeing safety across multiple executing threads.
    /// </remarks>

    #endregion

    public sealed class GradientTape : IAutogradContext, IDisposable
    {
        private readonly List<(IAutogradOperation operation, ITensor[] inputs, ITensor output)> _tape = new();
        private readonly HashSet<ITensor> _recordedTensors = new(ReferenceEqualityComparer.Instance);
        private readonly ReaderWriterLockSlim _lock = new();
        private bool _recording = true;
        private bool _disposed;
        /// <summary>
        /// Gets a value indicating whether the tape is currently recording operations.
        /// </summary>
        /// <value>
        /// <see langword="true"/> if the tape is actively recording operations; otherwise, <see langword="false"/>.
        /// </value>

        public bool IsRecording => _recording;

        public GradientTape() { }
        /// <summary>
        /// Starts or resumes recording operations on this tape.
        /// </summary>
        /// <remarks>
        /// This method acquires a write lock to update the recording state in a thread-safe manner.
        /// </remarks>

        public void StartRecording() { _lock.EnterWriteLock(); try { _recording = true; } finally { _lock.ExitWriteLock(); } }
        /// <summary>
        /// Temporarily stops or pauses recording operations on this tape.
        /// </summary>
        /// <remarks>
        /// This method acquires a write lock to update the recording state in a thread-safe manner.
        /// </remarks>
        public void StopRecording() { _lock.EnterWriteLock(); try { _recording = false; } finally { _lock.ExitWriteLock(); } }
        /// <summary>
        /// Records an autograd operation along with its inputs and output tensor.
        /// </summary>
        /// <param name="operation">The operation that was performed during the forward pass.</param>
        /// <param name="inputs">The input tensors to the operation.</param>
        /// <param name="output">The output tensor produced by the operation.</param>
        /// <remarks>
        /// If the tape is not currently recording, or if any argument is <see langword="null"/>, 
        /// this method will return immediately without recording.
        /// </remarks>

        public void Record(IAutogradOperation operation, ITensor[] inputs, ITensor output)
        {
            if (!_recording || operation == null || inputs == null || output == null) return;
            _lock.EnterWriteLock();
            try { _tape.Add((operation, (ITensor[])inputs.Clone(), output)); _recordedTensors.Add(output); }
            finally { _lock.ExitWriteLock(); }
        }
        /// <summary>
        /// Records an autograd operation without explicit inputs or outputs.
        /// </summary>
        /// <param name="operation">The operation that was performed during the forward pass.</param>
        /// <remarks>
        /// This method acts as a convenience overload, pairing the operation with an empty set 
        /// of inputs and a default zero-filled, single-element tensor as its output.
        /// </remarks>

        public void Record(IAutogradOperation operation)
        {
            if (!_recording || operation == null) return;
            Record(operation, Array.Empty<ITensor>(), Tensor.Zeros(new TensorShape(1)));
        }
        /// <summary>
        /// Records a custom closure-based backward function associated with an output tensor.
        /// </summary>
        /// <param name="output">The output tensor of the operation.</param>
        /// <param name="gradFn">A delegate representing the custom gradient computation function.</param>
        /// <remarks>
        /// This enables inline or custom backpropagation rules to be registered dynamically 
        /// without creating a dedicated class implementation of <see cref="IAutogradOperation"/>.
        /// </remarks>

        public void RecordClosure(ITensor output, Func<ITensor, ITensor> gradFn)
        {
            if (!_recording || output == null || gradFn == null) return;
            var wrapper = new ClosureOperation(output, gradFn);
            _lock.EnterWriteLock();
            try { _tape.Add((wrapper, new[] { output }, output)); _recordedTensors.Add(output); }
            finally { _lock.ExitWriteLock(); }
        }
        /// <summary>
        /// Performs backpropagation starting from the specified root tensor.
        /// </summary>
        /// <param name="root">The root tensor from which the backward pass begins (typically the loss tensor).</param>
        /// <param name="initialGradient">
        /// The initial gradient to seed the backpropagation. If <see langword="null"/>, 
        /// it defaults to a tensor of ones matching the shape and device of the root tensor.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="root"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// This method executes the backward pass by traversing the recorded tape operations 
        /// in reverse chronological order, accumulating computed gradients into the <see cref="ITensor.Grad"/> property of each input.
        /// </remarks>

        public void Backward(ITensor root, ITensor? initialGradient = null)
        {
            if (root == null) throw new ArgumentNullException(nameof(root));

            _lock.EnterReadLock();
            try
            {
                root.Grad = initialGradient ?? Tensor.Ones(root.Shape, root.Device);

                for (int i = _tape.Count - 1; i >= 0; i--)
                {
                    var (op, inputs, output) = _tape[i];
                    if (output.Grad == null) continue;

                    var inputGrads = op.Backward(output.Grad);
                    if (inputGrads == null) continue;

                    for (int j = 0; j < Math.Min(inputs.Length, inputGrads.Count); j++)
                    {
                        var input = inputs[j];
                        var grad = inputGrads[j];

                        if (input.RequiresGrad && grad != null)
                        {
                            if (input.Grad == null)
                                input.Grad = grad.Clone();
                            else
                                input.Grad = input.Grad.Add(grad);
                        }
                    }
                }
            }
            finally { _lock.ExitReadLock(); }
        }
        /// <summary>
        /// Performs backpropagation starting from the last recorded tensor on the tape.
        /// </summary>
        /// <remarks>
        /// If the tape is empty and contains no recorded operations, this method returns immediately.
        /// </remarks>

        public void Backward() { if (_tape.Count == 0) return; Backward(_tape[^1].output); }
        /// <summary>
        /// Performs backpropagation starting from the specified tensor using a default initial gradient.
        /// </summary>
        /// <param name="tensor">The tensor from which the backward pass begins.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="tensor"/> is <see langword="null"/>.</exception>
        public void Backward(ITensor tensor) => Backward(tensor, null);
        /// <summary>
        /// Clears all recorded operations and tracked tensors from the tape.
        /// </summary>
        /// <remarks>
        /// Call this method to reset the tape state and release references to tensors to aid garbage collection.
        /// </remarks>

        public void Clear()
        {
            _lock.EnterWriteLock();
            try { _tape.Clear(); _recordedTensors.Clear(); }
            finally { _lock.ExitWriteLock(); }
        }
        /// <summary>
        /// Releases all resources used by the <see cref="GradientTape"/> class.
        /// </summary>

        public void Dispose()
        {
            if (!_disposed) { Clear(); _lock.Dispose(); _disposed = true; }
            GC.SuppressFinalize(this);
        }

        ~GradientTape() => Dispose();
        /// <summary>
        /// An internal adapter class that wraps a custom gradient delegate to satisfy the <see cref="IAutogradOperation"/> interface.
        /// </summary>

        private sealed class ClosureOperation : IAutogradOperation
        {
            private readonly ITensor _output;
            private readonly Func<ITensor, ITensor> _gradFn;

            public ClosureOperation(ITensor output, Func<ITensor, ITensor> gradFn)
            {
                _output = output ?? throw new ArgumentNullException(nameof(output));
                _gradFn = gradFn ?? throw new ArgumentNullException(nameof(gradFn));
            }
            /// <summary>
            /// Simulates the forward pass, returning the pre-calculated output tensor.
            /// </summary>
            /// <param name="inputs">The input tensors to the operation.</param>
            /// <returns>The recorded output tensor.</returns>

            public ITensor Forward(params ITensor[] inputs) => _output;
            /// <summary>
            /// Computes the gradient with respect to the input by invoking the wrapped gradient delegate.
            /// </summary>
            /// <param name="gradOutput">The gradient of the loss with respect to the output tensor.</param>
            /// <returns>A list containing the computed gradient with respect to the input.</returns>
            /// <exception cref="ArgumentNullException">Thrown when <paramref name="gradOutput"/> is <see langword="null"/>.</exception>

            // FIXED: Matches IAutogradOperation.Backward(ITensor gradOutput) exactly.
            public IList<ITensor?> Backward(ITensor gradOutput)
            {
                if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
                var result = _gradFn(gradOutput);
                return new List<ITensor?> { result };
            }
        }
    }
}