// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Tensors
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.Threading;
    using ArborNet.Core.Interfaces;
    /// <summary>
    /// Thread-local scope manager that tracks and deterministically disposes of intermediate 
    /// temporary tensors, preventing GC (Garbage Collector) thrashing and memory pool exhaustion.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This class implements the <see cref="IDisposable"/> interface to provide a structured, 
    /// deterministic cleanup mechanism for transient tensors within a lexical block (typically managed via a <c>using</c> statement).
    /// It maintains nested scopes using a thread-local stack structure to support hierarchical operations.
    /// </para>
    /// <para>
    /// Only intermediate tensors that do not require gradients (<c>RequiresGrad == false</c>) and implement 
    /// <see cref="IDisposable"/> are registered and tracked by the active scope.
    /// </para>
    /// <para>
    /// This class is thread-safe for multi-threaded tensor operations, as each thread maintains its own isolated 
    /// stack of active scopes via <see cref="ThreadLocal{T}"/>.
    /// </para>
    /// </remarks>
    /// <example>
    /// The following example demonstrates how to use the <see cref="TensorScope"/> to manage temporary tensors:
    /// <code>
    /// using (new TensorScope())
    /// {
    ///     var tempTensor = Tensor.CreateSomeIntermediate();
    ///     TensorScope.Register(tempTensor);
    ///     // Perform operations...
    /// } // tempTensor is automatically disposed here
    /// </code>
    /// </example>

    #endregion

    public sealed class TensorScope : IDisposable
    {
        /// <summary>
        /// A thread-local stack maintaining the active nested tensor scopes for the current thread of execution.
        /// </summary>
        private static readonly ThreadLocal<Stack<TensorScope>> _activeScopes =
            new ThreadLocal<Stack<TensorScope>>(() => new Stack<TensorScope>());

        /// <summary>
        /// Collection of disposable tensor backends tracked within the current scope instance.
        /// </summary>
        private readonly List<IDisposable> _trackedBackends = new();

        /// <summary>
        /// Tracks whether this scope instance has been disposed to ensure idempotent cleanup.
        /// </summary>
        private bool _isDisposed;

        /// <summary>
        /// Initializes a new instance of the <see cref="TensorScope"/> class and pushes it 
        /// onto the current thread's active scopes stack.
        /// </summary>
        public TensorScope()
        {
            _activeScopes.Value!.Push(this);
        }
        /// <summary>
        /// Registers a tensor backend for automatic disposal at the end of the scope's lifetime.
        /// Only registers intermediate non-parameter tensors (RequiresGrad == false).
        /// </summary>
        /// <param name="tensor">The tensor instance containing resources to be registered for disposal.</param>
        /// <remarks>
        /// <para>
        /// If the <paramref name="tensor"/> is <see langword="null"/>, has <see cref="ITensor.RequiresGrad"/> set to <see langword="true"/>,
        /// does not implement <see cref="IDisposable"/>, or if no active scope exists on the current thread,
        /// the registration request is silently ignored.
        /// </para>
        /// <para>
        /// This method targets the innermost active scope on the current thread's stack.
        /// </para>
        /// </remarks>

        public static void Register(ITensor tensor)
        {
            if (tensor == null || tensor.RequiresGrad) return;

            var scopes = _activeScopes.Value;
            if (scopes != null && scopes.Count > 0)
            {
                var activeScope = scopes.Peek();
                if (tensor is IDisposable disposable)
                {
                    activeScope._trackedBackends.Add(disposable);
                }
            }
        }
        /// <summary>
        /// Disposes all registered temporary tensor backends within this scope, clears the tracking list, 
        /// and pops the scope from the thread-local active scopes stack.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This method is idempotent. If called multiple times, subsequent calls will return immediately without side effects.
        /// </para>
        /// <para>
        /// Disposing a scope will release all tracked resources, helping to prevent memory leaks in tight loops.
        /// It also pops itself from the thread-local stack, restoring the previous nested scope (if any) as the active scope.
        /// </para>
        /// </remarks>

        public void Dispose()
        {
            if (_isDisposed) return;

            foreach (var item in _trackedBackends)
            {
                item.Dispose();
            }
            _trackedBackends.Clear();

            var scopes = _activeScopes.Value;
            if (scopes != null && scopes.Count > 0)
            {
                scopes.Pop();
            }

            _isDisposed = true;
        }
    }
}