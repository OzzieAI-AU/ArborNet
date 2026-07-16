using System;
using System.Collections.Generic;
using System.Threading;
using ArborNet.Core.Interfaces;

namespace ArborNet.Core.Tensors
{
    /// <summary>
    /// Thread-local scope manager that tracks and deterministically disposes of intermediate 
    /// temporary tensors, preventing GC thrashing and memory pool exhaustion.
    /// </summary>
    public sealed class TensorScope : IDisposable
    {
        private static readonly ThreadLocal<Stack<TensorScope>> _activeScopes =
            new ThreadLocal<Stack<TensorScope>>(() => new Stack<TensorScope>());

        private readonly List<IDisposable> _trackedBackends = new();
        private bool _isDisposed;

        public TensorScope()
        {
            _activeScopes.Value!.Push(this);
        }

        /// <summary>
        /// Registers a tensor backend for automatic disposal at the end of the scope's lifetime.
        /// Only registers intermediate non-parameter tensors (RequiresGrad == false).
        /// </summary>
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