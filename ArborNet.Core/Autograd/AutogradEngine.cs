using System;
using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Core.Autograd
{
    /// <summary>
    /// Thread-safe mathematical backpropagation execution engine.
    /// Preserves parameter gradients to support gradient accumulation.
    /// </summary>
    public static class AutogradEngine
    {
        public static List<ITensor> TopologicalSort(ITensor root)
        {
            var sorted = new List<ITensor>();
            var visited = new HashSet<ITensor>();

            void Visit(ITensor node)
            {
                if (node == null) return;

                // Resolve wrappers to the core underlying backend node
                ITensor underlying = node;
                while (true)
                {
                    if (underlying is Tensor t) { underlying = t._backend; continue; }
                    if (underlying is Variable v) { underlying = v._inner; continue; }
                    break;
                }

                if (visited.Contains(underlying)) return;
                visited.Add(underlying);

                if (underlying.Inputs != null)
                {
                    foreach (var input in underlying.Inputs)
                    {
                        Visit(input);
                    }
                }

                sorted.Add(underlying);
            }

            Visit(root);
            sorted.Reverse(); // Outputs sorted from loss back to leaf inputs
            return sorted;
        }

        public static void Backward(ITensor root, ITensor? initialGradient = null)
        {
            if (root == null) throw new ArgumentNullException(nameof(root));

            var sortedNodes = TopologicalSort(root);

            // Initialize the root gradient
            ITensor underlyingRoot = root;
            while (true)
            {
                if (underlyingRoot is Tensor t) { underlyingRoot = t._backend; continue; }
                if (underlyingRoot is Variable v) { underlyingRoot = v._inner; continue; }
                break;
            }

            underlyingRoot.Grad = initialGradient ?? Tensor.Ones(root.Shape, root.Device);

            // Process each node in exact topological order
            foreach (var node in sortedNodes)
            {
                if (node.Grad == null || node.GradFn == null) continue;

                // Evaluates the local derivative and accumulates it to immediate inputs
                node.GradFn(node.Grad);
            }
        }
    }
}
