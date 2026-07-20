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
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Provides a thread-safe execution engine for mathematical backpropagation (reverse-mode automatic differentiation).
    /// This engine is responsible for performing topological sorting on the computation graph and executing
    /// the backward pass, while preserving parameter gradients to support gradient accumulation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The <see cref="AutogradEngine"/> is a core component of the neural network framework, enabling automated
    /// gradient computation for arbitrary computational graphs. It operates by traversing the graph of execution
    /// nodes, sorting them topologically, and sequentially executing their respective gradient functions.
    /// </para>
    /// <para>
    /// This class is stateless and thread-safe, assuming that the underlying tensor operations and node
    /// structures themselves are handled in a thread-safe manner or are isolated per thread of execution.
    /// </para>
    /// </remarks>

    #endregion

    public static class AutogradEngine
    {
        /// <summary>
        /// Performs a topological sort on the computation graph starting from the specified root tensor.
        /// </summary>
        /// <param name="root">The root tensor (typically representing the final loss) from which the traversal begins.</param>
        /// <returns>
        /// A <see cref="List{ITensor}"/> containing the computation graph nodes sorted in reverse topological order
        /// (from the root/loss down to the leaf inputs). If the <paramref name="root"/> is <see langword="null"/>,
        /// an empty list is returned.
        /// </returns>
        /// <remarks>
        /// <para>
        /// This method automatically resolves wrapped tensors (such as <see cref="Tensor"/> wrappers or <see cref="Variable"/> references)
        /// to their underlying backend implementations to prevent duplicate visits and ensure the integrity of the graph traversal.
        /// </para>
        /// <para>
        /// The sorting is performed using a depth-first search (DFS) post-order traversal, which is subsequently reversed
        /// to provide an execution path suitable for backpropagation. This ensures that a node's gradients are fully accumulated
        /// from all its consumers before its own gradient function (<see cref="ITensor.GradFn"/>) is invoked.
        /// </para>
        /// </remarks>
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
        /// <summary>
        /// Executes the backward pass of the autograd system, computing gradients of all reachable nodes in the
        /// computation graph with respect to the specified root tensor.
        /// </summary>
        /// <param name="root">The root tensor (usually a scalar representing the final loss value) to start backpropagation from.</param>
        /// <param name="initialGradient">
        /// An optional initial upstream gradient to seed the backpropagation process.
        /// If <see langword="null"/>, a tensor of ones matching the shape and device of <paramref name="root"/> is automatically instantiated.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="root"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// <para>
        /// This method performs reverse-mode automatic differentiation by first constructing a topologically sorted execution path
        /// from the specified root node down to the leaf nodes.
        /// </para>
        /// <para>
        /// It then traverses the sorted nodes in order, invoking each node's local gradient evaluation function
        /// (<see cref="ITensor.GradFn"/>) to compute and accumulate derivatives down to the input parameters.
        /// </para>
        /// <para>
        /// The root node's gradient is seeded using the provided <paramref name="initialGradient"/>, or defaults to a tensor of ones.
        /// This matches standard backpropagation behavior where <c>dLoss/dLoss = 1</c>.
        /// </para>
        /// </remarks>

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