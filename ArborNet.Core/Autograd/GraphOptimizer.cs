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
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Provides high-performance graph optimization techniques for computational graphs,
    /// primarily focusing on static graph simplification such as constant folding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Optimization reduces execution time and memory footprint during model evaluation 
    /// by pre-computing static subgraphs. This is particularly effective for inference 
    /// pipelines or parts of the network that do not undergo gradient updates.
    /// </para>
    /// <para>
    /// This class is stateless and thread-safe, enabling concurrent optimization operations
    /// across different computational graphs.
    /// </para>
    /// </remarks>
    /// <threadsafety static="true" instance="true" />

    #endregion

    public static class GraphOptimizer
    {
        /// <summary>
        /// Optimizes the computational graph originating from the specified root tensor by applying constant folding.
        /// </summary>
        /// <param name="root">The root <see cref="ITensor"/> of the computational graph to optimize. This acts as the sink node from which topological traversal begins.</param>
        /// <returns>
        /// An optimized <see cref="ITensor"/> representing the root of the simplified graph. 
        /// Returns the original <paramref name="root"/> if no optimizations could be applied or if the root itself is not a candidate for folding.
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="root"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// <para>
        /// The optimization algorithm performs a topological sort of the graph starting from the root node.
        /// For each node in the sorted order, if neither the node itself nor any of its input dependencies 
        /// require gradients (<c>RequiresGrad == false</c>), the node is eagerly evaluated, and its operation 
        /// is replaced with a constant <see cref="Tensor"/> containing the materialized result.
        /// </para>
        /// <para>
        /// This method performs safe evaluation of static subgraphs without altering the behavior of 
        /// nodes that require gradient computation during backward passes. It utilizes 
        /// <see cref="ReferenceEqualityComparer"/> to prevent issues with value-based equality during node lookups.
        /// </para>
        /// </remarks>
        /// <example>
        /// The following example demonstrates how to apply graph optimization to a static tensor operation:
        /// <code>
        /// ITensor a = Tensor.Constant(new float[] { 1, 2 }, new long[] { 2 });
        /// ITensor b = Tensor.Constant(new float[] { 3, 4 }, new long[] { 2 });
        /// ITensor c = a.Add(b); // RequiresGrad is false
        /// 
        /// ITensor optimized = GraphOptimizer.Optimize(c);
        /// // 'optimized' is now a single folded constant tensor with values { 4, 6 }
        /// </code>
        /// </example>
        public static ITensor Optimize(ITensor root)
        {
            if (root == null) throw new ArgumentNullException(nameof(root));

            var sorted = AutogradEngine.TopologicalSort(root);
            var foldedTensors = new Dictionary<ITensor, ITensor>(ReferenceEqualityComparer.Instance);

            foreach (var node in sorted)
            {
                if (!node.RequiresGrad && node.Inputs != null && node.Inputs.Length > 0 && node.Inputs.All(i => !i.RequiresGrad))
                {
                    var foldedVal = Tensor.FromArray(node.ToArray(), node.Shape, node.Device);
                    foldedTensors[node] = foldedVal;
                }
            }

            if (foldedTensors.TryGetValue(root, out var optimizedRoot))
            {
                return optimizedRoot;
            }

            return root;
        }
    }
}