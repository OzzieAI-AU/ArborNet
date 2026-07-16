using System;
using System.Collections.Generic;
using System.Linq;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Core.Autograd
{
    /// <summary>
    /// Implements high-performance graph optimization techniques including constant folding
    /// on the computational graph.
    /// </summary>
    public static class GraphOptimizer
    {
        /// <summary>
        /// Optimizes the computational graph originating from the root tensor.
        /// Applies constant folding to pre-compute operations with static inputs.
        /// </summary>
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