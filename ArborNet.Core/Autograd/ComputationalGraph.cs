using System;
using System.Collections.Generic;
using System.Linq;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Devices;

namespace ArborNet.Core.Autograd
{
    /// <summary>
    /// Manages a computational graph for automatic differentiation (autograd) operations.
    /// Tracks <see cref="ComputeNode"/> instances and orchestrates gradient propagation during the backward pass.
    /// </summary>
    /// <remarks>
    /// This class is thread-safe for modifications to the node collection. 
    /// The backward pass assumes that each <see cref="ComputeNode"/> knows its dependencies 
    /// and will recursively propagate gradients to its predecessors.
    /// </remarks>
    public class ComputationalGraph
    {
        /// <summary>
        /// The list of compute nodes in the order they were added.
        /// </summary>
        private readonly List<ComputeNode> _nodes = new();

        /// <summary>
        /// Synchronization primitive used to ensure thread-safe access to the nodes collection.
        /// </summary>
        private readonly object _lock = new();

        /// <summary>
        /// Adds a compute node to the computational graph.
        /// </summary>
        /// <param name="node">The compute node to add.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="node"/> is <see langword="null"/>.</exception>
        public void AddNode(ComputeNode node)
        {
            if (node == null) throw new ArgumentNullException(nameof(node));
            lock (_lock)
            {
                _nodes.Add(node);
            }
        }

        /// <summary>
        /// Performs the backward pass through the entire computational graph,
        /// propagating gradients from the output node back to all preceding nodes.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The method creates a copy of the current nodes, reverses their order,
        /// and initiates the backward pass on the final node with a tensor of ones.
        /// </para>
        /// <para>
        /// Each node's <see cref="ComputeNode.Backward(ITensor)"/> implementation is responsible 
        /// for accumulating gradients into its dependencies.
        /// </para>
        /// </remarks>
        public void Backward()
        {
            if (_nodes.Count == 0) return;

            List<ComputeNode> reversed;
            lock (_lock)
            {
                reversed = new List<ComputeNode>(_nodes);
            }
            reversed.Reverse();

            ITensor gradOutput = Tensor.Ones(new TensorShape(), Device.CPU);
            reversed[0].Backward(gradOutput);
        }

        /// <summary>
        /// Removes all nodes from the computational graph.
        /// </summary>
        /// <remarks>
        /// This operation is thread-safe.
        /// </remarks>
        public void Clear()
        {
            lock (_lock)
            {
                _nodes.Clear();
            }
        }

        /// <summary>
        /// Gets a read-only collection of all compute nodes currently in the graph.
        /// </summary>
        /// <value>
        /// A thread-safe snapshot of the nodes as a read-only list.
        /// </value>
        public IReadOnlyList<ComputeNode> Nodes
        {
            get { lock (_lock) return _nodes.AsReadOnly(); }
        }
    }
}