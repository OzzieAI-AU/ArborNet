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
    using ArborNet.Core.Devices;
    /// <summary>
    /// Manages a computational graph for automatic differentiation (autograd) operations.
    /// Tracks <see cref="ComputeNode"/> instances and orchestrates gradient propagation during the backward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This class is thread-safe for modifications to the node collection. 
    /// The backward pass assumes that each <see cref="ComputeNode"/> knows its dependencies 
    /// and will recursively propagate gradients to its predecessors.
    /// </para>
    /// <para>
    /// Nodes are processed in reverse order of their registration during the backward pass, which is a standard
    /// heuristic for reverse-mode automatic differentiation assuming a topological order has been maintained during construction.
    /// </para>
    /// </remarks>

    #endregion

    public class ComputationalGraph
    {
        /// <summary>
        /// The list of compute nodes in the order they were registered in the graph.
        /// </summary>
        /// <remarks>
        /// This chronological sequence represents the execution history of the forward pass.
        /// Reversing this collection provides a reverse topological ordering suitable for reverse-mode automatic differentiation.
        /// </remarks>
        private readonly List<ComputeNode> _nodes = new();

        /// <summary>
        /// Synchronization primitive used to ensure thread-safe access and mutation of the nodes collection.
        /// </summary>
        private readonly object _lock = new();
        /// <summary>
        /// Adds a compute node to the computational graph.
        /// </summary>
        /// <param name="node">The <see cref="ComputeNode"/> instance to add to the tracking list.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="node"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// This method is thread-safe and safely appends the node to the end of the evaluation sequence.
        /// </remarks>

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
        /// If the graph contains no nodes, this method returns immediately without performing any operations.
        /// </para>
        /// <para>
        /// The method creates a copy of the current nodes under a lock to ensure thread safety, reverses their order,
        /// and initiates the backward pass on the final node (which is assumed to be the scalar output or loss of the graph)
        /// with a tensor of ones acting as the initial incoming gradient.
        /// </para>
        /// <para>
        /// Each node's <see cref="ComputeNode.Backward(ITensor)"/> implementation is responsible 
        /// for computing and accumulating gradients into its dependencies.
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
        /// Removes all registered <see cref="ComputeNode"/> instances from the computational graph.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This operation is thread-safe. Clearing the graph does not mutate the internal state of the individual nodes themselves,
        /// but dissociates them from this tracking instance, allowing resources to be reclaimed.
        /// </para>
        /// </remarks>

        public void Clear()
        {
            lock (_lock)
            {
                _nodes.Clear();
            }
        }
        /// <summary>
        /// Gets a read-only collection of all compute nodes currently registered in the computational graph.
        /// </summary>
        /// <value>
        /// A thread-safe, read-only snapshot wrapper of the nodes collection as an <see cref="IReadOnlyList{ComputeNode}"/>.
        /// </value>
        /// <remarks>
        /// <para>
        /// While retrieving this property is thread-safe, the returned collection is a wrapper around the underlying live list.
        /// If another thread modifies the graph (e.g., via <see cref="AddNode"/> or <see cref="Clear"/>) while a caller is 
        /// enumerating this collection, an <see cref="InvalidOperationException"/> will be thrown. 
        /// </para>
        /// <para>
        /// Callers requiring concurrent iteration should copy the elements of this property under an external lock.
        /// </para>
        /// </remarks>

        public IReadOnlyList<ComputeNode> Nodes
        {
            get { lock (_lock) return _nodes.AsReadOnly(); }
        }
    }
}