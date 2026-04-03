using System;
using System.Collections.Generic;
using System.Linq;
using ArborNet.Core.Interfaces;

namespace ArborNet.Core.Autograd
{
    /// <summary>
    /// Represents a node in the computational graph for automatic differentiation.
    /// </summary>
    /// <remarks>
    /// This class encapsulates an operation and its inputs/outputs within the computational graph.
    /// It is a core component of the autograd engine, enabling both forward computation and
    /// gradient propagation during the backward pass using the chain rule.
    /// </remarks>
    public class ComputeNode
    {
        /// <summary>
        /// The autograd operation associated with this node.
        /// </summary>
        private IAutogradOperation _operation;

        /// <summary>
        /// The input tensors to the operation.
        /// </summary>
        private ITensor[] _inputs;

        /// <summary>
        /// The output tensor produced by the forward pass.
        /// </summary>
        private ITensor _output;

        /// <summary>
        /// Initializes a new instance of the ComputeNode class.
        /// </summary>
        /// <param name="operation">The autograd operation.</param>
        /// <param name="inputs">The input tensors.</param>
        public ComputeNode(IAutogradOperation operation, params ITensor[] inputs)
        {
            _operation = operation;
            _inputs = inputs;
            _output = operation.Forward(inputs);
        }

        /// <summary>
        /// Gets the output tensor.
        /// </summary>
        public ITensor Output => _output;

        /// <summary>
        /// Performs the backward pass, propagating the gradient.
        /// </summary>
        /// <param name="gradOutput">The gradient with respect to the output.</param>
        public void Backward(ITensor gradOutput)
        {
            var grads = _operation.Backward(gradOutput);
            int i = 0;
            foreach (var grad in grads)
            {
                if (_inputs[i].RequiresGrad)
                {
                    if (_inputs[i].Grad == null)
                    {
                        _inputs[i].Grad = grad;
                    }
                    else
                    {
                        _inputs[i].Grad = _inputs[i].Grad.Add(grad);
                    }
                }
                i++;
            }
        }
    }
}