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
    /// <summary>
    /// Represents a node within the execution/computational graph constructed during the forward pass 
    /// of automatic differentiation (autograd).
    /// </summary>
    /// <remarks>
    /// This class serves as the binding element between an operation (<see cref="IAutogradOperation"/>), 
    /// its input arguments (<see cref="ITensor"/>), and the resulting output tensor. It plays a critical role 
    /// in reverse-mode automatic differentiation by tracking dependencies and providing the mechanism (<see cref="Backward"/>) 
    /// to propagate gradients backward from the output back to the inputs using the chain rule.
    /// </remarks>

    #endregion

    public class ComputeNode
    {
        /// <summary>
        /// The mathematical or structural operation associated with this computational node.
        /// </summary>
        private IAutogradOperation _operation;

        /// <summary>
        /// The array of input tensors that were passed into the operation during the forward pass.
        /// </summary>
        private ITensor[] _inputs;

        /// <summary>
        /// The cached output tensor resulting from the execution of the forward pass.
        /// </summary>
        private ITensor _output;

        /// <summary>
        /// Initializes a new instance of the <see cref="ComputeNode"/> class, immediately executing 
        /// the forward pass of the specified operation.
        /// </summary>
        /// <param name="operation">The autograd operation to execute as part of this node.</param>
        /// <param name="inputs">The collection of input tensors to be supplied to the operation.</param>
        /// <exception cref="NullReferenceException">
        /// Thrown if <paramref name="operation"/> is <see langword="null"/>.
        /// </exception>
        public ComputeNode(IAutogradOperation operation, params ITensor[] inputs)
        {
            _operation = operation;
            _inputs = inputs;
            _output = operation.Forward(inputs);
        }
        /// <summary>
        /// Gets the output tensor produced by the forward pass of this computational node.
        /// </summary>
        /// <value>
        /// An <see cref="ITensor"/> containing the evaluated result of the operation.
        /// </value>

        public ITensor Output => _output;
        /// <summary>
        /// Executes the backward pass for this node, calculating the local gradients and 
        /// propagating them back to the input tensors.
        /// </summary>
        /// <param name="gradOutput">The incoming gradient of the objective function (loss) with respect to the output of this node.</param>
        /// <remarks>
        /// This method retrieves the gradients of the operation with respect to each input via the 
        /// <see cref="_operation"/>'s backward pass. For each input tensor, if the tensor tracks gradients 
        /// (i.e., <see cref="ITensor.RequiresGrad"/> is <see langword="true"/>), the calculated gradient is either assigned 
        /// directly (if no gradient has been accumulated yet) or accumulated (added) to the existing gradient 
        /// to properly support node reuse and multiple paths in the computational graph.
        /// </remarks>
        /// <exception cref="NullReferenceException">
        /// Thrown if <see cref="_operation"/> or <see cref="_inputs"/> contains elements that are null.
        /// </exception>
        /// <exception cref="IndexOutOfRangeException">
        /// Thrown if the number of gradients returned by the operation's backward pass does not align with the number of input tensors.
        /// </exception>

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