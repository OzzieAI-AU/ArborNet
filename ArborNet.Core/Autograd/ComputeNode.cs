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
        private IAutogradOperation _operation;
        private ITensor[] _inputs;
        private uint[] _inputVersions; // Store captured versions
        private ITensor _output;

        public ComputeNode(IAutogradOperation operation, params ITensor[] inputs)
        {
            _operation = operation;
            _inputs = inputs;
            // Snapshot the version at computation time
            _inputVersions = inputs.Select(x => x.Version).ToArray();
            _output = operation.Forward(inputs);
        }

        public ITensor Output => _output;

        public void Backward(ITensor gradOutput)
        {
            var grads = _operation.Backward(gradOutput);
            int i = 0;
            foreach (var grad in grads)
            {
                if (_inputs[i].RequiresGrad)
                {
                    // Validate that the tensor wasn't mutated after graph creation
                    if (_inputs[i].Version != _inputVersions[i])
                    {
                        throw new InvalidOperationException(
                            "Tensor modified in-place after being used in a gradient computation. " +
                            "This invalidates the autograd graph.");
                    }

                    if (_inputs[i].Grad == null)
                        _inputs[i].Grad = grad;
                    else
                        _inputs[i].Grad = _inputs[i].Grad.Add(grad);
                }
                i++;
            }
        }
    }
}