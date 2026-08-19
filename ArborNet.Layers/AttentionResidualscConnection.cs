// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Layers
{

    #region Using Statements:

    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    using System.Linq;

    #endregion


    public sealed class AttentionResidualscConnection
    {
        private readonly int _layerIndex;
        private readonly ITensor _routingWeights;

        public AttentionResidualscConnection(int layerIndex, Device device)
        {
            _layerIndex = layerIndex;
            _routingWeights = Tensor.Ones(new TensorShape(layerIndex + 1), device);
            _routingWeights.RequiresGrad = true;
        }

        public ITensor Route(List<ITensor> history, ITensor currentOutput)
        {
            var device = currentOutput.Device;

            // Native tensor operation creates a valid autograd backwards link
            var probs = _routingWeights.Softmax(-1);

            var accumulated = Tensor.Zeros(currentOutput.Shape, device);

            for (int i = 0; i < history.Count; i++)
            {
                var prob = probs.Slice((i, i + 1, 1)).Reshape(1, 1, 1).BroadcastTo(accumulated.Shape);
                accumulated = accumulated.Add(history[i].Multiply(prob));
            }

            var currentProb = probs.Slice((_layerIndex, _layerIndex + 1, 1)).Reshape(1, 1, 1).BroadcastTo(accumulated.Shape);
            accumulated = accumulated.Add(currentOutput.Multiply(currentProb));

            return accumulated;
        }

        public IEnumerable<ITensor> Parameters()
        {
            yield return _routingWeights;
        }
    }
}