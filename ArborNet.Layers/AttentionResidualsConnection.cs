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


    public sealed class AttentionResidualsConnection
    {
        private readonly int _layerIndex;
        private readonly ITensor _routingWeights;

        public AttentionResidualsConnection(int layerIndex, Device device)
        {
            _layerIndex = layerIndex;
            _routingWeights = Tensor.Ones(new TensorShape(layerIndex + 1), device);
            _routingWeights.RequiresGrad = true;
        }

        public ITensor Route(List<ITensor> history, ITensor currentOutput)
        {
            var device = currentOutput.Device;
            var accumulated = Tensor.Zeros(currentOutput.Shape, device);

            float[] weights = _routingWeights.ToArray();
            float sumExp = weights.Select(MathF.Exp).Sum() + 1e-10f;

            for (int i = 0; i < history.Count; i++)
            {
                float normalizedWeight = MathF.Exp(weights[i]) / sumExp;
                accumulated = accumulated.Add(history[i].Multiply(normalizedWeight));
            }

            float currentNormalizedWeight = MathF.Exp(weights[_layerIndex]) / sumExp;
            accumulated = accumulated.Add(currentOutput.Multiply(currentNormalizedWeight));

            return accumulated;
        }

        public IEnumerable<ITensor> Parameters()
        {
            yield return _routingWeights;
        }
    }
}