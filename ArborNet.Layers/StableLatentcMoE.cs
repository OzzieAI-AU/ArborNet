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
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    #endregion


    public sealed class StableLatentcMoE : BaseLayer
    {
        private readonly int _dModel;
        private readonly int _numExperts;
        private readonly List<Linear> _experts;
        private readonly Linear _gateProj;

        public StableLatentcMoE(int dModel, int numExperts, int activeExperts, int expertCapacity, Device device)
        {
            _dModel = dModel;
            _numExperts = numExperts;
            this.device = device;

            _gateProj = new Linear(dModel, numExperts, device);

            _experts = new List<Linear>(numExperts);
            for (int i = 0; i < numExperts; i++)
            {
                _experts.Add(new Linear(dModel, dModel, device));
            }
        }

        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input);
            int batch = input.Shape[0];
            int seqLen = input.Shape[1];
            int totalTokens = batch * seqLen;

            var flatInput = input.Reshape(totalTokens, _dModel);

            var gateLogits = _gateProj.Forward(flatInput); // [totalTokens, numExperts]
            var gateProbs = gateLogits.Softmax(-1); // [totalTokens, numExperts]

            ITensor? combinedOutput = null;

            // Fully differentiable Dense MoE (Preserves Autograd Graph)
            for (int e = 0; e < _numExperts; e++)
            {
                var expertOut = _experts[e].Forward(flatInput); // [totalTokens, dModel]
                var probSlice = gateProbs.Slice((0, totalTokens, 1), (e, e + 1, 1)); // [totalTokens, 1]
                var probBroadcast = probSlice.BroadcastTo(expertOut.Shape); // [totalTokens, dModel]

                var weightedExpertOut = expertOut.Multiply(probBroadcast);

                if (combinedOutput == null)
                    combinedOutput = weightedExpertOut;
                else
                    combinedOutput = combinedOutput.Add(weightedExpertOut);
            }

            return combinedOutput!.Reshape(batch, seqLen, _dModel);
        }

        public override IEnumerable<ITensor> Parameters()
        {
            var p = new List<ITensor>();
            p.AddRange(_gateProj.Parameters());
            foreach (var expert in _experts)
            {
                p.AddRange(expert.Parameters());
            }
            return p;
        }
    }
}