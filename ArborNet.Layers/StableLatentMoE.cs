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


    public sealed class StableLatentMoE : BaseLayer
    {
        private readonly int _dModel;
        private readonly int _numExperts;
        private readonly int _activeExperts;
        private readonly int _expertCapacity;
        private readonly List<Linear> _experts;
        private readonly Linear _gateProj;

        public StableLatentMoE(int dModel, int numExperts, int activeExperts, int expertCapacity, Device device)
        {
            _dModel = dModel;
            _numExperts = numExperts;
            _activeExperts = activeExperts;
            _expertCapacity = expertCapacity;
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

            float[] probsData = gateProbs.ToArray();
            float[] inputData = flatInput.ToArray();
            float[] outputData = new float[totalTokens * _dModel];

            int[] expertTokenCount = new int[_numExperts];
            object syncLock = new object();

            Parallel.For(0, totalTokens, t =>
            {
                var tokenProbs = new List<(float Prob, int Index)>();
                for (int e = 0; e < _numExperts; e++)
                {
                    tokenProbs.Add((probsData[t * _numExperts + e], e));
                }
                var topExperts = tokenProbs.OrderByDescending(x => x.Prob).Take(_activeExperts).ToList();
                float sumProb = topExperts.Sum(x => x.Prob) + 1e-10f;

                var tokenVec = new float[_dModel];
                Array.Copy(inputData, t * _dModel, tokenVec, 0, _dModel);

                float[] combinedOutput = new float[_dModel];

                foreach (var expertInfo in topExperts)
                {
                    int expertIdx = expertInfo.Index;
                    float weight = expertInfo.Prob / sumProb;

                    bool withinCapacity;
                    lock (syncLock)
                    {
                        if (expertTokenCount[expertIdx] < _expertCapacity)
                        {
                            expertTokenCount[expertIdx]++;
                            withinCapacity = true;
                        }
                        else
                        {
                            withinCapacity = false;
                        }
                    }

                    if (withinCapacity)
                    {
                        var expTensor = Tensor.FromArray(tokenVec, new TensorShape(1, _dModel), device);
                        var expOut = _experts[expertIdx].Forward(expTensor).ToArray();
                        for (int d = 0; d < _dModel; d++)
                        {
                            combinedOutput[d] += expOut[d] * weight;
                        }
                    }
                    else
                    {
                        // Soft Token Dropping
                        for (int d = 0; d < _dModel; d++)
                        {
                            combinedOutput[d] += tokenVec[d] * weight;
                        }
                    }
                }

                for (int d = 0; d < _dModel; d++)
                {
                    outputData[t * _dModel + d] = combinedOutput[d];
                }
            });

            var result = Tensor.FromArray(outputData, new TensorShape(batch, seqLen, _dModel), device);

            if (input.RequiresGrad)
            {
                result.GradFn = gradOutput =>
                {
                    _gateProj.Forward(flatInput).AccumulateGrad(gradOutput.Reshape(totalTokens, _dModel).Mean(1, true).BroadcastTo(gateLogits.Shape));
                    return gradOutput;
                };
            }

            return result;
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