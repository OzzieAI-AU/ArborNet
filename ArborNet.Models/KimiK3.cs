// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Models
{

    #region Using Statements:

    using ArborNet.Activations;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Functional;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;

    #endregion


    // =====================================================================================
    // 5. KIMI K3 DECODER BLOCK
    // =====================================================================================

    public sealed class KimiK3Block : BaseLayer
    {
        private readonly LayerNorm _norm1;
        private readonly DeltaAttention _kdaAttention;
        private readonly LayerNorm _norm2;
        private readonly StableLatentMoE _latentMoE;
        private readonly AttentionResidualsConnection _attnRes;

        public KimiK3Block(int layerIndex, int dModel, int nHeads, int numExperts, int activeExperts, int expertCapacity, Device device)
        {
            _norm1 = new LayerNorm(new[] { dModel });
            _kdaAttention = new DeltaAttention(dModel, nHeads, device);
            _norm2 = new LayerNorm(new[] { dModel });
            _latentMoE = new StableLatentMoE(dModel, numExperts, activeExperts, expertCapacity, device);
            _attnRes = new AttentionResidualsConnection(layerIndex, device);

            this.device = device;
        }

        public ITensor Forward(ITensor input, List<ITensor> history)
        {
            var norm1Out = _norm1.Forward(input);
            var attnOut = _kdaAttention.Forward(norm1Out);

            var routed1 = _attnRes.Route(history, attnOut);
            var x = input.Add(routed1);

            var norm2Out = _norm2.Forward(x);
            var moeOut = _latentMoE.Forward(norm2Out);
            x = x.Add(moeOut);

            return x;
        }

        public override ITensor Forward(ITensor input)
        {
            throw new NotSupportedException("KimiK3Block requires Layer History. Use Block.Forward(input, history) instead.");
        }

        public override IEnumerable<ITensor> Parameters()
        {
            var p = new List<ITensor>();
            p.AddRange(_norm1.Parameters());
            p.AddRange(_kdaAttention.Parameters());
            p.AddRange(_norm2.Parameters());
            p.AddRange(_latentMoE.Parameters());
            p.AddRange(_attnRes.Parameters());
            return p;
        }
    }

    // =====================================================================================
    // 6. MAIN FRONTIER MODEL: KIMI K3
    // =====================================================================================

    public sealed class KimiK3 : BaseModel
    {
        private readonly Embedding _tokenEmbedding;
        private readonly Embedding _positionEmbedding;
        private readonly List<KimiK3Block> _blocks;
        private readonly LayerNorm _finalNorm;
        private readonly Linear _outputHead;
        private readonly int _maxSeqLen;
        private readonly int _vocabSize;
        private readonly int _dModel;

        public KimiK3(
            int vocabSize,
            int dModel,
            int nHeads,
            int numLayers,
            int numExperts,
            int activeExperts,
            int expertCapacity,
            int maxSeqLen,
            Device? device = null)
        {
            device ??= Device.CPU;
            _vocabSize = vocabSize;
            _dModel = dModel;
            _maxSeqLen = maxSeqLen;

            _tokenEmbedding = new Embedding(vocabSize, dModel);
            _positionEmbedding = new Embedding(maxSeqLen, dModel);

            _blocks = new List<KimiK3Block>(numLayers);
            for (int i = 0; i < numLayers; i++)
            {
                _blocks.Add(new KimiK3Block(i, dModel, nHeads, numExperts, activeExperts, expertCapacity, device));
            }

            _finalNorm = new LayerNorm(new[] { dModel });
            _outputHead = new Linear(dModel, vocabSize, device);

            RegisterParameters();
        }

        private void RegisterParameters()
        {
            parameters.Clear();
            parameters.AddRange(_tokenEmbedding.Parameters());
            parameters.AddRange(_positionEmbedding.Parameters());
            foreach (var block in _blocks)
            {
                parameters.AddRange(block.Parameters());
            }
            parameters.AddRange(_finalNorm.Parameters());
            parameters.AddRange(_outputHead.Parameters());
        }

        public override ITensor Forward(ITensor input)
        {
            if (input.Shape.Rank != 2)
                throw new ArgumentException("Input must be a 2D tensor [batchSize, seqLen].", nameof(input));

            int batchSize = input.Shape[0];
            int seqLen = input.Shape[1];

            if (seqLen > _maxSeqLen)
                throw new ArgumentException($"Sequence length {seqLen} exceeds maximum window size {_maxSeqLen}.", nameof(input));

            var x = _tokenEmbedding.Forward(input);

            var posData = new float[batchSize * seqLen];
            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    posData[b * seqLen + t] = t;
                }
            }
            var posIds = Tensor.FromArray(posData, new TensorShape(batchSize, seqLen), input.Device);
            var posEmb = _positionEmbedding.Forward(posIds);
            x = x.Add(posEmb);

            var layerHistory = new List<ITensor>();

            // Reason state transitions
            var prevStateHigh = x.Clone();

            for (int i = 0; i < _blocks.Count; i++)
            {
                x = _blocks[i].Forward(x, layerHistory);
                layerHistory.Add(x);

                // Check for dynamic hidden state halting to skip redundant reasoning depths
                if (i > 0)
                {
                    var diff = x.Subtract(prevStateHigh);
                    float stateDiff = diff.Pow(2).Sum().ToScalar();
                    if (stateDiff < 1e-4f)
                    {
                        break;
                    }
                }
                prevStateHigh = x.Clone();
            }

            x = _finalNorm.Forward(x);

            // Flatten 3D context to 2D before linear projection to prevent rank exceptions
            var flatX = x.Reshape(batchSize * seqLen, _dModel);
            var logits = _outputHead.Forward(flatX).Reshape(batchSize, seqLen, _vocabSize);

            return logits;
        }

        public ITensor GenerateNextToken(ITensor input)
        {
            var logits = Forward(input); // [batchSize, seqLen, vocabSize]
            int batchSize = logits.Shape[0];
            int seqLen = logits.Shape[1];

            var lastLogits = logits.Slice(
                (0, batchSize, 1),
                (seqLen - 1, seqLen, 1),
                (0, _vocabSize, 1)
            );

            return lastLogits.Reshape(batchSize, _vocabSize).ArgMax(-1);
        }

        public override IEnumerable<ITensor> Parameters() => parameters;
    }
}