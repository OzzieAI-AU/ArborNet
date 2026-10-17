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
    using ArborNet.Core.Devices;
    using ArborNet.Core.Initializers;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Metrics;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Text.Json;

    public class RecurrentDepthTransformer : BaseModel, IRecurrentLayer
    {
        private readonly int _vocabSize;
        private readonly int _hiddenDim;
        private readonly int _ffnDim;
        private readonly int _numHeads;
        private readonly int _maxSequenceLength;
        private readonly int _factorDim;
        private int _defaultLoops;
        private readonly int _maxLoops;
        private int _activeLoops;

        // Certified autograd layers
        private readonly Embedding _tok;
        private readonly Embedding _pos;
        private readonly Linear _factorProj;
        private readonly LoopedTransformerCore _core;
        private readonly Linear _head;

        public int DefaultLoops
        {
            get => _defaultLoops;
            set
            {
                if (value < 1 || value > _maxLoops)
                    throw new ArgumentOutOfRangeException(nameof(value), $"Default loops must be in [1, {_maxLoops}].");
                _defaultLoops = value;
            }
        }

        public int MaxLoops => _maxLoops;
        public int RecurrentSteps => _activeLoops;
        public int HiddenDim => _hiddenDim;
        public int FfnDim => _ffnDim;
        public int NumHeads => _numHeads;
        public int VocabSize => _vocabSize;
        public int MaxSequenceLength => _maxSequenceLength;

        public int ActiveLoops
        {
            get => _activeLoops;
            set
            {
                if (value < 1 || value > _maxLoops)
                    throw new ArgumentOutOfRangeException(nameof(value), $"Active loops must be between 1 and MaxLoops ({_maxLoops}).");
                _activeLoops = value;
            }
        }

        public RecurrentDepthTransformer(
            int vocabSize = 42,
            int hiddenDim = 64,
            int ffnDim = 128,
            int numHeads = 4,
            int maxSequenceLength = 64,
            int defaultLoops = 4,
            int maxLoops = 16,
            int factorDim = 24,
            Device? device = null)
        {
            if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
            if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
            if (ffnDim <= 0) throw new ArgumentOutOfRangeException(nameof(ffnDim));
            if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
            if (maxLoops < 1) throw new ArgumentOutOfRangeException(nameof(maxLoops));
            if (defaultLoops < 1 || defaultLoops > maxLoops)
                throw new ArgumentOutOfRangeException(nameof(defaultLoops));

            device ??= Device.CPU;
            this.device = device;

            _vocabSize = vocabSize;
            _hiddenDim = hiddenDim;
            _ffnDim = ffnDim;
            _numHeads = numHeads;
            _maxSequenceLength = maxSequenceLength;
            _defaultLoops = defaultLoops;
            _activeLoops = defaultLoops;
            _maxLoops = maxLoops;
            _factorDim = factorDim;

            // 1. Layer-backed Factorized Embeddings (Full autograd support)
            _tok = new Embedding(vocabSize, factorDim);
            _pos = new Embedding(maxSequenceLength, hiddenDim);
            _factorProj = new Linear(factorDim, hiddenDim, device);

            // 2. Layer-backed Recurrent Transformer Core
            _core = new LoopedTransformerCore(hiddenDim, ffnDim, numHeads, maxLoops, defaultLoops);

            // 3. Layer-backed Logit Head
            _head = new Linear(hiddenDim, vocabSize, device);

            RebuildParameters();
        }

        private void RebuildParameters()
        {
            parameters.Clear();
            var seen = new HashSet<ITensor>(ReferenceEqualityComparer.Instance);

            void AddUnique(IEnumerable<ITensor> tensors)
            {
                foreach (var t in tensors)
                {
                    if (t != null && seen.Add(t))
                        parameters.Add(t);
                }
            }

            AddUnique(_tok.Parameters());
            AddUnique(_pos.Parameters());
            AddUnique(_factorProj.Parameters());
            if (_core != null) AddUnique(_core.Parameters());
            AddUnique(_head.Parameters());
        }

        public override ITensor Forward(ITensor inputTokens) => Forward(inputTokens, null);

        public ITensor Forward(ITensor inputTokens, int? loops = null)
        {
            ArgumentNullException.ThrowIfNull(inputTokens);

            int effectiveLoops = loops ?? _activeLoops;
            if (effectiveLoops < 1 || effectiveLoops > _maxLoops)
                throw new ArgumentOutOfRangeException(nameof(loops));

            if (loops.HasValue) _activeLoops = effectiveLoops;

            int batch = inputTokens.Shape.Rank >= 2 ? inputTokens.Shape[0] : 1;
            int seqLen = inputTokens.Shape.Rank >= 2 ? inputTokens.Shape[1] : inputTokens.Shape[0];

            // 1. Token embedding via Embedding layer (Registers NativeEmbeddingGrad closure)
            // Input: [batch, seqLen] -> [batch * seqLen, factorDim]
            ITensor tokEmbed = _tok.Forward(inputTokens).Reshape(batch * seqLen, _factorDim);

            // 2. Factor projection via Linear layer (Registers Linear closure)
            // [batch * seqLen, factorDim] @ W + b -> [batch * seqLen, hiddenDim]
            ITensor h = _factorProj.Forward(tokEmbed);

            // 3. Add positional embeddings (Registers Add closure)
            float[] posIds = new float[seqLen];
            for (int i = 0; i < seqLen; i++) posIds[i] = i;
            var posTensor = Tensor.FromArray(posIds, new TensorShape(1, seqLen), inputTokens.Device);
            ITensor posEmbed = _pos.Forward(posTensor).Reshape(seqLen, _hiddenDim);

            h = h.Add(posEmbed);

            // 4. Shared Recurrent Core
            h = _core.Forward(h, batch, seqLen, effectiveLoops);

            // 5. Output Head via Linear layer (Registers Linear closure)
            // [batch * seqLen, hiddenDim] -> [batch, seqLen, vocabSize]
            ITensor logits = _head.Forward(h);
            return logits.Reshape(batch, seqLen, _vocabSize);
        }

        public long ComputeTotalFlops(int batchSize, int seqLen, int loops)
        {
            long tokens = (long)batchSize * seqLen;
            long prelude = 2L * tokens * _factorDim * _hiddenDim;
            long coda = 2L * tokens * _hiddenDim * _vocabSize;
            long corePerLoop = (4L * tokens * _hiddenDim * _hiddenDim) + (6L * tokens * _hiddenDim * _ffnDim);
            return prelude + coda + (loops * corePerLoop);
        }

        public override void To(Device targetDevice)
        {
            base.To(targetDevice);
            _core?.To(targetDevice);
            RebuildParameters();
        }

        public int ComputeTotalUniqueParameters()
        {
            int total = 0;
            var visited = new HashSet<ITensor>(ReferenceEqualityComparer.Instance);
            foreach (var param in Parameters())
            {
                if (param != null && visited.Add(param))
                    total += param.Shape.TotalElements;
            }
            return total;
        }

        public void ExportOzm(string filePath)
        {
            var manifest = new
            {
                Format = "OZM-1.0",
                Architecture = nameof(RecurrentDepthTransformer),
                HiddenDim = _hiddenDim,
                FfnDim = _ffnDim,
                NumHeads = _numHeads,
                Recurrence = _activeLoops,
                DefaultLoops = _defaultLoops,
                MaxLoops = _maxLoops,
                UniqueParameters = ComputeTotalUniqueParameters(),
                TimestampUtc = DateTime.UtcNow.ToString("o")
            };

            string json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(filePath, json);
        }

        public KnowledgeDensityMetrics EvaluateKnowledgeDensity(BaseModel model, double evalLoss, int vocabSize, ITensor? sampleHiddenStates = null)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));

            return KnowledgeDensityEvaluator.Evaluate(model: model, evalLoss, vocabSize, sampleHiddenStates);
        }
    }
}