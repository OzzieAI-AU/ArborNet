namespace ArborNet.Models
{
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using global::ArborNet.Core.Devices;
    using global::ArborNet.Core.Interfaces;
    using global::ArborNet.Core.Models;
    using global::ArborNet.Layers;
    using System;
    using System.Collections.Generic;
    using System.Linq;


    using System;
    using System.Collections.Generic;
    using System.Text;

    /// <summary>
    /// Pure 2D Prime Geometric Transformer.
    /// Input: [1, Window] or [Window] indices -> Embeds to [Window, D]
    /// Output: [1, VocabSize] prediction logits for next-token generation.
    /// </summary>
    public sealed class PrimeGeometricTransformer2D : BaseModel
    {
        private readonly int _vocabSize;
        private readonly int _dModel;
        private readonly int _window;
        private readonly int _layers;

        private readonly Embedding _tokEmbedding;
        private readonly PrimeLorentzianPositionalEncoding2D _primePE;
        private readonly List<PrimeGeometricBlock2D> _blocks;
        private readonly RmsNorm _finalNorm;

        public int Window => _window;
        public int HiddenDim => _dModel;
        public int VocabSize => _vocabSize;
        public Embedding TokenEmbedding => _tokEmbedding;

        public PrimeGeometricTransformer2D(
            int vocabSize,
            int dModel,
            int nHeads,
            int dFf,
            int window,
            int numLayers,
            Device? device = null)
        {
            device ??= Device.CPU;
            this.device = device;

            _vocabSize = vocabSize;
            _dModel = dModel;
            _window = window;
            _layers = numLayers;

            _tokEmbedding = new Embedding(vocabSize, dModel);
            _primePE = new PrimeLorentzianPositionalEncoding2D(dModel, maxLen: window, hausdorffDim: 1.2618595, device);

            _blocks = new List<PrimeGeometricBlock2D>(numLayers);
            for (int i = 0; i < numLayers; i++)
            {
                _blocks.Add(new PrimeGeometricBlock2D(dModel, nHeads, dFf, device));
            }

            _finalNorm = new RmsNorm(dModel);
            RegisterParameters();
        }

        private void RegisterParameters()
        {
            parameters.Clear();
            parameters.AddRange(_tokEmbedding.Parameters());
            parameters.AddRange(_primePE.Parameters());

            foreach (var block in _blocks)
            {
                parameters.AddRange(block.Parameters());
            }

            parameters.AddRange(_finalNorm.Parameters());
        }

        public override ITensor Forward(ITensor input)
        {
            // Input token shape: [1, Window] -> Reshape to 1D sequence [Window]
            int T = input.Shape.TotalElements;
            if (T > _window)
            {
                throw new ArgumentException($"Sequence length {T} exceeds window limit {_window}.");
            }

            var flatIndices = (input.Shape.Rank == 1) ? input : input.Reshape(T);

            // 1. Embedding lookup produces strictly 2D tensor: [T, D]
            var x = _tokEmbedding.Forward(flatIndices);

            // 2. Add Lorentzian Prime Positional Encoding in 2D: [T, D] + [T, D]
            x = _primePE.Forward(x);

            // 3. Process blocks strictly in 2D: [T, D]
            foreach (var block in _blocks)
            {
                x = block.Forward(x);
            }

            // 4. Extract last token row: [1, D]
            var lastToken = x.Slice((T - 1, T, 1), (0, _dModel, 1));
            var normed = _finalNorm.Forward(lastToken);

            // 5. Weight-tied output projection: [1, D] @ [D, V] = [1, V]
            var tokWeight = _tokEmbedding.Parameters().First();
            var logits = normed.MatMul(tokWeight.Transpose(new[] { 1, 0 }));

            return logits.Clip(-20.0f, 20.0f);
        }

        public int ComputeUniqueParameters()
        {
            int total = 0;
            var visited = new HashSet<ITensor>(ReferenceEqualityComparer.Instance);
            foreach (var p in Parameters())
            {
                if (p != null && visited.Add(p))
                {
                    total += p.Shape.TotalElements;
                }
            }
            return total;
        }
    }
}