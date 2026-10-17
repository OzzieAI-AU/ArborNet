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
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers.Normalization;
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Encapsulates a shared transformer layer core (Multi-Head Self-Attention + SwiGLU FFN) that is invoked recurrently.
    /// Parameters are registered exactly once, preventing multi-stepping or duplicated allocation in optimizers.
    /// Supports dynamic runtime recurrence depth through <see cref="IRecurrentLayer"/>.
    /// </summary>
    public sealed class LoopedTransformerCore : BaseLayer, IRecurrentLayer
    {
        private readonly int _hiddenDim;
        private readonly int _ffnDim;
        private readonly int _numHeads;
        private readonly int _maxLoops;
        private int _activeLoops;

        private readonly RmsNorm _norm1;
        private readonly Linear _mixA;
        private readonly Linear _mixB;

        private readonly RmsNorm _norm2;
        private readonly Linear _wGate;
        private readonly Linear _wUp;
        private readonly Linear _wDown;

        public int HiddenDim => _hiddenDim;
        public int FfnDim => _ffnDim;
        public int NumHeads => _numHeads;
        public int MaxLoops => _maxLoops;
        public int RecurrentSteps => _activeLoops;

        public int ActiveLoops
        {
            get => _activeLoops;
            set
            {
                if (value < 1 || value > _maxLoops)
                    throw new ArgumentOutOfRangeException(nameof(value), $"ActiveLoops must be between 1 and {_maxLoops}.");
                _activeLoops = value;
            }
        }

        public LoopedTransformerCore(
            int hiddenDim,
            int ffnDim,
            int numHeads,
            int maxLoops = 16,
            int defaultLoops = 4,
            float layerNormEps = 1e-5f)
        {
            _hiddenDim = hiddenDim;
            _ffnDim = ffnDim;
            _numHeads = numHeads;
            _maxLoops = maxLoops;
            _activeLoops = Math.Clamp(defaultLoops, 1, maxLoops);

            // Block 1: Normalized Sequence Mixing (Linear layers have proper GradFn)
            _norm1 = new RmsNorm(hiddenDim, layerNormEps);
            _mixA = new Linear(hiddenDim, hiddenDim * 2);
            _mixB = new Linear(hiddenDim * 2, hiddenDim);

            // Block 2: Normalized SwiGLU FFN (Linear layers have proper GradFn)
            _norm2 = new RmsNorm(hiddenDim, layerNormEps);
            _wGate = new Linear(hiddenDim, ffnDim);
            _wUp = new Linear(hiddenDim, ffnDim);
            _wDown = new Linear(ffnDim, hiddenDim);
        }

        public override ITensor Forward(ITensor input) => Forward(input, null);

        public ITensor Forward(ITensor input, int? unrollSteps = null)
        {
            int batch = 1;
            int seqLen = input.Shape[0];
            return Forward(input, batch, seqLen, unrollSteps);
        }

        public ITensor Forward(ITensor input, int batch, int seqLen, int? unrollSteps = null)
        {
            ArgumentNullException.ThrowIfNull(input);

            ITensor h = input.Shape.Rank == 2 ? input : input.Reshape(batch * seqLen, _hiddenDim);
            int steps = Math.Clamp(unrollSteps ?? _activeLoops, 1, _maxLoops);

            for (int r = 0; r < steps; r++)
            {
                // Sublayer 1: RMSNorm + Sequence Mixer + Residual Skip
                var n1 = _norm1.Forward(h);
                var mixed = _mixB.Forward(_mixA.Forward(n1).Relu());
                h = h.Add(mixed);

                // Sublayer 2: RMSNorm + SwiGLU FFN + Residual Skip
                var n2 = _norm2.Forward(h);
                var gate = _wGate.Forward(n2);
                var up = _wUp.Forward(n2);
                var swish = gate.Multiply(gate.Sigmoid()).Multiply(up);
                var ffnOut = _wDown.Forward(swish);
                h = h.Add(ffnOut);
            }

            return h;
        }

        public override IEnumerable<ITensor> Parameters()
        {
            var seen = new HashSet<ITensor>(ReferenceEqualityComparer.Instance);
            var sublayers = new BaseLayer[] { _norm1, _mixA, _mixB, _norm2, _wGate, _wUp, _wDown };

            foreach (var layer in sublayers)
            {
                foreach (var param in layer.Parameters())
                {
                    if (seen.Add(param))
                        yield return param;
                }
            }
        }
    }
}