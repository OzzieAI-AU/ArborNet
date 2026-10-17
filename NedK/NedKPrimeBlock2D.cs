#nullable enable
namespace NedK
{
    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;

    /// <summary>
    /// NedK-Prime 2D recurrent core.
    /// Architecture id: dense-ffn-parallel-deltanorm.
    /// One parallel attention + SwiGLU write per step; RMSNorm on the write only.
    /// Stream loop-norm is not applied to the residual stream (preserves token magnitude).
    /// </summary>
    public sealed class NedKPrimeBlock2D : BaseLayer
    {
        public const string ArchitectureId = "dense-ffn-parallel-deltanorm";

        /// <summary>Scale for a unit-RMS parallel write (1.0 = undamped normalized write).</summary>
        public const float ParallelWriteScale = 1.0f;

        private readonly RmsNorm _norm1;
        private readonly SubquadraticAttention2D _attn;
        private readonly RmsNorm _norm2;
        private readonly Linear _wGate;
        private readonly Linear _wUp;
        private readonly Linear _wDown;
        private readonly RmsNorm _loopNorm;
        private readonly float _residualScale;

        public int DModel { get; }
        public int DFf { get; }
        public int MaxWindow { get; }
        public float ResidualScale => _residualScale;

        public NedKPrimeBlock2D(int dModel, int dFf, int maxWindow, Device device, float residualScale = ParallelWriteScale)
        {
            if (dModel <= 0) throw new ArgumentOutOfRangeException(nameof(dModel));
            if (dFf <= 0) throw new ArgumentOutOfRangeException(nameof(dFf));
            if (maxWindow <= 0) throw new ArgumentOutOfRangeException(nameof(maxWindow));
            if (float.IsNaN(residualScale) || float.IsInfinity(residualScale) || residualScale <= 0f)
                throw new ArgumentOutOfRangeException(nameof(residualScale));

            this.device = device;
            DModel = dModel;
            DFf = dFf;
            MaxWindow = maxWindow;
            _residualScale = residualScale;

            _norm1 = new RmsNorm(dModel);
            _attn = new SubquadraticAttention2D(dModel, maxWindow, device);
            _norm2 = new RmsNorm(dModel);
            _wGate = new Linear(dModel, dFf, device);
            _wUp = new Linear(dModel, dFf, device);
            _wDown = new Linear(dFf, dModel, device);
            _loopNorm = new RmsNorm(dModel);
        }

        public override ITensor Forward(ITensor x)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));

            // One forward per module so module-local backward caches stay valid.
            var nAttn = _norm1.Forward(x);
            var attnOut = _attn.Forward(nAttn);

            var nFfn = _norm2.Forward(x);
            var gate = _wGate.Forward(nFfn);
            var up = _wUp.Forward(nFfn);
            var swish = gate.Multiply(gate.Sigmoid()).Multiply(up);
            var ffnOut = _wDown.Forward(swish);

            var write = _loopNorm.Forward(attnOut.Add(ffnOut));
            return Math.Abs(_residualScale - 1.0f) < 1e-6f
                ? x.Add(write)
                : x.Add(write.Multiply(_residualScale));
        }

        /// <summary>
        /// Yield every trainable tensor once (order must stay stable for OZM save/load).
        /// </summary>
        public override IEnumerable<ITensor> Parameters()
        {
            foreach (var p in _norm1.Parameters()) yield return p;
            foreach (var p in _attn.Parameters()) yield return p;
            foreach (var p in _norm2.Parameters()) yield return p;
            foreach (var p in _wGate.Parameters()) yield return p;
            foreach (var p in _wUp.Parameters()) yield return p;
            foreach (var p in _wDown.Parameters()) yield return p;
            foreach (var p in _loopNorm.Parameters()) yield return p;
        }
    }
}
