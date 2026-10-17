namespace ArborNet.Layers
{
    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;

    /// <summary>
    /// Transformer block configured exclusively for 2D Tensor operations [T, D].
    /// Chains RmsNorm -> SubquadraticAttention2D -> RmsNorm -> SwiGLU FFN.
    /// </summary>
    public sealed class PrimeGeometricBlock2D : BaseLayer
    {
        private readonly RmsNorm _norm1;
        private readonly SubquadraticAttention2D _attn;
        private readonly RmsNorm _norm2;
        private readonly Linear _wGate;
        private readonly Linear _wUp;
        private readonly Linear _wDown;

        public PrimeGeometricBlock2D(int dModel, int nHeads, int dFf, Device device)
        {
            this.device = device;

            _norm1 = new RmsNorm(dModel);
            _attn = new SubquadraticAttention2D(dModel, nHeads, device);

            _norm2 = new RmsNorm(dModel);
            _wGate = new Linear(dModel, dFf, device);
            _wUp = new Linear(dModel, dFf, device);
            _wDown = new Linear(dFf, dModel, device);
        }

        public override ITensor Forward(ITensor x)
        {
            // All operations remain strictly 2D: [T, D]
            var n1 = _norm1.Forward(x);
            var attnOut = _attn.Forward(n1);
            var h1 = x.Add(attnOut);

            var n2 = _norm2.Forward(h1);
            var gate = _wGate.Forward(n2);
            var up = _wUp.Forward(n2);
            var swish = gate.Multiply(gate.Sigmoid()).Multiply(up);
            var ffnOut = _wDown.Forward(swish);

            return h1.Add(ffnOut);
        }

        public override IEnumerable<ITensor> Parameters()
        {
            foreach (var p in _norm1.Parameters()) yield return p;
            foreach (var p in _attn.Parameters()) yield return p;
            foreach (var p in _norm2.Parameters()) yield return p;
            foreach (var p in _wGate.Parameters()) yield return p;
            foreach (var p in _wUp.Parameters()) yield return p;
            foreach (var p in _wDown.Parameters()) yield return p;
        }
    }
}