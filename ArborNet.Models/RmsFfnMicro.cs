namespace ArborNet.Models
{


    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Layers;


    /// <summary>
    /// FFN-only compile / GradientCheck vehicle for tape-only RmsNorm.
    /// No attention, no public MHA, no Linear×4. Lock 2×256, T=64, I=512.
    /// </summary>
    public sealed class RmsFfnMicro : BaseModel
    {
        public const int Vocab = 256;
        public const int Width = 256;
        public const int Seq = 64;
        public const int Hidden = 512;
        public const int Depth = 2;

        private readonly Embedding _embed;
        private readonly RmsNorm _rms0;
        private readonly RmsNorm _rms1;
        private readonly RmsNorm _rmsFinal;
        private readonly Linear _up0;
        private readonly Linear _down0;
        private readonly Linear _up1;
        private readonly Linear _down1;
        private readonly Linear _head;

        public RmsFfnMicro()
        {
            _embed = new Embedding(Vocab, Width);
            _rms0 = new RmsNorm(Width);
            _rms1 = new RmsNorm(Width);
            _rmsFinal = new RmsNorm(Width);
            _up0 = new Linear(Width, Hidden);
            _down0 = new Linear(Hidden, Width);
            _up1 = new Linear(Width, Hidden);
            _down1 = new Linear(Hidden, Width);
            _head = new Linear(Width, Vocab); // always-bias head
        }

        public RmsNorm Rms0 => _rms0;
        public RmsNorm Rms1 => _rms1;
        public RmsNorm RmsFinal => _rmsFinal;

        public override IEnumerable<ITensor> Parameters()
        {
            foreach (var p in _embed.Parameters()) yield return p;
            foreach (var p in _rms0.Parameters()) yield return p;
            foreach (var p in _up0.Parameters()) yield return p;
            foreach (var p in _down0.Parameters()) yield return p;
            foreach (var p in _rms1.Parameters()) yield return p;
            foreach (var p in _up1.Parameters()) yield return p;
            foreach (var p in _down1.Parameters()) yield return p;
            foreach (var p in _rmsFinal.Parameters()) yield return p;
            foreach (var p in _head.Parameters()) yield return p;
        }

        public override ITensor Forward(ITensor tokens)
        {
            if (tokens is null) throw new ArgumentNullException(nameof(tokens));
            if (tokens.Shape.Rank != 2)
                throw new ArgumentException("RmsFfnMicro expects tokens [B, T].", nameof(tokens));

            var b = tokens.Shape[0];
            var t = tokens.Shape[1];
            var bt = b * t;

            var h = _embed.Forward(tokens); // [B, T, D]
            h = ResidualFfn(h, _rms0, _up0, _down0, b, t, bt);
            h = ResidualFfn(h, _rms1, _up1, _down1, b, t, bt);

            var n = _rmsFinal.Forward(h);
            var flat = n.Reshape(bt, Width); // rank-2 VJP into Linear
            return _head.Forward(flat);
        }

        private static ITensor ResidualFfn(
            ITensor h,
            RmsNorm rms,
            Linear up,
            Linear down,
            int b,
            int t,
            int bt)
        {
            var n = rms.Forward(h);
            var flat = n.Reshape(bt, Width);
            var hid = up.Forward(flat).Relu();
            var outFlat = down.Forward(hid);
            var outSeq = outFlat.Reshape(b, t, Width);
            return h.Add(outSeq);
        }
    }
}
