// -----------------------------------------------------------------------------------------
// Scale-only RMSNorm — sibling of LayerNorm, not a subclass. No Fluent X.* API.
// -----------------------------------------------------------------------------------------

namespace ArborNet.Layers
{
    #region Using Statements

    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;

    #endregion

    /// <summary>
    /// Scale-only RMSNorm. Sibling of LayerNorm, not a subclass.
    /// Tape ops: Multiply / Mean / Add / Sqrt / Divide only. No beta, no Pow, no Rsqrt, no Fluent X.*.
    /// </summary>
    public sealed class RmsNorm : BaseLayer
    {
        private readonly ITensor _gamma;
        private readonly float _eps;
        private readonly int _d;

        public RmsNorm(int d, float eps = 1e-6f)
        {
            if (d <= 0) throw new ArgumentOutOfRangeException(nameof(d));
            if (!(eps > 0f)) throw new ArgumentOutOfRangeException(nameof(eps));
            _d = d;
            _eps = eps;
            _gamma = Tensor.Ones(new TensorShape(_d));
            _gamma.RequiresGrad = true;
        }

        public ITensor Gamma => _gamma;

        public override IEnumerable<ITensor> Parameters()
        {
            yield return _gamma;
        }

        public override ITensor Forward(ITensor x)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            var rank = x.Shape.Rank;
            if (rank < 2)
                throw new ArgumentException("RmsNorm expects rank >= 2 [..., D].", nameof(x));
            var last = x.Shape[rank - 1];
            if (last != _d)
                throw new ArgumentException($"Last dim must be {_d}, got {last}.", nameof(x));

            // x / sqrt(mean(x^2) + eps) * gamma
            var m2 = x.Multiply(x).Mean(-1, keepDims: true);
            var rms = m2.Add(_eps).Sqrt();
            var z = x.Divide(rms);
            return z.Multiply(_gamma);
        }
    }
}
