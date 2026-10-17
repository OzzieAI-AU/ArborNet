using System;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Layers;
using ArborNet.Core.Tensors;

namespace ArborNet.Layers
{
    /// <summary>
    /// RMS sibling with one shared scale. Not a subclass of RmsNorm or LayerNorm.
    /// Not GroupRmsNorm (P=C) and not UnitRmsNorm (affine off).
    /// Not inserted into RmsFfnMicro. Staging this file is not a Knowledge Density raise.
    /// Tape: m2 = mean(x*x, last, keepDims); inv = Divide(1, Sqrt(m2+eps)); y = (x*inv)*gamma.
    /// Residual Add stays in the caller. No beta.
    /// </summary>
    public sealed class ScalarRmsNorm : BaseLayer
    {
        private readonly int _features;
        private readonly float _eps;
        private readonly ITensor _gamma;
        private readonly ITensor _one;

        public ScalarRmsNorm(int features, float eps = 1e-6f)
        {
            if (features <= 0)
                throw new ArgumentOutOfRangeException(nameof(features), "last dimension must be > 0");
            if (!(eps > 0f) || float.IsNaN(eps) || float.IsInfinity(eps))
                throw new ArgumentOutOfRangeException(nameof(eps), "eps must be > 0");

            _features = features;
            _eps = eps;

            // Shape [1], value 1. Not Tensor.Ones and not a D-sized LayerScale.
            _gamma = Tensor.FromArray(new float[] { 1f }, new TensorShape(new int[] { 1 }));
            _gamma.RequiresGrad = true;
            _one = Tensor.FromArray(new float[] { 1f }, new TensorShape(new int[] { 1 }));
            _one.RequiresGrad = false;
        }

        public override ITensor[] Parameters()
        {
            return new ITensor[] { _gamma };
        }

        public override ITensor Forward(ITensor input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            int last = LastDimension(input);
            if (last <= 0)
                throw new ArgumentException("last dimension must be > 0", nameof(input));
            if (last != _features)
                throw new ArgumentException("last dimension must match constructed features", nameof(input));

            ITensor m2 = input.Multiply(input).Mean(-1, keepDims: true);
            ITensor inv = _one.Divide(m2.Add(_eps).Sqrt());
            return input.Multiply(inv).Multiply(_gamma);
        }

        private static int LastDimension(ITensor input)
        {
            if (input.Shape == null)
                throw new ArgumentException("shape is required", nameof(input));
            int rank = input.Shape.Rank;
            if (rank < 1)
                throw new ArgumentException("rank must be >= 1", nameof(input));
            return input.Shape[rank - 1];
        }
    }
}
