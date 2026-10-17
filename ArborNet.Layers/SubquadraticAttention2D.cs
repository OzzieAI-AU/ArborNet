namespace ArborNet.Layers
{
    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;

    /// <summary>
    /// Pure 2D Linear Causal Attention.
    /// Operates entirely within the computational autograd tape using continuous Rank-2 matrix multiplications.
    /// Completely eliminates O(N^2) token-token dot product matrices.
    /// </summary>
    public sealed class SubquadraticAttention2D : BaseLayer
    {
        private readonly int _dModel;
        private readonly Linear _qProj;
        private readonly Linear _kProj;
        private readonly Linear _vProj;
        private readonly Linear _oProj;
        private readonly ITensor _causalTril;

        public SubquadraticAttention2D(int dModel, int maxWindow, Device device)
        {
            this.device = device;
            _dModel = dModel;

            _qProj = new Linear(dModel, dModel, device);
            _kProj = new Linear(dModel, dModel, device);
            _vProj = new Linear(dModel, dModel, device);
            _oProj = new Linear(dModel, dModel, device);

            // Precompute Lower-Triangular Causal Decay Matrix: L[i, j] = 1 if j <= i else 0
            float[] tril = new float[maxWindow * maxWindow];
            for (int i = 0; i < maxWindow; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    tril[i * maxWindow + j] = 1.0f;
                }
            }

            _causalTril = Tensor.FromArray(tril, new TensorShape(maxWindow, maxWindow), device);
            _causalTril.RequiresGrad = false;
        }

        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input);
            int T = input.Shape[0];

            // 1. Differentiable Projections: [T, D]
            var Q = _qProj.Forward(input);
            var K = _kProj.Forward(input);
            var V = _vProj.Forward(input);

            // 2. Kernel Activation: phi(x) = Relu(x) + 1e-4f (Positive kernel map for linear attention)
            var qPhi = Q.Relu().Add(1e-4f);
            var kPhi = K.Relu().Add(1e-4f);

            // 3. Slice Precomputed Causal Mask: [T, T]
            var tril = (T == _causalTril.Shape[0])
                ? _causalTril
                : _causalTril.Slice((0, T, 1), (0, T, 1));

            // 4. Causally Sum Keys: [T, D] = [T, T] @ [T, D]
            var kSum = tril.MatMul(kPhi);

            // 5. Normalizer denominator: q_t * sum_{j <= t} k_j
            var normalizer = qPhi.Multiply(kSum).Sum(-1, keepDims: true).Add(1e-6f);

            // 6. Causal Value Accumulation: Weighted KV representations
            var kvElement = kPhi.Multiply(V);
            var causalNumerator = tril.MatMul(kvElement);
            var context = qPhi.Multiply(causalNumerator).Divide(normalizer.BroadcastTo(causalNumerator.Shape));

            // 7. Output Projection Head
            return _oProj.Forward(context);
        }

        public override IEnumerable<ITensor> Parameters()
        {
            foreach (var p in _qProj.Parameters()) yield return p;
            foreach (var p in _kProj.Parameters()) yield return p;
            foreach (var p in _vProj.Parameters()) yield return p;
            foreach (var p in _oProj.Parameters()) yield return p;
        }
    }
}