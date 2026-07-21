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

    #region Using Statements:

    using ArborNet.Core.Activations;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Functional;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;
    using System.Collections.Generic;
    using System.Linq;

    #endregion

    /// <summary>
    /// Implements the Kimi Delta Attention (KDA) hybrid linear/quadratic mechanism.
    /// Resolves long-context dependencies in O(N) complexity using projected positive state-kernels.
    /// </summary>
    public sealed class DeltaAttention : BaseLayer
    {
        private readonly int _dModel;
        private readonly int _nHeads;
        private readonly int _dHead;
        private readonly ITensor _wq, _wk, _wv, _wo;
        private readonly SiTU _situ;

        public DeltaAttention(int dModel, int nHeads, Device device)
        {
            _dModel = dModel;
            _nHeads = nHeads;
            _dHead = dModel / nHeads;
            this.device = device;

            _wq = Initializers.XavierUniform(new TensorShape(dModel, dModel), device);
            _wk = Initializers.XavierUniform(new TensorShape(dModel, dModel), device);
            _wv = Initializers.XavierUniform(new TensorShape(dModel, dModel), device);
            _wo = Initializers.XavierUniform(new TensorShape(dModel, dModel), device);

            _wq.RequiresGrad = _wk.RequiresGrad = _wv.RequiresGrad = _wo.RequiresGrad = true;
            _situ = new SiTU();
        }

        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input);
            int batch = input.Shape[0];
            int seqLen = input.Shape[1];

            // Flatten 3D input to 2D for linear projection compatibility
            var flatInput = input.Reshape(batch * seqLen, _dModel);

            var Q_all = flatInput.MatMul(_wq).Reshape(batch, _nHeads, seqLen, _dHead);
            var K_all = flatInput.MatMul(_wk).Reshape(batch, _nHeads, seqLen, _dHead);
            var V_all = flatInput.MatMul(_wv).Reshape(batch, _nHeads, seqLen, _dHead);

            var qFeat = _situ.Forward(Q_all);
            var kFeat = _situ.Forward(K_all);

            var batchOutputs = new List<ITensor>();

            // Calculate O(N) linear attention slice-by-slice to prevent 3D/4D MatMul exceptions
            for (int b = 0; b < batch; b++)
            {
                var headOutputs = new List<ITensor>();
                for (int h = 0; h < _nHeads; h++)
                {
                    // Slice 4D maps to 2D spaces: [seqLen, dHead]
                    var q_bh = qFeat.Slice((b, b + 1, 1), (h, h + 1, 1), (0, seqLen, 1), (0, _dHead, 1)).Reshape(seqLen, _dHead);
                    var k_bh = kFeat.Slice((b, b + 1, 1), (h, h + 1, 1), (0, seqLen, 1), (0, _dHead, 1)).Reshape(seqLen, _dHead);
                    var v_bh = V_all.Slice((b, b + 1, 1), (h, h + 1, 1), (0, seqLen, 1), (0, _dHead, 1)).Reshape(seqLen, _dHead);

                    var kt_bh = k_bh.Transpose(new[] { 1, 0 }); // [dHead, seqLen]
                    var kvContext = kt_bh.MatMul(v_bh); // [dHead, dHead]

                    var numerator = q_bh.MatMul(kvContext); // [seqLen, dHead]
                    var kSum = k_bh.Sum(0, keepDims: true); // [1, dHead]

                    var denominator = q_bh.Multiply(kSum.BroadcastTo(q_bh.Shape)).Sum(1, keepDims: true).Add(1e-6f); // [seqLen, 1]

                    var out_bh = numerator.Divide(denominator.BroadcastTo(numerator.Shape)); // [seqLen, dHead]
                    headOutputs.Add(out_bh);
                }

                // Concat attention heads: [seqLen, dModel]
                var concatHeads = headOutputs[0].Concat(headOutputs.Skip(1), 1);
                batchOutputs.Add(concatHeads);
            }

            // Concat batch elements: [batch * seqLen, dModel]
            var flatOutput = batchOutputs[0].Concat(batchOutputs.Skip(1), 0);

            // Out-projection mapping to 3D shape: [batch, seqLen, dModel]
            return flatOutput.MatMul(_wo).Reshape(batch, seqLen, _dModel);
        }

        public override IEnumerable<ITensor> Parameters()
        {
            yield return _wq;
            yield return _wk;
            yield return _wv;
            yield return _wo;
        }
    }
}