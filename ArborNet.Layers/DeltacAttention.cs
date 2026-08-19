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
    using ArborNet.Core.Initializers;
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
    public sealed class DeltacAttention : BaseLayer
    {
        private readonly int _dModel;
        private readonly int _nHeads;
        private readonly int _dHead;
        private readonly ITensor _wq, _wk, _wv, _wo;
        private readonly SiTU _situ;

        public DeltacAttention(int dModel, int nHeads, Device device)
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

            var flatInput = input.Reshape(batch * seqLen, _dModel);

            var Q_all = flatInput.MatMul(_wq).Reshape(batch, seqLen, _nHeads, _dHead);
            var K_all = flatInput.MatMul(_wk).Reshape(batch, seqLen, _nHeads, _dHead);
            var V_all = flatInput.MatMul(_wv).Reshape(batch, seqLen, _nHeads, _dHead);

            var qFeat = _situ.Forward(Q_all);
            var kFeat = _situ.Forward(K_all);

            var batchOutputs = new List<ITensor>();

            // Calculate O(N) CAUSAL linear attention slice-by-slice
            for (int b = 0; b < batch; b++)
            {
                var headOutputs = new List<ITensor>();
                for (int h = 0; h < _nHeads; h++)
                {
                    var q_bh = qFeat.Slice((b, b + 1, 1), (0, seqLen, 1), (h, h + 1, 1), (0, _dHead, 1)).Reshape(seqLen, _dHead);
                    var k_bh = kFeat.Slice((b, b + 1, 1), (0, seqLen, 1), (h, h + 1, 1), (0, _dHead, 1)).Reshape(seqLen, _dHead);
                    var v_bh = V_all.Slice((b, b + 1, 1), (0, seqLen, 1), (h, h + 1, 1), (0, _dHead, 1)).Reshape(seqLen, _dHead);

                    var timeOutputs = new List<ITensor>();
                    ITensor kvContext = Tensor.Zeros(new TensorShape(_dHead, _dHead), device);
                    ITensor kSum = Tensor.Zeros(new TensorShape(1, _dHead), device);

                    for (int t = 0; t < seqLen; t++)
                    {
                        var q_t = q_bh.Slice((t, t + 1, 1), (0, _dHead, 1)); // [1, dHead]
                        var k_t = k_bh.Slice((t, t + 1, 1), (0, _dHead, 1)); // [1, dHead]
                        var v_t = v_bh.Slice((t, t + 1, 1), (0, _dHead, 1)); // [1, dHead]

                        // Causal accumulation: kvContext += k_t^T * v_t
                        var kv_t = k_t.Transpose(new[] { 1, 0 }).MatMul(v_t); // [dHead, dHead]
                        kvContext = kvContext.Add(kv_t);
                        kSum = kSum.Add(k_t);

                        // Projection num = q_t * kvContext
                        var num = q_t.MatMul(kvContext); // [1, dHead]
                        //var den = q_t.Multiply(kSum).Sum(1, keepDims: true).Add(1e-6f); // [1, 1]
                        // To strictly guarantee a positive denominator:
                        var den = q_t.Multiply(kSum).Sum(1, keepDims: true).Abs().Add(1e-6f);

                        var out_t = num.Divide(den.BroadcastTo(num.Shape)); // [1, dHead]
                        timeOutputs.Add(out_t);
                    }

                    var out_bh = timeOutputs[0].Concat(timeOutputs.Skip(1), 0); // [seqLen, dHead]
                    headOutputs.Add(out_bh);
                }

                var concatHeads = headOutputs[0].Concat(headOutputs.Skip(1), 1);
                batchOutputs.Add(concatHeads);
            }

            var flatOutput = batchOutputs[0].Concat(batchOutputs.Skip(1), 0);
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