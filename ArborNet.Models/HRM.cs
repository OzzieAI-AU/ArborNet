// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Models
{
    #region Using Statements:

    using ArborNet.Activations;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Initializers;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;
    using System;
    using System.Collections.Generic;
    using System.Linq;

    #endregion

    /// <summary>
    /// Implements the Hierarchical Reasoning Model (HRM) architecture.
    /// Emulates the hierarchical and multi-timescale reasoning processes of the prefrontal cortex,
    /// combining a high-level strategic planning module with a low-level execution module.
    /// </summary>
    /// <remarks>
    /// This implementation uses only ArborNet native primitives and layers. It is fully device-aware 
    /// (CPU/CUDA), supports dynamic attention-stability halting, and integrates seamlessly with
    /// the framework's automatic differentiation (autograd) engine.
    /// 
    /// **2D Matrix Guaranteed:** All internal tensors are strictly flattened to 2D before linear projection 
    /// or scaled dot-product attention to ensure complete compatibility and faultless execution 
    /// with backend MatMul dimension requirements.
    /// </remarks>
    public sealed class HRM : BaseModel
    {
        private readonly int _vocabSize;
        private readonly int _dModel;
        private readonly int _nHeads;
        private readonly int _dHead;
        private readonly int _dFF;
        private readonly int _maxSeqLen;
        private readonly int _lCycles;
        private readonly float _dropoutRate;
        private readonly float _haltThreshold;

        // Embedding Layers
        private readonly Embedding _tokenEmbedding;
        private readonly Embedding _positionEmbedding;
        private readonly Embedding _puzzleEmbedding;

        // High-Level Module (Planning) - Native 2D projections
        private readonly ITensor _wqHigh, _wkHigh, _wvHigh, _woHigh;

        // Low-Level Module (Execution) - Native 2D projections
        private readonly ITensor _wqLow, _wkLow, _wvLow, _woLow;

        // Recurrent State Transitions
        private readonly ITensor _rnnWHigh, _rnnBHigh;
        private readonly ITensor _rnnWLow, _rnnBLow;

        // Feed-Forward Networks & Output Projections
        private readonly Linear _ffn1;
        private readonly Linear _ffn2;
        private readonly Linear _outputHead;

        // Regularization
        private readonly Dropout _dropout;
        private readonly Dropout _attnDropout;

        public bool IsTraining => isTraining;

        /// <summary>
        /// Initializes a new instance of the <see cref="HRM"/> model.
        /// </summary>
        public HRM(
            int vocabSize,
            int dModel,
            int nHeads,
            int dFF,
            int maxSeqLen,
            int lCycles = 8,
            float dropoutRate = 0.1f,
            float haltThreshold = 1e-4f,
            Device? device = null)
        {
            if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
            if (dModel <= 0) throw new ArgumentOutOfRangeException(nameof(dModel));
            if (nHeads <= 0 || dModel % nHeads != 0)
                throw new ArgumentException("dModel must be divisible by nHeads.");
            if (dFF <= 0) throw new ArgumentOutOfRangeException(nameof(dFF));
            if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
            if (lCycles <= 0) throw new ArgumentOutOfRangeException(nameof(lCycles));

            _vocabSize = vocabSize;
            _dModel = dModel;
            _nHeads = nHeads;
            _dHead = dModel / nHeads;
            _dFF = dFF;
            _maxSeqLen = maxSeqLen;
            _lCycles = lCycles;
            _dropoutRate = dropoutRate;
            _haltThreshold = haltThreshold;

            device ??= Device.CPU;
            this.currentDevice = device;

            // Initialize Embeddings
            _tokenEmbedding = new Embedding(vocabSize, dModel);
            _positionEmbedding = new Embedding(maxSeqLen, dModel);
            _puzzleEmbedding = new Embedding(vocabSize, dModel);

            // High-Level (Planning) Attention Weights
            _wqHigh = Initializers.XavierUniform(new TensorShape(dModel, dModel), device);
            _wkHigh = Initializers.XavierUniform(new TensorShape(dModel, dModel), device);
            _wvHigh = Initializers.XavierUniform(new TensorShape(dModel, dModel), device);
            _woHigh = Initializers.XavierUniform(new TensorShape(dModel, dModel), device);
            _wqHigh.RequiresGrad = _wkHigh.RequiresGrad = _wvHigh.RequiresGrad = _woHigh.RequiresGrad = true;

            // Low-Level (Execution) Attention Weights
            _wqLow = Initializers.XavierUniform(new TensorShape(dModel, dModel), device);
            _wkLow = Initializers.XavierUniform(new TensorShape(dModel, dModel), device);
            _wvLow = Initializers.XavierUniform(new TensorShape(dModel, dModel), device);
            _woLow = Initializers.XavierUniform(new TensorShape(dModel, dModel), device);
            _wqLow.RequiresGrad = _wkLow.RequiresGrad = _wvLow.RequiresGrad = _woLow.RequiresGrad = true;

            // Recurrent Weights
            _rnnWHigh = Initializers.XavierUniform(new TensorShape(dModel, dModel), device);
            _rnnBHigh = Tensor.Zeros(new TensorShape(dModel), device);
            _rnnWLow = Initializers.XavierUniform(new TensorShape(dModel, dModel), device);
            _rnnBLow = Tensor.Zeros(new TensorShape(dModel), device);
            _rnnWHigh.RequiresGrad = _rnnBHigh.RequiresGrad = _rnnWLow.RequiresGrad = _rnnBLow.RequiresGrad = true;

            // FFN & Output Layers
            _ffn1 = new Linear(dModel, dFF, device);
            _ffn2 = new Linear(dFF, dModel, device);
            _outputHead = new Linear(dModel, vocabSize, device);

            // Regularization Layers
            _dropout = new Dropout(dropoutRate);
            _attnDropout = new Dropout(dropoutRate);

            // Register Parameters for Auto-Optimization and Device Migration
            RegisterParameters();
        }

        private void RegisterParameters()
        {
            parameters.AddRange(_tokenEmbedding.Parameters());
            parameters.AddRange(_positionEmbedding.Parameters());
            parameters.AddRange(_puzzleEmbedding.Parameters());

            parameters.Add(_wqHigh);
            parameters.Add(_wkHigh);
            parameters.Add(_wvHigh);
            parameters.Add(_woHigh);

            parameters.Add(_wqLow);
            parameters.Add(_wkLow);
            parameters.Add(_wvLow);
            parameters.Add(_woLow);

            parameters.Add(_rnnWHigh);
            parameters.Add(_rnnBHigh);
            parameters.Add(_rnnWLow);
            parameters.Add(_rnnBLow);

            parameters.AddRange(_ffn1.Parameters());
            parameters.AddRange(_ffn2.Parameters());
            parameters.AddRange(_outputHead.Parameters());
        }

        /// <summary>
        /// Transitions the model and all nested layers into training mode.
        /// </summary>
        public override void Train()
        {
            base.Train();
            _tokenEmbedding.Train();
            _positionEmbedding.Train();
            _puzzleEmbedding.Train();
            _ffn1.Train();
            _ffn2.Train();
            _outputHead.Train();
            _dropout.Train();
            _attnDropout.Train();
        }

        /// <summary>
        /// Transitions the model and all nested layers into evaluation (inference) mode.
        /// </summary>
        public override void Eval()
        {
            base.Eval();
            _tokenEmbedding.Eval();
            _positionEmbedding.Eval();
            _puzzleEmbedding.Eval();
            _ffn1.Eval();
            _ffn2.Eval();
            _outputHead.Eval();
            _dropout.Eval();
            _attnDropout.Eval();
        }

        /// <summary>
        /// Executes the forward pass of the Hierarchical Reasoning Model.
        /// </summary>
        /// <param name="input">Input token IDs tensor with shape [batchSize, seqLen].</param>
        /// <returns>Computed logits tensor with shape [batchSize, seqLen, vocabSize].</returns>
        public override ITensor Forward(ITensor input)
        {
            return Forward(input, null);
        }

        /// <summary>
        /// Executes the forward pass of the HRM including optional puzzle metadata contexts.
        /// </summary>
        /// <param name="input">Input token IDs tensor with shape [batchSize, seqLen].</param>
        /// <param name="puzzle">Optional puzzle-grid spatial embedding input of shape [batchSize, seqLen].</param>
        /// <returns>Computed logits tensor with shape [batchSize, seqLen, vocabSize].</returns>
        /// <exception cref="ArgumentException">Thrown when input structural dimensions mismatch constraints.</exception>
        public ITensor Forward(ITensor input, ITensor? puzzle)
        {
            if (input.Shape.Rank != 2)
                throw new ArgumentException("Input must be a 2D tensor [batchSize, seqLen].", nameof(input));
            if (input.Shape[1] > _maxSeqLen)
                throw new ArgumentException($"Sequence length {input.Shape[1]} exceeds maximum limit {_maxSeqLen}.", nameof(input));

            int batchSize = input.Shape[0];
            int seqLen = input.Shape[1];

            // 1. Compute Multimodal Embeddings (Immediate flattening for 2D safety)
            var embeddings = _tokenEmbedding.Forward(input); // [batchSize, seqLen, dModel]
            var flatEmbeddings = embeddings.Reshape(batchSize * seqLen, _dModel);

            if (puzzle != null)
            {
                var puzzleEmb = _puzzleEmbedding.Forward(puzzle).Reshape(batchSize * seqLen, _dModel);
                flatEmbeddings = flatEmbeddings.Add(puzzleEmb);
            }

            // Apply Trainable Positional Embeddings
            var posIdsData = new float[batchSize * seqLen];
            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    posIdsData[b * seqLen + t] = t;
                }
            }
            var posIds = Tensor.FromArray(posIdsData, new TensorShape(batchSize, seqLen), input.Device);
            var posEmb = _positionEmbedding.Forward(posIds).Reshape(batchSize * seqLen, _dModel);

            flatEmbeddings = flatEmbeddings.Add(posEmb);

            // 2. Initialize Recurrent Latent States in strictly 2D structures
            var stateHigh = Tensor.Zeros(new TensorShape(batchSize * seqLen, _dModel), input.Device);
            var stateLow = Tensor.Zeros(new TensorShape(batchSize * seqLen, _dModel), input.Device);

            ITensor? prevAttnWeightsHigh = null;

            // Generate sequence causal masks on targeted computing devices
            var maskData = new float[seqLen * seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    maskData[i * seqLen + j] = j > i ? -1e9f : 0f;
                }
            }
            var causalMask2D = Tensor.FromArray(maskData, new TensorShape(seqLen, seqLen), input.Device);

            // 3. Multi-Timescale Recurrent Reasoning Loop
            for (int cycle = 0; cycle < _lCycles; cycle++)
            {
                // --- HIGH-LEVEL MODULE (Abstract Planning) ---
                var highInput = flatEmbeddings.Add(stateHigh); // [batch*seq, dModel]

                // Compute self-attention manually mapping 2D slices to bypass backend dimension restrictions
                var (highOutput, attnWeights) = Compute2DAttention(
                    highInput, batchSize, seqLen, _wqHigh, _wkHigh, _wvHigh, _woHigh, causalMask2D);

                // Planning convergence halting condition
                if (cycle > 0 && prevAttnWeightsHigh != null)
                {
                    var diff = attnWeights.Subtract(prevAttnWeightsHigh);
                    float attnDiff = diff.Pow(2f).Sum().ToScalar();

                    // Early exit if planning weights stabilize
                    if (attnDiff < _haltThreshold && cycle < 8)
                    {
                        break;
                    }
                }
                prevAttnWeightsHigh = attnWeights.Clone();

                // State transition (planning phase update - 2D Compatible)
                stateHigh = stateHigh.Add(highOutput)
                                     .MatMul(_rnnWHigh)
                                     .Add(_rnnBHigh.Reshape(1, _dModel).BroadcastTo(new TensorShape(batchSize * seqLen, _dModel)))
                                     .Tanh();

                if (IsTraining && _dropoutRate > 0)
                {
                    stateHigh = _dropout.Forward(stateHigh);
                }

                // --- LOW-LEVEL MODULE (Detailed Computations) ---
                var lowInput = flatEmbeddings.Add(stateHigh); // [batch*seq, dModel]

                var (lowOutput, _) = Compute2DAttention(
                    lowInput, batchSize, seqLen, _wqLow, _wkLow, _wvLow, _woLow, causalMask2D);

                // State transition (execution phase update - 2D Compatible)
                stateLow = stateLow.Add(lowOutput)
                                   .MatMul(_rnnWLow)
                                   .Add(_rnnBLow.Reshape(1, _dModel).BroadcastTo(new TensorShape(batchSize * seqLen, _dModel)))
                                   .Tanh();

                if (IsTraining && _dropoutRate > 0)
                {
                    stateLow = _dropout.Forward(stateLow);
                }
            }

            // 4. Feed-Forward Neural Network (FFN) -> All strictly operating on 2D projections
            var ffn1Out = _ffn1.Forward(stateLow).Tanh();
            if (IsTraining && _dropoutRate > 0)
            {
                ffn1Out = _dropout.Forward(ffn1Out);
            }
            var ffnOutput = _ffn2.Forward(ffn1Out);

            // 5. Output Projections (Logits Mapping)
            var logitsFlat = _outputHead.Forward(ffnOutput); // [batchSize * seqLen, vocabSize]

            // Restore batch sequence structural shapes safely
            var logits = logitsFlat.Reshape(batchSize, seqLen, _vocabSize);

            return logits;
        }

        /// <summary>
        /// Explicit 2D Scaled Dot-Product Attention pipeline preventing backend 4D MatMul crashes.
        /// Slices operations down to standard matrix multiplication dimensions.
        /// </summary>
        private (ITensor Output, ITensor AttnWeights) Compute2DAttention(
            ITensor flatInput, int batchSize, int seqLen,
            ITensor wq, ITensor wk, ITensor wv, ITensor wo, ITensor causalMask2D)
        {
            var Q_all = flatInput.MatMul(wq).Reshape(batchSize, seqLen, _nHeads, _dHead);
            var K_all = flatInput.MatMul(wk).Reshape(batchSize, seqLen, _nHeads, _dHead);
            var V_all = flatInput.MatMul(wv).Reshape(batchSize, seqLen, _nHeads, _dHead);

            var batchContexts = new List<ITensor>();
            var batchAttns = new List<ITensor>();

            float scale = MathF.Sqrt(_dHead);

            for (int b = 0; b < batchSize; b++)
            {
                var headContexts = new List<ITensor>();
                var headAttns = new List<ITensor>();

                for (int h = 0; h < _nHeads; h++)
                {
                    // Isolated 2D Sequence x Features mapping: [seqLen, dHead]
                    var q_bh = Q_all.Slice((b, b + 1, 1), (0, seqLen, 1), (h, h + 1, 1), (0, _dHead, 1)).Reshape(seqLen, _dHead);
                    var k_bh = K_all.Slice((b, b + 1, 1), (0, seqLen, 1), (h, h + 1, 1), (0, _dHead, 1)).Reshape(seqLen, _dHead);
                    var v_bh = V_all.Slice((b, b + 1, 1), (0, seqLen, 1), (h, h + 1, 1), (0, _dHead, 1)).Reshape(seqLen, _dHead);

                    var scores = q_bh.MatMul(k_bh.Transpose(new[] { 1, 0 })).Divide(scale); // [seqLen, seqLen]

                    if (causalMask2D != null)
                    {
                        scores = scores.Add(causalMask2D);
                    }

                    var attn_bh = scores.Softmax(-1);
                    if (IsTraining && _dropoutRate > 0)
                    {
                        attn_bh = _attnDropout.Forward(attn_bh);
                    }

                    var context_bh = attn_bh.MatMul(v_bh); // [seqLen, dHead]

                    headContexts.Add(context_bh);
                    headAttns.Add(attn_bh);
                }

                var concatHeads = headContexts[0];
                if (headContexts.Count > 1)
                    concatHeads = concatHeads.Concat(headContexts.Skip(1), 1); // [seqLen, dModel]
                batchContexts.Add(concatHeads);

                var concatAttns = headAttns[0];
                if (headAttns.Count > 1)
                    concatAttns = concatAttns.Concat(headAttns.Skip(1), 0); // [nHeads * seqLen, seqLen]
                batchAttns.Add(concatAttns);
            }

            var flatContext = batchContexts[0];
            if (batchContexts.Count > 1)
                flatContext = flatContext.Concat(batchContexts.Skip(1), 0); // [batch * seqLen, dModel]

            var output = flatContext.MatMul(wo); // [batch * seqLen, dModel]

            var flatAttns = batchAttns[0];
            if (batchAttns.Count > 1)
                flatAttns = flatAttns.Concat(batchAttns.Skip(1), 0); // [batch * nHeads * seqLen, seqLen]

            return (output, flatAttns);
        }

        /// <summary>
        /// Autoregressively generates next token predictions for each sequence batch in a single forward pass.
        /// </summary>
        /// <param name="input">Batch input token sequences shape [batchSize, seqLen].</param>
        /// <param name="puzzle">Optional puzzle grids background layout context shape [batchSize, seqLen].</param>
        /// <returns>A flat 1D tensor [batchSize] of predicted next-token indices.</returns>
        public ITensor GenerateNextToken(ITensor input, ITensor? puzzle = null)
        {
            var logits = Forward(input, puzzle); // [batchSize, seqLen, vocabSize]
            int batchSize = logits.Shape[0];
            int seqLen = logits.Shape[1];

            // Slice out final prediction index coordinates: [batch, 1, vocabSize]
            var lastLogits = logits.Slice(
                (0, batchSize, 1),
                (seqLen - 1, seqLen, 1),
                (0, _vocabSize, 1)
            );

            // Reshape to 2D [batchSize, vocabSize] and return ArgMax across columns
            return lastLogits.Reshape(batchSize, _vocabSize).ArgMax(-1);
        }
    }
}