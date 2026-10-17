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

    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Initializers;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;

    #endregion

    /// <summary>
    /// Implements the Holonomic Looped GPT architecture.
    /// Replaces standard unrolled transformer depth with a weight-tied recurrent core
    /// that lifts hidden states into a higher-dimensional reasoning manifold (d_reason = 2x to 4x d_model),
    /// performs K iterative self-attention and non-linear reasoning passes with constant prompt injection,
    /// and projects back into d_model via a soft-Stiefel constrained right-inverse linear mapping.
    /// </summary>
    public sealed class HolonomicLoopedGPT : BaseModel
    {
        private readonly int _vocabSize;
        private readonly int _dModel;
        private readonly int _dReason;
        private readonly int _nHeadsReason;
        private readonly int _nLoops;
        private readonly int _maxSeqLen;
        private readonly float _lambdaOrtho;

        // Embedding infrastructure
        private readonly Embedding _tokenEmbedding;
        private readonly PositionalEncoding _posEmbedding;
        private readonly Embedding _stepEmbedding;

        // Manifold lift and projection fibers
        private readonly Linear _wUp;
        private readonly Linear _wDown;

        // Shared recurrent reasoning cell (Registered only ONCE)
        private readonly TransformerBlock _sharedBlock;

        // Output normalization and projection head
        private readonly LayerNorm _finalNorm;
        private readonly Linear _outputHead;

        public int NumLoops => _nLoops;
        public int DModel => _dModel;
        public int DReason => _dReason;

        /// <summary>
        /// Initializes a new instance of the <see cref="HolonomicLoopedGPT"/> architecture.
        /// </summary>
        /// <param name="vocabSize">Vocabulary size.</param>
        /// <param name="dModel">Input and output token representation dimension.</param>
        /// <param name="nHeads">Attention heads for the base model.</param>
        /// <param name="maxSeqLen">Maximum context sequence length.</param>
        /// <param name="nLoops">Number of recurrence cycles K through the shared transformer cell.</param>
        /// <param name="liftFactor">Dimension multiplier for higher-dimensional reasoning space (e.g. 2 or 4).</param>
        /// <param name="lambdaOrtho">Weight factor for the soft right-inverse orthogonality regularization.</param>
        /// <param name="device">Target computation device.</param>
        public HolonomicLoopedGPT(
            int vocabSize,
            int dModel = 768,
            int nHeads = 12,
            int maxSeqLen = 1024,
            int nLoops = 6,
            int liftFactor = 2,
            float lambdaOrtho = 0.01f,
            Device? device = null)
        {
            if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
            if (dModel <= 0 || dModel % nHeads != 0)
                throw new ArgumentException("dModel must be positive and divisible by nHeads.");
            if (nLoops <= 0) throw new ArgumentOutOfRangeException(nameof(nLoops));
            if (liftFactor < 1) throw new ArgumentOutOfRangeException(nameof(liftFactor));

            device ??= Device.CPU;
            base.device = device;

            _vocabSize = vocabSize;
            _dModel = dModel;
            _dReason = dModel * liftFactor;
            _nHeadsReason = nHeads * liftFactor;
            _nLoops = nLoops;
            _maxSeqLen = maxSeqLen;
            _lambdaOrtho = lambdaOrtho;

            // 1. Encodings
            _tokenEmbedding = new Embedding(vocabSize, dModel);
            _posEmbedding = new PositionalEncoding(dModel, maxSeqLen, device);
            _stepEmbedding = new Embedding(nLoops, _dReason);

            // 2. High-dimensional fiber lift & projection
            _wUp = new Linear(dModel, _dReason, device);
            _wDown = new Linear(_dReason, dModel, device);

            // 3. Single shared transformer reasoning cell
            _sharedBlock = new TransformerBlock(_dReason, _nHeadsReason);

            // 4. Final normalization and vocabulary projection
            _finalNorm = new LayerNorm(new[] { dModel });
            _outputHead = new Linear(dModel, vocabSize, device);

            RegisterUniqueParameters();
        }

        /// <summary>
        /// Guarantees that the shared transformer body and projection matrices are registered 
        /// strictly ONCE in the optimizer parameter graph, preventing gradient duplication.
        /// </summary>
        private void RegisterUniqueParameters()
        {
            parameters.Clear();
            var seen = new HashSet<ITensor>(ReferenceEqualityComparer.Instance);

            void AddUnique(IEnumerable<ITensor> tensors)
            {
                foreach (var t in tensors)
                {
                    if (t != null && seen.Add(t))
                    {
                        parameters.Add(t);
                    }
                }
            }

            AddUnique(_tokenEmbedding.Parameters());
            AddUnique(_posEmbedding.Parameters());
            AddUnique(_stepEmbedding.Parameters());
            AddUnique(_wUp.Parameters());
            AddUnique(_sharedBlock.Parameters());
            AddUnique(_wDown.Parameters());
            AddUnique(_finalNorm.Parameters());
            AddUnique(_outputHead.Parameters());
        }

        /// <summary>
        /// Executes the forward pass:
        /// 1. Embeds tokens and applies positional encodings: h0
        /// 2. Lifts to higher dimension: r0 = W_up(h0)
        /// 3. Loops through shared TransformerBlock K times with token residual injection:
        ///    r_{k+1} = TransformerBlock(r_k + W_up(h0) + StepEmb(k))
        /// 4. Projects back to d_model: h = h0 + W_down(r_K)
        /// 5. Computes logits: head(finalNorm(h))
        /// </summary>
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (input.Shape.Rank != 2)
                throw new ArgumentException("Input tensor must be 2D [batchSize, seqLen].", nameof(input));

            int batchSize = input.Shape[0];
            int seqLen = input.Shape[1];

            if (seqLen > _maxSeqLen)
                throw new ArgumentException($"Sequence length {seqLen} exceeds max sequence length {_maxSeqLen}.");

            // Input token bounds validation
            float[] tokenData = input.ToArray();
            for (int i = 0; i < tokenData.Length; i++)
            {
                int id = (int)tokenData[i];
                if (id < 0 || id >= _vocabSize)
                    throw new IndexOutOfRangeException($"Token ID {id} at index {i} is out of vocabulary bounds [0, {_vocabSize - 1}].");
            }

            // Step 1: Base embeddings
            var tokEmb = _tokenEmbedding.Forward(input);
            var h0 = _posEmbedding.Forward(tokEmb); // [B, T, dModel]

            // Step 2: Manifold lift
            var h0Flat = h0.Reshape(batchSize * seqLen, _dModel);
            var injectedUpFlat = _wUp.Forward(h0Flat); // [B*T, dReason]
            var r = injectedUpFlat.Reshape(batchSize, seqLen, _dReason);

            // Step 3: Recurrent reasoning loop
            for (int k = 0; k < _nLoops; k++)
            {
                // Retrieve learned step embedding for iteration k
                var stepIdTensor = Tensor.FromArray(new float[] { k }, new TensorShape(1), input.Device);
                var stepVec = _stepEmbedding.Forward(stepIdTensor).Reshape(1, 1, _dReason);
                var stepBroadcast = stepVec.BroadcastTo(r.Shape);

                // Inject original token state and step vector into recurrent state
                var injectedPrompt = injectedUpFlat.Reshape(batchSize, seqLen, _dReason);
                var rInput = r.Add(injectedPrompt).Add(stepBroadcast);

                // Reentrant forward execution through the single tied TransformerBlock
                r = _sharedBlock.Forward(rInput);
            }

            // Step 4: Projection down to d_model and residual outer skip
            var rFlat = r.Reshape(batchSize * seqLen, _dReason);
            var projectedDownFlat = _wDown.Forward(rFlat);
            var projectedDown = projectedDownFlat.Reshape(batchSize, seqLen, _dModel);

            var hFinal = h0.Add(projectedDown);

            // Step 5: Normalization and Output Head
            var normed = _finalNorm.Forward(hFinal);
            var flatNormed = normed.Reshape(batchSize * seqLen, _dModel);
            var logitsFlat = _outputHead.Forward(flatNormed);

            return logitsFlat.Reshape(batchSize, seqLen, _vocabSize);
        }

        /// <summary>
        /// Computes the soft right-inverse penalty: lambda * ||W_down @ W_up - I||_F^2
        /// Encourages the recurrent lifted manifold to project losslessly back to the input state.
        /// </summary>
        public ITensor ComputeOrthogonalityLoss()
        {
            var wUpWeight = _wUp.Parameters().GetEnumerator();
            var wDownWeight = _wDown.Parameters().GetEnumerator();

            if (!wUpWeight.MoveNext() || !wDownWeight.MoveNext())
                return Tensor.FromScalar(0f, device);

            var wU = wUpWeight.Current;     // [d_model, d_reason]
            var wD = wDownWeight.Current;   // [d_reason, d_model]

            // Product: [d_reason, d_reason] or [d_model, d_model]
            var product = wU.MatMul(wD);
            var identity = Tensor.Eye(product.Shape[0], device);

            var diff = product.Subtract(identity);
            var frobeniusNormSq = diff.Multiply(diff).Sum();

            return frobeniusNormSq.Multiply(_lambdaOrtho);
        }

        public override IEnumerable<ITensor> Parameters() => parameters;
    }
}
