using ArborNet.Core;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Models;
using ArborNet.Fluent;
using ArborNet.Core.Initializers;
using ArborNet.Layers.Fractal;
using System;
using System.Collections.Generic;

namespace ArborNet.Models
{
    /// <summary>
    /// An advanced Large Language Model combining structured Fractal Weight Initialization
    /// with O(N) Subquadratic Sparse Attention. Completely native to ArborNet.
    /// </summary>
    public class FractalLLM : BaseModel
    {
        private readonly FractalLinear _tokenEmbedding;
        private readonly FractalLinear _outputHead;
        private readonly List<FractalTransformerBlock> _layers;
        private readonly int _maxSeqLen;

        // Dynamic variance modulator state
        private readonly float _varianceSignatureIntensity = 15.0f;

        public FractalLLM(int vocabSize, int nLayers, int nHeads, int dModel, int dFF, int maxSeqLen, FractalType initType)
        {
            if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
            if (nLayers <= 0) throw new ArgumentOutOfRangeException(nameof(nLayers));
            if (dModel % nHeads != 0) throw new ArgumentException("dModel must be divisible by nHeads.");

            _maxSeqLen = maxSeqLen;

            // We use FractalLinear without bias as a high-speed Embedding lookup alternative 
            // via One-Hot to Dense projection, ensuring weights are fractally bound.
            _tokenEmbedding = new FractalLinear(vocabSize, dModel, initType, false);

            _layers = new List<FractalTransformerBlock>();
            for (int i = 0; i < nLayers; i++)
            {
                _layers.Add(new FractalTransformerBlock(dModel, dFF, nHeads, initType));
            }

            _outputHead = new FractalLinear(dModel, vocabSize, initType, false);

            // Register all Autograd Parameters to BaseModel
            parameters.AddRange(_tokenEmbedding.Parameters());
            foreach (var layer in _layers) parameters.AddRange(layer.Parameters());
            parameters.AddRange(_outputHead.Parameters());
        }

        /// <summary>
        /// Highly expressive Fluent Forward Pass.
        /// </summary>
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            // Assume input is one-hot encoded sequence: [batch, seqLen, vocabSize]
            var x = X.Of(input);

            // 1. Fractal Token Embedding
            x = x.Apply(_tokenEmbedding);

            // 2. Transformer Blocks (Subquadratic Attention + FFN)
            foreach (var block in _layers)
            {
                x = x.Apply(block);
            }

            // 3. Final Layer Norm
            x = x.LayerNorm();

            // 4. Custom Variance Modulation (As seen in the original FractalLLMT script)
            // We apply a mathematically precise modulation just before the logits
            x = ApplyVarianceModulator(x.Tensor);

            // 5. Output Projection to Vocabulary space
            var logits = x.Apply(_outputHead);

            return logits.Tensor;
        }

        /// <summary>
        /// Extracts dynamic variance and scales the token structural signature.
        /// Built entirely using native ArborNet ITensor primitives.
        /// </summary>
        private X ApplyVarianceModulator(ITensor stateVector)
        {
            // 1. Mean across the embedding dimension (axis -1)
            ITensor mean = stateVector.Mean(-1);

            // 2. Variance = Mean((x - mean)^2)
            ITensor diff = stateVector.Subtract(mean);
            ITensor diffSq = diff.Pow(2.0f);
            ITensor variance = diffSq.Mean(-1);

            // 3. Signature = variance * 15.0f
            // Convert the primitive float into an ArborNet scalar tensor first!
            ITensor intensityScalar = ArborNet.Core.Tensors.Tensor.FromScalar(_varianceSignatureIntensity);
            ITensor signature = variance.Multiply(intensityScalar);

            // 4. Apply structural signature mapping
            return (stateVector.Multiply(signature)).ToX();
        }

        public override IEnumerable<ITensor> Parameters() => parameters;
    }
}