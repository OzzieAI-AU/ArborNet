// -----------------------------------------------------------------------------------------
// Project:      ArborNet
// Description:  Autoregressive Text Generation Pipeline
// -----------------------------------------------------------------------------------------

namespace ArborNet.Generation
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Data;

    public class TextGenerator
    {
        private readonly IModel _model;
        private readonly ITokenizer _tokenizer;
        private readonly Device _device;
        private readonly int _maxContextLength;

        public TextGenerator(IModel model, ITokenizer tokenizer, Device device, int maxContextLength = 512)
        {
            _model = model;
            _tokenizer = tokenizer;
            _device = device;
            _maxContextLength = maxContextLength;

            // Ensure model is in inference mode (freezes dropout/batchnorm)
            _model.Eval();
        }

        /// <summary>
        /// Predicts text sequentially and yields tokens as they are generated.
        /// </summary>
        public IEnumerable<string> GenerateStream(
            string prompt,
            int maxTokensToGenerate = 100,
            float temperature = 0.7f,
            int topK = 50,
            float topP = 0.9f)
        {
            // 1. Encode prompt
            List<int> contextTokens = _tokenizer.Encode(prompt);

            for (int step = 0; step < maxTokensToGenerate; step++)
            {
                // Truncate to context window limits if necessary
                var window = contextTokens.Skip(Math.Max(0, contextTokens.Count - _maxContextLength)).ToList();
                int seqLen = window.Count;

                // Create input tensor: Shape [Batch=1, SeqLen]
                float[] inputData = window.Select(t => (float)t).ToArray();
                ITensor inputTensor = Tensor.FromArray(inputData, new TensorShape(1, seqLen), _device);

                // 2. Forward Pass through the LLM (KimiK3, Llama3, etc.)
                using (new TensorScope()) // Automatically clean up intermediate computation graph
                {
                    // Output shape: [1, SeqLen, VocabSize]
                    ITensor logits = _model.Forward(inputTensor);

                    // 3. Extract logits for the VERY LAST token in the sequence
                    int vocabSize = logits.Shape[2];
                    ITensor lastTokenLogits = logits.Slice(
                        (0, 1, 1),              // Batch 0
                        (seqLen - 1, seqLen, 1),// Last time step
                        (0, vocabSize, 1)       // All vocab logits
                    ).Reshape(vocabSize);

                    // 4. Sample the next token
                    int nextTokenId = Sampler.SampleToken(lastTokenLogits, temperature, topK, topP);

                    // Append to rolling context
                    contextTokens.Add(nextTokenId);

                    // Decode and yield the newly generated string chunk
                    string decodedPiece = _tokenizer.Decode(new List<int> { nextTokenId });

                    // Stop generation if we hit an End-Of-Sequence token (assuming ID 2 is EOS for this demo)
                    if (nextTokenId == 2)
                        yield break;

                    yield return decodedPiece;
                }
            }
        }
    }
}