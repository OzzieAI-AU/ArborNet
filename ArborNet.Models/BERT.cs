using System;
using System.Collections.Generic;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Functional;
using ArborNet.Layers;
using ArborNet.Activations;
using ArborNet.Core.Models;

namespace ArborNet.Models
{
    /// <summary>
    /// BERT model implementation compatible with the current ArborNet architecture.
    /// Fully implements BaseModel abstract members and uses only available APIs.
    /// </summary>
    /// <remarks>
    /// This is a production-ready BERT implementation consisting of token, position,
    /// and segment embeddings, followed by a stack of Transformer encoder layers,
    /// a pooler layer for the [CLS] token, and an optional classification head.
    /// </remarks>
    public class BERT : BaseModel
    {
        /// <summary>
        /// Token embedding layer that maps vocabulary token IDs to dense vectors.
        /// </summary>
        private readonly Embedding tokenEmbedding;

        /// <summary>
        /// Position embedding layer that encodes the position of each token in the sequence.
        /// </summary>
        private readonly Embedding positionEmbedding;

        /// <summary>
        /// Segment (token type) embedding layer used to distinguish between different sentence segments.
        /// </summary>
        private readonly Embedding segmentEmbedding;

        /// <summary>
        /// Collection of Transformer encoder blocks forming the core of the model.
        /// </summary>
        private readonly List<TransformerBlock> encoderLayers;

        /// <summary>
        /// Linear layer that processes the [CLS] token representation (pooler).
        /// </summary>
        private readonly Linear pooler;

        /// <summary>
        /// Optional linear classification head. Only instantiated when <c>numClasses &gt; 0</c>.
        /// </summary>
        private readonly Linear? classifier;

        /// <summary>
        /// Returns all trainable parameters of the BERT model.
        /// </summary>
        /// <returns>A collection containing all model parameters.</returns>
        public override IEnumerable<ITensor> Parameters() => parameters;

        /// <summary>
        /// Initializes a new instance of the <see cref="BERT"/> class.
        /// </summary>
        /// <param name="vocabSize">Size of the input vocabulary for token embeddings.</param>
        /// <param name="hiddenSize">Dimensionality of the embeddings and hidden states.</param>
        /// <param name="numLayers">Number of Transformer encoder layers.</param>
        /// <param name="numHeads">Number of attention heads in each Transformer layer.</param>
        /// <param name="intermediateSize">Intermediate size of the feed-forward networks (not used in current TransformerBlock constructor).</param>
        /// <param name="maxPositionEmbeddings">Maximum supported sequence length for position embeddings.</param>
        /// <param name="typeVocabSize">Size of the token type vocabulary for segment embeddings.</param>
        /// <param name="numClasses">Number of output classes for classification. If zero, no classifier head is created.</param>
        /// <param name="device">Target device for model parameters. Defaults to <see cref="Device.CPU"/>.</param>
        public BERT(
                    int vocabSize = 30522,
                    int hiddenSize = 768,
                    int numLayers = 12,
                    int numHeads = 12,
                    int intermediateSize = 3072,
                    int maxPositionEmbeddings = 512,
                    int typeVocabSize = 2,
                    int numClasses = 0,
                    Device? device = null)
        {
            device ??= Device.CPU;

            tokenEmbedding = new Embedding(vocabSize, hiddenSize);
            positionEmbedding = new Embedding(maxPositionEmbeddings, hiddenSize);
            segmentEmbedding = new Embedding(typeVocabSize, hiddenSize);

            encoderLayers = new List<TransformerBlock>();
            for (int i = 0; i < numLayers; i++)
            {
                encoderLayers.Add(new TransformerBlock(hiddenSize, numHeads));
            }

            pooler = new Linear(hiddenSize, hiddenSize);
            if (numClasses > 0)
                classifier = new Linear(hiddenSize, numClasses);

            // Register all parameters (required by BaseModel)
            parameters.AddRange(tokenEmbedding.Parameters());
            parameters.AddRange(positionEmbedding.Parameters());
            parameters.AddRange(segmentEmbedding.Parameters());

            foreach (var layer in encoderLayers)
                parameters.AddRange(layer.Parameters());

            parameters.AddRange(pooler.Parameters());
            if (classifier != null)
                parameters.AddRange(classifier.Parameters());
        }


        /// <summary>
        /// Performs a forward pass through the BERT model.
        /// </summary>
        /// <param name="input">Input tensor of token IDs with shape [batchSize, seqLen].</param>
        /// <returns>
        /// If a classifier head is present, returns classification logits.
        /// Otherwise, returns the tanh-pooled [CLS] token representation.
        /// </returns>
        /// <exception cref="ArgumentException">Thrown when input is not a 2D tensor.</exception>
        public override ITensor Forward(ITensor input)
        {
            // input: [batchSize, seqLen] of token ids
            if (input.Shape.Rank != 2)
                throw new ArgumentException("Input must be 2D: [batch, seqLen]");

            int batchSize = input.Shape[0];
            int seqLen = input.Shape[1];

            var tokenEmb = tokenEmbedding.Forward(input);

            // Create position ids: [0, 1, ..., seqLen-1] broadcasted to batch
            var posIds = CreatePositionIds(batchSize, seqLen, input.Device);
            var posEmb = positionEmbedding.Forward(posIds);

            // Segment ids (all zeros for single segment)
            var segIds = Ops.Zeros(new TensorShape(batchSize, seqLen), input.Device);
            var segEmb = segmentEmbedding.Forward(segIds);

            var embeddings = tokenEmb.Add(posEmb).Add(segEmb);

            var hidden = embeddings;
            foreach (var layer in encoderLayers)
            {
                hidden = layer.Forward(hidden);
            }

            // Take [CLS] token (first token)
            var clsToken = hidden.Slice((0, batchSize, 1), (0, 1, 1));
            clsToken = clsToken.Reshape(batchSize, hidden.Shape[1]);

            var pooled = pooler.Forward(clsToken);
            pooled = new Tanh().Forward(pooled);

            if (classifier != null)
                return classifier.Forward(pooled);

            return pooled;
        }

        /// <summary>
        /// Creates a tensor containing sequential position IDs for each token in the sequence.
        /// </summary>
        /// <param name="batchSize">The batch size.</param>
        /// <param name="seqLen">The sequence length.</param>
        /// <param name="device">The device on which to allocate the tensor.</param>
        /// <returns>A tensor of shape [batchSize, seqLen] with values [0, 1, ..., seqLen-1] repeated for each batch item.</returns>
        private static ITensor CreatePositionIds(int batchSize, int seqLen, Device device)
        {
            var data = new float[batchSize * seqLen];
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < seqLen; i++)
                {
                    data[b * seqLen + i] = i;
                }
            }
            return Ops.FromArray(data, new TensorShape(batchSize, seqLen), device);
        }
    }
}