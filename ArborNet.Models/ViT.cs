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

    using ArborNet.Core.Functional;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;
    using System.Collections.Generic;
    /// <summary>
    /// Implements the Vision Transformer (ViT) architecture for image classification tasks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This implementation follows the architecture described in the paper:
    /// <see href="https://arxiv.org/abs/2010.11929">"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"</see>.
    /// </para>
    /// <para>
    /// Input images are split into fixed-size patches, linearly projected (via a 2D convolution),
    /// prepended with a learnable class token, augmented with learnable 1D position embeddings,
    /// and then passed through a standard Transformer Encoder stack. The class token embedding at 
    /// the output of the transformer is used as the representative vector for the classification head.
    /// </para>
    /// </remarks>

    #endregion

    public class ViT : BaseModel
    {
        /// <summary>
        /// Convolutional layer that projects image patches into embedding vectors.
        /// </summary>
        private readonly Conv2D patchEmbed;

        /// <summary>
        /// Learnable class token prepended to the sequence of patch embeddings.
        /// </summary>
        private readonly ITensor classToken;

        /// <summary>
        /// Positional encoding added to the combined class token and patch embeddings.
        /// </summary>
        private readonly PositionalEncoding posEmbed;

        /// <summary>
        /// Collection of transformer encoder blocks (layers).
        /// </summary>
        private readonly List<TransformerBlock> blocks;

        /// <summary>
        /// Final linear classification head that maps the class token embedding to class logits.
        /// </summary>
        private readonly Linear head;
        /// <summary>
        /// Retrieves all trainable parameters associated with the Vision Transformer model, 
        /// including layer weights, biases, and learnable embeddings.
        /// </summary>
        /// <returns>An enumerable collection of parameter tensors (<see cref="ITensor"/>).</returns>

        public override IEnumerable<ITensor> Parameters() => parameters;

        /// <summary>
        /// Initializes a new instance of the <see cref="ViT"/> class.
        /// </summary>
        /// <param name="imageSize">Height and width of the square input images.</param>
        /// <param name="patchSize">Size of each square patch the image is divided into.</param>
        /// <param name="inChannels">Number of input channels (typically 3 for RGB images).</param>
        /// <param name="embedDim">Embedding dimension for patches and transformer hidden size.</param>
        /// <param name="numHeads">Number of attention heads in each transformer block.</param>
        /// <param name="numLayers">Number of transformer blocks in the encoder.</param>
        /// <param name="numClasses">Number of output classes for classification.</param>
        public ViT(int imageSize = 224, int patchSize = 16, int inChannels = 3, int embedDim = 768,
                   int numHeads = 12, int numLayers = 12, int numClasses = 1000)
        {
            patchEmbed = new Conv2D(inChannels, embedDim, patchSize, patchSize);
            classToken = Tensor.Ones(new TensorShape(1, 1, embedDim));
            posEmbed = new PositionalEncoding(embedDim, (imageSize / patchSize) * (imageSize / patchSize) + 1);
            blocks = new List<TransformerBlock>();
            for (int i = 0; i < numLayers; i++)
                blocks.Add(new TransformerBlock(embedDim, numHeads));
            head = new Linear(embedDim, numClasses);

            parameters.AddRange(patchEmbed.Parameters());
            parameters.Add(classToken);
            parameters.AddRange(posEmbed.Parameters());
            foreach (var b in blocks)
                parameters.AddRange(b.Parameters());
            parameters.AddRange(head.Parameters());
        }
        /// <summary>
        /// Executes the forward pass of the Vision Transformer model.
        /// </summary>
        /// <param name="x">The input image tensor of shape <c>(BatchSize, Channels, Height, Width)</c>.</param>
        /// <returns>The computed class logits tensor of shape <c>(BatchSize, NumClasses)</c>.</returns>
        /// <remarks>
        /// The input images are first projected to sequence patches. A learnable class token is 
        /// prepended, positional encodings are added, and the tokens are passed through the transformer encoder blocks.
        /// Finally, the class token representation is sliced out and passed to the classification head.
        /// </remarks>

        public override ITensor Forward(ITensor x)
        {
            int batchSize = x.Shape[0];
            int seqLen = x.Shape[1] * x.Shape[2]; // after patch embedding

            x = patchEmbed.Forward(x);
            x = x.Reshape(batchSize, seqLen, x.Shape[1]);
            x = x.Transpose(new[] { 0, 2, 1 });

            // Repeat class token for batch
            var clsTokens = classToken.Reshape(1, 1, classToken.Shape[1])
                .BroadcastTo(new TensorShape(batchSize, 1, classToken.Shape[1]));

            x = Ops.Concat(new[] { clsTokens, x }, axis: 1);
            x = posEmbed.Forward(x);

            foreach (var block in blocks)
                x = block.Forward(x);

            // Take CLS token
            var cls = x.Slice((0, batchSize, 1), (0, 1, 1), (0, x.Shape[2], 1)).Reshape(batchSize, x.Shape[2]);
            return head.Forward(cls);
        }
    }
}