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
    using ArborNet.Core.Functional;
    using ArborNet.Core.Initializers;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;

    #endregion

    public class ViT : BaseModel
    {
        private readonly Conv2D patchEmbed;
        private readonly ITensor classToken;
        private readonly PositionalEncoding posEmbed;
        private readonly List<TransformerBlock> blocks;
        private readonly Linear head;
        private readonly int _embedDim;
        private readonly int _numPatches;

        public override IEnumerable<ITensor> Parameters() => parameters;

        public ViT(int imageSize = 224, int patchSize = 16, int inChannels = 3, int embedDim = 768,
                   int numHeads = 12, int numLayers = 12, int numClasses = 1000, Device? device = null)
        {
            device ??= Device.CPU;
            base.device = device;
            _embedDim = embedDim;

            int gridH = imageSize / patchSize;
            int gridW = imageSize / patchSize;
            _numPatches = gridH * gridW;

            // 1. Convolutional patch projection
            patchEmbed = new Conv2D(inChannels, embedDim, patchSize, patchSize, 0, device: device);

            // 2. Trainable class token (RequiresGrad = true, initialized from normal distribution)
            classToken = Initializers.Normal(new TensorShape(1, 1, embedDim), device);
            classToken.RequiresGrad = true;

            // 3. Positional encoding covering [class_token + patches]
            posEmbed = new PositionalEncoding(embedDim, _numPatches + 1, device);

            // 4. Transformer encoder blocks
            blocks = new List<TransformerBlock>();
            for (int i = 0; i < numLayers; i++)
            {
                blocks.Add(new TransformerBlock(embedDim, numHeads));
            }

            // 5. Classification head
            head = new Linear(embedDim, numClasses, device);

            // Parameter registration
            parameters.Clear();
            parameters.AddRange(patchEmbed.Parameters());
            parameters.Add(classToken);
            parameters.AddRange(posEmbed.Parameters());
            foreach (var b in blocks)
                parameters.AddRange(b.Parameters());
            parameters.AddRange(head.Parameters());
        }

        public override ITensor Forward(ITensor x)
        {
            if (x.Shape.Rank != 4)
                throw new ArgumentException("ViT expects input shape [Batch, Channels, Height, Width].", nameof(x));

            int batchSize = x.Shape[0];

            // 1. Patch projection: [B, C, H, W] -> [B, embedDim, GridH, GridW]
            var patches = patchEmbed.Forward(x);
            int gridH = patches.Shape[2];
            int gridW = patches.Shape[3];
            int numPatches = gridH * gridW;

            // 2. Flatten spatial grid into sequence: [B, embedDim, numPatches] -> Transpose -> [B, numPatches, embedDim]
            var flatPatches = patches.Reshape(batchSize, _embedDim, numPatches).Transpose(new[] { 0, 2, 1 });

            // 3. Broadcast and prepend trainable class token: [1, 1, embedDim] -> [B, 1, embedDim]
            var clsTokens = classToken.BroadcastTo(new TensorShape(batchSize, 1, _embedDim));
            var sequence = clsTokens.Concat(new[] { flatPatches }, axis: 1); // [B, 1 + numPatches, embedDim]

            // 4. Add position encodings
            sequence = posEmbed.Forward(sequence);

            // 5. Transformer encoder stack
            foreach (var block in blocks)
            {
                sequence = block.Forward(sequence);
            }

            // 6. Slice the [CLS] token representation at index 0
            var clsOut = sequence.Slice((0, batchSize, 1), (0, 1, 1), (0, _embedDim, 1))
                                 .Reshape(batchSize, _embedDim);

            return head.Forward(clsOut);
        }
    }
}
