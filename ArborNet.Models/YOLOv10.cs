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

    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Layers;
    using System.Collections.Generic;
    /// <summary>
    /// Implements a YOLOv10-style object detection model using a simplified, highly optimized architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The model consists of an initial stem convolution for spatial downsampling and channel expansion,
    /// a stack of ConvNeXt blocks for robust high-level feature extraction, and a final 1×1 convolutional
    /// head that predicts class scores and bounding box offsets (class logits + 4 box values per anchor/grid).
    /// </para>
    /// <para>
    /// This implementation inherits from <see cref="BaseModel"/> and manages its own internal parameter collection
    /// for training, tracking, and gradient-based optimization.
    /// </para>
    /// </remarks>

    #endregion

    public class YOLOv10 : BaseModel
    {
        /// <summary>
        /// The initial stem convolution layer that downsamples the input image.
        /// </summary>
        private readonly Conv2D stem;

        /// <summary>
        /// The list of ConvNeXt blocks forming the main feature extractor backbone.
        /// </summary>
        private readonly List<ConvNeXtBlock> blocks;

        /// <summary>
        /// The final 1×1 convolution head that outputs class logits and bounding box coordinates.
        /// </summary>
        private readonly Conv2D head;
        /// <summary>
        /// Returns all trainable parameters (weights, biases, etc.) of the model.
        /// </summary>
        /// <returns>An enumerable collection of all <see cref="ITensor"/> parameters utilized by this model's sub-layers.</returns>
        /// <remarks>
        /// This method aggregates the parameters of the <see cref="stem"/> layer, all sequential <see cref="blocks"/>, 
        /// and the prediction <see cref="head"/> layer to assist optimizer classes in updating weights.
        /// </remarks>

        public override IEnumerable<ITensor> Parameters() => parameters;

        /// <summary>
        /// Initializes a new instance of the <see cref="YOLOv10"/> class.
        /// </summary>
        /// <param name="numClasses">The number of object classes to predict. Defaults to 80 (COCO dataset).</param>
        public YOLOv10(int numClasses = 80)
        {
            stem = new Conv2D(3, 32, 3, 2, 1);
            blocks = new List<ConvNeXtBlock>();
            for (int i = 0; i < 12; i++)
                blocks.Add(new ConvNeXtBlock(32));
            head = new Conv2D(32, numClasses + 4, 1, 1, 0); // cls + box

            parameters.AddRange(stem.Parameters());
            foreach (var b in blocks) parameters.AddRange(b.Parameters());
            parameters.AddRange(head.Parameters());
        }
        /// <summary>
        /// Performs a forward pass of the input tensor through the YOLOv10 network architecture.
        /// </summary>
        /// <param name="x">The input tensor, typically representing an image batch with a shape of <c>(Batch, Channels, Height, Width)</c>.</param>
        /// <returns>
        /// A tensor of predicted values with shape <c>(Batch, Classes + 4, OutputHeight, OutputWidth)</c> containing raw class scores and bounding box regressions.
        /// </returns>
        /// <remarks>
        /// The forward propagation execution flow is as follows:
        /// <list type="number">
        /// <item><description>Pass the input through the initial <see cref="stem"/> layer to reduce spatial dimensions and increase channels, followed by a ReLU activation.</description></item>
        /// <item><description>Iteratively route the feature map through the 12 sequential <see cref="ConvNeXtBlock"/> backbone layers for feature enhancement.</description></item>
        /// <item><description>Compute final predictions using the 1x1 convolution <see cref="head"/> layer.</description></item>
        /// </list>
        /// </remarks>

        public override ITensor Forward(ITensor x)
        {
            x = stem.Forward(x).Relu();
            foreach (var b in blocks)
                x = b.Forward(x);
            return head.Forward(x);
        }
    }
}