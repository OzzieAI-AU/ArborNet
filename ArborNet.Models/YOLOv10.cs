using ArborNet.Core.Interfaces;
using ArborNet.Core.Models;
using ArborNet.Layers;
using System.Collections.Generic;

namespace ArborNet.Models
{
    /// <summary>
    /// Implements a YOLOv10-style object detection model using a simplified architecture.
    /// </summary>
    /// <remarks>
    /// The model consists of an initial stem convolution, a stack of ConvNeXt blocks for feature extraction,
    /// and a final 1×1 convolutional head that predicts class scores and bounding box offsets (class + 4 box values).
    /// This implementation inherits from <see cref="BaseModel"/> and manages its own parameter collection.
    /// </remarks>
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
        /// Returns all trainable parameters of the model.
        /// </summary>
        /// <returns>An enumerable collection of all <see cref="ITensor"/> parameters used by this model.</returns>
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
        /// Performs a forward pass of the input through the YOLOv10 model.
        /// </summary>
        /// <param name="x">The input tensor, typically of shape (batch, 3, height, width).</param>
        /// <returns>The output tensor containing classification scores and bounding box predictions.</returns>
        public override ITensor Forward(ITensor x)
        {
            x = stem.Forward(x).Relu();
            foreach (var b in blocks)
                x = b.Forward(x);
            return head.Forward(x);
        }
    }
}