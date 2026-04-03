using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Models;
using ArborNet.Layers;
using System.Collections.Generic;

namespace ArborNet.Models
{
    /// <summary>
    /// Implements the ConvNeXt architecture, a modernized convolutional neural network 
    /// that achieves transformer-level performance using depthwise convolutions and 
    /// inverted bottlenecks.
    /// </summary>
    /// <remarks>
    /// ConvNeXt consists of a 4x4 stem convolution, four hierarchical stages of ConvNeXt blocks,
    /// downsampling layers between stages, and a final classification head. The design draws 
    /// inspiration from Vision Transformers while preserving the efficiency of CNNs.
    /// </remarks>
    public class ConvNeXt : BaseModel
    {
        /// <summary>
        /// Initial stem convolution that converts the 3-channel input image into 
        /// patch embeddings using a 4x4 kernel with stride 4.
        /// </summary>
        private readonly Conv2D stem;

        /// <summary>
        /// Layer normalization applied after the stem convolution.
        /// </summary>
        private readonly LayerNorm stemNorm;

        /// <summary>
        /// Four stages of the network. Each stage contains a list of ConvNeXt blocks
        /// with consistent channel dimensions within the stage.
        /// </summary>
        private readonly List<List<ConvNeXtBlock>> stages;

        /// <summary>
        /// Downsampling convolution layers used between the four stages (3 downsampling layers total).
        /// Each uses a 2x2 kernel with stride 2.
        /// </summary>
        private readonly List<Conv2D> downsamples;

        /// <summary>
        /// Layer normalization applied before the classification head.
        /// </summary>
        private readonly LayerNorm headNorm;

        /// <summary>
        /// Final linear classification layer that projects the pooled features to class logits.
        /// </summary>
        private readonly Linear head;

        /// <summary>
        /// Initializes a new instance of the <see cref="ConvNeXt"/> class.
        /// </summary>
        /// <param name="numClasses">The number of output classes for the classification head.</param>
        /// <param name="depths">Array of 4 integers specifying the number of ConvNeXt blocks in each stage.</param>
        /// <param name="dims">Array of 4 integers specifying the channel dimensions for each of the 4 stages.</param>
        /// <param name="device">The computation device to place model parameters on. Defaults to <see cref="Device.CPU"/> if null.</param>
        public ConvNeXt(int numClasses, int[] depths, int[] dims, Device device = null)
        {
            device ??= Device.CPU;
            stem = new Conv2D(3, dims[0], 4, 4, 0);
            stemNorm = new LayerNorm(new[] { dims[0] });
            stages = new List<List<ConvNeXtBlock>>();
            downsamples = new List<Conv2D>();

            for (int i = 0; i < 4; i++)
            {
                var stage = new List<ConvNeXtBlock>();
                for (int j = 0; j < depths[i]; j++)
                    stage.Add(new ConvNeXtBlock(dims[i]));
                stages.Add(stage);
            }

            for (int i = 1; i < 4; i++)
                downsamples.Add(new Conv2D(dims[i - 1], dims[i], 2, 2, 0));

            headNorm = new LayerNorm(new[] { dims[3] });
            head = new Linear(dims[3], numClasses);

            // Populating the inherited 'parameters' list directly
            parameters.AddRange(stem.Parameters());
            parameters.AddRange(stemNorm.Parameters());
            foreach (var stage in stages)
                foreach (var b in stage)
                    parameters.AddRange(b.Parameters());
            foreach (var ds in downsamples)
                parameters.AddRange(ds.Parameters());
            parameters.AddRange(headNorm.Parameters());
            parameters.AddRange(head.Parameters());
        }

        /// <summary>
        /// Performs a forward pass through the ConvNeXt model.
        /// </summary>
        /// <param name="x">The input tensor of shape (batch_size, 3, height, width).</param>
        /// <returns>The output logits tensor of shape (batch_size, numClasses).</returns>
        /// <remarks>
        /// The forward pass applies the stem, processes through four hierarchical stages 
        /// (with downsampling between stages), applies final layer normalization, performs 
        /// global average pooling over the spatial dimensions, and finally applies the linear head.
        /// </remarks>
        public override ITensor Forward(ITensor x)
        {
            x = stem.Forward(x);
            x = stemNorm.Forward(x);
            for (int i = 0; i < 4; i++)
            {
                if (i > 0)
                    x = downsamples[i - 1].Forward(x);
                foreach (var block in stages[i])
                    x = block.Forward(x);
            }
            x = headNorm.Forward(x);

            // Global average pooling over spatial dimensions (height and width)
            x = x.Mean(-1).Mean(-1);

            return head.Forward(x);
        }
    }
}