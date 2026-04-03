using ArborNet.Core.Interfaces;
using ArborNet.Core.Models;
using ArborNet.Layers;
using System.Collections.Generic;
using System.Linq;

namespace ArborNet.Models
{
    /// <summary>
    /// Implements a U-Net style convolutional neural network architecture.
    /// </summary>
    /// <remarks>
    /// This model follows an encoder-decoder structure with skip connections via element-wise addition.
    /// The encoder progressively downsamples the feature maps while increasing the number of channels.
    /// The decoder reduces the channel depth and restores spatial information using learned convolutions
    /// and residual connections from the encoder path.
    /// </remarks>
    public class UNet : BaseModel
    {
        /// <summary>
        /// Convolutional layers primarily used in the encoder and bottleneck stages.
        /// </summary>
        private readonly Conv2D conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9;

        /// <summary>
        /// Convolutional layers used in the decoder/upsampling path.
        /// </summary>
        private readonly Conv2D up1, up2, up3, up4;

        /// <summary>
        /// Returns all trainable parameters from the model.
        /// </summary>
        /// <returns>An enumerable collection of all <see cref="ITensor"/> parameters used by the network.</returns>
        public override IEnumerable<ITensor> Parameters() => parameters;

        /// <summary>
        /// Initializes a new instance of the <see cref="UNet"/> class.
        /// </summary>
        /// <param name="inChannels">Number of channels in the input tensor. Default is 3 (RGB).</param>
        /// <param name="outChannels">Number of channels in the output tensor. Default is 1.</param>
        /// <param name="features">Base number of feature maps in the first convolution layer. Default is 64.</param>
        public UNet(int inChannels = 3, int outChannels = 1, int features = 64)
        {
            conv1 = new Conv2D(inChannels, features, 3, 1, 1);
            conv2 = new Conv2D(features, features * 2, 3, 2, 1);
            conv3 = new Conv2D(features * 2, features * 4, 3, 2, 1);
            conv4 = new Conv2D(features * 4, features * 8, 3, 2, 1);
            conv5 = new Conv2D(features * 8, features * 8, 3, 1, 1);
            conv6 = new Conv2D(features * 8, features * 4, 3, 1, 1);
            conv7 = new Conv2D(features * 4, features * 2, 3, 1, 1);
            conv8 = new Conv2D(features * 2, features, 3, 1, 1);
            conv9 = new Conv2D(features, outChannels, 1, 1, 0);

            up1 = new Conv2D(features * 8, features * 4, 3, 1, 1);
            up2 = new Conv2D(features * 4, features * 2, 3, 1, 1);
            up3 = new Conv2D(features * 2, features, 3, 1, 1);
            up4 = new Conv2D(features, features, 3, 1, 1);

            parameters.AddRange(new[] { conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, up1, up2, up3, up4 }.SelectMany(l => l.Parameters()));
        }

        /// <summary>
        /// Performs a forward pass of the input through the U-Net architecture.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The output tensor after passing through the encoder-decoder network with skip connections.</returns>
        /// <remarks>
        /// Architecture flow:
        /// <list type="bullet">
        ///   <item>Encoder: conv1 → conv2 → conv3 → conv4 → conv5 (bottleneck)</item>
        ///   <item>Decoder with skips: conv6 + e4, conv7 + e3, conv8 + e2, followed by up4 and final conv9.</item>
        /// </list>
        /// All intermediate activations use ReLU.
        /// </remarks>
        public override ITensor Forward(ITensor x)
        {
            var e1 = conv1.Forward(x).Relu();
            var e2 = conv2.Forward(e1).Relu();
            var e3 = conv3.Forward(e2).Relu();
            var e4 = conv4.Forward(e3).Relu();
            var b = conv5.Forward(e4).Relu();
            var d4 = conv6.Forward(b).Relu().Add(e4);
            var d3 = conv7.Forward(d4).Relu().Add(e3);
            var d2 = conv8.Forward(d3).Relu().Add(e2);
            var d1 = conv9.Forward(up4.Forward(d2).Relu());
            return d1;
        }
    }
}