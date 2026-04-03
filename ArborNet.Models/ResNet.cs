using System;
using System.Collections.Generic;
using System.Linq;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Models;
using ArborNet.Layers;
using ArborNet.Activations;

namespace ArborNet.Models
{
    /// <summary>
    /// Implements the ResNet architecture from "Deep Residual Learning for Image Recognition"
    /// (He et al., 2015). Supports multiple variants including ResNet-18, ResNet-34, 
    /// ResNet-50, ResNet-101, and ResNet-152.
    /// </summary>
    /// <remarks>
    /// The model consists of a convolutional stem, four stages of residual blocks 
    /// (BasicBlock or BottleneckBlock), adaptive average pooling, and a final 
    /// fully-connected classification head.
    /// </remarks>
    public class ResNet : BaseModel
    {
        /// <summary>
        /// The sequence of layers that constitute the ResNet model.
        /// </summary>
        private readonly List<ILayer> _layers = new();
        
        /// <summary>
        /// The final fully-connected layer for classification.
        /// </summary>
        private readonly Linear _fc;
        
        /// <summary>
        /// Indicates whether bottleneck blocks should be used (true for ResNet-50 and deeper).
        /// </summary>
        private readonly bool _isBottleneck;

        /// <summary>
        /// Initializes a new instance of the <see cref="ResNet"/> class.
        /// </summary>
        /// <param name="numClasses">Number of output classes. Default is 1000 (ImageNet).</param>
        /// <param name="variant">ResNet variant to construct. Supported values: "ResNet18", "ResNet34", 
        /// "ResNet50", "ResNet101", "ResNet152". Defaults to "ResNet18".</param>
        /// <param name="device">Target computation device. If null, defaults to <see cref="Device.CPU"/>.</param>
        public ResNet(int numClasses = 1000, string variant = "ResNet18", Device? device = null)
        {
            device ??= Device.CPU;
            _isBottleneck = variant.Contains("50") || variant.Contains("101") || variant.Contains("152");

            int[] layerConfig = variant switch
            {
                "ResNet18" => new[] { 2, 2, 2, 2 },
                "ResNet34" => new[] { 3, 4, 6, 3 },
                "ResNet50" => new[] { 3, 4, 6, 3 },
                "ResNet101" => new[] { 3, 4, 23, 3 },
                "ResNet152" => new[] { 3, 8, 36, 3 },
                _ => new[] { 2, 2, 2, 2 }
            };

            int inChannels = 64;

            // Stem: Note the added 'device' arguments
            _layers.Add(new Conv2D(3, 64, 7, 2, 3, false));
            _layers.Add(new BatchNorm(64));
            _layers.Add(new ActivationLayer(new ReLU())); // Now works if ReLU inherits BaseLayer
            _layers.Add(new MaxPool2D(3, 2, 1));

            inChannels = MakeLayer(64, layerConfig[0], _isBottleneck ? 4 : 1, inChannels, 1, device);
            inChannels = MakeLayer(128, layerConfig[1], _isBottleneck ? 4 : 1, inChannels, 2, device);
            inChannels = MakeLayer(256, layerConfig[2], _isBottleneck ? 4 : 1, inChannels, 2, device);
            inChannels = MakeLayer(512, layerConfig[3], _isBottleneck ? 4 : 1, inChannels, 2, device);

            _layers.Add(new AdaptiveAvgPool2D(1));
            _fc = new Linear(512 * (_isBottleneck ? 4 : 1), numClasses, device);

            parameters.AddRange(_layers.SelectMany(l => l.Parameters()));
            parameters.AddRange(_fc.Parameters());
        }

        /// <summary>
        /// Creates a stage consisting of multiple residual blocks.
        /// </summary>
        /// <param name="planes">Base number of output channels for the blocks in this stage.</param>
        /// <param name="blocks">Number of residual blocks to create in this stage.</param>
        /// <param name="expansion">Channel expansion factor (1 for BasicBlock, 4 for BottleneckBlock).</param>
        /// <param name="inChannels">Number of input channels from the previous stage.</param>
        /// <param name="stride">Stride used for the first block in the stage (for downsampling).</param>
        /// <param name="device">Target computation device for the layers.</param>
        /// <returns>The number of output channels after this stage.</returns>
        private int MakeLayer(int planes, int blocks, int expansion, int inChannels, int stride, Device device)
        {
            for (int i = 0; i < blocks; i++)
            {
                int s = (i == 0) ? stride : 1;
                // Explicitly cast to the shared base type (BaseLayer or ILayer)
                ILayer block = _isBottleneck
                    ? (ILayer)new BottleneckBlock(inChannels, planes, s, expansion, device)
                    : (ILayer)new BasicBlock(inChannels, planes, s, expansion, device);
                _layers.Add(block); _layers.Add(block);
                inChannels = planes * expansion;
            }
            return inChannels;
        }

        /// <summary>
        /// Performs a forward pass through the ResNet model.
        /// </summary>
        /// <param name="x">Input tensor of shape [batch, channels, height, width].</param>
        /// <returns>Output tensor of shape [batch, numClasses] containing class logits.</returns>
        /// <exception cref="ArgumentException">Thrown when the input tensor is not 4-dimensional.</exception>
        public override ITensor Forward(ITensor x)
        {
            if (x.Shape.Rank != 4)
                throw new ArgumentException("Input must be 4D [B, C, H, W]");

            foreach (var layer in _layers)
                x = layer.Forward(x);

            x = x.Reshape(x.Shape[0], -1);
            return _fc.Forward(x);
        }

        /// <summary>
        /// Returns all trainable parameters in the model.
        /// </summary>
        /// <returns>Collection of all parameter tensors from all layers and the final classifier.</returns>
        public override IEnumerable<ITensor> Parameters() => parameters;
    }

    /// <summary>
    /// Basic residual block used in ResNet-18 and ResNet-34.
    /// Consists of two 3x3 convolutions with a shortcut connection.
    /// </summary>
    public class BasicBlock : BaseLayer
    {
        /// <summary>
        /// First convolution layer in the residual block.
        /// </summary>
        private readonly Conv2D conv1, conv2;
        
        /// <summary>
        /// Batch normalization layers corresponding to the convolutions.
        /// </summary>
        private readonly BatchNorm bn1, bn2;
        
        /// <summary>
        /// Optional downsampling convolution for shortcut connection when dimensions differ.
        /// </summary>
        private readonly Conv2D? downsample;

        /// <summary>
        /// Initializes a new instance of the <see cref="BasicBlock"/> class.
        /// </summary>
        /// <param name="inChannels">Number of input channels.</param>
        /// <param name="planes">Number of output channels for the main path convolutions.</param>
        /// <param name="stride">Stride for the first convolution (used for spatial downsampling).</param>
        /// <param name="expansion">Channel expansion factor (typically 1 for basic blocks).</param>
        /// <param name="device">Target computation device.</param>
        public BasicBlock(int inChannels, int planes, int stride, int expansion, Device device)
        {
            conv1 = new Conv2D(inChannels, planes, 3, stride, 1);
            bn1 = new BatchNorm(planes);
            conv2 = new Conv2D(planes, planes, 3, 1, 1);
            bn2 = new BatchNorm(planes);

            if (stride != 1 || inChannels != planes)
                downsample = new Conv2D(inChannels, planes, 1, stride, 0);
        }

        /// <summary>
        /// Executes the forward pass of the basic residual block.
        /// </summary>
        /// <param name="x">Input tensor.</param>
        /// <returns>Output tensor after residual connection and ReLU activation.</returns>
        public override ITensor Forward(ITensor x)
        {
            var residual = x;
            x = new ReLU().Forward(bn1.Forward(conv1.Forward(x)));
            x = bn2.Forward(conv2.Forward(x));
            if (downsample != null) residual = downsample.Forward(residual);
            return new ReLU().Forward(x.Add(residual));
        }

        /// <summary>
        /// Returns all trainable parameters contained in this block.
        /// </summary>
        /// <returns>Enumerable collection of all weight and bias tensors.</returns>
        public override IEnumerable<ITensor> Parameters()
        {
            // FIXED: Yield tensors from layer.Parameters(), not the layer objects themselves (Conv2D is ILayer, not ITensor)
            foreach (var p in conv1.Parameters()) yield return p;
            foreach (var p in bn1.Parameters()) yield return p;
            foreach (var p in conv2.Parameters()) yield return p;
            foreach (var p in bn2.Parameters()) yield return p;
            if (downsample != null)
                foreach (var p in downsample.Parameters()) yield return p;
        }
    }

    /// <summary>
    /// Bottleneck residual block used in ResNet-50, ResNet-101, and ResNet-152.
    /// Uses 1x1, 3x3, and 1x1 convolutions with an expansion factor of 4.
    /// </summary>
    public class BottleneckBlock : BaseLayer
    {
        /// <summary>
        /// First 1x1 convolution that reduces channel dimension.
        /// </summary>
        private readonly Conv2D conv1, conv2, conv3;
        
        /// <summary>
        /// Batch normalization layers for each convolution in the bottleneck.
        /// </summary>
        private readonly BatchNorm bn1, bn2, bn3;
        
        /// <summary>
        /// Optional downsampling layer for the residual connection when needed.
        /// </summary>
        private readonly Conv2D? downsample;

        /// <summary>
        /// Initializes a new instance of the <see cref="BottleneckBlock"/> class.
        /// </summary>
        /// <param name="inChannels">Number of input channels.</param>
        /// <param name="planes">Base number of channels (bottleneck width).</param>
        /// <param name="stride">Stride for the 3x3 convolution.</param>
        /// <param name="expansion">Channel expansion factor (typically 4 for bottleneck blocks).</param>
        /// <param name="device">Target computation device.</param>
        public BottleneckBlock(int inChannels, int planes, int stride, int expansion, Device device)
        {
            conv1 = new Conv2D(inChannels, planes, 1, 1, 0);
            bn1 = new BatchNorm(planes);
            conv2 = new Conv2D(planes, planes, 3, stride, 1);
            bn2 = new BatchNorm(planes);
            conv3 = new Conv2D(planes, planes * expansion, 1, 1, 0);
            bn3 = new BatchNorm(planes * expansion);

            if (stride != 1 || inChannels != planes * expansion)
                downsample = new Conv2D(inChannels, planes * expansion, 1, stride, 0);
        }

        /// <summary>
        /// Executes the forward pass of the bottleneck residual block.
        /// </summary>
        /// <param name="x">Input tensor.</param>
        /// <returns>Output tensor after residual addition and final ReLU activation.</returns>
        public override ITensor Forward(ITensor x)
        {
            var residual = x;
            x = new ReLU().Forward(bn1.Forward(conv1.Forward(x)));
            x = new ReLU().Forward(bn2.Forward(conv2.Forward(x)));
            x = bn3.Forward(conv3.Forward(x));
            if (downsample != null) residual = downsample.Forward(residual);
            return new ReLU().Forward(x.Add(residual));
        }

        /// <summary>
        /// Returns all trainable parameters contained in this block.
        /// </summary>
        /// <returns>Enumerable collection of all weight and bias tensors.</returns>
        public override IEnumerable<ITensor> Parameters()
        {
            // FIXED: Yield tensors from layer.Parameters(), not the layer objects themselves
            foreach (var p in conv1.Parameters()) yield return p;
            foreach (var p in bn1.Parameters()) yield return p;
            foreach (var p in conv2.Parameters()) yield return p;
            foreach (var p in bn2.Parameters()) yield return p;
            foreach (var p in conv3.Parameters()) yield return p;
            foreach (var p in bn3.Parameters()) yield return p;
            if (downsample != null)
                foreach (var p in downsample.Parameters()) yield return p;
        }
    }

    /// <summary>
    /// 2D max pooling layer.
    /// </summary>
    /// <remarks>
    /// Current implementation is a placeholder. A production version would delegate 
    /// to the appropriate backend implementation.
    /// </remarks>
    public class MaxPool2D : BaseLayer
    {
        /// <summary>
        /// Size of the pooling kernel.
        /// </summary>
        private readonly int kernelSize, stride, padding;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MaxPool2D"/> class.
        /// </summary>
        /// <param name="kernelSize">Size of the pooling window.</param>
        /// <param name="stride">Stride between pooling windows.</param>
        /// <param name="padding">Zero-padding added to the input borders.</param>
        public MaxPool2D(int kernelSize = 2, int stride = 2, int padding = 0) 
            => (this.kernelSize, this.stride, this.padding) = (kernelSize, stride, padding);
        
        /// <summary>
        /// Performs the max pooling operation.
        /// </summary>
        /// <param name="x">Input tensor.</param>
        /// <returns>Output tensor after max pooling.</returns>
        public override ITensor Forward(ITensor x) => x; // production impl would use backend maxpool
        
        /// <summary>
        /// Returns the parameters of this layer (none for pooling).
        /// </summary>
        /// <returns>Empty collection of tensors.</returns>
        public override IEnumerable<ITensor> Parameters() => Array.Empty<ITensor>();
    }

    /// <summary>
    /// Adaptive average pooling layer that reduces spatial dimensions to a fixed output size.
    /// </summary>
    public class AdaptiveAvgPool2D : BaseLayer
    {
        /// <summary>
        /// Desired output spatial size.
        /// </summary>
        private readonly int outputSize;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="AdaptiveAvgPool2D"/> class.
        /// </summary>
        /// <param name="outputSize">Target height and width after pooling (default 1 for global pooling).</param>
        public AdaptiveAvgPool2D(int outputSize = 1) => this.outputSize = outputSize;
        
        /// <summary>
        /// Performs adaptive average pooling by computing mean across spatial dimensions.
        /// </summary>
        /// <param name="x">Input tensor.</param>
        /// <returns>Tensor with reduced spatial dimensions.</returns>
        public override ITensor Forward(ITensor x) => x.Mean(new[] { -1, -2 });
        
        /// <summary>
        /// Returns the parameters of this layer (none for pooling).
        /// </summary>
        /// <returns>Empty collection of tensors.</returns>
        public override IEnumerable<ITensor> Parameters() => Array.Empty<ITensor>();
    }
}