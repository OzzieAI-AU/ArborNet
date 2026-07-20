// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Layers
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;
    using ArborNet.Core.Functional;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Implements a 3D convolutional layer.
    /// </summary>
    /// <remarks>
    /// Applies a 3D convolution over an input tensor using a set of learnable filters.
    /// The input tensor is expected to have shape [batch, channels, depth, height, width].
    /// This layer manages its own weight and optional bias tensors and integrates with the
    /// framework's parameter collection system through <see cref="BaseLayer"/>.
    /// </remarks>

    #endregion

    public class Conv3D : BaseLayer
    {
        /// <summary>
        /// Gets the learnable weights of the 3D convolutional kernel.
        /// </summary>
        /// <value>
        /// An <see cref="ITensor"/> of shape [outChannels, inChannels, kernelDepth, kernelHeight, kernelWidth].
        /// </value>
        public ITensor Weight { get; private set; }
        /// <summary>
        /// Gets the optional learnable bias tensor.
        /// </summary>
        /// <value>
        /// An <see cref="ITensor"/> of shape [outChannels], or <see langword="null"/> if bias is disabled.
        /// </value>
        public ITensor? Bias { get; private set; }

        private readonly int inChannels, outChannels, kernelDepth, kernelHeight, kernelWidth;
        private readonly int _stride;
        private readonly int _padding;

        public Conv3D(int inChannels, int outChannels, int kernelDepth, int kernelHeight, int kernelWidth,
                      bool hasBias = true, int stride = 1, int padding = 0)
        {
            this.inChannels = inChannels;
            this.outChannels = outChannels;
            this.kernelDepth = kernelDepth;
            this.kernelHeight = kernelHeight;
            this.kernelWidth = kernelWidth;
            _stride = stride;
            _padding = padding;

            Weight = Initializers.XavierUniform(new TensorShape(outChannels, inChannels, kernelDepth, kernelHeight, kernelWidth));
            Weight.RequiresGrad = true;

            if (hasBias)
            {
                Bias = Tensor.Zeros(new TensorShape(outChannels));
                Bias.RequiresGrad = true;
            }
        }
        /// <summary>
        /// Executes the forward pass of the 3D convolutional layer.
        /// </summary>
        /// <param name="input">The input tensor, expected to be 5D with shape [batch, channels, depth, height, width].</param>
        /// <returns>A new <see cref="ITensor"/> containing the activation outputs of the convolution.</returns>
        /// <exception cref="ArgumentException">
        /// Thrown if the <paramref name="input"/> is not 5D, or if its channel dimension does not match <see cref="inChannels"/>.
        /// </exception>
        /// <exception cref="InvalidOperationException">
        /// Thrown if the output dimensions (computed from input dimensions, kernel size, stride, and padding) are non-positive.
        /// </exception>

        public override ITensor Forward(ITensor input)
        {
            if (input.Shape.Rank != 5)
                throw new ArgumentException("Conv3D expects 5D input [B, C, D, H, W]");

            if (input.Shape[1] != inChannels)
                throw new ArgumentException($"Input channels ({input.Shape[1]}) does not match expected inChannels ({inChannels})");

            int batch = input.Shape[0];
            int inD = input.Shape[2];
            int inH = input.Shape[3];
            int inW = input.Shape[4];

            int outD = (inD + 2 * _padding - kernelDepth) / _stride + 1;
            int outH = (inH + 2 * _padding - kernelHeight) / _stride + 1;
            int outW = (inW + 2 * _padding - kernelWidth) / _stride + 1;

            if (outD <= 0 || outH <= 0 || outW <= 0)
                throw new InvalidOperationException("Output dimensions are non-positive. Check kernel/stride/padding.");

            var outputShape = new TensorShape(batch, outChannels, outD, outH, outW);
            float[] inData = input.ToArray();
            float[] wData = Weight.ToArray();
            float[] outData = new float[outputShape.TotalElements];

            int inStrideC = inD * inH * inW;
            int inStrideD = inH * inW;
            int inStrideH = inW;
            int wStrideC = kernelDepth * kernelHeight * kernelWidth;
            int outStrideC = outD * outH * outW;
            int outStrideD = outH * outW;
            int outStrideH = outW;

            // Fused parallel forward pass
            Parallel.For(0, batch * outChannels, idx =>
            {
                int b = idx / outChannels;
                int oc = idx % outChannels;

                for (int od = 0; od < outD; od++)
                {
                    for (int oh = 0; oh < outH; oh++)
                    {
                        for (int ow = 0; ow < outW; ow++)
                        {
                            float sum = 0f;
                            int outIdx = b * outChannels * outStrideC + oc * outStrideC + od * outStrideD + oh * outStrideH + ow;

                            for (int ic = 0; ic < inChannels; ic++)
                            {
                                int inChannelOffset = b * inChannels * inStrideC + ic * inStrideC;
                                int wChannelOffset = oc * inChannels * wStrideC + ic * wStrideC;

                                for (int kd = 0; kd < kernelDepth; kd++)
                                {
                                    int id = od * _stride - _padding + kd;
                                    if (id < 0 || id >= inD) continue;

                                    int inDepthOffset = inChannelOffset + id * inStrideD;
                                    int wDepthOffset = wChannelOffset + kd * kernelHeight * kernelWidth;

                                    for (int kh = 0; kh < kernelHeight; kh++)
                                    {
                                        int ih = oh * _stride - _padding + kh;
                                        if (ih < 0 || ih >= inH) continue;

                                        int inRowOffset = inDepthOffset + ih * inStrideH;
                                        int wRowOffset = wDepthOffset + kh * kernelWidth;

                                        for (int kw = 0; kw < kernelWidth; kw++)
                                        {
                                            int iw = ow * _stride - _padding + kw;
                                            if (iw >= 0 && iw < inW)
                                            {
                                                sum += inData[inRowOffset + iw] * wData[wRowOffset + kw];
                                            }
                                        }
                                    }
                                }
                            }
                            outData[outIdx] = sum;
                        }
                    }
                }
            });

            var result = Tensor.FromArray(outData, outputShape, input.Device);

            if (Bias != null)
            {
                var biasReshaped = Bias.Reshape(new int[] { 1, outChannels, 1, 1, 1 });
                result = result.Add(biasReshaped.BroadcastTo(result.Shape));
            }

            if (input.RequiresGrad || Weight.RequiresGrad)
            {
                var capturedInput = input;
                var capturedWeight = Weight;

                result.GradFn = gradOutput =>
                {
                    var goData = gradOutput.ToArray();
                    var gradInputData = new float[capturedInput.Shape.TotalElements];
                    var gradWeightData = new float[capturedWeight.Shape.TotalElements];

                    // Lock-free weight gradients partitioned by output channels
                    Parallel.For(0, outChannels, oc =>
                    {
                        for (int b = 0; b < batch; b++)
                        {
                            for (int od = 0; od < outD; od++)
                            {
                                for (int oh = 0; oh < outH; oh++)
                                {
                                    for (int ow = 0; ow < outW; ow++)
                                    {
                                        int outIdx = b * outChannels * outStrideC + oc * outStrideC + od * outStrideD + oh * outStrideH + ow;
                                        float goVal = goData[outIdx];

                                        for (int ic = 0; ic < inChannels; ic++)
                                        {
                                            int inChannelOffset = b * inChannels * inStrideC + ic * inStrideC;
                                            int wChannelOffset = oc * inChannels * wStrideC + ic * wStrideC;

                                            for (int kd = 0; kd < kernelDepth; kd++)
                                            {
                                                int id = od * _stride - _padding + kd;
                                                if (id < 0 || id >= inD) continue;

                                                int inDepthOffset = inChannelOffset + id * inStrideD;
                                                int wDepthOffset = wChannelOffset + kd * kernelHeight * kernelWidth;

                                                for (int kh = 0; kh < kernelHeight; kh++)
                                                {
                                                    int ih = oh * _stride - _padding + kh;
                                                    if (ih < 0 || ih >= inH) continue;

                                                    int inRowOffset = inDepthOffset + ih * inStrideH;
                                                    int wRowOffset = wDepthOffset + kh * kernelWidth;

                                                    for (int kw = 0; kw < kernelWidth; kw++)
                                                    {
                                                        int iw = ow * _stride - _padding + kw;
                                                        if (iw >= 0 && iw < inW)
                                                        {
                                                            int wIdx = wRowOffset + kw;
                                                            gradWeightData[wIdx] += inData[inRowOffset + iw] * goVal;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });

                    // Lock-free input gradients partitioned by batch and input channels
                    Parallel.For(0, batch * inChannels, index =>
                    {
                        int b = index / inChannels;
                        int ic = index % inChannels;

                        int inChannelOffset = b * inChannels * inStrideC + ic * inStrideC;

                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            int wChannelOffset = oc * inChannels * wStrideC + ic * wStrideC;

                            for (int od = 0; od < outD; od++)
                            {
                                for (int oh = 0; oh < outH; oh++)
                                {
                                    for (int ow = 0; ow < outW; ow++)
                                    {
                                        int outIdx = b * outChannels * outStrideC + oc * outStrideC + od * outStrideD + oh * outStrideH + ow;
                                        float goVal = goData[outIdx];

                                        for (int kd = 0; kd < kernelDepth; kd++)
                                        {
                                            int id = od * _stride - _padding + kd;
                                            if (id < 0 || id >= inD) continue;

                                            int inDepthOffset = inChannelOffset + id * inStrideD;
                                            int wDepthOffset = wChannelOffset + kd * kernelHeight * kernelWidth;

                                            for (int kh = 0; kh < kernelHeight; kh++)
                                            {
                                                int ih = oh * _stride - _padding + kh;
                                                if (ih < 0 || ih >= inH) continue;

                                                int inRowOffset = inDepthOffset + ih * inStrideH;
                                                int wRowOffset = wDepthOffset + kh * kernelWidth;

                                                for (int kw = 0; kw < kernelWidth; kw++)
                                                {
                                                    int iw = ow * _stride - _padding + kw;
                                                    if (iw >= 0 && iw < inW)
                                                    {
                                                        gradInputData[inRowOffset + iw] += wData[wRowOffset + kw] * goVal;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });

                    var gradInput = Tensor.FromArray(gradInputData, capturedInput.Shape, input.Device);
                    var gradWeight = Tensor.FromArray(gradWeightData, capturedWeight.Shape, input.Device);

                    if (capturedWeight.RequiresGrad)
                    {
                        capturedWeight.AccumulateGrad(gradWeight);
                    }

                    if (Bias != null && Bias.RequiresGrad)
                    {
                        var gradBias = gradOutput.Sum(4, false).Sum(3, false).Sum(2, false).Sum(0, false);
                        Bias.AccumulateGrad(gradBias);
                    }

                    if (capturedInput.RequiresGrad)
                    {
                        capturedInput.AccumulateGrad(gradInput);
                        capturedInput.GradFn?.Invoke(gradInput);
                    }

                    return gradInput;
                };
            }

            return result;
        }
        /// <summary>
        /// Enumerates all of the learnable parameter tensors associated with this layer.
        /// </summary>
        /// <returns>An enumerable collection containing the layer's <see cref="Weight"/>, and <see cref="Bias"/> (if configured).</returns>

        public override IEnumerable<ITensor> Parameters()
        {
            yield return Weight;
            if (Bias != null) yield return Bias;
        }
    }
}