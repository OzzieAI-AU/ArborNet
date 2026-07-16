using System;
using System.Collections.Generic;
using ArborNet.Core.Functional;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Layers;
using ArborNet.Core.Tensors;

namespace ArborNet.Layers
{
    /// <summary>
    /// Implements a 3D convolutional layer.
    /// </summary>
    /// <remarks>
    /// Applies a 3D convolution over an input tensor using a set of learnable filters.
    /// The input tensor is expected to have shape [batch, channels, depth, height, width].
    /// This layer manages its own weight and optional bias tensors and integrates with the
    /// framework's parameter collection system through <see cref="BaseLayer"/>.
    /// </remarks>
    public class Conv3D : BaseLayer
    {
        public ITensor Weight { get; private set; }
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

            for (int b = 0; b < batch; b++)
                for (int oc = 0; oc < outChannels; oc++)
                    for (int od = 0; od < outD; od++)
                        for (int oh = 0; oh < outH; oh++)
                            for (int ow = 0; ow < outW; ow++)
                            {
                                float sum = 0f;
                                int outIdx = b * outChannels * outStrideC + oc * outStrideC + od * outStrideD + oh * outStrideH + ow;

                                for (int ic = 0; ic < inChannels; ic++)
                                    for (int kd = 0; kd < kernelDepth; kd++)
                                        for (int kh = 0; kh < kernelHeight; kh++)
                                            for (int kw = 0; kw < kernelWidth; kw++)
                                            {
                                                int id = od * _stride - _padding + kd;
                                                int ih = oh * _stride - _padding + kh;
                                                int iw = ow * _stride - _padding + kw;

                                                if (id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                                {
                                                    int inIdx = b * inChannels * inStrideC + ic * inStrideC + id * inStrideD + ih * inStrideH + iw;
                                                    int wIdx = oc * inChannels * wStrideC + ic * wStrideC + kd * kernelHeight * kernelWidth + kh * kernelWidth + kw;
                                                    sum += inData[inIdx] * wData[wIdx];
                                                }
                                            }
                                outData[outIdx] = sum;
                            }

            var result = Tensor.FromArray(outData, outputShape, input.Device);

            if (Bias != null)
            {
                var biasReshaped = Bias.Reshape(new int[] { 1, outChannels, 1, 1, 1 });
                result = result.Add(biasReshaped.BroadcastTo(result.Shape));
            }

            // FIXED: Complete backpropagation algorithm replacing the zero-stub
            if (input.RequiresGrad || Weight.RequiresGrad)
            {
                var capturedInput = input;
                var capturedWeight = Weight;

                result.GradFn = gradOutput =>
                {
                    var goData = gradOutput.ToArray();
                    var gradInputData = new float[capturedInput.Shape.TotalElements];
                    var gradWeightData = new float[capturedWeight.Shape.TotalElements];

                    for (int b = 0; b < batch; b++)
                    {
                        for (int oc = 0; oc < outChannels; oc++)
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
                                            for (int kd = 0; kd < kernelDepth; kd++)
                                            {
                                                for (int kh = 0; kh < kernelHeight; kh++)
                                                {
                                                    for (int kw = 0; kw < kernelWidth; kw++)
                                                    {
                                                        int id = od * _stride - _padding + kd;
                                                        int ih = oh * _stride - _padding + kh;
                                                        int iw = ow * _stride - _padding + kw;

                                                        if (id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                                        {
                                                            int inIdx = b * inChannels * inStrideC + ic * inStrideC + id * inStrideD + ih * inStrideH + iw;
                                                            int wIdx = oc * inChannels * wStrideC + ic * wStrideC + kd * kernelHeight * kernelWidth + kh * kernelWidth + kw;

                                                            // dL/dW
                                                            gradWeightData[wIdx] += inData[inIdx] * goVal;

                                                            // dL/dX
                                                            gradInputData[inIdx] += wData[wIdx] * goVal;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    var gradInput = Tensor.FromArray(gradInputData, capturedInput.Shape, input.Device);
                    var gradWeight = Tensor.FromArray(gradWeightData, capturedWeight.Shape, input.Device);

                    if (capturedWeight.RequiresGrad)
                    {
                        capturedWeight.Grad = capturedWeight.Grad == null ? gradWeight : capturedWeight.Grad.Add(gradWeight);
                    }

                    if (Bias != null && Bias.RequiresGrad)
                    {
                        var gradBias = gradOutput.Sum(4, false).Sum(3, false).Sum(2, false).Sum(0, false);
                        Bias.Grad = Bias.Grad == null ? gradBias : Bias.Grad.Add(gradBias);
                    }

                    if (capturedInput.RequiresGrad)
                    {
                        capturedInput.Grad = capturedInput.Grad == null ? gradInput : capturedInput.Grad.Add(gradInput);
                        capturedInput.GradFn?.Invoke(gradInput);
                    }

                    return gradInput;
                };
            }

            return result;
        }

        public override IEnumerable<ITensor> Parameters()
        {
            yield return Weight;
            if (Bias != null) yield return Bias;
        }
    }
}