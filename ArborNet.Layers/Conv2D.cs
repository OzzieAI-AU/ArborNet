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
    using ArborNet.Activations;
    using ArborNet.Core;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Core.Functional;
    using System.Threading.Tasks;
    using ArborNet.Core.Layers;
    /// <summary>
    /// Production-grade 2D convolutional layer with thread-safe atomic autograd support.
    /// Handles forward validation, multi-threaded CPU convolution, and registers 
    /// lock-free parallelized backward passes for autograd gradient accumulation.
    /// </summary>

    #endregion


    public class Conv2D : BaseLayer
    {
        private readonly int _inChannels;
        private readonly int _outChannels;
        private readonly int _kernelSize;
        private readonly int _stride;
        private readonly int _padding;
        private readonly bool _useBias;

        private readonly ITensor _weight;
        private readonly ITensor? _bias;

        public Conv2D(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0, bool useBias = true, Device? device = null)
        {
            _inChannels = inChannels;
            _outChannels = outChannels;
            _kernelSize = kernelSize;
            _stride = stride;
            _padding = padding;
            _useBias = useBias;

            var dev = device ?? Device.CPU;
            _weight = Initializers.XavierUniform(new TensorShape(outChannels, inChannels, kernelSize, kernelSize), dev);
            _weight.RequiresGrad = true;

            if (_useBias)
            {
                _bias = Tensor.Zeros(new TensorShape(outChannels), dev);
                _bias.RequiresGrad = true;
            }
        }
        /// <summary>
        /// Computes the forward pass of the 2D convolution operation and sets up the autograd backward graph if required.
        /// </summary>
        /// <param name="input">The 4D input tensor with shape <c>[Batch, Channels, Height, Width]</c>.</param>
        /// <returns>The output <see cref="ITensor"/> containing the convolved features.</returns>
        /// <exception cref="ArgumentException">Thrown if the input rank is not 4 or if the input channels do not match <see cref="_inChannels"/>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the computed spatial dimensions result in a non-positive size.</exception>

        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input, expectedRank: 4);
            if (input.Shape[1] != _inChannels)
                throw new ArgumentException($"Channel mismatch: Expected {_inChannels}, got {input.Shape[1]}.");

            int batch = input.Shape[0];
            int inH = input.Shape[2];
            int inW = input.Shape[3];

            int outH = (inH + 2 * _padding - _kernelSize) / _stride + 1;
            int outW = (inW + 2 * _padding - _kernelSize) / _stride + 1;

            if (outH <= 0 || outW <= 0)
                throw new InvalidOperationException("Invalid spatial configuration resulting in negative dimensions.");

            var outputShape = new TensorShape(batch, _outChannels, outH, outW);
            var inData = input.ToArray();
            var wData = _weight.ToArray();
            var outData = new float[outputShape.TotalElements];

            int inStrideC = inH * inW;
            int inStrideH = inW;
            int wStrideC = _kernelSize * _kernelSize;
            int outStrideC = outH * outW;
            int outStrideH = outW;

            Parallel.For(0, batch * _outChannels, idx =>
            {
                int b = idx / _outChannels;
                int oc = idx % _outChannels;

                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        float sum = 0f;
                        int outIdx = b * _outChannels * outStrideC + oc * outStrideC + oh * outStrideH + ow;

                        for (int ic = 0; ic < _inChannels; ic++)
                        {
                            int inChannelOffset = b * _inChannels * inStrideC + ic * inStrideC;
                            int wChannelOffset = oc * _inChannels * wStrideC + ic * wStrideC;

                            for (int kh = 0; kh < _kernelSize; kh++)
                            {
                                int ih = oh * _stride - _padding + kh;
                                if (ih < 0 || ih >= inH) continue;

                                int inRowOffset = inChannelOffset + ih * inStrideH;
                                int wRowOffset = wChannelOffset + kh * _kernelSize;

                                for (int kw = 0; kw < _kernelSize; kw++)
                                {
                                    int iw = ow * _stride - _padding + kw;
                                    if (iw >= 0 && iw < inW)
                                    {
                                        sum += inData[inRowOffset + iw] * wData[wRowOffset + kw];
                                    }
                                }
                            }
                        }
                        outData[outIdx] = sum;
                    }
                }
            });

            var result = Tensor.FromArray(outData, outputShape, input.Device);

            if (input.RequiresGrad || _weight.RequiresGrad)
            {
                var capturedInput = input;
                var capturedWeight = _weight;

                result.GradFn = gradOutput =>
                {
                    var goData = gradOutput.ToArray();
                    var gradInputData = new float[capturedInput.Shape.TotalElements];
                    var gradWeightData = new float[capturedWeight.Shape.TotalElements];

                    // FIXED: Lock-free weight gradient accumulation.
                    // Each thread 'oc' exclusively owns its segment of 'gradWeightData'.
                    Parallel.For(0, _outChannels, oc =>
                    {
                        for (int b = 0; b < batch; b++)
                        {
                            for (int oh = 0; oh < outH; oh++)
                            {
                                for (int ow = 0; ow < outW; ow++)
                                {
                                    int outIdx = b * _outChannels * outStrideC + oc * outStrideC + oh * outStrideH + ow;
                                    float goVal = goData[outIdx];

                                    for (int ic = 0; ic < _inChannels; ic++)
                                    {
                                        int wChannelOffset = oc * _inChannels * wStrideC + ic * wStrideC;
                                        for (int kh = 0; kh < _kernelSize; kh++)
                                        {
                                            int ih = oh * _stride - _padding + kh;
                                            if (ih < 0 || ih >= inH) continue;

                                            int wRowOffset = wChannelOffset + kh * _kernelSize;
                                            for (int kw = 0; kw < _kernelSize; kw++)
                                            {
                                                int iw = ow * _stride - _padding + kw;
                                                if (iw >= 0 && iw < inW)
                                                {
                                                    int inIdx = b * _inChannels * inStrideC + ic * inStrideC + ih * inStrideH + iw;
                                                    int wIdx = wRowOffset + kw;

                                                    gradWeightData[wIdx] += inData[inIdx] * goVal;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });

                    // FIXED: Lock-free input gradient accumulation.
                    // Each thread (b, ic) exclusively owns its segment of 'gradInputData'.
                    Parallel.For(0, batch * _inChannels, index =>
                    {
                        int b = index / _inChannels;
                        int ic = index % _inChannels;

                        for (int oc = 0; oc < _outChannels; oc++)
                        {
                            int wChannelOffset = oc * _inChannels * wStrideC + ic * wStrideC;
                            for (int oh = 0; oh < outH; oh++)
                            {
                                for (int ow = 0; ow < outW; ow++)
                                {
                                    int outIdx = b * _outChannels * outStrideC + oc * outStrideC + oh * outStrideH + ow;
                                    float goVal = goData[outIdx];

                                    for (int kh = 0; kh < _kernelSize; kh++)
                                    {
                                        int ih = oh * _stride - _padding + kh;
                                        if (ih < 0 || ih >= inH) continue;

                                        int wRowOffset = wChannelOffset + kh * _kernelSize;
                                        for (int kw = 0; kw < _kernelSize; kw++)
                                        {
                                            int iw = ow * _stride - _padding + kw;
                                            if (iw >= 0 && iw < inW)
                                            {
                                                int inIdx = b * _inChannels * inStrideC + ic * inStrideC + ih * inStrideH + iw;
                                                int wIdx = wRowOffset + kw;

                                                gradInputData[inIdx] += wData[wIdx] * goVal;
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

                    if (_useBias && _bias != null && _bias.RequiresGrad)
                    {
                        var gradBias = gradOutput.Sum(3, false).Sum(2, false).Sum(0, false);
                        _bias.AccumulateGrad(gradBias);
                    }

                    if (capturedInput.RequiresGrad)
                    {
                        capturedInput.AccumulateGrad(gradInput);
                        capturedInput.GradFn?.Invoke(gradInput);
                    }

                    return gradInput;
                };
            }

            if (_useBias && _bias != null)
            {
                var biasReshaped = _bias.Reshape(new int[] { 1, _outChannels, 1, 1 });
                result = result.Add(biasReshaped.BroadcastTo(result.Shape));
            }

            return result;
        }
        /// <summary>
        /// Enumerates all learnable parameters associated with this convolutional layer.
        /// </summary>
        /// <returns>An enumerable sequence containing the weight tensor and, if configured, the bias tensor.</returns>

        public override IEnumerable<ITensor> Parameters()
        {
            yield return _weight;
            if (_useBias && _bias != null) yield return _bias;
        }
    }
}