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

    using ArborNet.Core.Backends;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Functional;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Native.PInvoke;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;
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
        private readonly int _inChannels;
        private readonly int _outChannels;
        private readonly int _kDepth;
        private readonly int _kHeight;
        private readonly int _kWidth;
        private readonly int _stride;
        private readonly int _padding;
        private readonly bool _useBias;

        private readonly ITensor _weight;
        private readonly ITensor? _bias;

        public Conv3D(int inChannels, int outChannels, int kernelDepth, int kernelHeight, int kernelWidth, bool hasBias = true, int stride = 1, int padding = 0)
        {
            _inChannels = inChannels;
            _outChannels = outChannels;
            _kDepth = kernelDepth;
            _kHeight = kernelHeight;
            _kWidth = kernelWidth;
            _stride = stride;
            _padding = padding;
            _useBias = hasBias;

            this.device = Device.CUDA;
            if (this.device.Type == DeviceType.CUDA && !CUDA.IsAvailable())
            {
                this.device = Device.CPU;
            }

            _weight = Initializers.XavierUniform(new TensorShape(outChannels, inChannels, kernelDepth, kernelHeight, kernelWidth), this.device);
            _weight.RequiresGrad = true;

            if (_useBias)
            {
                _bias = Tensor.Zeros(new TensorShape(outChannels), this.device);
                _bias.RequiresGrad = true;
            }
        }

        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input, expectedRank: 5);

            int batch = input.Shape[0];
            int inD = input.Shape[2];
            int inH = input.Shape[3];
            int inW = input.Shape[4];

            int outD = (inD + 2 * _padding - _kDepth) / _stride + 1;
            int outH = (inH + 2 * _padding - _kHeight) / _stride + 1;
            int outW = (inW + 2 * _padding - _kWidth) / _stride + 1;

            var outShape = new TensorShape(batch, _outChannels, outD, outH, outW);

            if (input.Device.Type == DeviceType.CUDA && CUDA.IsAvailable())
            {
                var result = new Tensor(new CudaBackend(outShape, input.RequiresGrad || _weight.RequiresGrad, this.device));

                var inRaw = Tensor.Unwrap(input) as CudaBackend ?? throw new InvalidOperationException("Input must reside on CUDA GPU.");
                var wRaw = Tensor.Unwrap(_weight) as CudaBackend ?? throw new InvalidOperationException("Weights must reside on CUDA GPU.");
                var resRaw = Tensor.Unwrap(result) as CudaBackend ?? throw new InvalidOperationException("Initialization failed.");

                CUDA.NativeConv3DForward(
                    inRaw.DevicePointer, wRaw.DevicePointer, resRaw.DevicePointer,
                    batch, _inChannels, inD, inH, inW, _outChannels, outD, outH, outW, _kDepth, _kHeight, _kWidth, _stride, _padding);

                if (input.RequiresGrad || _weight.RequiresGrad)
                {
                    var capturedInput = input;
                    var capturedWeight = _weight;

                    result.GradFn = gradOutput =>
                    {
                        var goRaw = Tensor.Unwrap(gradOutput) as CudaBackend ?? throw new InvalidOperationException("Gradient output must reside on CUDA.");

                        if (capturedWeight.RequiresGrad)
                        {
                            var gradWeight = new Tensor(new CudaBackend(capturedWeight.Shape, false, this.device));
                            var gwRaw = Tensor.Unwrap(gradWeight) as CudaBackend ?? throw new InvalidOperationException("Gradient allocation failed.");

                            CUDA.NativeConv3DGradWeight(
                                inRaw.DevicePointer, goRaw.DevicePointer, gwRaw.DevicePointer,
                                batch, _inChannels, inD, inH, inW, _outChannels, outD, outH, outW, _kDepth, _kHeight, _kWidth, _stride, _padding);

                            capturedWeight.AccumulateGrad(gradWeight);
                        }

                        var gradInput = new Tensor(new CudaBackend(capturedInput.Shape, false, this.device));
                        if (capturedInput.RequiresGrad)
                        {
                            var giRaw = Tensor.Unwrap(gradInput) as CudaBackend ?? throw new InvalidOperationException("Gradient allocation failed.");

                            CUDA.NativeConv3DGradInput(
                                goRaw.DevicePointer, wRaw.DevicePointer, giRaw.DevicePointer,
                                batch, _inChannels, inD, inH, inW, _outChannels, outD, outH, outW, _kDepth, _kHeight, _kWidth, _stride, _padding);

                            capturedInput.AccumulateGrad(gradInput);
                            capturedInput.GradFn?.Invoke(gradInput);
                        }

                        if (_useBias && _bias != null && _bias.RequiresGrad)
                        {
                            var gradBias = gradOutput.Sum(4, false).Sum(3, false).Sum(2, false).Sum(0, false);
                            _bias.AccumulateGrad(gradBias);
                        }

                        return gradInput;
                    };
                }

                if (_useBias && _bias != null)
                {
                    var biasReshaped = _bias.Reshape(1, _outChannels, 1, 1, 1);
                    result = (Tensor)result.Add(biasReshaped.BroadcastTo(result.Shape));
                }

                return result;
            }
            else
            {
                // HIGH-PERFORMANCE PARALLEL CPU FALLBACK
                float[] inData = input.ToArray();
                float[] wData = _weight.ToArray();
                float[] outData = new float[outShape.TotalElements];

                Parallel.For(0, batch, b =>
                {
                    for (int oc = 0; oc < _outChannels; oc++)
                    {
                        for (int od = 0; od < outD; od++)
                        {
                            for (int oh = 0; oh < outH; oh++)
                            {
                                for (int ow = 0; ow < outW; ow++)
                                {
                                    float sum = 0f;
                                    for (int ic = 0; ic < _inChannels; ic++)
                                    {
                                        for (int kd = 0; kd < _kDepth; kd++)
                                        {
                                            int id = od * _stride - _padding + kd;
                                            if (id >= 0 && id < inD)
                                            {
                                                for (int kh = 0; kh < _kHeight; kh++)
                                                {
                                                    int ih = oh * _stride - _padding + kh;
                                                    if (ih >= 0 && ih < inH)
                                                    {
                                                        for (int kw = 0; kw < _kWidth; kw++)
                                                        {
                                                            int iw = ow * _stride - _padding + kw;
                                                            if (iw >= 0 && iw < inW)
                                                            {
                                                                int inIdx = (((b * _inChannels + ic) * inD + id) * inH + ih) * inW + iw;
                                                                int wIdx = ((((oc * _inChannels + ic) * _kDepth + kd) * _kHeight + kh) * _kWidth + kw);
                                                                sum += inData[inIdx] * wData[wIdx];
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    int outIdx = (((b * _outChannels + oc) * outD + od) * outH + oh) * outW + ow;
                                    outData[outIdx] = sum;
                                }
                            }
                        }
                    }
                });

                var result = Tensor.FromArray(outData, outShape, input.Device);

                if (_useBias && _bias != null)
                {
                    var biasReshaped = _bias.Reshape(1, _outChannels, 1, 1, 1);
                    result = (Tensor)result.Add(biasReshaped.BroadcastTo(result.Shape));
                }

                return result;
            }
        }

        public override IEnumerable<ITensor> Parameters()
        {
            yield return _weight;
            if (_useBias && _bias != null) yield return _bias;
        }
    }
}