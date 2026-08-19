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

    using ArborNet.Activations;
    using ArborNet.Core;
    using ArborNet.Core.Backends;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Initializers;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Native.PInvoke;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;

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

            this.device = device ?? Device.CUDA;
            if (this.device.Type == DeviceType.CUDA && !CUDA.IsAvailable())
            {
                this.device = Device.CPU;
            }

            _weight = Initializers.XavierUniform(new TensorShape(outChannels, inChannels, kernelSize, kernelSize), this.device);
            _weight.RequiresGrad = true;

            if (_useBias)
            {
                _bias = Tensor.Zeros(new TensorShape(outChannels), this.device);
                _bias.RequiresGrad = true;
            }
        }

        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input, expectedRank: 4);

            int batch = input.Shape[0];
            int inH = input.Shape[2];
            int inW = input.Shape[3];

            int outH = (inH + 2 * _padding - _kernelSize) / _stride + 1;
            int outW = (inW + 2 * _padding - _kernelSize) / _stride + 1;

            var outShape = new TensorShape(batch, _outChannels, outH, outW);

            if (input.Device.Type == DeviceType.CUDA && CUDA.IsAvailable())
            {
                var result = new Tensor(new CudaBackend(outShape, input.RequiresGrad || _weight.RequiresGrad, this.device));

                var inRaw = Tensor.Unwrap(input) as CudaBackend ?? throw new InvalidOperationException("Input must reside on CUDA GPU.");
                var wRaw = Tensor.Unwrap(_weight) as CudaBackend ?? throw new InvalidOperationException("Weights must reside on CUDA GPU.");
                var resRaw = Tensor.Unwrap(result) as CudaBackend ?? throw new InvalidOperationException("Initialization failed.");

                CUDA.NativeConv2DForward(
                    inRaw.DevicePointer, wRaw.DevicePointer, resRaw.DevicePointer,
                    batch, _inChannels, inH, inW, _outChannels, outH, outW, _kernelSize, _kernelSize, _stride, _padding);

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

                            CUDA.NativeConv2DGradWeight(
                                inRaw.DevicePointer, goRaw.DevicePointer, gwRaw.DevicePointer,
                                batch, _inChannels, inH, inW, _outChannels, outH, outW, _kernelSize, _kernelSize, _stride, _padding);

                            capturedWeight.AccumulateGrad(gradWeight);
                        }

                        var gradInput = new Tensor(new CudaBackend(capturedInput.Shape, false, this.device));
                        if (capturedInput.RequiresGrad)
                        {
                            var giRaw = Tensor.Unwrap(gradInput) as CudaBackend ?? throw new InvalidOperationException("Gradient allocation failed.");

                            CUDA.NativeConv2DGradInput(
                                goRaw.DevicePointer, wRaw.DevicePointer, giRaw.DevicePointer,
                                batch, _inChannels, inH, inW, _outChannels, outH, outW, _kernelSize, _kernelSize, _stride, _padding);

                            capturedInput.AccumulateGrad(gradInput);
                            capturedInput.GradFn?.Invoke(gradInput);
                        }

                        if (_useBias && _bias != null && _bias.RequiresGrad)
                        {
                            var gradBias = gradOutput.Sum(3, false).Sum(2, false).Sum(0, false);
                            _bias.AccumulateGrad(gradBias);
                        }

                        return gradInput;
                    };
                }

                if (_useBias && _bias != null)
                {
                    var biasReshaped = _bias.Reshape(1, _outChannels, 1, 1);
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
                        for (int oh = 0; oh < outH; oh++)
                        {
                            for (int ow = 0; ow < outW; ow++)
                            {
                                float sum = 0f;
                                for (int ic = 0; ic < _inChannels; ic++)
                                {
                                    for (int kh = 0; kh < _kernelSize; kh++)
                                    {
                                        int ih = oh * _stride - _padding + kh;
                                        if (ih >= 0 && ih < inH)
                                        {
                                            for (int kw = 0; kw < _kernelSize; kw++)
                                            {
                                                int iw = ow * _stride - _padding + kw;
                                                if (iw >= 0 && iw < inW)
                                                {
                                                    int inIdx = ((b * _inChannels + ic) * inH + ih) * inW + iw;
                                                    int wIdx = ((oc * _inChannels + ic) * _kernelSize + kh) * _kernelSize + kw;
                                                    sum += inData[inIdx] * wData[wIdx];
                                                }
                                            }
                                        }
                                    }
                                }
                                int outIdx = ((b * _outChannels + oc) * outH + oh) * outW + ow;
                                outData[outIdx] = sum;
                            }
                        }
                    }
                });

                var result = Tensor.FromArray(outData, outShape, input.Device);

                if (input.RequiresGrad || _weight.RequiresGrad)
                {
                    var capturedInput = input;
                    var capturedWeight = _weight;

                    result.GradFn = gradOutput =>
                    {
                        float[] goData = gradOutput.ToArray();

                        if (capturedWeight.RequiresGrad)
                        {
                            float[] gwData = new float[capturedWeight.Shape.TotalElements];
                            Parallel.For(0, _outChannels, oc =>
                            {
                                for (int ic = 0; ic < _inChannels; ic++)
                                {
                                    for (int kh = 0; kh < _kernelSize; kh++)
                                    {
                                        for (int kw = 0; kw < _kernelSize; kw++)
                                        {
                                            float sum = 0f;
                                            for (int b = 0; b < batch; b++)
                                            {
                                                for (int oh = 0; oh < outH; oh++)
                                                {
                                                    int ih = oh * _stride - _padding + kh;
                                                    if (ih >= 0 && ih < inH)
                                                    {
                                                        for (int ow = 0; ow < outW; ow++)
                                                        {
                                                            int iw = ow * _stride - _padding + kw;
                                                            if (iw >= 0 && iw < inW)
                                                            {
                                                                int inIdx = ((b * _inChannels + ic) * inH + ih) * inW + iw;
                                                                int goIdx = ((b * _outChannels + oc) * outH + oh) * outW + ow;
                                                                sum += inData[inIdx] * goData[goIdx];
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            int wIdx = ((oc * _inChannels + ic) * _kernelSize + kh) * _kernelSize + kw;
                                            gwData[wIdx] = sum;
                                        }
                                    }
                                }
                            });
                            capturedWeight.AccumulateGrad(Tensor.FromArray(gwData, capturedWeight.Shape, input.Device));
                        }

                        var gradInput = Tensor.Zeros(capturedInput.Shape, input.Device);
                        if (capturedInput.RequiresGrad)
                        {
                            float[] giData = new float[capturedInput.Shape.TotalElements];
                            Parallel.For(0, batch, b =>
                            {
                                for (int ic = 0; ic < _inChannels; ic++)
                                {
                                    for (int ih = 0; ih < inH; ih++)
                                    {
                                        for (int iw = 0; iw < inW; iw++)
                                        {
                                            float sum = 0f;
                                            for (int oc = 0; oc < _outChannels; oc++)
                                            {
                                                for (int kh = 0; kh < _kernelSize; kh++)
                                                {
                                                    int oh = ih + _padding - kh;
                                                    if (oh % _stride == 0)
                                                    {
                                                        oh /= _stride;
                                                        if (oh >= 0 && oh < outH)
                                                        {
                                                            for (int kw = 0; kw < _kernelSize; kw++)
                                                            {
                                                                int ow = iw + _padding - kw;
                                                                if (ow % _stride == 0)
                                                                {
                                                                    ow /= _stride;
                                                                    if (ow >= 0 && ow < outW)
                                                                    {
                                                                        int goIdx = ((b * _outChannels + oc) * outH + oh) * outW + ow;
                                                                        int wIdx = ((oc * _inChannels + ic) * _kernelSize + kh) * _kernelSize + kw;
                                                                        sum += goData[goIdx] * wData[wIdx];
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            int inIdx = ((b * _inChannels + ic) * inH + ih) * inW + iw;
                                            giData[inIdx] = sum;
                                        }
                                    }
                                }
                            });
                            gradInput = Tensor.FromArray(giData, capturedInput.Shape, input.Device);
                            capturedInput.AccumulateGrad(gradInput);
                            capturedInput.GradFn?.Invoke(gradInput);
                        }

                        if (_useBias && _bias != null && _bias.RequiresGrad)
                        {
                            var gradBias = gradOutput.Sum(3, false).Sum(2, false).Sum(0, false);
                            _bias.AccumulateGrad(gradBias);
                        }

                        return gradInput;
                    };
                }

                if (_useBias && _bias != null)
                {
                    var biasReshaped = _bias.Reshape(1, _outChannels, 1, 1);
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