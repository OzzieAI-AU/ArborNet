using System;
using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Devices;
using ArborNet.Core.Functional;
using ArborNet.Core.Layers;

namespace ArborNet.Layers
{
    public class Conv1D : BaseLayer
    {
        private readonly int _inChannels;
        private readonly int _outChannels;
        private readonly int _kernelSize;
        private readonly int _stride;
        private readonly int _padding;
        private readonly bool _useBias;

        private readonly ITensor _weight;
        private readonly ITensor? _bias;

        public Conv1D(int inChannels, int outChannels, int kernelSize,
                      int stride = 1, int padding = 0, bool useBias = true,
                      Device? device = null)
        {
            _inChannels = inChannels;
            _outChannels = outChannels;
            _kernelSize = kernelSize;
            _stride = stride;
            _padding = padding;
            _useBias = useBias;

            var dev = device ?? Device.CPU;
            _weight = Initializers.XavierUniform(new TensorShape(outChannels, inChannels, kernelSize), dev);
            _weight.RequiresGrad = true;

            if (_useBias)
            {
                _bias = Tensor.Zeros(new TensorShape(outChannels), dev);
                _bias.RequiresGrad = true;
            }
        }

        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input, expectedRank: 3);
            int batch = input.Shape[0];
            int inLen = input.Shape[2];
            int outLen = (inLen + 2 * _padding - _kernelSize) / _stride + 1;

            if (outLen <= 0)
                throw new InvalidOperationException("Output length is non-positive. Check kernel/stride/padding settings.");

            var outputShape = new TensorShape(batch, _outChannels, outLen);
            var inData = input.ToArray();
            var wData = _weight.ToArray();
            var outData = new float[outputShape.TotalElements];

            // Direct 1D Convolution Cross-Correlation
            for (int b = 0; b < batch; b++)
            {
                for (int oc = 0; oc < _outChannels; oc++)
                {
                    for (int ol = 0; ol < outLen; ol++)
                    {
                        float sum = 0f;
                        int outIdx = b * _outChannels * outLen + oc * outLen + ol;

                        for (int ic = 0; ic < _inChannels; ic++)
                        {
                            for (int k = 0; k < _kernelSize; k++)
                            {
                                int inPos = ol * _stride - _padding + k;
                                if (inPos >= 0 && inPos < inLen)
                                {
                                    int inIdx = b * _inChannels * inLen + ic * inLen + inPos;
                                    int wIdx = oc * _inChannels * _kernelSize + ic * _kernelSize + k;
                                    sum += inData[inIdx] * wData[wIdx];
                                }
                            }
                        }
                        outData[outIdx] = sum;
                    }
                }
            }

            var result = Tensor.FromArray(outData, outputShape, input.Device);

            // ANALYTICAL BACKWARD PASS FOR 1D CONVOLUTION
            if (input.RequiresGrad || _weight.RequiresGrad)
            {
                var capturedInput = input;
                var capturedWeight = _weight;

                result.GradFn = gradOutput =>
                {
                    var goData = gradOutput.ToArray();
                    var gradInputData = new float[capturedInput.Shape.TotalElements];
                    var gradWeightData = new float[capturedWeight.Shape.TotalElements];

                    for (int b = 0; b < batch; b++)
                    {
                        for (int oc = 0; oc < _outChannels; oc++)
                        {
                            for (int ol = 0; ol < outLen; ol++)
                            {
                                int outIdx = b * _outChannels * outLen + oc * outLen + ol;
                                float goVal = goData[outIdx];

                                for (int ic = 0; ic < _inChannels; ic++)
                                {
                                    for (int k = 0; k < _kernelSize; k++)
                                    {
                                        int inPos = ol * _stride - _padding + k;
                                        if (inPos >= 0 && inPos < inLen)
                                        {
                                            int inIdx = b * _inChannels * inLen + ic * inLen + inPos;
                                            int wIdx = oc * _inChannels * _kernelSize + ic * _kernelSize + k;

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

                    var gradInput = Tensor.FromArray(gradInputData, capturedInput.Shape, input.Device);
                    var gradWeight = Tensor.FromArray(gradWeightData, capturedWeight.Shape, input.Device);

                    if (capturedWeight.RequiresGrad)
                    {
                        capturedWeight.Grad = capturedWeight.Grad == null ? gradWeight : capturedWeight.Grad.Add(gradWeight);
                    }

                    if (_useBias && _bias != null && _bias.RequiresGrad)
                    {
                        // Gradient w.r.t bias is accumulated across batch and spatial length dimensions
                        var gradBias = gradOutput.Sum(2, false).Sum(0, false);
                        _bias.Grad = _bias.Grad == null ? gradBias : _bias.Grad.Add(gradBias);
                    }

                    if (capturedInput.RequiresGrad)
                    {
                        capturedInput.Grad = capturedInput.Grad == null ? gradInput : capturedInput.Grad.Add(gradInput);
                        capturedInput.GradFn?.Invoke(gradInput);
                    }

                    return gradInput;
                };
            }

            if (_useBias && _bias != null)
            {
                var biasReshaped = _bias.Reshape(new int[] { 1, _outChannels, 1 });
                result = result.Add(biasReshaped.BroadcastTo(result.Shape));
            }

            return result;
        }

        public override IEnumerable<ITensor> Parameters()
        {
            yield return _weight;
            if (_useBias && _bias != null) yield return _bias;
        }
    }
}
