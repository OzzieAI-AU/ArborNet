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

    using ArborNet.Core.Devices;
    using ArborNet.Core.Functional;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;
    /// <summary>
    /// Represents a 1D Convolutional Layer (Conv1D) for neural networks.
    /// Applies a 1D convolution over an input signal composed of several input channels.
    /// </summary>
    /// <remarks>
    /// The input tensor is expected to have a shape of <c>(Batch, InChannels, InputLength)</c>.
    /// The output tensor will have a shape of <c>(Batch, OutChannels, OutputLength)</c>, where
    /// <c>OutputLength = (InputLength + 2 * Padding - KernelSize) / Stride + 1</c>.
    /// </remarks>

    #endregion

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
        /// <summary>
        /// Performs the forward pass of the 1D convolution layer.
        /// </summary>
        /// <param name="input">The input tensor of shape <c>(Batch, InChannels, InputLength)</c>.</param>
        /// <returns>A new tensor containing the results of the 1D convolution of shape <c>(Batch, OutChannels, OutputLength)</c>.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the calculated output length is non-positive.</exception>

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

            Parallel.For(0, batch * _outChannels, idx =>
            {
                int b = idx / _outChannels;
                int oc = idx % _outChannels;

                for (int ol = 0; ol < outLen; ol++)
                {
                    float sum = 0f;
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
                    outData[b * _outChannels * outLen + oc * outLen + ol] = sum;
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

                    // Lock-Free Weight Gradient Evaluation: OutChannels Partitioning
                    Parallel.For(0, _outChannels, oc =>
                    {
                        for (int b = 0; b < batch; b++)
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

                                            gradWeightData[wIdx] += inData[inIdx] * goVal;
                                        }
                                    }
                                }
                            }
                        }
                    });

                    // Lock-Free Input Gradient Evaluation: InputChannels Partitioning
                    Parallel.For(0, batch * _inChannels, index =>
                    {
                        int b = index / _inChannels;
                        int ic = index % _inChannels;

                        for (int oc = 0; oc < _outChannels; oc++)
                        {
                            for (int ol = 0; ol < outLen; ol++)
                            {
                                int outIdx = b * _outChannels * outLen + oc * outLen + ol;
                                float goVal = goData[outIdx];

                                for (int k = 0; k < _kernelSize; k++)
                                {
                                    int inPos = ol * _stride - _padding + k;
                                    if (inPos >= 0 && inPos < inLen)
                                    {
                                        int inIdx = b * _inChannels * inLen + ic * inLen + inPos;
                                        int wIdx = oc * _inChannels * _kernelSize + ic * _kernelSize + k;

                                        gradInputData[inIdx] += wData[wIdx] * goVal;
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
                        var gradBias = gradOutput.Sum(2, false).Sum(0, false);
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
                var biasReshaped = _bias.Reshape(new int[] { 1, _outChannels, 1 });
                result = result.Add(biasReshaped.BroadcastTo(result.Shape));
            }

            return result;
        }
        /// <summary>
        /// Returns an enumerator that iterates through the trainable parameters (weights and biases) of this layer.
        /// </summary>
        /// <returns>An enumerable collection of trainable tensors.</returns>

        public override IEnumerable<ITensor> Parameters()
        {
            yield return _weight;
            if (_useBias && _bias != null) yield return _bias;
        }
    }
}
