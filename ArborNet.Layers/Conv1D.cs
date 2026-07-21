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
        private readonly Conv2D _fused2DConv;

        public Conv1D(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0, bool useBias = true, Device? device = null)
        {
            // We project 1D operations onto a height-1 2D Conv plane to avoid CPU fallback loops.
            _fused2DConv = new Conv2D(inChannels, outChannels, kernelSize, stride, padding, useBias, device ?? Device.CUDA);
        }

        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input, expectedRank: 3);

            int batch = input.Shape[0];
            int channels = input.Shape[1];
            int length = input.Shape[2];

            // Reshape [B, C, L] to 4D [B, C, 1, L]
            var reshapedInput = input.Reshape(batch, channels, 1, length);
            var output2D = _fused2DConv.Forward(reshapedInput);

            // Reshape back to 3D: [Batch, OutChannels, OutSeqLen]
            return output2D.Reshape(output2D.Shape[0], output2D.Shape[1], output2D.Shape[3]);
        }

        public override IEnumerable<ITensor> Parameters()
        {
            return _fused2DConv.Parameters();
        }
    }
}