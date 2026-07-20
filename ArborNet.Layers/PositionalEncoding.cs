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
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using System.Collections.Generic;
    using ArborNet.Core.Layers;
    /// <summary>
    /// Implements a Sinusoidal Positional Encoding layer, typically used in Transformer models.
    /// This layer injects positional information into input sequence embeddings by adding precomputed 
    /// wave-based (sine and cosine) patterns.
    /// </summary>

    #endregion

    public class PositionalEncoding : BaseLayer
    {
        private ITensor _pe;
        private readonly int _maxLen;
        private readonly int _dModel;

        public PositionalEncoding(int dModel, int maxLen = 512, Device device = null)
        {
            if (dModel % 2 != 0)
                throw new ArgumentException("dModel must be even for sinusoidal encoding.");

            _maxLen = maxLen;
            _dModel = dModel;
            device ??= Device.CPU;

            var data = new float[maxLen * dModel];
            for (int pos = 0; pos < maxLen; pos++)
            {
                for (int i = 0; i < dModel; i += 2)
                {
                    double angle = pos / Math.Pow(10000.0, (double)i / dModel);
                    int idx = pos * dModel + i;
                    data[idx] = (float)Math.Sin(angle);
                    data[idx + 1] = (float)Math.Cos(angle);
                }
            }

            _pe = Tensor.FromArray(data, new TensorShape(maxLen, dModel), device);
            this.device = device;
        }
        /// <summary>
        /// Moves the layer, including its internal positional encoding tensor, to the specified target device.
        /// </summary>
        /// <param name="targetDevice">The target hardware device (e.g., CPU or CUDA device).</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="targetDevice"/> is null.</exception>

        public override void To(Device targetDevice)
        {
            if (targetDevice == null) throw new ArgumentNullException(nameof(targetDevice));
            base.To(targetDevice);
            _pe = _pe.To(targetDevice);
        }
        /// <summary>
        /// Performs the forward pass by adding the positional encodings to the input tensor.
        /// </summary>
        /// <param name="input">The input tensor of shape (..., seqLen, dModel).</param>
        /// <returns>A new tensor with positional encodings added to the input.</returns>
        /// <exception cref="ArgumentException">
        /// Thrown if the input tensor has fewer than 2 dimensions, or if the input sequence length 
        /// exceeds the maximum supported sequence length (<see cref="_maxLen"/>).
        /// </exception>

        public override ITensor Forward(ITensor input)
        {
            var shape = input.Shape;
            if (shape.Rank < 2)
                throw new ArgumentException("Input must have at least 2 dimensions (..., seqLen, dModel)");

            int seqLen = shape[shape.Rank - 2];
            if (seqLen > _maxLen)
                throw new ArgumentException($"Sequence length {seqLen} exceeds maximum {_maxLen}");

            if (_pe.Device != input.Device)
            {
                _pe = _pe.To(input.Device);
                this.device = input.Device;
            }

            var peSlice = _pe.Slice((0, seqLen, 1), (0, _dModel, 1));
            var peReshaped = shape.Rank == 2 ? peSlice : peSlice.Reshape(1, seqLen, _dModel);

            return input.Add(peReshaped);
        }
        /// <summary>
        /// Returns the parameters of this layer.
        /// </summary>
        /// <returns>An enumerable containing the internal positional encoding tensor.</returns>

        public override IEnumerable<ITensor> Parameters()
        {
            yield return _pe;
        }
    }
}
