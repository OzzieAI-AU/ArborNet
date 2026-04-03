using System;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using System.Collections.Generic;

namespace ArborNet.Layers
{
    /// <summary>
    /// Sinusoidal positional encoding as used in the original Transformer paper.
    /// Fully implements BaseLayer / ITensor contract.
    /// </summary>
    public class PositionalEncoding : BaseLayer
    {
        /// <summary>
        /// Pre-computed sinusoidal positional encodings tensor of shape (maxLen, dModel).
        /// </summary>
        private readonly ITensor _pe;

        /// <summary>
        /// Maximum sequence length for which positional encodings were pre-computed.
        /// </summary>
        private readonly int _maxLen;

        /// <summary>
        /// Model dimension (embedding size). Must be even.
        /// </summary>
        private readonly int _dModel;

        /// <summary>
        /// Initializes a new instance of the <see cref="PositionalEncoding"/> class.
        /// </summary>
        /// <param name="dModel">The dimensionality of the embeddings (must be even).</param>
        /// <param name="maxLen">The maximum sequence length to precompute encodings for (default: 512).</param>
        /// <param name="device">The device to store the positional encodings on. If null, defaults to CPU.</param>
        /// <exception cref="ArgumentException">Thrown when <paramref name="dModel"/> is odd.</exception>
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
        }

        /// <summary>
        /// Adds the corresponding sinusoidal positional encodings to the input tensor.
        /// </summary>
        /// <param name="input">The input tensor. Must have at least 2 dimensions with shape (..., seqLen, dModel).</param>
        /// <returns>A new tensor containing the input with positional encodings added.</returns>
        /// <exception cref="ArgumentException">
        /// Thrown when the input rank is less than 2 or when the sequence length exceeds the precomputed maximum length.
        /// </exception>
        public override ITensor Forward(ITensor input)
        {
            var shape = input.Shape;
            if (shape.Rank < 2)
                throw new ArgumentException("Input must have at least 2 dimensions (..., seqLen, dModel)");

            int seqLen = shape[shape.Rank - 2];
            if (seqLen > _maxLen)
                throw new ArgumentException($"Sequence length {seqLen} exceeds maximum {_maxLen}");

            // Slice PE to current sequence length
            var peSlice = _pe.Slice((0, seqLen, 1), (0, _dModel, 1));

            // Broadcast to match input shape (add batch dimension if needed)
            var targetShape = shape.Rank == 2
                ? new TensorShape(seqLen, _dModel)
                : new TensorShape(shape[0], seqLen, _dModel);

            var peBroadcast = peSlice.ReshapeWithBroadcast(targetShape, 0);

            return input.Add(peBroadcast);
        }

        /// <summary>
        /// Returns the positional encoding tensor.
        /// </summary>
        /// <returns>An enumerable containing the positional encoding tensor.</returns>
        /// <remarks>
        /// Although positional encodings are fixed (non-trainable) in the classic Transformer,
        /// they are exposed via the <see cref="BaseLayer.Parameters"/> contract to maintain
        /// compatibility with optimizers and layer management systems.
        /// </remarks>
        public override IEnumerable<ITensor> Parameters()
        {
            yield return _pe; // positional encodings are learnable in some variants, but usually fixed
        }
    }
}