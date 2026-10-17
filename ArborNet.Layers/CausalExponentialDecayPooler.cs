using System;

namespace ArborNet.Layers
{
    /// <summary>
    /// Provides causal, order-preserving temporal decay pooling over sequence embeddings.
    /// Replaces parameter-heavy sequence-flattening projections with a parameter-efficient
    /// causal accumulator to maximize knowledge density.
    /// </summary>
    public sealed class CausalExponentialDecayPooler
    {
        private readonly int _channelDim;
        private readonly float _initialDecay;

        /// <summary>
        /// Gets the channel dimension of input tokens.
        /// </summary>
        public int ChannelDim => _channelDim;

        /// <summary>
        /// Gets the initial decay factor applied to past temporal states.
        /// </summary>
        public float InitialDecay => _initialDecay;

        /// <summary>
        /// Initializes a new instance of the <see cref="CausalExponentialDecayPooler"/> class.
        /// </summary>
        /// <param name="channelDim">Feature channel dimension (d).</param>
        /// <param name="initialDecay">Initial decay scalar in range (0, 1).</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if channelDim <= 0 or initialDecay not in (0, 1).</exception>
        public CausalExponentialDecayPooler(int channelDim, float initialDecay = 0.85f)
        {
            if (channelDim <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(channelDim), "Channel dimension must be positive.");
            }

            if (initialDecay <= 0.0f || initialDecay >= 1.0f)
            {
                throw new ArgumentOutOfRangeException(nameof(initialDecay), "Initial decay factor must reside strictly in (0, 1).");
            }

            _channelDim = channelDim;
            _initialDecay = initialDecay;
        }

        /// <summary>
        /// Causally pools a sequence buffer of shape [sequenceLength, channelDim] into a single representation [channelDim].
        /// </summary>
        /// <param name="sequenceBuffer">Contiguous flat array of layout [sequenceLength * channelDim].</param>
        /// <param name="sequenceLength">Number of temporal tokens in the window.</param>
        /// <param name="decayFactor">Exponential decay base factor in (0, 1).</param>
        /// <returns>Allocated pooled channel vector of length <see cref="ChannelDim"/>.</returns>
        public float[] PoolCausalSequence(ReadOnlySpan<float> sequenceBuffer, int sequenceLength, float decayFactor)
        {
            if (sequenceLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be positive.");
            }

            if (sequenceBuffer.Length < sequenceLength * _channelDim)
            {
                throw new ArgumentException("Input buffer does not match sequenceLength * channelDim.", nameof(sequenceBuffer));
            }

            float[] pooled = new float[_channelDim];

            // Compute normalized causal decay weights: w_t = decay^(T - 1 - t)
            float[] weights = new float[sequenceLength];
            float sumWeights = 0.0f;
            float currentWeight = 1.0f;

            for (int t = sequenceLength - 1; t >= 0; t--)
            {
                weights[t] = currentWeight;
                sumWeights += currentWeight;
                currentWeight *= decayFactor;
            }

            float invSum = sumWeights > 0.0f ? (1.0f / sumWeights) : 1.0f;

            // Causally accumulate weighted features across time
            for (int t = 0; t < sequenceLength; t++)
            {
                float normWeight = weights[t] * invSum;
                int timeOffset = t * _channelDim;

                for (int c = 0; c < _channelDim; c++)
                {
                    pooled[c] += sequenceBuffer[timeOffset + c] * normWeight;
                }
            }

            return pooled;
        }

        /// <summary>
        /// Executes forward causal pooling with autodiff-compatible tensor buffers.
        /// </summary>
        /// <param name="inputTensor">Flattened float array representing [batchSize, sequenceLength, channelDim].</param>
        /// <param name="batchSize">Batch dimension size.</param>
        /// <param name="sequenceLength">Sequence dimension size.</param>
        /// <param name="learnableLogDecay">Learnable scalar parameterizing logit/log-decay factor.</param>
        /// <returns>Pooled feature array representing [batchSize, channelDim].</returns>
        public float[] Forward(float[] inputTensor, int batchSize, int sequenceLength, float learnableLogDecay)
        {
            // TODO: Bind directly to ArborNet ITensor graph execution once custom tape ops are registered.
            // Map unconstrained log-decay scalar into valid (0, 1) decay via sigmoid:
            float decayFactor = 1.0f / (1.0f + MathF.Exp(-learnableLogDecay));
            float[] output = new float[batchSize * _channelDim];

            for (int b = 0; b < batchSize; b++)
            {
                int batchInOffset = b * sequenceLength * _channelDim;
                int batchOutOffset = b * _channelDim;

                var slice = new ReadOnlySpan<float>(inputTensor, batchInOffset, sequenceLength * _channelDim);
                float[] pooledBatch = PoolCausalSequence(slice, sequenceLength, decayFactor);

                Array.Copy(pooledBatch, 0, output, batchOutOffset, _channelDim);
            }

            return output;
        }
    }
}
