using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Layers;

namespace ArborNet.Layers
{
    /// <summary>
    /// Implements a single Mistral transformer block consisting of pre-norm self-attention
    /// and a feed-forward network with residual connections.
    /// </summary>
    /// <remarks>
    /// This block follows a standard pre-norm transformer architecture:
    /// <list type="bullet">
    ///   <item>LayerNorm → Multi-Head Attention → Residual</item>
    ///   <item>LayerNorm → Feed-Forward (Linear + ReLU + Linear) → Residual</item>
    /// </list>
    /// </remarks>
    public class MistralBlock : BaseLayer
    {
        /// <summary>
        /// Layer normalization applied before the self-attention sub-layer.
        /// </summary>
        private readonly LayerNorm norm1;

        /// <summary>
        /// Multi-head self-attention mechanism.
        /// </summary>
        private readonly MultiHeadAttention attention;

        /// <summary>
        /// Layer normalization applied before the feed-forward sub-layer.
        /// </summary>
        private readonly LayerNorm norm2;

        /// <summary>
        /// First linear projection in the feed-forward network (expands to 4× hidden dimension).
        /// </summary>
        private readonly Linear ff1;

        /// <summary>
        /// Second linear projection in the feed-forward network (projects back to hidden dimension).
        /// </summary>
        private readonly Linear ff2;

        /// <summary>
        /// Aggregated list of all trainable parameters from the sub-layers.
        /// </summary>
        private readonly List<ITensor> _parameters = new();

        /// <summary>
        /// Initializes a new instance of the <see cref="MistralBlock"/> class.
        /// </summary>
        /// <param name="hiddenDim">The hidden dimension size of the model.</param>
        /// <param name="numHeads">The number of attention heads.</param>
        /// <param name="kvHeads">The number of key-value heads (for grouped-query attention).</param>
        /// <param name="slidingWindow">The sliding window size for attention masking.</param>
        public MistralBlock(int hiddenDim, int numHeads, int kvHeads, int slidingWindow)
        {
            norm1 = new LayerNorm(new[] { hiddenDim });
            attention = new MultiHeadAttention(hiddenDim, numHeads);
            norm2 = new LayerNorm(new[] { hiddenDim });
            ff1 = new Linear(hiddenDim, hiddenDim * 4);
            ff2 = new Linear(hiddenDim * 4, hiddenDim);

            _parameters.AddRange(norm1.Parameters());
            _parameters.AddRange(attention.Parameters());
            _parameters.AddRange(norm2.Parameters());
            _parameters.AddRange(ff1.Parameters());
            _parameters.AddRange(ff2.Parameters());
        }

        /// <summary>
        /// Performs a forward pass through the Mistral transformer block.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The output tensor after applying attention and feed-forward layers with residual connections.</returns>
        public override ITensor Forward(ITensor x)
        {
            var residual = x;
            x = norm1.Forward(x);
            x = attention.Forward(x);
            x = x.Add(residual);

            residual = x;
            x = norm2.Forward(x);
            x = ff2.Forward(ff1.Forward(x).Relu());
            return x.Add(residual);
        }

        /// <summary>
        /// Returns all trainable parameters contained within this block.
        /// </summary>
        /// <returns>A collection of all parameter tensors from the sub-layers.</returns>
        public override IEnumerable<ITensor> Parameters() => _parameters;
    }
}