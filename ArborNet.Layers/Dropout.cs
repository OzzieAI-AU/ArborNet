using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using System;
using System.Collections.Generic;

namespace ArborNet.Layers
{
    /// <summary>
    /// Implements Dropout regularization as a neural network layer.
    /// Randomly sets input elements to zero with probability <c>p</c> and scales the remaining values
    /// to maintain the original expected value.
    /// </summary>
    /// <remarks>
    /// Dropout is a widely used regularization technique that helps prevent overfitting by 
    /// introducing noise during the forward pass. This implementation applies dropout 
    /// consistently during the forward pass (typically used in training mode).
    /// </remarks>
    public class Dropout : BaseLayer
    {
        /// <summary>
        /// The probability of dropping (zeroing) each element.
        /// </summary>
        private readonly float p;

        /// <summary>
        /// Random number generator instance.
        /// </summary>
        private readonly Random random = new Random();

        /// <summary>
        /// Initializes a new instance of the <see cref="Dropout"/> class.
        /// </summary>
        /// <param name="p">The dropout probability. Must be a value between 0.0 and 1.0 (inclusive). Defaults to 0.5.</param>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown when <paramref name="p"/> is less than 0 or greater than 1.
        /// </exception>
        public Dropout(float p = 0.5f)
        {
            if (p < 0 || p > 1) throw new ArgumentOutOfRangeException(nameof(p));
            this.p = p;
        }

        /// <summary>
        /// Performs the forward pass through the dropout layer.
        /// </summary>
        /// <param name="input">The input tensor to apply dropout to.</param>
        /// <returns>
        /// A tensor of the same shape as <paramref name="input"/> with dropout applied.
        /// </returns>
        /// <remarks>
        /// If <see cref="p"/> is exactly 0, the input tensor is returned unchanged.
        /// Otherwise, a binary mask is generated using random values and the input is 
        /// element-wise multiplied by the mask and scaled by <c>1 / (1 - p)</c>.
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            if (p == 0) return input;

            var mask = Tensor.Rand(input.Shape).GreaterThan(Tensor.FromScalar(p));
            var scale = 1f / (1f - p);
            return input.Multiply(mask).Multiply(scale);
        }

        /// <summary>
        /// Returns the trainable parameters of this layer.
        /// </summary>
        /// <returns>An empty collection, as the dropout layer contains no trainable parameters.</returns>
        public override IEnumerable<ITensor> Parameters() => Array.Empty<ITensor>();
    }
}