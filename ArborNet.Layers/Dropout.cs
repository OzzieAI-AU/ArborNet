using System;
using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Layers;
using ArborNet.Core.Tensors;

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
        private readonly float p;
        private readonly Random random = new Random();

        public Dropout(float p = 0.5f)
        {
            if (p < 0 || p > 1) throw new ArgumentOutOfRangeException(nameof(p));
            this.p = p;
        }

        public override ITensor Forward(ITensor input)
        {
            // Do not apply dropout noise during validation or inference
            if (!IsTraining || p == 0) return input;

            var mask = Tensor.Rand(input.Shape).GreaterThan(Tensor.FromScalar(p));
            var scale = 1f / (1f - p);
            return input.Multiply(mask).Multiply(scale);
        }

        public override IEnumerable<ITensor> Parameters() => Array.Empty<ITensor>();
    }
}