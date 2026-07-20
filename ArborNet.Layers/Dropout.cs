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
    using System.Collections.Generic;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;
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

    #endregion

    public class Dropout : BaseLayer
    {
        private readonly float p;
        private readonly Random random = new Random();

        public Dropout(float p = 0.5f)
        {
            if (p < 0 || p > 1) throw new ArgumentOutOfRangeException(nameof(p));
            this.p = p;
        }
        /// <summary>
        /// Applies the dropout regularization mask to the input tensor during the training phase.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> to process.</param>
        /// <returns>
        /// If the layer is in training mode and <see cref="p"/> is greater than 0, returns the element-wise scaled tensor with randomly dropped units.
        /// Otherwise, returns the original <paramref name="input"/> tensor unmodified.
        /// </returns>

        public override ITensor Forward(ITensor input)
        {
            // Do not apply dropout noise during validation or inference
            if (!IsTraining || p == 0) return input;

            var mask = Tensor.Rand(input.Shape).GreaterThan(Tensor.FromScalar(p));
            var scale = 1f / (1f - p);
            return input.Multiply(mask).Multiply(scale);
        }
        /// <summary>
        /// Returns the parameters of this layer.
        /// </summary>
        /// <returns>An empty enumerable of <see cref="ITensor"/> because the dropout layer contains no trainable parameters.</returns>

        public override IEnumerable<ITensor> Parameters() => Array.Empty<ITensor>();
    }
}