// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Optimizers
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Implements the RMSProp (Root Mean Squared Propagation) optimization algorithm.
    /// </summary>
    /// <remarks>
    /// RMSProp is an adaptive learning rate optimization method that utilizes a moving average 
    /// of squared gradients to normalize the gradient. This helps in handling non-stationary 
    /// objectives and mitigates the diminishing learning rate problem found in AdaGrad.
    /// </remarks>

    #endregion

    public class RMSProp : IOptimizer
    {
        /// <summary>
        /// Gets or sets the learning rate (step size) for the optimization step.
        /// </summary>
        public double LearningRate { get; set; }
        private readonly float _alpha;
        private readonly float _epsilon;
        private readonly Dictionary<ITensor, ITensor> _v = new();

        public RMSProp(double learningRate = 0.001, float alpha = 0.99f, float epsilon = 1e-8f)
        {
            LearningRate = learningRate;
            _alpha = alpha;
            _epsilon = epsilon;
        }
        /// <summary>
        /// Performs a single optimization step, updating the model parameters using their current gradients.
        /// </summary>
        /// <param name="parameters">An enumerable of <see cref="ITensor"/> representing the parameters to update.</param>
        /// <remarks>
        /// This method computes the running average of squared gradients in-place on-device,
        /// and applies the normalized updates directly to the parameter tensors.
        /// Parameters without calculated gradients (where <c>Grad</c> is null) are ignored.
        /// </remarks>

        public void Step(IEnumerable<ITensor> parameters)
        {
            foreach (var param in parameters)
            {
                if (param.Grad == null) continue;

                if (!_v.TryGetValue(param, out var v))
                {
                    v = Tensor.Zeros(param.Shape, param.Device);
                    _v[param] = v;
                }

                var gradSq = param.Grad.Multiply(param.Grad);
                v.MultiplyInPlace(_alpha);
                v.AddInPlace(gradSq.Multiply(1f - _alpha));

                var denom = v.Sqrt().Add(Tensor.FromScalar(_epsilon, param.Device));
                var update = param.Grad.Divide(denom).Multiply((float)LearningRate);

                // Run fully in-place on-device
                param.SubtractInPlace(update);
            }
        }
        /// <summary>
        /// Resets the gradients of all trainable parameters in the provided collection to zero.
        /// </summary>
        /// <param name="parameters">An enumerable of <see cref="ITensor"/> representing the parameters whose gradients should be cleared.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="parameters"/> is null.</exception>

        public void ZeroGrad(IEnumerable<ITensor> parameters)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));

            foreach (var param in parameters)
            {
                if (param != null && param.RequiresGrad)
                {
                    param.Grad = Tensor.Zeros(param.Shape, param.Device);
                }
            }
        }
    }
}
