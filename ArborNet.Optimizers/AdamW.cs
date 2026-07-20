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
    /// Implements the AdamW optimization algorithm with decoupled weight decay as described in 
    /// "Decoupled Weight Decay Regularization" by Ilya Loshchilov and Frank Hutter.
    /// </summary>
    /// <remarks>
    /// AdamW modifies the typical L2 regularization by applying weight decay directly to the weights 
    /// rather than combining it with the gradient updates, preventing the weight decay from being 
    /// scaled by the historical gradient variances.
    /// </remarks>

    #endregion

    public class AdamW : IOptimizer
    {
        /// <summary>
        /// Gets or sets the learning rate for the optimization updates.
        /// </summary>
        /// <value>The learning rate scale factor.</value>
        public double LearningRate { get; set; }
        private readonly double _beta1;
        private readonly double _beta2;
        private readonly double _eps;
        private readonly double _weightDecay;
        private int _timestep;
        private readonly Dictionary<ITensor, (ITensor m, ITensor v)> _state = new();

        public AdamW(double learningRate = 1e-3, double beta1 = 0.9, double beta2 = 0.999,
                    double eps = 1e-8, double weightDecay = 0.0)
        {
            LearningRate = learningRate;
            _beta1 = beta1;
            _beta2 = beta2;
            _eps = eps;
            _weightDecay = weightDecay;
        }
        /// <summary>
        /// Performs a single optimization step and updates the provided parameter tensors in-place.
        /// </summary>
        /// <param name="parameters">An enumerable of <see cref="ITensor"/> parameters to be updated.</param>
        /// <remarks>
        /// This method increments the internal step counter, applies decoupled weight decay, updates 
        /// first and second moment tracking buffers, computes bias-corrected adaptive learning rates, 
        /// and applies the final parameter update step in-place on the device.
        /// </remarks>

        public void Step(IEnumerable<ITensor> parameters)
        {
            _timestep++;

            foreach (var p in parameters)
            {
                if (p == null || !p.RequiresGrad || p.Grad == null) continue;

                var grad = p.Grad;

                // 1. FIXED: Decoupled Weight Decay executed in-place on-device
                if (_weightDecay > 0)
                {
                    p.SubtractInPlace(p.Multiply((float)(_weightDecay * LearningRate)));
                }

                if (!_state.TryGetValue(p, out var state))
                {
                    state.m = Tensor.Zeros(p.Shape, p.Device);
                    state.v = Tensor.Zeros(p.Shape, p.Device);
                    _state[p] = state;
                }

                var (m, v) = state;

                // 2. FIXED: Update moments in-place on-device
                m.MultiplyInPlace((float)_beta1);
                m.AddInPlace(grad.Multiply(1f - (float)_beta1));

                v.MultiplyInPlace((float)_beta2);
                v.AddInPlace(grad.Multiply(grad).Multiply(1f - (float)_beta2));

                double biasCorrection1 = 1.0 - Math.Pow(_beta1, _timestep);
                double biasCorrection2 = 1.0 - Math.Pow(_beta2, _timestep);
                float stepSize = (float)(LearningRate * Math.Sqrt(biasCorrection2) / biasCorrection1);

                // 3. FIXED: Subtract update vector in-place on-device
                var denom = v.Sqrt().Add((float)_eps);
                var update = m.Divide(denom).Multiply(stepSize);

                p.SubtractInPlace(update);

                _state[p] = (m, v);
            }
        }
        /// <summary>
        /// Resets the gradients of all active parameter tensors to <see langword="null"/>.
        /// </summary>
        /// <param name="parameters">An enumerable of <see cref="ITensor"/> parameters whose gradients should be cleared.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="parameters"/> is <see langword="null"/>.</exception>

        public void ZeroGrad(IEnumerable<ITensor> parameters)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));
            foreach (var param in parameters)
            {
                if (param != null && param.RequiresGrad)
                {
                    param.Grad = null;
                }
            }
        }
    }
}