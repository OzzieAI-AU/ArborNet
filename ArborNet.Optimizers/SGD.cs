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
    /// Represents a Stochastic Gradient Descent (SGD) optimizer with optional support for momentum and weight decay (L2 regularization).
    /// </summary>
    /// <remarks>
    /// This optimizer updates model parameters iteratively using the gradients calculated during backward propagation.
    /// It can use momentum to accelerate gradients in consistent directions and weight decay to apply L2 regularization.
    /// </remarks>

    #endregion

    public class SGD : IOptimizer
    {
        /// <summary>
        /// Gets or sets the learning rate, which controls the step size taken during parameter updates.
        /// </summary>
        /// <value>The learning rate as a double-precision floating-point value.</value>
        public double LearningRate { get; set; }
        /// <summary>
        /// Gets or sets the momentum factor. A value greater than zero accelerates gradient descent and dampens oscillations.
        /// </summary>
        /// <value>The momentum factor as a double-precision floating-point value.</value>
        public double Momentum { get; set; }
        /// <summary>
        /// Gets or sets the weight decay (L2 penalty) factor. A value greater than zero helps prevent overfitting by penalizing large weights.
        /// </summary>
        /// <value>The weight decay factor as a double-precision floating-point value.</value>
        public double WeightDecay { get; set; }
        private readonly Dictionary<ITensor, ITensor> velocity = new();

        public SGD(double learningRate = 0.01, double momentum = 0.0, double weightDecay = 0.0)
        {
            LearningRate = learningRate;
            Momentum = momentum;
            WeightDecay = weightDecay;
        }
        /// <summary>
        /// Performs a single optimization step, updating the parameters in-place using their accumulated gradients.
        /// </summary>
        /// <param name="parameters">An enumerable of <see cref="ITensor"/> parameters to be updated.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="parameters"/> is null.</exception>

        public void Step(IEnumerable<ITensor> parameters)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));

            foreach (var param in parameters)
            {
                if (param == null || !param.RequiresGrad || param.Grad == null) continue;

                var grad = param.Grad;
                if (WeightDecay > 0)
                {
                    grad = grad.Add(param.Multiply(WeightDecay));
                }

                ITensor update;
                if (Momentum > 0)
                {
                    if (!velocity.TryGetValue(param, out var v))
                    {
                        v = Tensor.Zeros(param.Shape, param.Device);
                        velocity[param] = v;
                    }

                    v.MultiplyInPlace((float)Momentum);
                    v.AddInPlace(grad);
                    update = v;
                }
                else
                {
                    update = grad;
                }

                // Apply update directly on-device
                param.SubtractInPlace(update.Multiply((float)LearningRate));
            }
        }
        /// <summary>
        /// Resets the gradients of all tracked parameters to null, preparing them for the next training iteration.
        /// </summary>
        /// <param name="parameters">An enumerable of <see cref="ITensor"/> parameters whose gradients are to be cleared.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="parameters"/> is null.</exception>

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
