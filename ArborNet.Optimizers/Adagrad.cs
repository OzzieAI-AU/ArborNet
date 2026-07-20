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
    /// Implements the Adagrad (Adaptive Gradient) optimization algorithm.
    /// Adagrad adapts the learning rate to the parameters, performing larger updates
    /// for infrequent parameters and smaller updates for frequent parameters, which
    /// makes it well-suited for dealing with sparse data.
    /// </summary>

    #endregion

    public class Adagrad : IOptimizer
    {
        /// <summary>
        /// Gets or sets the learning rate (step size) used for updating the parameters.
        /// </summary>
        /// <value>
        /// The step size used to scale the gradient updates during optimization.
        /// </value>
        public double LearningRate { get; set; }
        private readonly double epsilon;
        private readonly Dictionary<ITensor, ITensor> accumulatedSquares = new();

        public Adagrad(double learningRate = 0.01, double epsilon = 1e-10)
        {
            LearningRate = learningRate;
            this.epsilon = epsilon;
        }
        /// <summary>
        /// Performs a single optimization step, updating the provided parameters in-place based on their gradients.
        /// </summary>
        /// <param name="parameters">The collection of parameter tensors to be updated. Only parameters that require gradients and have gradients computed will be modified.</param>
        /// <exception cref="ArgumentNullException">Thrown when the <paramref name="parameters"/> sequence is null.</exception>

        public void Step(IEnumerable<ITensor> parameters)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));

            foreach (var param in parameters)
            {
                if (param == null || !param.RequiresGrad || param.Grad == null) continue;

                var grad = param.Grad;

                if (!accumulatedSquares.TryGetValue(param, out var accum))
                {
                    accum = Tensor.Zeros(param.Shape, param.Device);
                    accumulatedSquares[param] = accum;
                }

                accum.AddInPlace(grad.Multiply(grad));

                var denom = accum.Sqrt().Add(Tensor.FromScalar((float)epsilon, param.Device));
                var update = grad.Divide(denom).Multiply((float)LearningRate);

                // Run fully in-place on-device
                param.SubtractInPlace(update);
            }
        }
        /// <summary>
        /// Resets the gradients of all parameters that require gradients to zero.
        /// </summary>
        /// <param name="parameters">The collection of parameter tensors whose gradients should be cleared.</param>
        /// <exception cref="ArgumentNullException">Thrown when the <paramref name="parameters"/> sequence is null.</exception>

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
