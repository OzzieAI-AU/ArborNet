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
    /// Implements the Adam (Adaptive Moment Estimation) optimization algorithm.
    /// </summary>
    /// <remarks>
    /// Adam is an algorithm for first-order gradient-based optimization of stochastic objective functions,
    /// based on adaptive estimates of lower-order moments.
    /// </remarks>

    #endregion

    public class Adam : IOptimizer
    {
        /// <summary>
        /// Gets or sets the learning rate (step size) for parameter updates.
        /// </summary>
        /// <value>The current learning rate scale factor.</value>
        public double LearningRate { get; set; }
        private readonly double beta1;
        private readonly double beta2;
        private readonly double eps;
        private readonly double weightDecay;
        private int timestep;
        private readonly Dictionary<ITensor, (ITensor m, ITensor v)> state = new();

        public Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weightDecay = 0.0)
        {
            LearningRate = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
            this.weightDecay = weightDecay;
        }
        /// <summary>
        /// Performs a single optimization step, updating the parameters in-place using their computed gradients.
        /// </summary>
        /// <param name="parameters">An enumerable collection of parameter tensors to be optimized.</param>
        /// <exception cref="ArgumentNullException">Thrown when the <paramref name="parameters"/> collection is null.</exception>

        public void Step(IEnumerable<ITensor> parameters)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));

            timestep++;

            foreach (var param in parameters)
            {
                if (param == null || !param.RequiresGrad || param.Grad == null) continue;

                var grad = param.Grad;
                if (weightDecay > 0)
                    grad = grad.Add(param.Multiply(weightDecay));

                if (!state.TryGetValue(param, out var s))
                {
                    s.m = Tensor.Zeros(param.Shape, param.Device);
                    s.v = Tensor.Zeros(param.Shape, param.Device);
                    state[param] = s;
                }

                s.m.MultiplyInPlace((float)beta1);
                s.m.AddInPlace(grad.Multiply(1f - (float)beta1));

                s.v.MultiplyInPlace((float)beta2);
                s.v.AddInPlace(grad.Multiply(grad).Multiply(1f - (float)beta2));

                double biasCorrection1 = 1.0 - Math.Pow(beta1, timestep);
                double biasCorrection2 = 1.0 - Math.Pow(beta2, timestep);
                float stepSize = (float)(LearningRate * Math.Sqrt(biasCorrection2) / biasCorrection1);

                var denom = s.v.Sqrt().Add(Tensor.FromScalar((float)eps, param.Device));
                var update = s.m.Divide(denom).Multiply(stepSize);

                // Run fully in-place on-device
                param.SubtractInPlace(update);
                state[param] = s;
            }
        }
        /// <summary>
        /// Clears the gradients of all active parameters in the specified collection by setting them to zero.
        /// </summary>
        /// <param name="parameters">An enumerable collection of parameter tensors whose gradients should be reset.</param>
        /// <exception cref="ArgumentNullException">Thrown when the <paramref name="parameters"/> collection is null.</exception>

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