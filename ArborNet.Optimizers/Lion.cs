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
    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Implements the Lion (EvoSign) optimization algorithm discovered by Google Brain.
    /// Lion is a memory-efficient, high-speed optimizer that tracks only the first momentum buffer
    /// and uses the sign function to update parameters.
    /// </summary>
    /// <remarks>
    /// Compared to Adam, Lion only tracks the first momentum (storing half the auxiliary state memory)
    /// and applies an element-wise sign operation for updates, resulting in uniform update magnitudes
    /// and potentially faster convergence on modern hardware.
    /// </remarks>

    public sealed class Lion : IOptimizer
    {
        private double _lr;
        private readonly double _beta1;
        private readonly double _beta2;
        private readonly double _weightDecay;
        private readonly Dictionary<ITensor, ITensor> _momentum = new();
        /// <summary>
        /// Gets or sets the learning rate (step size) used for parameter updates.
        /// </summary>
        /// <value>The positive scalar value representing the learning rate.</value>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the assigned value is less than or equal to zero.</exception>

        public double LearningRate
        {
            get => _lr;
            set
            {
                if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value), "Learning rate must be positive.");
                _lr = value;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Lion"/> optimizer.
        /// </summary>
        /// <param name="learningRate">Learning rate step factor.</param>
        /// <param name="beta1">Momentum factor for update calculation. Default is 0.9.</param>
        /// <param name="beta2">Momentum factor for EMA history tracking. Default is 0.99.</param>
        /// <param name="weightDecay">Decoupled L2 weight decay factor.</param>
        public Lion(double learningRate = 1e-4, double beta1 = 0.9, double beta2 = 0.99, double weightDecay = 0.0)
        {
            if (learningRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
            if (beta1 < 0 || beta1 >= 1)
                throw new ArgumentOutOfRangeException(nameof(beta1), "Beta1 must be in the range [0, 1).");
            if (beta2 < 0 || beta2 >= 1)
                throw new ArgumentOutOfRangeException(nameof(beta2), "Beta2 must be in the range [0, 1).");
            if (weightDecay < 0)
                throw new ArgumentOutOfRangeException(nameof(weightDecay), "Weight decay must be non-negative.");

            _lr = learningRate;
            _beta1 = beta1;
            _beta2 = beta2;
            _weightDecay = weightDecay;
        }
        /// <summary>
        /// Performs a single optimization step, updating the provided parameters in-place.
        /// </summary>
        /// <param name="parameters">An enumerable of parameter tensors to update. Tensors without gradients or not requiring gradients are skipped.</param>
        /// <exception cref="ArgumentNullException">Thrown if the <paramref name="parameters"/> collection is null.</exception>

        public void Step(IEnumerable<ITensor> parameters)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));

            using (new TensorScope())
            {
                foreach (var p in parameters)
                {
                    if (p == null || !p.RequiresGrad || p.Grad == null) continue;

                    var grad = p.Grad;

                    // 1. Initialize momentum if not already present
                    if (!_momentum.TryGetValue(p, out var m))
                    {
                        m = Tensor.Zeros(p.Shape, p.Device);
                        _momentum[p] = m;
                    }

                    // 2. Perform decoupled weight decay step (in-place on-device scaling)
                    if (_weightDecay > 0)
                    {
                        p.MultiplyInPlace(1f - (float)(_weightDecay * _lr));
                    }

                    // 3. Compute the update step factor: c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                    var c = m.Multiply((float)_beta1).Add(grad.Multiply(1f - (float)_beta1));

                    // 4. Update the model weights using the sign of c_t: theta_t -= lr * sign(c_t)
                    var update = c.Sign().Multiply((float)_lr);
                    p.SubtractInPlace(update);

                    // 5. Update momentum EMA history: m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
                    m.MultiplyInPlace((float)_beta2);
                    m.AddInPlace(grad.Multiply(1f - (float)_beta2));
                }
            }
        }
        /// <summary>
        /// Resets the gradients of all target parameters to a zero-filled tensor matching their shape and device.
        /// </summary>
        /// <param name="parameters">An enumerable of parameter tensors whose gradients should be cleared.</param>
        /// <exception cref="ArgumentNullException">Thrown if the <paramref name="parameters"/> collection is null.</exception>

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