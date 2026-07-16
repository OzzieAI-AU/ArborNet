using System;
using System.Collections.Generic;
using System.Linq;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Core.Native.AMP
{
    /// <summary>
    /// Implements a high-performance Automatic Mixed Precision (AMP) Gradient Scaler.
    /// Prevents numerical underflow (gradients becoming 0.0) and overflow (NaN/Inf) 
    /// when training neural networks with half-precision floating-point values.
    /// </summary>
    public sealed class MixedPrecisionScaler
    {
        private float _scale;
        private readonly float _growthFactor;
        private readonly float _backoffFactor;
        private readonly int _growthInterval;
        private int _consecutiveSuccessfulSteps;

        /// <summary>
        /// Gets the current loss scale factor.
        /// </summary>
        public float CurrentScale => _scale;

        /// <summary>
        /// Initializes a new instance of the <see cref="MixedPrecisionScaler"/> class.
        /// </summary>
        /// <param name="initialScale">The starting scale factor. Defaults to 65536.0f.</param>
        /// <param name="growthFactor">Multiplier used to increase the scale factor. Defaults to 2.0f.</param>
        /// <param name="backoffFactor">Multiplier used to decrease the scale factor. Defaults to 0.5f.</param>
        /// <param name="growthInterval">Number of consecutive successful steps without NaNs/Infs before growing scale.</param>
        public MixedPrecisionScaler(
            float initialScale = 65536f,
            float growthFactor = 2f,
            float backoffFactor = 0.5f,
            int growthInterval = 2000)
        {
            _scale = initialScale;
            _growthFactor = growthFactor;
            _backoffFactor = backoffFactor;
            _growthInterval = growthInterval;
            _consecutiveSuccessfulSteps = 0;
        }

        /// <summary>
        /// Multiplies the computed loss tensor by the scale factor before backpropagation.
        /// </summary>
        public ITensor Scale(ITensor loss)
        {
            if (loss == null) throw new ArgumentNullException(nameof(loss));
            return loss.Multiply(_scale);
        }

        /// <summary>
        /// Divides the gradients of the parameters by the loss scale factor to restore original values.
        /// </summary>
        public void Unscale(IEnumerable<ITensor> parameters)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));

            foreach (var param in parameters)
            {
                if (param == null || !param.RequiresGrad || param.Grad == null) continue;

                // Divide the gradient by the current loss scale factor
                var unscaledGrad = param.Grad.Divide(_scale);
                param.Grad = unscaledGrad;
            }
        }

        /// <summary>
        /// Steps the optimizer, skipping updates and backing off scale if NaNs or Infinities are detected.
        /// </summary>
        /// <param name="optimizer">The optimizer used for parameter updates.</param>
        /// <param name="parameters">The parameters to optimize.</param>
        /// <returns>True if the step succeeded; False if NaNs/Infs were detected and the step was skipped.</returns>
        public bool Step(IOptimizer optimizer, IEnumerable<ITensor> parameters)
        {
            if (optimizer == null) throw new ArgumentNullException(nameof(optimizer));
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));

            var parameterList = parameters.ToList();

            // Check if any gradient contains NaN or Infinity values
            if (HasInvalidGradients(parameterList))
            {
                // Decrease scale and skip optimization step to purge bad activations
                _scale = Math.Max(1.0f, _scale * _backoffFactor);
                _consecutiveSuccessfulSteps = 0;

                // Clear out corrupt gradients
                optimizer.ZeroGrad(parameterList);
                return false;
            }

            // Normal parameter step
            optimizer.Step(parameterList);

            // Increment scale growth counter
            _consecutiveSuccessfulSteps++;
            if (_consecutiveSuccessfulSteps >= _growthInterval)
            {
                _scale *= _growthFactor;
                _consecutiveSuccessfulSteps = 0;
            }

            return true;
        }

        /// <summary>
        /// Scans parameters to find any NaN or Infinite values in their gradients.
        /// </summary>
        private static bool HasInvalidGradients(IEnumerable<ITensor> parameters)
        {
            foreach (var param in parameters)
            {
                if (param?.Grad == null) continue;

                var data = param.Grad.ToArray();
                for (int i = 0; i < data.Length; i++)
                {
                    float val = data[i];
                    if (float.IsNaN(val) || float.IsInfinity(val))
                    {
                        return true;
                    }
                }
            }
            return false;
        }
    }
}