// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Native.AMP
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.Linq;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Implements a high-performance Automatic Mixed Precision (AMP) Gradient Scaler.
    /// Prevents numerical underflow (gradients becoming 0.0) and overflow (NaN/Inf) 
    /// when training neural networks with half-precision floating-point values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// During mixed-precision training, some gradient values can underflow to zero due to the limited dynamic range of 16-bit floats (FP16/BF16). 
    /// To prevent this, the <see cref="MixedPrecisionScaler"/> multiplies the loss by a scale factor before backpropagation.
    /// Consequently, gradients are scaled up, preserving small values.
    /// </para>
    /// <para>
    /// Before the optimizer updates the model weights, the gradients must be unscaled using <see cref="Unscale(IEnumerable{ITensor})"/> 
    /// to ensure weight updates are applied at the correct magnitude.
    /// If any gradient contains non-finite values (<see cref="float.IsNaN(float)"/> or <see cref="float.IsInfinity(float)"/>), 
    /// the step is skipped, the scale factor is reduced by the backoff factor, and the gradients are cleared.
    /// If no non-finite gradients are encountered for <c>growthInterval</c> consecutive steps, the scale factor is increased.
    /// </para>
    /// </remarks>

    #endregion

    public sealed class MixedPrecisionScaler
    {
        /// <summary>
        /// The current dynamic scale factor applied to the loss tensor.
        /// </summary>
        private float _scale;

        /// <summary>
        /// The multiplier factor applied to <see cref="_scale"/> to increase it when no invalid gradients are detected.
        /// </summary>
        private readonly float _growthFactor;

        /// <summary>
        /// The multiplier factor applied to <see cref="_scale"/> to decrease it when invalid gradients (NaN/Infinity) are detected.
        /// </summary>
        private readonly float _backoffFactor;

        /// <summary>
        /// The number of consecutive successful optimization steps required before growing the scale factor.
        /// </summary>
        private readonly int _growthInterval;

        /// <summary>
        /// Tracks the number of consecutive optimization steps that completed successfully without encountering invalid gradients.
        /// </summary>
        private int _consecutiveSuccessfulSteps;
        /// <summary>
        /// Gets the current loss scale factor.
        /// </summary>
        /// <value>
        /// A <see cref="float"/> representing the multiplier currently used to scale the loss and unscale the gradients.
        /// </value>

        public float CurrentScale => _scale;

        /// <summary>
        /// Initializes a new instance of the <see cref="MixedPrecisionScaler"/> class with configurable scaling parameters.
        /// </summary>
        /// <param name="initialScale">The starting scale factor. Highly recommended to start large (e.g., <c>65536.0f</c>) to prevent early underflow.</param>
        /// <param name="growthFactor">The multiplier used to increase the scale factor when no invalid gradients are found over the growth interval. Default is <c>2.0f</c>.</param>
        /// <param name="backoffFactor">The multiplier used to decrease the scale factor when NaNs or Infinities are detected. Default is <c>0.5f</c>.</param>
        /// <param name="growthInterval">The number of consecutive successful steps without detecting invalid gradients required to trigger a scale growth. Default is <c>2000</c>.</param>
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
        /// Multiplies the computed loss tensor by the current scale factor.
        /// </summary>
        /// <param name="loss">The loss tensor to be scaled prior to backpropagation.</param>
        /// <returns>An <see cref="ITensor"/> representing the scaled loss.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="loss"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// Scaling the loss ensures that gradients computed during the backward pass are scaled by the same factor,
        /// thereby preventing underflow in representations with low dynamic range.
        /// </remarks>

        public ITensor Scale(ITensor loss)
        {
            if (loss == null) throw new ArgumentNullException(nameof(loss));
            return loss.Multiply(_scale);
        }
        /// <summary>
        /// Divides the gradients of the parameters by the current loss scale factor to restore their original magnitude.
        /// </summary>
        /// <param name="parameters">An enumerable collection of parameters containing the gradients to be unscaled.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="parameters"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// This method should be called after backpropagation and before the optimizer steps, allowing gradient clipping 
        /// and other operations to work on the actual unscaled gradient values.
        /// Only parameters requiring gradients (<c>RequiresGrad == true</c>) with non-null gradients will be processed.
        /// </remarks>

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
        /// Updates the network parameters using the provided optimizer, adjusting or growing the loss scale factor dynamically.
        /// </summary>
        /// <param name="optimizer">The optimizer to use for performing the parameter updates.</param>
        /// <param name="parameters">The collection of parameters whose gradients will be evaluated and updated.</param>
        /// <returns>
        /// <see langword="true"/> if the step succeeded and parameters were updated; 
        /// <see langword="false"/> if invalid gradients (NaN or Infinity) were detected, indicating that the step was skipped, 
        /// the scale factor was reduced, and the gradients were cleared.
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="optimizer"/> or <paramref name="parameters"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// This method encapsulates the logic for detecting gradient overflow (NaNs/Infs). If an overflow is found:
        /// <list type="number">
        /// <item><description>The scale is multiplied by the backoff factor (clamped to a minimum of <c>1.0f</c>).</description></item>
        /// <item><description>The consecutive success counter is reset to zero.</description></item>
        /// <item><description>The step is skipped, and gradients are zeroed to discard contaminated values.</description></item>
        /// </list>
        /// If the step is successful, the scale is grown by the growth factor if the growth interval threshold is reached.
        /// </remarks>

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
        /// Scans the gradients of the provided parameters to identify any invalid floating-point values (NaN or Infinity).
        /// </summary>
        /// <param name="parameters">The collection of parameters whose gradients will be scanned.</param>
        /// <returns>
        /// <see langword="true"/> if any parameter has a gradient containing at least one NaN or Infinity value; otherwise, <see langword="false"/>.
        /// </returns>
        /// <remarks>
        /// This method flattens the gradient values into an array to perform the check. Null parameters or parameters 
        /// without gradients are ignored during the scan.
        /// </remarks>

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