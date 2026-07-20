// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Losses
{

    #region Using Statements:

    using System;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Represents the Kullback-Leibler (KL) Divergence loss function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Kullback-Leibler divergence measures how one probability distribution diverges 
    /// from a second expected probability distribution.
    /// </para>
    /// <para>
    /// This implementation expects the <paramref name="targets"/> to be given as probabilities 
    /// and the <paramref name="predictions"/> to be given as log-probabilities (e.g., from a LogSoftmax layer).
    /// </para>
    /// <para>
    /// The forward formula computed per element is: 
    /// <c>loss = target * (log(target) - prediction)</c>
    /// </para>
    /// </remarks>

    #endregion

    public class KLDiv : BaseLoss
    {
        /// <summary>
        /// Small constant added to predictions for numerical stability to avoid 
        /// taking the logarithm of zero.
        /// </summary>
        private const float EPS = 1e-10f;
        /// <summary>
        /// Computes the Kullback-Leibler Divergence loss between the predicted 
        /// log-probabilities and target probabilities.
        /// </summary>
        /// <param name="predictions">The tensor containing the predicted log-probabilities.</param>
        /// <param name="targets">The tensor containing the ground-truth target probabilities.</param>
        /// <param name="reduction">
        /// Specifies the reduction reduction to apply to the output. 
        /// Options include:
        /// <list type="bullet">
        /// <item><description><c>"none"</c>: No reduction is applied.</description></item>
        /// <item><description><c>"mean"</c>: The sum of the output is divided by the number of elements.</description></item>
        /// <item><description><c>"sum"</c>: The output will be summed.</description></item>
        /// </list>
        /// Default is <c>"mean"</c>.
        /// </param>
        /// <returns>An <see cref="ITensor"/> containing the calculated KL divergence loss.</returns>
        /// <exception cref="ArgumentNullException">Thrown if either <paramref name="predictions"/> or <paramref name="targets"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown if input shapes or devices are incompatible.</exception>

        public override ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean")
        {
            ValidateInputs(predictions, targets);

            var device = predictions.Device;
            var epsTensor = Tensor.FromScalar(EPS, device);
            var clampedPred = predictions.Add(epsTensor);

            var logTargets = targets.Log();
            var diff = logTargets.Subtract(clampedPred);
            var weighted = targets.Multiply(diff);
            var loss = weighted;

            loss = ApplyReduction(loss, reduction, predictions);

            return loss;
        }
    }
}