using System;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Losses
{
    /// <summary>
    /// Kullback-Leibler Divergence loss.
    /// Expects targets as probabilities and predictions as log-probabilities.
    /// </summary>
    /// <remarks>
    /// The Kullback-Leibler divergence measures how one probability distribution diverges 
    /// from a second expected probability distribution. This implementation follows the 
    /// forward formula: target * (log(target) - predictions), where predictions are 
    /// expected to be in log space.
    /// </remarks>
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
        /// <param name="predictions">Tensor of predicted log-probabilities.</param>
        /// <param name="targets">Tensor of target probabilities.</param>
        /// <param name="reduction">
        /// Specifies the reduction to apply to the output: "none", "mean", or "sum". 
        /// Default is "mean".
        /// </param>
        /// <returns>The computed KL divergence loss after the specified reduction.</returns>
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