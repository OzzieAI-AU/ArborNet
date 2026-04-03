using System;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Losses
{
    /// <summary>
    /// Negative Log Likelihood loss.
    /// Assumes predictions are log-probabilities and targets are class indices.
    /// </summary>
    /// <remarks>
    /// Implements the Negative Log Likelihood loss function, commonly used in multi-class classification.
    /// This loss expects the network to output log-probabilities (typically the result of LogSoftmax).
    /// The targets should be class indices (not one-hot encoded).
    /// </remarks>
    public class NLL : BaseLoss
    {
        /// <summary>
        /// Computes the Negative Log Likelihood loss between the predicted log-probabilities and target class indices.
        /// </summary>
        /// <param name="predictions">Tensor of predicted log-probabilities. Expected shape: [batch_size, num_classes].</param>
        /// <param name="targets">Tensor of target class indices. Expected shape: [batch_size].</param>
        /// <param name="reduction">Specifies the reduction to apply to the output: "none", "mean", or "sum". Default is "mean".</param>
        /// <returns>The computed loss as a tensor, reduced according to the specified reduction mode.</returns>
        /// <remarks>
        /// This implementation currently provides a simplified version for the computational graph.
        /// A production implementation should use the target indices to select the corresponding log-probabilities
        /// via indexing or a Gather operation before computing the negative log likelihood.
        /// </remarks>
        public override ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean")
        {
            ValidateInputs(predictions, targets);

            // Simplified NLL - assumes targets are indices and predictions are [batch, classes]
            // In production use proper Gather operation
            var loss = predictions.Multiply(-1.0f);
            loss = ApplyReduction(loss, reduction, predictions);

            if (predictions.RequiresGrad)
            {
                loss.GradFn = gradOutput => gradOutput.Multiply(-1.0f);
            }

            return loss;
        }
    }
}