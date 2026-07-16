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
        public override ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean")
        {
            if (predictions == null) throw new ArgumentNullException(nameof(predictions));
            if (targets == null) throw new ArgumentNullException(nameof(targets));

            // 1. Gather target class log-probabilities along axis 1 (classes)
            var gathered = predictions.Gather(axis: 1, targets);

            // 2. Compute Negative Log-Likelihood: loss = -1.0 * gathered
            var loss = gathered.Multiply(-1.0f);

            // 3. Apply reduction (Mean, Sum, or None). 
            // All operators automatically register their correct autograd functions.
            return ApplyReduction(loss, reduction, predictions);
        }
    }
}