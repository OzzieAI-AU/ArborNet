using System;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Activations;

namespace ArborNet.Losses
{
    /// <summary>
    /// Implements the categorical cross-entropy loss function.
    /// </summary>
    /// <remarks>
    /// Cross-entropy loss measures the difference between the predicted probability distribution
    /// and the true distribution (one-hot or sparse labels). This implementation combines softmax
    /// and negative log-likelihood in a single operation for numerical stability and efficiency.
    /// It supports both dense (one-hot encoded) and sparse (class index) target formats.
    /// </remarks>
    public class CrossEntropy : BaseLoss
    {
        public override ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean")
        {
            if (predictions == null) throw new ArgumentNullException(nameof(predictions));
            if (targets == null) throw new ArgumentNullException(nameof(targets));

            // 1. Apply Softmax to predictions along the class dimension
            var probs = new Softmax(-1).Forward(predictions);

            // 2. Compute Log Probabilities
            var logProbs = probs.Log();

            ITensor loss;
            if (targets.Shape.Equals(probs.Shape))
            {
                // Dense targets (One-Hot Encoded)
                loss = targets.Multiply(logProbs).Multiply(-1.0f);
            }
            else
            {
                // Sparse targets (Class Indices): Gather negative log probs
                loss = logProbs.Negate().Gather(axis: 1, targets);
            }

            // 3. Apply reduction (Mean, Sum, or None)
            return ApplyReduction(loss, reduction, predictions);
        }
    }
}
