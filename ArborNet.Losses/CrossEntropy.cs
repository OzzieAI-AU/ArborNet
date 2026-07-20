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
    using ArborNet.Activations;
    /// <summary>
    /// Implements the categorical cross-entropy loss function.
    /// </summary>
    /// <remarks>
    /// Cross-entropy loss measures the difference between the predicted probability distribution
    /// and the true distribution (one-hot or sparse labels). This implementation combines softmax
    /// and negative log-likelihood in a single operation for numerical stability and efficiency.
    /// It supports both dense (one-hot encoded) and sparse (class index) target formats.
    /// </remarks>

    #endregion

    public class CrossEntropy : BaseLoss
    {
        /// <summary>
        /// Computes the forward pass of the cross-entropy loss function.
        /// </summary>
        /// <param name="predictions">The predicted raw outputs (logits) tensor from the network model.</param>
        /// <param name="targets">The target values tensor. This can be one-hot encoded (dense) with the same shape as <paramref name="predictions"/>, or class indices (sparse).</param>
        /// <param name="reduction">The reduction method to apply to the output. Options are "mean" (average loss over the batch), "sum" (sum of losses), or "none" (unreduced element-wise loss). Defaults to "mean".</param>
        /// <returns>An <see cref="ITensor"/> containing the calculated and reduced loss.</returns>
        /// <exception cref="ArgumentNullException">Thrown when either <paramref name="predictions"/> or <paramref name="targets"/> is null.</exception>
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
