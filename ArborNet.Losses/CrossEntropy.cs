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
        public override ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean")
        {
            if (predictions == null) throw new ArgumentNullException(nameof(predictions));
            if (targets == null) throw new ArgumentNullException(nameof(targets));

            var probs = new Softmax(-1).Forward(predictions);
            var logProbs = probs.Log();

            ITensor loss;
            if (targets.Shape.Equals(probs.Shape))
            {
                // FIX: Sum across the class dimension (-1) before global reduction!
                loss = targets.Multiply(logProbs).Multiply(-1.0f).Sum(-1);
            }
            else
            {
                loss = logProbs.Negate().Gather(axis: 1, targets);
            }

            return ApplyReduction(loss, reduction, predictions);
        }
    }
}