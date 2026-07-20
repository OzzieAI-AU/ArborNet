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

    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using System;

    #endregion

    /// <summary>
    /// Provides a comprehensive set of high-quality, numerically stable loss functions
    /// for supervised learning. All methods are pure (do not modify inputs), device-aware,
    /// and fully compatible with the <see cref="ITensor"/> abstraction and autograd system.
    /// </summary>
    public static class Losses
    {

        public static ITensor MeanSquaredError(ITensor predictions, ITensor targets)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (!predictions.Shape.Equals(targets.Shape))
                throw new ArgumentException("Predictions and targets must have the same shape.");

            var diff = predictions.Subtract(targets);
            var squared = diff.Multiply(diff);
            return squared.Mean();
        }

        public static ITensor CrossEntropy(ITensor logits, ITensor oneHotTargets)
        {
            if (logits is null) throw new ArgumentNullException(nameof(logits));
            if (oneHotTargets is null) throw new ArgumentNullException(nameof(oneHotTargets));
            if (!logits.Shape.Equals(oneHotTargets.Shape))
                throw new ArgumentException("Logits and one-hot targets must have the same shape.");

            var probs = new Activations.Softmax(axis: -1).Forward(logits);
            var logProbs = probs.Log();
            var loss = oneHotTargets.Multiply(logProbs).Multiply(-1.0f);
            return loss.Sum(axis: -1).Mean();
        }

        public static ITensor BinaryCrossEntropy(ITensor predictions, ITensor targets)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (!predictions.Shape.Equals(targets.Shape))
                throw new ArgumentException("Predictions and targets must have the same shape.");

            var device = predictions.Device;
            var eps = Tensor.FromScalar(1e-7f, device);
            var ones = Tensor.Ones(predictions.Shape, device);
            var oneMinusEps = ones.Subtract(eps);

            var clamped = predictions.Where(
                predictions.LessEqual(eps),
                eps,
                predictions.Where(
                    predictions.GreaterThan(oneMinusEps),
                    oneMinusEps,
                    predictions
                )
            );

            var logPred = clamped.Log();
            var logOneMinusPred = Tensor.Ones(clamped.Shape, device).Subtract(clamped).Log();

            var loss = targets.Multiply(logPred)
                              .Add(Tensor.Ones(targets.Shape, device).Subtract(targets).Multiply(logOneMinusPred))
                              .Multiply(-1.0f);

            return loss.Mean();
        }

        public static ITensor Hinge(ITensor predictions, ITensor targets)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (!predictions.Shape.Equals(targets.Shape))
                throw new ArgumentException("Predictions and targets must have the same shape.");

            var device = predictions.Device;
            var prod = targets.Multiply(predictions);
            var margin = Tensor.Ones(prod.Shape, device).Subtract(prod);
            var zero = Tensor.Zeros(margin.Shape, device);

            var hinge = margin.Where(margin.GreaterThan(zero), margin, zero);
            return hinge.Mean();
        }

        public static ITensor Huber(ITensor predictions, ITensor targets, float delta = 1.0f)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (!predictions.Shape.Equals(targets.Shape))
                throw new ArgumentException("Predictions and targets must have the same shape.");

            var device = predictions.Device;
            var diff = predictions.Subtract(targets);
            var absDiff = diff.Abs();
            var deltaTensor = Tensor.FromScalar(delta, device);

            var quadratic = diff.Multiply(diff).Multiply(0.5f);
            var linear = absDiff.Multiply(deltaTensor).Subtract(deltaTensor.Multiply(deltaTensor).Multiply(0.5f));

            var mask = absDiff.LessEqual(deltaTensor);
            var loss = mask.Where(mask, quadratic, linear);
            return loss.Mean();
        }

        public static ITensor KLDiv(ITensor logPredictions, ITensor targets)
        {
            if (logPredictions is null) throw new ArgumentNullException(nameof(logPredictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (!logPredictions.Shape.Equals(targets.Shape))
                throw new ArgumentException("Log-predictions and targets must have the same shape.");

            var device = logPredictions.Device;
            var eps = Tensor.FromScalar(1e-10f, device);
            var clamped = logPredictions.Add(eps);

            var loss = targets.Multiply(targets.Log().Subtract(clamped));
            return loss.Sum();
        }

        /// <summary>
        /// Computes the Negative Log Likelihood (NLL) loss.
        /// Gathers target class log-probabilities along the class dimension and averages them.
        /// </summary>
        public static ITensor NLL(ITensor logProbs, ITensor targets)
        {
            if (logProbs is null) throw new ArgumentNullException(nameof(logProbs));
            if (targets is null) throw new ArgumentNullException(nameof(targets));

            var gathered = logProbs.Gather(axis: 1, targets);
            return gathered.Multiply(-1.0f).Mean();
        }
    }
}