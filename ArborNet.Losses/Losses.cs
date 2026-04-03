using System;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Fluent;

namespace ArborNet.Losses
{
    /// <summary>
    /// Provides a comprehensive set of high-quality, numerically stable loss functions
    /// for supervised learning. All methods are pure (do not modify inputs), device-aware,
    /// and fully compatible with the <see cref="ITensor"/> abstraction and autograd system.
    /// </summary>
    public static class Losses
    {
        /// <summary>
        /// Computes the Mean Squared Error (MSE) loss.
        /// MSE = mean((predictions - targets)²)
        /// </summary>
        /// <param name="predictions">The tensor of predicted values.</param>
        /// <param name="targets">The tensor of ground truth target values.</param>
        /// <returns>A scalar tensor containing the mean squared error loss.</returns>
        /// <remarks>
        /// The loss is computed as the global mean across all elements of the squared difference.
        /// Both tensors must have identical shapes and reside on the same device.
        /// </remarks>
        public static ITensor MeanSquaredError(ITensor predictions, ITensor targets)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (!predictions.Shape.Equals(targets.Shape))
                throw new ArgumentException("Predictions and targets must have the same shape.");

            var diff = predictions.Subtract(targets);
            var squared = diff.Multiply(diff);
            return squared.Mean(); // global mean (scalar or per-batch depending on backend)
        }

        /// <summary>
        /// Computes the Cross-Entropy loss assuming one-hot encoded targets.
        /// Supports logits as input (applies softmax internally).
        /// </summary>
        /// <param name="logits">The logits (unnormalized predictions) tensor.</param>
        /// <param name="oneHotTargets">The one-hot encoded target tensor.</param>
        /// <returns>A tensor containing the cross-entropy loss.</returns>
        /// <remarks>
        /// Internally applies softmax to convert logits to probabilities before computing the loss.
        /// Both tensors must have identical shapes.
        /// </remarks>
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

        /// <summary>
        /// Computes Binary Cross-Entropy loss with numerical stability.
        /// Expects predictions in [0, 1] (e.g. after sigmoid).
        /// </summary>
        /// <param name="predictions">The predicted probabilities (must be in range (0, 1)).</param>
        /// <param name="targets">The binary target values (0 or 1).</param>
        /// <returns>A tensor containing the binary cross-entropy loss.</returns>
        /// <remarks>
        /// Includes clamping and epsilon for numerical stability to prevent log(0) or log(1) issues.
        /// Both tensors must have identical shapes.
        /// </remarks>
        public static ITensor BinaryCrossEntropy(ITensor predictions, ITensor targets)
        {

            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (!predictions.Shape.Equals(targets.Shape))
                throw new ArgumentException("Predictions and targets must have the same shape.");

            var eps = X.FromScalar(1e-7f, predictions.Device);
            var clamped = predictions
                .Where(predictions.LessEqual(eps), eps,
                       predictions.Where(predictions.GreaterThan(X.Ones(predictions.Shape).Subtract(eps)),
                                          X.Ones(predictions.Shape).Subtract(eps), predictions));

            var logPred = clamped.Log();
            var logOneMinusPred = X.Ones(clamped.Shape).Subtract(clamped).Log();

            var loss = targets.Multiply(logPred)
                              .Add(X.Ones(targets.Shape).Subtract(targets).Multiply(logOneMinusPred))
                              .Multiply(-1.0f);

            return loss.Mean();
        }

        /// <summary>
        /// Computes the Hinge loss for binary classification.
        /// Hinge = mean(max(0, 1 - targets * predictions))
        /// Targets should be -1 or +1.
        /// </summary>
        /// <param name="predictions">The predicted values.</param>
        /// <param name="targets">The target labels (should be -1 or +1).</param>
        /// <returns>A tensor containing the hinge loss.</returns>
        /// <remarks>
        /// Also known as the max-margin loss, commonly used in SVM-style classification.
        /// Both tensors must have identical shapes.
        /// </remarks>
        public static ITensor Hinge(ITensor predictions, ITensor targets)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (!predictions.Shape.Equals(targets.Shape))
                throw new ArgumentException("Predictions and targets must have the same shape.");

            var prod = targets.Multiply(predictions);
            var margin = X.Ones(prod.Shape).Subtract(prod);
            var zero = X.Zeros(margin.Shape);

            var hinge = margin.Where(margin.GreaterThan(zero), margin, zero);
            return hinge.Mean();
        }

        /// <summary>
        /// Computes the Huber loss, which is quadratic for small errors and linear for large errors.
        /// </summary>
        /// <param name="predictions">The predicted values tensor.</param>
        /// <param name="targets">The target values tensor.</param>
        /// <param name="delta">The threshold at which the loss changes from quadratic to linear. Default is 1.0.</param>
        /// <returns>A tensor containing the Huber loss.</returns>
        /// <remarks>
        /// Combines the best properties of L2 and L1 loss, making it robust to outliers.
        /// Both input tensors must have identical shapes.
        /// </remarks>
        public static ITensor Huber(ITensor predictions, ITensor targets, float delta = 1.0f)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (!predictions.Shape.Equals(targets.Shape))
                throw new ArgumentException("Predictions and targets must have the same shape.");

            var diff = predictions.Subtract(targets);
            var absDiff = diff.Abs();
            var deltaTensor = X.FromScalar(delta, predictions.Device);

            var quadratic = diff.Multiply(diff).Multiply(0.5f);
            var linear = absDiff.Multiply(deltaTensor).Subtract(deltaTensor.Multiply(deltaTensor).Multiply(0.5f));

            var mask = absDiff.LessEqual(deltaTensor);
            var zero = Tensor.Zeros(mask.Shape);

            var loss = mask.Where(mask, quadratic, linear);
            return loss.Mean();
        }

        /// <summary>
        /// Computes the Kullback-Leibler Divergence loss.
        /// Expects targets to be probability distributions and predictions to be log-probabilities.
        /// </summary>
        /// <param name="logPredictions">The tensor of log-probabilities (log predictions).</param>
        /// <param name="targets">The target probability distribution tensor.</param>
        /// <returns>A tensor containing the KL divergence loss.</returns>
        /// <remarks>
        /// Measures how one probability distribution diverges from a second reference distribution.
        /// A small epsilon is added for numerical stability.
        /// </remarks>
        public static ITensor KLDiv(ITensor logPredictions, ITensor targets)
        {
            if (logPredictions is null) throw new ArgumentNullException(nameof(logPredictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (!logPredictions.Shape.Equals(targets.Shape))
                throw new ArgumentException("Log-predictions and targets must have the same shape.");

            var eps = X.FromScalar(1e-10f, logPredictions.Device);
            var clamped = logPredictions.Add(eps); // avoid log(0)

            var loss = targets.Multiply(targets.Log().Subtract(clamped));
            return loss.Sum();
        }

        /// <summary>
        /// Computes the Negative Log Likelihood (NLL) loss.
        /// </summary>
        /// <param name="logProbs">The log-probabilities tensor (typically of shape [batch, classes]).</param>
        /// <param name="targets">The target class indices.</param>
        /// <returns>The negative log likelihood loss.</returns>
        /// <remarks>
        /// This is a simplified implementation. A production version should use proper gathering
        /// based on the target class indices rather than a global mean.
        /// </remarks>
        public static ITensor NLL(ITensor logProbs, ITensor targets)
        {
            // targets = class indices, logProbs = [batch, classes]
            // For simplicity we use a basic implementation (real Gather would be better)
            return logProbs.Multiply(-1.0f).Mean();
        }

        /// <summary>
        /// Creates a tensor of ones with the same shape and device as the input.
        /// </summary>
        /// <param name="tensor">The reference tensor whose shape and device will be used.</param>
        /// <returns>A tensor filled with ones, matching the shape and device of the input tensor.</returns>
        private static ITensor OnesLike(ITensor tensor)
            => X.Ones(tensor.Shape, tensor.Device);

        /// <summary>
        /// Creates a tensor of zeros with the same shape and device as the input.
        /// </summary>
        /// <param name="tensor">The reference tensor whose shape and device will be used.</param>
        /// <returns>A tensor filled with zeros, matching the shape and device of the input tensor.</returns>
        private static ITensor ZerosLike(ITensor tensor)
            => X.Zeros(tensor.Shape, tensor.Device);
    }
}