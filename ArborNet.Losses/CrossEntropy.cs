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
        /// <summary>
        /// Computes the cross-entropy loss between predictions and targets.
        /// </summary>
        /// <param name="predictions">The raw predictions (logits) from the model. 
        /// Must be a tensor of shape (batch_size, num_classes) or similar.</param>
        /// <param name="targets">The target labels. When the shape matches <paramref name="predictions"/>, 
        /// targets are treated as one-hot encoded. Otherwise, they are treated as class indices.</param>
        /// <param name="reduction">The reduction to apply to the output: 'none', 'mean', or 'sum'. 
        /// Default is 'mean'.</param>
        /// <returns>The computed loss as a tensor. When reduced with 'mean' or 'sum', returns a scalar tensor.</returns>
        /// <remarks>
        /// The forward pass applies a softmax activation along the last dimension of the predictions,
        /// computes the negative log likelihood with the targets, and applies the specified reduction.
        /// If <see cref="ITensor.RequiresGrad"/> is true on the predictions, a gradient function is
        /// attached for automatic differentiation during backpropagation.
        /// </remarks>
        public override ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean")
        {
            ValidateInputs(predictions, targets);

            var probs = new Softmax(-1).Forward(predictions);
            var logProbs = probs.Log();

            ITensor loss;
            if (targets.Shape.Equals(probs.Shape))
            {
                loss = targets.Multiply(logProbs).Multiply(-1.0f);
            }
            else
            {
                loss = logProbs.Negate().Where(targets, logProbs.Negate(), Tensor.Zeros(logProbs.Shape));
            }

            loss = ApplyReduction(loss, reduction, predictions);

            if (predictions.RequiresGrad)
            {
                loss.GradFn = gradOutput =>
                {
                    var grad = probs.Subtract(targets);
                    return grad.Multiply(gradOutput);
                };
            }

            return loss;
        }
    }
}