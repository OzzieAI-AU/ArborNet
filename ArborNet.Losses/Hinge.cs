using System;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Losses
{
    /// <summary>
    /// Hinge loss for binary classification with full autograd support.
    /// Targets should be -1 or +1.
    /// </summary>
    public class Hinge : BaseLoss
    {
        /// <summary>
        /// Computes the hinge loss between predictions and targets.
        /// </summary>
        /// <param name="predictions">The predicted values (typically model logits).</param>
        /// <param name="targets">The target values. Must be -1 or +1 for binary classification.</param>
        /// <param name="reduction">The reduction to apply to the output: 
        /// "none" | "mean" (default) | "sum".</param>
        /// <returns>The computed hinge loss as an <see cref="ITensor"/>.</returns>
        /// <remarks>
        /// <para>
        /// The hinge loss for each element is defined as:
        /// <c>max(0, 1 - target * prediction)</c>
        /// </para>
        /// <para>
        /// This implementation supports full automatic differentiation. When 
        /// <see cref="ITensor.RequiresGrad"/> is <see langword="true"/>, a 
        /// <see cref="ITensor.GradFn"/> is attached that computes the gradient 
        /// with respect to the predictions.
        /// </para>
        /// <para>
        /// The gradient is <c>-target</c> where the margin is positive, and zero otherwise.
        /// </para>
        /// </remarks>
        public override ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean")
        {
            ValidateInputs(predictions, targets);

            var one = Tensor.Ones(predictions.Shape, predictions.Device);
            var prod = targets.Multiply(predictions);
            var margin = one.Subtract(prod);
            var zero = Tensor.Zeros(margin.Shape, predictions.Device);

            var hinge = margin.Where(margin.GreaterThan(zero), margin, zero);
            var loss = hinge;

            loss = ApplyReduction(loss, reduction, predictions);

            if (predictions.RequiresGrad)
            {
                loss.GradFn = gradOutput =>
                {
                    var gradMask = margin.GreaterThan(zero);
                    var gradPred = targets.Multiply(-1.0f).Where(gradMask, targets.Multiply(-1.0f), zero);
                    return gradPred.Multiply(gradOutput);
                };
            }

            return loss;
        }
    }
}