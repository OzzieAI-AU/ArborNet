using System;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Losses
{
    /// <summary>
    /// Implements the Binary Cross Entropy (BCE) loss function, commonly used for binary classification tasks.
    /// </summary>
    /// <remarks>
    /// The binary cross entropy loss is defined as:
    /// <c>- (y * log(p) + (1 - y) * log(1 - p))</c>
    /// where <c>y</c> is the target label (0 or 1) and <c>p</c> is the predicted probability.
    /// 
    /// This implementation includes input validation, numerical stability clipping of predictions
    /// to the range [eps, 1-eps], and support for automatic differentiation when gradients are required.
    /// </remarks>
    public class BinaryCrossEntropy : BaseLoss
    {
        /// <summary>
        /// Small constant value used to clip predicted probabilities to prevent 
        /// taking the logarithm of zero, which would cause numerical instability.
        /// </summary>
        private const float EPS = 1e-12f;

        /// <summary>
        /// Computes the binary cross entropy loss between predicted probabilities and target labels.
        /// </summary>
        /// <param name="predictions">The predicted probabilities. Expected values are in the range (0, 1).</param>
        /// <param name="targets">The ground truth binary labels (0 or 1).</param>
        /// <param name="reduction">Specifies the reduction to apply to the output: 
        /// "none" | "mean" | "sum". Default is "mean".</param>
        /// <returns>The computed binary cross entropy loss as a tensor.</returns>
        /// <remarks>
        /// Predictions are clamped to [EPS, 1-EPS] to ensure numerical stability.
        /// If <paramref name="predictions"/> requires gradients, a custom gradient function 
        /// is attached to the output tensor to support backpropagation.
        /// </remarks>
        public override ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean")
        {
            ValidateInputs(predictions, targets);

            var device = predictions.Device;
            var one = Tensor.Ones(predictions.Shape, device);
            var epsTensor = Tensor.FromScalar(EPS, device);

            var clamped = predictions.Where(
                predictions.LessEqual(epsTensor), 
                epsTensor, 
                predictions.Where(
                    predictions.GreaterThan(one.Subtract(epsTensor)), 
                    one.Subtract(epsTensor), 
                    predictions
                )
            );

            var logPred = clamped.Log();
            var logOneMinus = one.Subtract(clamped).Log();
            var loss = targets.Multiply(logPred)
                              .Add(one.Subtract(targets).Multiply(logOneMinus))
                              .Multiply(-1.0f);

            loss = ApplyReduction(loss, reduction, predictions);

            if (predictions.RequiresGrad)
            {
                loss.GradFn = gradOutput =>
                {
                    var grad = clamped.Subtract(targets)
                                     .Divide(clamped.Multiply(one.Subtract(clamped)));
                    return grad.Multiply(gradOutput);
                };
            }

            return loss;
        }
    }
}