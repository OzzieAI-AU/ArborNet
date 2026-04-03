using System;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core;

namespace ArborNet.Losses
{
    /// <summary>
    /// Implements the Huber loss function, a robust loss that is quadratic for small errors
    /// and linear for large errors, making it less sensitive to outliers than MSE.
    /// </summary>
    /// <remarks>
    /// The Huber loss transitions from L2 to L1 behavior at the <see cref="delta"/> threshold.
    /// This implementation supports both forward computation and automatic differentiation.
    /// </remarks>
    public class Huber : BaseLoss
    {
        /// <summary>
        /// The threshold parameter that determines where the loss function changes 
        /// from quadratic to linear behavior.
        /// </summary>
        private readonly float delta;

        /// <summary>
        /// Initializes a new instance of the <see cref="Huber"/> class.
        /// </summary>
        /// <param name="delta">The threshold value. Must be greater than zero. 
        /// Errors smaller than this are treated as quadratic loss; larger errors are treated linearly.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="delta"/> is less than or equal to zero.</exception>
        public Huber(float delta = 1.0f)
        {
            if (delta <= 0) throw new ArgumentOutOfRangeException(nameof(delta));
            this.delta = delta;
        }

        /// <summary>
        /// Computes the Huber loss between predictions and target values.
        /// </summary>
        /// <param name="predictions">The predicted values tensor.</param>
        /// <param name="targets">The ground truth target values tensor.</param>
        /// <param name="reduction">The type of reduction to apply to the output: 
        /// "none", "mean" (default), or "sum".</param>
        /// <returns>A tensor containing the computed Huber loss, reduced according to the specified strategy.</returns>
        /// <remarks>
        /// The loss is defined as:
        /// <list type="bullet">
        ///   <item><description>0.5 * err², if |err| ≤ delta (quadratic region)</description></item>
        ///   <item><description>delta * |err| - 0.5 * delta², if |err| &gt; delta (linear region)</description></item>
        /// </list>
        /// When <see cref="ITensor.RequiresGrad"/> is true on the predictions tensor, 
        /// a custom gradient function is attached for backpropagation.
        /// </remarks>
        public override ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean")
        {
            ValidateInputs(predictions, targets);

            var device = predictions.Device;
            var err = predictions.Subtract(targets);
            var absErr = err.Abs();
            var deltaTensor = Tensor.FromScalar(delta, device);
            var zero = Tensor.Zeros(err.Shape, device);

            var quadratic = err.Multiply(err).Multiply(0.5f);
            var linear = absErr.Multiply(deltaTensor).Subtract(deltaTensor.Multiply(deltaTensor).Multiply(0.5f));

            var isQuadratic = absErr.LessEqual(deltaTensor);
            var loss = isQuadratic.Where(isQuadratic, quadratic, linear);

            loss = ApplyReduction(loss, reduction, predictions);

            if (predictions.RequiresGrad)
            {
                loss.GradFn = gradOutput =>
                {
                    var positive = err.GreaterThan(zero);
                    var sign = positive.Where(positive, Tensor.Ones(err.Shape, device), Tensor.Ones(err.Shape, device).Negate());
                    var grad = err.Where(isQuadratic, err, deltaTensor.Multiply(sign));
                    return grad.Multiply(gradOutput);
                };
            }

            return loss;
        }
    }
}