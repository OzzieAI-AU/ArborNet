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
    using ArborNet.Core;
    /// <summary>
    /// Implements the Huber loss function, a robust loss that is quadratic for small errors
    /// and linear for large errors, making it less sensitive to outliers than MSE.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Huber loss transitions from L2 (MSE) behavior to L1 (MAE) behavior at the <see cref="delta"/> threshold.
    /// This hybrid nature provides the benefits of being differentiable at zero (unlike absolute error) and 
    /// robust to extreme outliers (unlike squared error).
    /// </para>
    /// <para>
    /// This implementation supports both forward evaluation and backward automatic differentiation via gradient tape.
    /// </para>
    /// </remarks>
    /// <seealso cref="BaseLoss"/>
    /// <seealso cref="ITensor"/>

    #endregion

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
        /// Computes the Huber loss between the predicted outputs and the target ground truth values.
        /// </summary>
        /// <param name="predictions">The predicted <see cref="ITensor"/> values from the model.</param>
        /// <param name="targets">The ground truth target <see cref="ITensor"/> values.</param>
        /// <param name="reduction">The type of reduction to apply to the computed loss elements. 
        /// Supported values are "none", "mean" (default), or "sum".</param>
        /// <returns>An <see cref="ITensor"/> containing the computed and reduced Huber loss values.</returns>
        /// <remarks>
        /// <para>
        /// The element-wise Huber loss is defined mathematically as:
        /// <list type="bullet">
        ///   <item>
        ///     <description><c>0.5 * (predictions - targets)^2</c>, if <c>|predictions - targets| &lt;= delta</c></description>
        ///   </item>
        ///   <item>
        ///     <description><c>delta * (|predictions - targets| - 0.5 * delta)</c>, if <c>|predictions - targets| &gt; delta</c></description>
        ///   </item>
        /// </list>
        /// </para>
        /// <para>
        /// When <see cref="ITensor.RequiresGrad"/> is enabled on the predictions tensor, 
        /// a backward gradient function (<see cref="ITensor.GradFn"/>) is attached to the returned loss tensor 
        /// to support automatic differentiation and backpropagation.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="predictions"/> or <paramref name="targets"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown if the shapes of <paramref name="predictions"/> and <paramref name="targets"/> do not match.</exception>

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