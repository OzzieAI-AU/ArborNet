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
    /// <summary>
    /// Implements the Binary Cross Entropy (BCE) loss function, commonly used for binary classification tasks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The binary cross entropy loss measures the performance of a classification model whose output is a probability value between 0 and 1.
    /// The loss increases as the predicted probability diverges from the actual label.
    /// </para>
    /// <para>
    /// The mathematical formula for Binary Cross Entropy (BCE) for a single sample is:
    /// <c>L = - [y * log(p) + (1 - y) * log(1 - p)]</c>
    /// where:
    /// <list type="bullet">
    /// <item>
    /// <term>y</term>
    /// <description>is the binary target label (0 or 1).</description>
    /// </item>
    /// <item>
    /// <term>p</term>
    /// <description>is the predicted probability of the positive class.</description>
    /// </item>
    /// </list>
    /// </para>
    /// <para>
    /// This implementation includes:
    /// <list type="number">
    /// <item><description>Input validation (shape matching, device affinity).</description></item>
    /// <item><description>Numerical stability clipping of predicted probabilities to the range [<see cref="EPS"/>, 1.0f - <see cref="EPS"/>] to prevent taking <c>log(0)</c> or <c>log(1 - 1)</c>.</description></item>
    /// <item><description>Support for automatic differentiation (Autograd) by attaching a custom backward gradient function to the output tensor when the prediction tensor requires gradients.</description></item>
    /// </list>
    /// </para>
    /// </remarks>

    #endregion

    public class BinaryCrossEntropy : BaseLoss
    {
        /// <summary>
        /// Small constant value used to clip predicted probabilities to prevent 
        /// taking the logarithm of zero, which would cause numerical instability.
        /// </summary>
        private const float EPS = 1e-12f;
        /// <summary>
        /// Computes the forward pass of the binary cross entropy loss.
        /// </summary>
        /// <param name="predictions">The predicted probability tensor of shape matching <paramref name="targets"/>. Values should ideally reside in the range [0, 1].</param>
        /// <param name="targets">The ground truth tensor containing binary labels (0 or 1) of shape matching <paramref name="predictions"/>.</param>
        /// <param name="reduction">The reduction method to apply to the computed loss tensor. 
        /// Accepted values are <c>"none"</c> (no reduction), <c>"mean"</c> (average loss over all elements), or <c>"sum"</c> (sum of losses over all elements). 
        /// Defaults to <c>"mean"</c>.</param>
        /// <returns>A new <see cref="ITensor"/> containing the computed loss value(s).</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="predictions"/> or <paramref name="targets"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when the dimensions of <paramref name="predictions"/> and <paramref name="targets"/> do not match, or when an invalid <paramref name="reduction"/> type is specified.</exception>
        /// <remarks>
        /// <para>
        /// Predictions are automatically clamped internally to the range [<see cref="EPS"/>, <c>1.0f - </c> <see cref="EPS"/>] 
        /// before applying the logarithm, ensuring strict mathematical safety.
        /// </para>
        /// <para>
        /// If <paramref name="predictions"/> has <see cref="ITensor.RequiresGrad"/> set to <see langword="true"/>,
        /// a custom backpropagation closure is registered to the returned loss tensor's <see cref="ITensor.GradFn"/> property.
        /// The gradient formula applied during backward pass is:
        /// <c>d(Loss)/d(p) = (p - y) / (p * (1 - p))</c> (scaled by the incoming gradient and optional reduction factor).
        /// </para>
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