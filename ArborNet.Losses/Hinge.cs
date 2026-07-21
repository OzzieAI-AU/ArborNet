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
    /// Represents the Hinge Loss function, primarily used for training maximum-margin binary classifiers 
    /// such as Support Vector Machines (SVMs).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The hinge loss is defined for binary classification where the target values are encoded as -1 or +1.
    /// For a prediction <c>y_pred</c> and a target <c>y_true</c>, the element-wise loss is calculated as:
    /// <code>
    /// Loss = max(0, 1 - y_true * y_pred)
    /// </code>
    /// </para>
    /// <para>
    /// This implementation provides full support for automatic differentiation (autograd). When the input 
    /// <c>predictions</c> tensor requires gradients (<see cref="ITensor.RequiresGrad"/> is <see langword="true"/>), 
    /// a gradient function is automatically attached to the returned loss tensor to compute the 
    /// backpropagated gradients during the backward pass.
    /// </para>
    /// </remarks>

    #endregion

    public class Hinge : BaseLoss
    {
        /// <summary>
        /// Computes the hinge loss between the model predictions and the ground-truth targets.
        /// </summary>
        /// <param name="predictions">The predicted continuous-valued decision scores or logits from the model.</param>
        /// <param name="targets">The ground-truth binary targets. Values must be either -1.0 or +1.0.</param>
        /// <param name="reduction">
        /// The reduction technique to apply to the output tensor. Supported values are:
        /// <list type="bullet">
        /// <item>
        /// <term>"none"</term>
        /// <description>No reduction is performed; returns a tensor of the same shape containing element-wise losses.</description>
        /// </item>
        /// <item>
        /// <term>"mean"</term>
        /// <description>The element-wise losses are averaged into a single scalar tensor (default).</description>
        /// </item>
        /// <item>
        /// <term>"sum"</term>
        /// <description>The element-wise losses are summed into a single scalar tensor.</description>
        /// </item>
        /// </list>
        /// </param>
        /// <returns>An <see cref="ITensor"/> containing the computed hinge loss value(s).</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="predictions"/> or <paramref name="targets"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown if the shapes of <paramref name="predictions"/> and <paramref name="targets"/> are incompatible, or if an unsupported <paramref name="reduction"/> type is supplied.</exception>
        /// <remarks>
        /// <para>
        /// During backpropagation, the gradient of the loss with respect to the prediction <c>x</c> is computed as:
        /// <code>
        /// d(Loss)/dx = -y, if (1 - y * x) > 0
        /// d(Loss)/dx = 0, otherwise
        /// </code>
        /// where <c>y</c> represents the target value.
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
                    var gradInput = gradPred.Multiply(gradOutput);
                    predictions.AccumulateGrad(gradInput);
                    return gradInput;
                };
            }

            return loss;
        }
    }
}