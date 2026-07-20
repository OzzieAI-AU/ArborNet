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
    /// Represents the Mean Squared Error (MSE) loss function, which measures the average 
    /// of the squares of the errors (the average squared difference between the estimated 
    /// values and the actual value). This is commonly used for regression tasks.
    /// </summary>

    #endregion

    public class MSE : BaseLoss
    {
        /// <summary>
        /// Computes the forward pass of the Mean Squared Error (MSE) loss.
        /// If the prediction tensor requires gradients, this method will also register 
        /// the backward gradient computation function on the returned loss tensor.
        /// </summary>
        /// <param name="predictions">The predicted output tensor from the neural network.</param>
        /// <param name="targets">The ground truth target tensor of the same shape as predictions.</param>
        /// <param name="reduction">
        /// Specifies the reduction operation to apply to the computed loss. 
        /// Supported values are "mean" (default), "sum", and "none".
        /// </param>
        /// <returns>A tensor containing the computed loss value(s).</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="predictions"/> or <paramref name="targets"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the dimensions of <paramref name="predictions"/> and <paramref name="targets"/> do not match.</exception>
        public override ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean")
        {
            ValidateInputs(predictions, targets);

            var diff = predictions.Subtract(targets);
            var squared = diff.Multiply(diff);

            ITensor loss;
            bool isMean = reduction.ToLowerInvariant() != "sum" && reduction.ToLowerInvariant() != "none";
            int n = squared.Shape.TotalElements;
            if (reduction.ToLowerInvariant() == "sum")
                loss = squared.Sum();
            else if (reduction.ToLowerInvariant() == "none")
                loss = squared;
            else
                loss = squared.Mean();

            if (predictions.RequiresGrad)
            {
                loss.GradFn = gradOutput =>
                {
                    ITensor gradForSquared = gradOutput;
                    if (isMean)
                    {
                        gradForSquared = gradOutput.Divide((float)n);
                    }
                    var grad = diff.Multiply(2.0f).Multiply(gradForSquared);
                    if (predictions.Grad == null)
                    {
                        predictions.Grad = grad.Clone();
                    }
                    else
                    {
                        predictions.Grad = predictions.Grad.Add(grad);
                    }
                    predictions.GradFn?.Invoke(grad);
                    return grad;
                };
            }

            return loss;
        }
    }
}