using System;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Losses
{
    public class MSE : BaseLoss
    {
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