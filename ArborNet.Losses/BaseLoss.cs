using System;
using ArborNet.Core.Devices;
using ArborNet.Core.Tensors;

namespace ArborNet.Core.Interfaces
{

    /// <summary>
    /// Abstract base class providing common functionality for loss function implementations.
    /// </summary>
    /// <remarks>
    /// Derived classes must implement the <see cref="Forward"/> method. This base class supplies
    /// input validation and reduction utilities to ensure consistent behavior across loss functions.
    /// </remarks>
    public abstract class BaseLoss : ILoss
    {
        /// <summary>
        /// Computes the loss between the model predictions and the target values.
        /// </summary>
        /// <param name="predictions">The predicted output tensor from the model.</param>
        /// <param name="targets">The ground truth target tensor.</param>
        /// <param name="reduction">The reduction to apply to the output: 
        /// "mean" (default), "sum", or "none".</param>
        /// <returns>The computed loss as an <see cref="ITensor"/>.</returns>
        public abstract ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean");

        /// <summary>
        /// Validates that the predictions and targets tensors are valid for loss computation.
        /// </summary>
        /// <param name="predictions">The predictions tensor to validate.</param>
        /// <param name="targets">The targets tensor to validate.</param>
        /// <exception cref="ArgumentNullException">Thrown when either tensor is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when the shapes of the two tensors are not identical.</exception>
        protected void ValidateInputs(ITensor predictions, ITensor targets)
        {
            if (predictions == null) throw new ArgumentNullException(nameof(predictions));
            if (targets == null) throw new ArgumentNullException(nameof(targets));
            if (!predictions.Shape.Equals(targets.Shape))
                throw new ArgumentException("Predictions and targets must have identical shapes.");
        }

        /// <summary>
        /// Applies the requested reduction operation to the computed loss tensor.
        /// </summary>
        /// <param name="loss">The unreduced loss tensor.</param>
        /// <param name="reduction">The type of reduction to apply ("sum", "mean", or "none").</param>
        /// <param name="originalShapeTensor">The tensor containing the original shape information 
        /// (provided for potential shape-aware reductions in derived implementations).</param>
        /// <returns>The reduced loss tensor according to the specified reduction mode.</returns>
        protected ITensor ApplyReduction(ITensor loss, string reduction, ITensor originalShapeTensor)
        {
            return reduction.ToLowerInvariant() switch
            {
                "sum" => loss.Sum(),
                "none" => loss,
                _ => loss.Mean()
            };
        }
    }
}