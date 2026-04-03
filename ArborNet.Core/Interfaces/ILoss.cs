using System;
using ArborNet.Core.Devices;
using ArborNet.Core.Tensors;

namespace ArborNet.Core.Interfaces
{

    /// <summary>
    /// Core interface for all loss functions in ArborNet.
    /// Supports predictions/targets of any shape, optional reduction, and full autograd integration.
    /// All losses must be numerically stable and device-aware.
    /// </summary>
    public interface ILoss
    {
        /// <summary>
        /// Computes the loss between predictions and targets.
        /// </summary>
        /// <param name="predictions">Predicted tensor.</param>
        /// <param name="targets">Target tensor (must match predictions shape).</param>
        /// <param name="reduction">Reduction type: 'mean' (default), 'sum', or 'none'.</param>
        /// <returns>Loss tensor (scalar for mean/sum, same shape for 'none').</returns>
        ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean");
    }

    /// <summary>
    /// Base implementation for losses providing common validation and device handling.
    /// </summary>
    public abstract class BaseLoss : ILoss
    {
        /// <inheritdoc />
        public abstract ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean");

        /// <summary>
        /// Validates that the prediction and target tensors are valid for loss computation.
        /// </summary>
        /// <param name="predictions">Predicted tensor.</param>
        /// <param name="targets">Target tensor.</param>
        /// <exception cref="ArgumentNullException">
        /// Thrown when <paramref name="predictions"/> or <paramref name="targets"/> is <see langword="null"/>.
        /// </exception>
        /// <exception cref="ArgumentException">
        /// Thrown when the shapes of <paramref name="predictions"/> and <paramref name="targets"/> do not match.
        /// </exception>
        protected void ValidateInputs(ITensor predictions, ITensor targets)
        {
            if (predictions == null) throw new ArgumentNullException(nameof(predictions));
            if (targets == null) throw new ArgumentNullException(nameof(targets));
            if (!predictions.Shape.Equals(targets.Shape))
                throw new ArgumentException("Predictions and targets must have identical shapes.");
        }

        /// <summary>
        /// Applies the specified reduction operation to the computed loss tensor.
        /// </summary>
        /// <param name="loss">The loss tensor prior to reduction.</param>
        /// <param name="reduction">The reduction type: 'mean' (default), 'sum', or 'none'.</param>
        /// <param name="originalShapeTensor">Tensor containing original shape information 
        /// (reserved for future shape-aware reductions).</param>
        /// <returns>
        /// A reduced loss tensor: scalar for 'mean' or 'sum', or the original tensor for 'none'.
        /// </returns>
        /// <remarks>
        /// This method centralizes reduction logic across all loss implementations 
        /// to ensure consistent behavior and autograd compatibility.
        /// </remarks>
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