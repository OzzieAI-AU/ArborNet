// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Interfaces
{

    #region Using Statements:

    using System;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Tensors;

    #endregion

    /// <summary>
    /// Core interface for all loss functions within the ArborNet framework.
    /// </summary>
    /// <remarks>
    /// Loss functions evaluate the discrepancy between predictions and ground-truth targets. 
    /// Implementations must support multi-dimensional tensors of arbitrary shapes, optional mathematical reductions, 
    /// seamless integration with the autograd system for backward passes, and execute reliably across different compute devices.
    /// </remarks>
    /// <example>
    /// <para>
    /// The following example demonstrates how to implement a custom Mean Squared Error (MSE) loss
    /// by inheriting from the <see cref="BaseLoss"/> base class:
    /// </para>
    /// <code language="csharp">
    /// public class MeanSquaredError : BaseLoss
    /// {
    ///     public override ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean")
    ///     {
    ///         ValidateInputs(predictions, targets);
    ///         
    ///         // Compute element-wise squared error: (y_pred - y_true)^2
    ///         ITensor difference = predictions.Subtract(targets);
    ///         ITensor squaredDifference = difference.Multiply(difference);
    ///         
    ///         // Apply standard reduction rules
    ///         return ApplyReduction(squaredDifference, reduction, predictions);
    ///     }
    /// }
    /// </code>
    /// </example>
    public interface ILoss
    {
        /// <summary>
        /// Computes the loss value between the predicted outputs and the target values.
        /// </summary>
        /// <param name="predictions">The predicted <see cref="ITensor"/> output by the model.</param>
        /// <param name="targets">The ground-truth <see cref="ITensor"/> targets. Must be structurally compatible with <paramref name="predictions"/>.</param>
        /// <param name="reduction">
        /// The reduction operation to apply to the output. Supported values are:
        /// <list type="bullet">
        /// <item>
        /// <term>mean</term>
        /// <description>Calculates the average of the loss elements (default).</description>
        /// </item>
        /// <item>
        /// <term>sum</term>
        /// <description>Calculates the sum of the loss elements.</description>
        /// </item>
        /// <item>
        /// <term>none</term>
        /// <description>Applies no reduction and preserves the element-wise loss shape.</description>
        /// </item>
        /// </list>
        /// </param>
        /// <returns>A new <see cref="ITensor"/> representing the calculated (and optionally reduced) loss.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="predictions"/> or <paramref name="targets"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when the dimensions or shapes of <paramref name="predictions"/> and <paramref name="targets"/> do not match.</exception>
        ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean");

    }
    /// <summary>
    /// Base implementation for loss functions providing common input validation and standardized reduction handling.
    /// </summary>
    /// <remarks>
    /// This abstract class establishes standard preprocessing and postprocessing templates for concrete loss implementations,
    /// ensuring uniform behavior, strict validation, and autograd-compatible tensor reductions.
    /// </remarks>
    /// <seealso cref="ILoss"/>
    /// <seealso cref="ITensor"/>


    public abstract class BaseLoss : ILoss
    {
        /// <summary>
        /// Computes the loss value between the predicted outputs and the target values.
        /// </summary>
        /// <param name="predictions">The predicted <see cref="ITensor"/> output by the model.</param>
        /// <param name="targets">The ground-truth <see cref="ITensor"/> targets. Must be structurally compatible with <paramref name="predictions"/>.</param>
        /// <param name="reduction">
        /// The reduction operation to apply to the output. Supported values are:
        /// <list type="bullet">
        /// <item>
        /// <term>mean</term>
        /// <description>Calculates the average of the loss elements (default).</description>
        /// </item>
        /// <item>
        /// <term>sum</term>
        /// <description>Calculates the sum of the loss elements.</description>
        /// </item>
        /// <item>
        /// <term>none</term>
        /// <description>Applies no reduction and preserves the element-wise loss shape.</description>
        /// </item>
        /// </list>
        /// </param>
        /// <returns>A new <see cref="ITensor"/> representing the calculated (and optionally reduced) loss.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="predictions"/> or <paramref name="targets"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when the dimensions or shapes of <paramref name="predictions"/> and <paramref name="targets"/> do not match.</exception>
        public abstract ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean");
        /// <summary>
        /// Validates that the prediction and target tensors are non-null and have identical shapes.
        /// </summary>
        /// <param name="predictions">The predicted tensor to validate.</param>
        /// <param name="targets">The target tensor to validate.</param>
        /// <exception cref="ArgumentNullException">
        /// Thrown when <paramref name="predictions"/> or <paramref name="targets"/> is <see langword="null"/>.
        /// </exception>
        /// <exception cref="ArgumentException">
        /// Thrown when the shape of <paramref name="predictions"/> is not equal to the shape of <paramref name="targets"/>.
        /// </exception>

        protected void ValidateInputs(ITensor predictions, ITensor targets)
        {
            if (predictions == null) throw new ArgumentNullException(nameof(predictions));
            if (targets == null) throw new ArgumentNullException(nameof(targets));
            if (!predictions.Shape.Equals(targets.Shape))
                throw new ArgumentException("Predictions and targets must have identical shapes.");
        }
        /// <summary>
        /// Applies the specified reduction operation to the element-wise loss tensor.
        /// </summary>
        /// <param name="loss">The computed raw, element-wise loss tensor to be reduced.</param>
        /// <param name="reduction">The reduction type to apply: <c>"sum"</c>, <c>"none"</c>, or <c>"mean"</c> (default). This parameter is case-insensitive.</param>
        /// <param name="originalShapeTensor">An auxiliary tensor retaining the original input shape metadata, reserved for advanced shape-aware operations.</param>
        /// <returns>
        /// A reduced <see cref="ITensor"/>. Returns a scalar tensor for <c>"sum"</c> or <c>"mean"</c> reductions, 
        /// or the unmodified element-wise <paramref name="loss"/> tensor if <paramref name="reduction"/> is <c>"none"</c>.
        /// </returns>
        /// <exception cref="NullReferenceException">Thrown if the <paramref name="reduction"/> string is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentNullException">Thrown if the <paramref name="loss"/> tensor is <see langword="null"/>.</exception>
        /// <remarks>
        /// Centralizing the reduction logic ensures that all derived loss functions process scalar conversions uniformly,
        /// which guarantees robust and reliable backward graph construction during backpropagation.
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