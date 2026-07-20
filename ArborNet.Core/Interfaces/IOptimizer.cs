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

    using System.Collections.Generic;
    using ArborNet.Core.Interfaces;

    #endregion

    /// <summary>
    /// Defines the contract for optimization algorithms within the ArborNet framework.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This interface enables the framework to support multiple optimization strategies
    /// (such as SGD, Adam, RMSProp, etc.) while maintaining a consistent API for the training loop.
    /// Optimizers are responsible for the parameter update step after gradients have been computed via backpropagation.
    /// </para>
    /// <para>
    /// A typical training loop iteration follows this pattern:
    /// <list type="number">
    /// <item><description>Clear gradients using <see cref="ZeroGrad(IEnumerable{ITensor})"/>.</description></item>
    /// <item><description>Perform the forward pass to compute loss.</description></item>
    /// <item><description>Perform the backward pass to compute gradients.</description></item>
    /// <item><description>Update model parameters using <see cref="Step(IEnumerable{ITensor})"/>.</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public interface IOptimizer
    {
        /// <summary>
        /// Gets or sets the learning rate for the optimizer.
        /// </summary>
        /// <value>
        /// A <see cref="double"/> representing the current learning rate value.
        /// </value>
        /// <remarks>
        /// The learning rate controls the step size taken during parameter updates.
        /// Most optimizers perform best with carefully tuned learning rate schedules.
        /// </remarks>
        /// <exception cref="System.ArgumentOutOfRangeException">
        /// Thrown when setting a learning rate that is negative.
        /// </exception>
        double LearningRate { get; set; }
        /// <summary>
        /// Performs a single optimization step, updating the given parameters based on their computed gradients.
        /// </summary>
        /// <param name="parameters">The enumerable collection of parameters (<see cref="ITensor"/>) to update.</param>
        /// <remarks>
        /// This method applies the specific optimization algorithm to adjust each parameter's
        /// data using its associated gradient. It must be called after the backward pass (gradient calculation) has completed.
        /// </remarks>
        /// <exception cref="System.ArgumentNullException">
        /// Thrown when the <paramref name="parameters"/> collection is <see langword="null"/>.
        /// </exception>

        void Step(IEnumerable<ITensor> parameters);
        /// <summary>
        /// Resets the gradients of the given parameters to zero.
        /// </summary>
        /// <param name="parameters">The enumerable collection of parameters (<see cref="ITensor"/>) whose gradients are to be zeroed.</param>
        /// <remarks>
        /// Clearing gradients prevents accumulation from multiple backward passes.
        /// This is a critical step in the standard training iteration pattern before invoking backpropagation.
        /// </remarks>
        /// <exception cref="System.ArgumentNullException">
        /// Thrown when the <paramref name="parameters"/> collection is <see langword="null"/>.
        /// </exception>

        void ZeroGrad(IEnumerable<ITensor> parameters);
    }
}