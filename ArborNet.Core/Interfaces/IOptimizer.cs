using System.Collections.Generic;
using ArborNet.Core.Interfaces;

namespace ArborNet.Core.Interfaces
{
    /// <summary>
    /// Interface for optimizers in ArborNet.
    /// Defines the contract for optimization algorithms that update model parameters
    /// based on gradients, manage learning rates, and handle gradient zeroing.
    /// Implementations include algorithms like SGD, Adam, RMSProp, etc.
    /// </summary>
    /// <remarks>
    /// This interface enables the framework to support multiple optimization strategies
    /// while maintaining a consistent API for the training loop. Optimizers are responsible
    /// for the parameter update step after gradients have been computed via backpropagation.
    /// </remarks>
    public interface IOptimizer
    {
        /// <summary>
        /// Gets or sets the learning rate for the optimizer.
        /// </summary>
        /// <value>
        /// The current learning rate value.
        /// </value>
        /// <remarks>
        /// The learning rate controls the step size taken during parameter updates.
        /// Most optimizers perform best with carefully tuned learning rate schedules.
        /// </remarks>
        double LearningRate { get; set; }

        /// <summary>
        /// Performs a single optimization step, updating the given parameters based on their gradients.
        /// </summary>
        /// <param name="parameters">The enumerable collection of parameters to update.</param>
        /// <remarks>
        /// This method applies the specific optimization algorithm to adjust each parameter's
        /// data using its associated gradient. It should be called after the backward pass.
        /// </remarks>
        void Step(IEnumerable<ITensor> parameters);

        /// <summary>
        /// Zeros the gradients of the given parameters, typically called before backpropagation.
        /// </summary>
        /// <param name="parameters">The enumerable collection of parameters whose gradients to zero.</param>
        /// <remarks>
        /// Clearing gradients prevents accumulation from multiple backward passes.
        /// This is a critical step in the standard training iteration pattern.
        /// </remarks>
        void ZeroGrad(IEnumerable<ITensor> parameters);
    }
}