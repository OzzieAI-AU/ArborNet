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
    using ArborNet.Core.Tensors;

    /// <summary>
    /// Defines the contract for recurrent neural network layers that support variable depth unrolling,
    /// parameter deduplication, and execution cap controls.
    /// </summary>
    public interface IRecurrentLayer : ILayer
    {
        /// <summary>
        /// Gets the maximum allowed unroll iterations to prevent tape explosion and divergence.
        /// </summary>
        int MaxLoops { get; }

        /// <summary>
        /// Gets or sets the active recurrent loop depth (R) for forward evaluation.
        /// </summary>
        int ActiveLoops { get; set; }

        /// <summary>
        /// Executes forward propagation across a specified number of recurrent iterations.
        /// </summary>
        /// <param name="input">The input activation tensor.</param>
        /// <param name="unrollSteps">Optional recurrent step override (clamped to MaxLoops).</param>
        /// <returns>The resulting tensor after unrolled recurrence.</returns>
        ITensor Forward(ITensor input, int? unrollSteps = null);
    }
}
