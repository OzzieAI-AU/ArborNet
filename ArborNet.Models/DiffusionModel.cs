// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Models
{

    #region Using Statements:

    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Models;
    using System.Collections.Generic;
    /// <summary>
    /// Implements a Denoising Diffusion Probabilistic Model (DDPM) designed for generative machine learning tasks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This model precomputes a linear beta noise schedule and the corresponding cumulative
    /// alpha products used in the forward diffusion process. The denoising step is performed
    /// by an internal U-Net architecture.
    /// </para>
    /// <para>
    /// The class inherits parameter management from <see cref="BaseModel"/>, enabling tracking
    /// of all trainable weights and biases across the network.
    /// </para>
    /// </remarks>

    #endregion

    public class DiffusionModel : BaseModel
    {
        /// <summary>
        /// The total number of timesteps in the diffusion process.
        /// </summary>
        private readonly int numTimesteps;

        /// <summary>
        /// The beta schedule defining the variance of noise added at each timestep.
        /// Values increase linearly from 0.0001 to 0.02.
        /// </summary>
        private readonly float[] betas;

        /// <summary>
        /// Precomputed cumulative products of alpha values (ᾱ_t = ∏(1 - β_s) for s = 1 to t).
        /// Used for efficient sampling and variance calculation in the diffusion process.
        /// </summary>
        private readonly float[] alphasCumprod;

        /// <summary>
        /// The U-Net network responsible for predicting noise or denoising the input.
        /// </summary>
        private readonly UNet denoiser;
        /// <summary>
        /// Retrieves all trainable parameters associated with this diffusion model, including the parameters of the underlying denoiser.
        /// </summary>
        /// <returns>An <see cref="IEnumerable{ITensor}"/> representing the trainable parameters (weights, biases) of the model.</returns>

        public override IEnumerable<ITensor> Parameters() => parameters;

        /// <summary>
        /// Initializes a new instance of the <see cref="DiffusionModel"/> class.
        /// </summary>
        /// <param name="numTimesteps">The number of timesteps to use in the diffusion process. Default is 1000.</param>
        public DiffusionModel(int numTimesteps = 1000)
        {
            this.numTimesteps = numTimesteps;
            betas = new float[numTimesteps];
            for (int i = 0; i < numTimesteps; i++)
                betas[i] = 0.0001f + (0.02f - 0.0001f) * i / (numTimesteps - 1);

            alphasCumprod = new float[numTimesteps];
            alphasCumprod[0] = 1 - betas[0];
            for (int i = 1; i < numTimesteps; i++)
                alphasCumprod[i] = alphasCumprod[i - 1] * (1 - betas[i]);

            denoiser = new UNet(3, 3, 256);
            parameters.AddRange(denoiser.Parameters());
        }
        /// <summary>
        /// Executes the forward pass of the diffusion model by routing the input through the underlying denoiser network.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> representing the noisy sample at a given timestep.</param>
        /// <returns>The output <see cref="ITensor"/> representing the predicted noise or denoised representation.</returns>

        public override ITensor Forward(ITensor input)
        {
            return denoiser.Forward(input);
        }
    }
}