using ArborNet.Core.Interfaces;
using ArborNet.Core.Models;
using ArborNet.Core.Tensors;
using ArborNet.Models;
using System.Collections.Generic;

namespace ArborNet.Models
{
    /// <summary>
    /// Implements a denoising diffusion probabilistic model (DDPM) for generative tasks.
    /// </summary>
    /// <remarks>
    /// This model precomputes a linear beta noise schedule and the corresponding cumulative
    /// alpha products used in the forward diffusion process. The denoising step is performed
    /// by an internal UNet architecture. The class inherits parameter management from <see cref="BaseModel"/>.
    /// </remarks>
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
        /// Returns all trainable parameters managed by this model.
        /// </summary>
        /// <returns>An enumerable collection of <see cref="ITensor"/> containing all model parameters.</returns>
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
        /// Performs a forward pass through the diffusion model.
        /// </summary>
        /// <param name="input">The input tensor, typically a noisy sample.</param>
        /// <returns>The output tensor produced by the underlying denoiser network.</returns>
        public override ITensor Forward(ITensor input)
        {
            return denoiser.Forward(input);
        }
    }
}