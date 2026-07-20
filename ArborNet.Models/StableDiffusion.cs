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
    /// Implements the Stable Diffusion model, a latent diffusion model (LDM) architecture
    /// designed for high-resolution image synthesis and manipulation.
    /// </summary>
    /// <remarks>
    /// This model integrates a Variational Autoencoder (VAE) to compress images into a low-dimensional
    /// latent space, and a U-Net architecture to perform the iterative denoising process (diffusion)
    /// within that latent space. It inherits from <see cref="BaseModel"/> and aggregates the trainable
    /// parameters of both sub-networks.
    /// </remarks>

    #endregion

    public class StableDiffusion : BaseModel
    {

        /// <summary>
        /// The Variational Autoencoder (VAE) component responsible for encoding images
        /// into latent representations and decoding them back to pixel space.
        /// </summary>
        private readonly VAE vae;

        /// <summary>
        /// The U-Net component that performs the core noise prediction in the latent diffusion process.
        /// </summary>
        private readonly UNet unet;
        /// <summary>
        /// Retrieves all trainable parameter tensors from both the internal Variational Autoencoder (VAE) 
        /// and U-Net components.
        /// </summary>
        /// <returns>
        /// An <see cref="IEnumerable{ITensor}"/> sequence containing all parameters of the combined model.
        /// </returns>

        public override IEnumerable<ITensor> Parameters() => parameters;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusion"/> class.
        /// </summary>
        /// <remarks>
        /// Configures a VAE with 4 latent channels and a U-Net with 4 input channels,
        /// 4 output channels, and a base model dimension of 256. All parameters from
        /// both components are registered with the base model.
        /// </remarks>
        public StableDiffusion()
        {
            vae = new VAE(4);
            unet = new UNet(4, 4, 256);
            parameters.AddRange(vae.Parameters());
            parameters.AddRange(unet.Parameters());
        }
        /// <summary>
        /// Performs the forward pass of the Stable Diffusion model by routing the input through 
        /// the noise-prediction U-Net component.
        /// </summary>
        /// <param name="input">The input latent tensor representation, typically representing noisy latents.</param>
        /// <returns>
        /// An <see cref="ITensor"/> containing the predicted noise or processed latent output from the U-Net.
        /// </returns>

        public override ITensor Forward(ITensor input)
        {
            return unet.Forward(input);
        }
    }
}