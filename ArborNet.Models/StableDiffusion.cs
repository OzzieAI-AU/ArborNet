using ArborNet.Core.Interfaces;
using ArborNet.Core.Models;
using ArborNet.Core.Tensors;
using ArborNet.Models;
using System.Collections.Generic;

namespace ArborNet.Models
{
    /// <summary>
    /// Implements the Stable Diffusion model by combining a Variational Autoencoder (VAE)
    /// for latent space operations with a U-Net for the diffusion process.
    /// </summary>
    /// <remarks>
    /// This class inherits from <see cref="BaseModel"/> and aggregates trainable parameters
    /// from both the VAE and U-Net components. The forward pass is delegated to the U-Net.
    /// </remarks>
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
        /// Returns all trainable parameters from both the VAE and U-Net submodels.
        /// </summary>
        /// <returns>A collection containing all model parameters as <see cref="ITensor"/> instances.</returns>
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
        /// Performs the forward pass of the Stable Diffusion model.
        /// </summary>
        /// <param name="input">The input tensor, typically a latent representation or noise tensor.</param>
        /// <returns>The output tensor produced by the U-Net component.</returns>
        public override ITensor Forward(ITensor input)
        {
            return unet.Forward(input);
        }
    }
}