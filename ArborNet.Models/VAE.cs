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

    using System;
    using System.Collections.Generic;
    using ArborNet.Activations;
    using ArborNet.Core;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;
    using ArborNet.Layers.Normalization;
    /// <summary>
    /// Represents a Variational Autoencoder (VAE) neural network model.
    /// This model is designed to encode high-dimensional input into a lower-dimensional latent space
    /// and reconstruct it back, utilizing dynamic linear layer scaling based on input spatial resolution.
    /// </summary>

    #endregion

    public sealed class VAE : BaseModel
    {
        private readonly int _latentDim;
        private readonly Device _device;

        private readonly Conv2D _encConv1;
        private readonly BatchNorm _encBn1;
        private readonly Conv2D _encConv2;
        private readonly BatchNorm _encBn2;
        private readonly Conv2D _encConv3;
        private readonly BatchNorm _encBn3;

        private Linear _fcMu;
        private Linear _fcLogVar;
        private Linear _fcDecode;

        private readonly Conv2D _decConv1;
        private readonly BatchNorm _decBn1;
        private readonly Conv2D _decConv2;
        private readonly BatchNorm _decBn2;
        private readonly Conv2D _decConv3;
        private readonly BatchNorm _decBn3;
        private readonly Conv2D _decConv4;

        private int _flattenedSize = -1;

        public VAE(int latentDim = 128, Device? device = null)
        {
            _latentDim = latentDim;
            _device = device ?? Device.CPU;

            _encConv1 = new Conv2D(3, 64, kernelSize: 4, stride: 2, padding: 1, device: _device);
            _encBn1 = new BatchNorm(64);
            _encConv2 = new Conv2D(64, 128, 4, 2, 1, false, device: _device);
            _encBn2 = new BatchNorm(128);
            _encConv3 = new Conv2D(128, 256, 4, 2, 1, false, device: _device);
            _encBn3 = new BatchNorm(256);

            int initialFlat = 256 * 8 * 8;
            _fcMu = new Linear(initialFlat, latentDim, _device);
            _fcLogVar = new Linear(initialFlat, latentDim, _device);
            _fcDecode = new Linear(latentDim, initialFlat, _device);

            _decConv1 = new Conv2D(256, 128, 4, 2, 1, false, device: _device);
            _decBn1 = new BatchNorm(128);
            _decConv2 = new Conv2D(128, 64, 4, 2, 1, false, device: _device);
            _decBn2 = new BatchNorm(64);
            _decConv3 = new Conv2D(64, 32, 4, 2, 1, false, device: _device);
            _decBn3 = new BatchNorm(32);
            _decConv4 = new Conv2D(32, 3, 4, 2, 1, false, device: _device);

            RebuildParameters();
        }
        /// <summary>
        /// Consolidates all model parameters from the individual encoder, decoder, and fully connected 
        /// projection layers into the centralized parameters list.
        /// </summary>

        private void RebuildParameters()
        {
            parameters.Clear();
            parameters.AddRange(_encConv1.Parameters());
            parameters.AddRange(_encBn1.Parameters());
            parameters.AddRange(_encConv2.Parameters());
            parameters.AddRange(_encBn2.Parameters());
            parameters.AddRange(_encConv3.Parameters());
            parameters.AddRange(_encBn3.Parameters());
            parameters.AddRange(_fcMu.Parameters());
            parameters.AddRange(_fcLogVar.Parameters());
            parameters.AddRange(_fcDecode.Parameters());
            parameters.AddRange(_decConv1.Parameters());
            parameters.AddRange(_decBn1.Parameters());
            parameters.AddRange(_decConv2.Parameters());
            parameters.AddRange(_decBn2.Parameters());
            parameters.AddRange(_decConv3.Parameters());
            parameters.AddRange(_decBn3.Parameters());
            parameters.AddRange(_decConv4.Parameters());
        }
        /// <summary>
        /// Executes the forward pass of the model, returning only the reconstructed output tensor.
        /// </summary>
        /// <param name="input">The input tensor of shape [Batch, Channels, Height, Width].</param>
        /// <returns>An <see cref="ITensor"/> representing the reconstructed image tensor.</returns>

        public override ITensor Forward(ITensor input)
        {
            var (recon, _) = ForwardVAE(input);
            return recon;
        }
        /// <summary>
        /// Executes a complete forward pass through the VAE, returning both the reconstructed output 
        /// and the calculated Kullback-Leibler (KL) divergence loss.
        /// </summary>
        /// <param name="x">The input tensor, which must have a rank of 4 representing [Batch, Channels, Height, Width].</param>
        /// <returns>
        /// A tuple where:
        /// <list type="bullet">
        /// <item><description><c>Reconstruction</c>: The decoded reconstructed tensor of the same spatial shape as input.</description></item>
        /// <item><description><c>KL_Loss</c>: A scalar tensor representing the mean Kullback-Leibler divergence loss.</description></item>
        /// </list>
        /// </returns>
        /// <exception cref="ArgumentException">Thrown when the input tensor rank is not equal to 4.</exception>

        public (ITensor Reconstruction, ITensor KL_Loss) ForwardVAE(ITensor x)
        {
            if (x.Shape.Rank != 4)
                throw new ArgumentException("VAE expects input of shape [B, C, H, W].");

            int batch = x.Shape[0];
            int h = x.Shape[2];
            int w = x.Shape[3];

            int spatial = (h / 8) * (w / 8);
            int currentFlat = 256 * spatial;

            if (_flattenedSize != currentFlat)
            {
                _flattenedSize = currentFlat;
                _fcMu = new Linear(currentFlat, _latentDim, _device);
                _fcLogVar = new Linear(currentFlat, _latentDim, _device);
                _fcDecode = new Linear(_latentDim, currentFlat, _device);
                RebuildParameters();
            }

            var h1 = new ReLU().Forward(_encBn1.Forward(_encConv1.Forward(x)));
            var h2 = new ReLU().Forward(_encBn2.Forward(_encConv2.Forward(h1)));
            var h3 = new ReLU().Forward(_encBn3.Forward(_encConv3.Forward(h2)));

            var flat = h3.Reshape(batch, currentFlat);
            var mu = _fcMu.Forward(flat);
            var logVar = _fcLogVar.Forward(flat);

            var std = logVar.Multiply(0.5f).Exp();
            var eps = Tensor.Randn(mu.Shape, _device);
            var z = mu.Add(eps.Multiply(std));

            var decoded = _fcDecode.Forward(z);
            decoded = decoded.Reshape(batch, 256, h / 8, w / 8);

            decoded = new ReLU().Forward(_decBn1.Forward(_decConv1.Forward(decoded)));
            decoded = new ReLU().Forward(_decBn2.Forward(_decConv2.Forward(decoded)));
            decoded = new ReLU().Forward(_decBn3.Forward(_decConv3.Forward(decoded)));
            var reconstruction = new Sigmoid().Forward(_decConv4.Forward(decoded));

            var kl = logVar.Add(1.0f)
                           .Subtract(mu.Multiply(mu))
                           .Subtract(logVar.Exp())
                           .Multiply(-0.5f)
                           .Mean();

            return (reconstruction, kl);
        }
        /// <summary>
        /// Retrieves an enumerable collection of all trainable parameters (weights and biases) in the model.
        /// </summary>
        /// <returns>An <see cref="IEnumerable{ITensor}"/> containing all model parameters.</returns>

        public override IEnumerable<ITensor> Parameters() => parameters;
    }
}