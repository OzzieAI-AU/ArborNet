// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Initializers
{

    #region Using Statements:

    using System;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Provides static factory methods for initializing tensors using various weight initialization schemes.
    /// </summary>
    /// <remarks>
    /// In deep learning, proper weight initialization is critical to prevent vanishing or exploding gradients,
    /// ensuring stable forward and backward propagation. This class implements standard initializers, including
    /// Constant (Zeros, Ones), Uniform, Normal, Xavier/Glorot, and Kaiming/He methods.
    /// All methods support target device redirection (CPU/GPU) and preserve numerical precision.
    /// </remarks>

    #endregion

    public static class Initializers
    {
        /// <summary>
        /// Creates a new tensor of the specified <see cref="TensorShape"/> with all elements initialized to 0.0.
        /// </summary>
        /// <param name="shape">The dimensional configuration of the tensor to create.</param>
        /// <param name="device">
        /// The target execution/memory device (e.g., CPU, GPU). 
        /// If <see langword="null"/>, defaults to <see cref="Device.CPU"/>.
        /// </param>
        /// <returns>A new <see cref="ITensor"/> populated entirely with zeros on the specified <paramref name="device"/>.</returns>
        public static ITensor Zeros(TensorShape shape, Device device = null)
        {
            return Tensor.Zeros(shape, device ?? Device.CPU);
        }
        /// <summary>
        /// Creates a new tensor of the specified <see cref="TensorShape"/> with all elements initialized to 1.0.
        /// </summary>
        /// <param name="shape">The dimensional configuration of the tensor to create.</param>
        /// <param name="device">
        /// The target execution/memory device (e.g., CPU, GPU). 
        /// If <see langword="null"/>, defaults to <see cref="Device.CPU"/>.
        /// </param>
        /// <returns>A new <see cref="ITensor"/> populated entirely with ones on the specified <paramref name="device"/>.</returns>

        public static ITensor Ones(TensorShape shape, Device device = null)
        {
            return Tensor.Ones(shape, device ?? Device.CPU);
        }
        /// <summary>
        /// Creates a new tensor of the specified <see cref="TensorShape"/> populated with random values drawn from a uniform distribution.
        /// </summary>
        /// <remarks>
        /// The random values are typically generated in the half-open interval <c>[0.0, 1.0)</c>.
        /// </remarks>
        /// <param name="shape">The dimensional configuration of the tensor to create.</param>
        /// <param name="device">
        /// The target execution/memory device (e.g., CPU, GPU). 
        /// If <see langword="null"/>, defaults to <see cref="Device.CPU"/>.
        /// </param>
        /// <returns>A new <see cref="ITensor"/> populated with uniform random values on the specified <paramref name="device"/>.</returns>

        public static ITensor Uniform(TensorShape shape, Device device = null)
        {
            return Tensor.Rand(shape, device ?? Device.CPU);
        }
        /// <summary>
        /// Creates a new tensor of the specified <see cref="TensorShape"/> populated with random values drawn from a standard normal (Gaussian) distribution.
        /// </summary>
        /// <remarks>
        /// The generated values follow a distribution with a mean of 0.0 and a standard deviation of 1.0.
        /// </remarks>
        /// <param name="shape">The dimensional configuration of the tensor to create.</param>
        /// <param name="device">
        /// The target execution/memory device (e.g., CPU, GPU). 
        /// If <see langword="null"/>, defaults to <see cref="Device.CPU"/>.
        /// </param>
        /// <returns>A new <see cref="ITensor"/> populated with normally distributed values on the specified <paramref name="device"/>.</returns>

        public static ITensor Normal(TensorShape shape, Device device = null)
        {
            return Tensor.Randn(shape, device ?? Device.CPU);
        }
        /// <summary>
        /// Initializes a tensor using the Xavier (Glorot) Uniform initialization method.
        /// </summary>
        /// <remarks>
        /// This initialization is designed for deep networks utilizing symmetric activation functions such as tanh or sigmoid.
        /// It generates random values from a uniform distribution, scales them by <c>limit = sqrt(6 / (fanIn + fanOut))</c>,
        /// and applies a zero-centering shift of <c>-0.5 * limit</c>.
        /// <para/>
        /// The input connections (<c>fanIn</c>) and output connections (<c>fanOut</c>) are computed from the final two dimensions of the tensor's shape:
        /// <list type="bullet">
        /// <item><description><c>fanIn</c> is determined by <c>shape[shape.Rank - 2]</c></description></item>
        /// <item><description><c>fanOut</c> is determined by <c>shape[shape.Rank - 1]</c></description></item>
        /// </list>
        /// </remarks>
        /// <param name="shape">The <see cref="TensorShape"/> of the tensor. Must have a rank of 2 or higher.</param>
        /// <param name="device">
        /// The target execution/memory device (e.g., CPU, GPU). 
        /// If <see langword="null"/>, defaults to <see cref="Device.CPU"/>.
        /// </param>
        /// <returns>An initialized <see cref="ITensor"/> scaled according to Xavier Uniform parameters on the specified <paramref name="device"/>.</returns>
        /// <exception cref="System.ArgumentException">Thrown when the rank of <paramref name="shape"/> is less than 2.</exception>

        public static ITensor XavierUniform(TensorShape shape, Device device = null)
        {
            if (shape.Rank < 2)
                throw new ArgumentException("Xavier initialization requires at least 2D shape.");

            int fanIn = shape[shape.Rank - 2];
            int fanOut = shape[shape.Rank - 1];
            float limit = (float)Math.Sqrt(6.0 / (fanIn + fanOut));

            var tensor = Tensor.Rand(shape, device ?? Device.CPU);
            return tensor.Multiply(limit).Subtract(Tensor.FromScalar(0.5f * limit, device ?? Device.CPU));
        }
        /// <summary>
        /// Initializes a tensor using the Xavier (Glorot) Normal initialization method.
        /// </summary>
        /// <remarks>
        /// This initialization is designed for deep networks utilizing symmetric activation functions such as tanh or sigmoid.
        /// It generates random values from a zero-mean normal distribution with a standard deviation scaled to:
        /// <c>std = sqrt(2 / (fanIn + fanOut))</c>.
        /// <para/>
        /// The input connections (<c>fanIn</c>) and output connections (<c>fanOut</c>) are computed from the final two dimensions of the tensor's shape:
        /// <list type="bullet">
        /// <item><description><c>fanIn</c> is determined by <c>shape[shape.Rank - 2]</c></description></item>
        /// <item><description><c>fanOut</c> is determined by <c>shape[shape.Rank - 1]</c></description></item>
        /// </list>
        /// </remarks>
        /// <param name="shape">The <see cref="TensorShape"/> of the tensor. Must have a rank of 2 or higher.</param>
        /// <param name="device">
        /// The target execution/memory device (e.g., CPU, GPU). 
        /// If <see langword="null"/>, defaults to <see cref="Device.CPU"/>.
        /// </param>
        /// <returns>An initialized <see cref="ITensor"/> scaled according to Xavier Normal parameters on the specified <paramref name="device"/>.</returns>
        /// <exception cref="System.ArgumentException">Thrown when the rank of <paramref name="shape"/> is less than 2.</exception>

        public static ITensor XavierNormal(TensorShape shape, Device device = null)
        {
            if (shape.Rank < 2)
                throw new ArgumentException("Xavier initialization requires at least 2D shape.");

            int fanIn = shape[shape.Rank - 2];
            int fanOut = shape[shape.Rank - 1];
            float std = (float)Math.Sqrt(2.0 / (fanIn + fanOut));

            var tensor = Tensor.Randn(shape, device ?? Device.CPU);
            return tensor.Multiply(std);
        }
        /// <summary>
        /// Initializes a tensor using the Kaiming (He) Uniform initialization method.
        /// </summary>
        /// <remarks>
        /// This initialization is optimized for deep networks utilizing non-symmetric, rectified activation functions (e.g., ReLU, LeakyReLU).
        /// It generates uniform random values scaled by <c>limit = sqrt(6 / fanIn)</c> and shifted by a zero-centering factor of <c>-0.5 * limit</c>.
        /// <para/>
        /// The input connections count (<c>fanIn</c>) is derived from the second-to-last dimension of the tensor:
        /// <list type="bullet">
        /// <item><description><c>fanIn</c> is determined by <c>shape[shape.Rank - 2]</c></description></item>
        /// </list>
        /// </remarks>
        /// <param name="shape">The <see cref="TensorShape"/> of the tensor. Must have a rank of 2 or higher.</param>
        /// <param name="device">
        /// The target execution/memory device (e.g., CPU, GPU). 
        /// If <see langword="null"/>, defaults to <see cref="Device.CPU"/>.
        /// </param>
        /// <returns>An initialized <see cref="ITensor"/> scaled according to Kaiming Uniform parameters on the specified <paramref name="device"/>.</returns>
        /// <exception cref="System.ArgumentException">Thrown when the rank of <paramref name="shape"/> is less than 2.</exception>

        public static ITensor KaimingUniform(TensorShape shape, Device device = null)
        {
            if (shape.Rank < 2)
                throw new ArgumentException("Kaiming initialization requires at least 2D shape.");

            int fanIn = shape[shape.Rank - 2];
            float limit = (float)Math.Sqrt(6.0 / fanIn);

            var tensor = Tensor.Rand(shape, device ?? Device.CPU);
            return tensor.Multiply(limit).Subtract(Tensor.FromScalar(0.5f * limit, device ?? Device.CPU));
        }
        /// <summary>
        /// Initializes a tensor using the Kaiming (He) Normal initialization method.
        /// </summary>
        /// <remarks>
        /// This initialization is optimized for deep networks utilizing non-symmetric, rectified activation functions (e.g., ReLU, LeakyReLU).
        /// It generates random values from a zero-mean normal distribution with a standard deviation scaled to:
        /// <c>std = sqrt(2 / fanIn)</c>.
        /// <para/>
        /// The input connections count (<c>fanIn</c>) is derived from the second-to-last dimension of the tensor:
        /// <list type="bullet">
        /// <item><description><c>fanIn</c> is determined by <c>shape[shape.Rank - 2]</c></description></item>
        /// </list>
        /// </remarks>
        /// <param name="shape">The <see cref="TensorShape"/> of the tensor. Must have a rank of 2 or higher.</param>
        /// <param name="device">
        /// The target execution/memory device (e.g., CPU, GPU). 
        /// If <see langword="null"/>, defaults to <see cref="Device.CPU"/>.
        /// </param>
        /// <returns>An initialized <see cref="ITensor"/> scaled according to Kaiming Normal parameters on the specified <paramref name="device"/>.</returns>
        /// <exception cref="System.ArgumentException">Thrown when the rank of <paramref name="shape"/> is less than 2.</exception>

        public static ITensor KaimingNormal(TensorShape shape, Device device = null)
        {
            if (shape.Rank < 2)
                throw new ArgumentException("Kaiming initialization requires at least 2D shape.");

            int fanIn = shape[shape.Rank - 2];
            float std = (float)Math.Sqrt(2.0 / fanIn);

            var tensor = Tensor.Randn(shape, device ?? Device.CPU);
            return tensor.Multiply(std);
        }
    }
}