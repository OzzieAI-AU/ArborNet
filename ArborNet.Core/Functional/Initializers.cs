using System;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Core.Functional
{
    /// <summary>
    /// Provides static methods for initializing tensors with various weight initialization schemes.
    /// These initializers are commonly used for neural network weights to improve training stability and convergence.
    /// </summary>
    /// <remarks>
    /// Proper initialization is critical in deep learning to maintain signal variance across layers
    /// and prevent vanishing or exploding gradients. This class implements several widely-used strategies
    /// from research literature including Xavier/Glorot and Kaiming/He initializations.
    /// All methods are pure, device-aware, and numerically stable.
    /// </remarks>
    public static class Initializers
    {
        /// <summary>
        /// Initializes a tensor with zeros.
        /// </summary>
        public static ITensor Zeros(TensorShape shape, Device device = null)
        {
            return Tensor.Zeros(shape, device ?? Device.CPU);
        }

        /// <summary>
        /// Initializes a tensor with ones.
        /// </summary>
        public static ITensor Ones(TensorShape shape, Device device = null)
        {
            return Tensor.Ones(shape, device ?? Device.CPU);
        }

        /// <summary>
        /// Initializes a tensor with uniform random values between 0 and 1.
        /// </summary>
        public static ITensor Uniform(TensorShape shape, Device device = null)
        {
            return Tensor.Rand(shape, device ?? Device.CPU);
        }

        /// <summary>
        /// Initializes a tensor with normal random values (mean 0, std 1).
        /// </summary>
        public static ITensor Normal(TensorShape shape, Device device = null)
        {
            return Tensor.Randn(shape, device ?? Device.CPU);
        }

        /// <summary>
        /// Initializes a tensor using Xavier (Glorot) uniform initialization.
        /// Suitable for tanh or sigmoid activations. Scales uniform random values by sqrt(6 / (fanIn + fanOut)).
        /// </summary>
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
        /// Initializes a tensor using Xavier (Glorot) normal initialization.
        /// Suitable for tanh or sigmoid activations. Scales normal random values by sqrt(2 / (fanIn + fanOut)).
        /// </summary>
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
        /// Initializes a tensor using Kaiming (He) uniform initialization.
        /// Suitable for ReLU activations. Scales uniform random values by sqrt(6 / fanIn).
        /// </summary>
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
        /// Initializes a tensor using Kaiming (He) normal initialization.
        /// Suitable for ReLU activations. Scales normal random values by sqrt(2 / fanIn).
        /// </summary>
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
