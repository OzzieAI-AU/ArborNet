// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core
{

    #region Using Statements:

    using ArborNet.Activations;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using System.Collections;
    using System.Numerics;
    using ArborNet.Core.Backends;
    /// <summary>
    /// Provides extension methods for the <see cref="ITensor"/> interface to support scalar arithmetic operations,
    /// common neural network activation functions, and tensor data conversion utilities.
    /// </summary>
    /// <remarks>
    /// These methods enable fluent, expressive tensor manipulation commonly used in deep learning pipelines,
    /// shielding the consumer from underlying hardware-specific backend implementations (e.g., CPU or CUDA).
    /// </remarks>

    #endregion

    // ****************************************************************************
    // Project:     ArborNet
    // Description: A C# Machine Learning Library implemented in .NET 10 with 
    //              full CUDA support.
    // Author:      OzzieAI - Chris Sykes
    // License:     MIT License
    // ****************************************************************************
    public static class TensorScalarExtensions
    {
        /// <summary>
        /// Multiplies the tensor by a double-precision floating-point scalar value.
        /// </summary>
        /// <param name="t">The tensor to multiply.</param>
        /// <param name="scalar">The scalar multiplier.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise multiplication result.</returns>
        public static ITensor Multiply(this ITensor t, double scalar)
            => t.Multiply((float)scalar);
        /// <summary>
        /// Divides the tensor by a single-precision floating-point scalar value.
        /// </summary>
        /// <param name="t">The tensor to divide.</param>
        /// <param name="scalar">The scalar divisor.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise division result.</returns>

        public static ITensor Divide(this ITensor t, float scalar)
            => t.Multiply(1f / scalar);
        /// <summary>
        /// Applies the Hyperbolic Tangent (tanh) activation function element-wise to the tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>A new <see cref="ITensor"/> with the tanh activation applied.</returns>

        public static ITensor Tanh(this ITensor t)
            => new Tanh().Forward(t);
        /// <summary>
        /// Applies the Rectified Linear Unit (ReLU) activation function element-wise to the tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>A new <see cref="ITensor"/> with the ReLU activation applied.</returns>

        public static ITensor Relu(this ITensor t)
            => new ReLU().Forward(t);
        /// <summary>
        /// Applies the Gaussian Error Linear Unit (GELU) activation function element-wise to the tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>A new <see cref="ITensor"/> with the GELU activation applied.</returns>

        public static ITensor Gelu(this ITensor t)
            => new Gelu().Forward(t);
        /// <summary>
        /// Retrieves the underlying data of the tensor as a flat single-precision floating-point array.
        /// </summary>
        /// <param name="t">The tensor containing the data.</param>
        /// <returns>An array of single-precision floating-point numbers containing the tensor data.</returns>

        public static float[] Data(this ITensor t) => t.ToArray();
        /// <summary>
        /// Converts the tensor data into a flat array of the specified struct type.
        /// </summary>
        /// <typeparam name="T">The target value type (supported types: <see cref="double"/>, <see cref="float"/>, <see cref="Complex"/>).</typeparam>
        /// <param name="tensor">The source tensor.</param>
        /// <returns>An array of type <typeparamref name="T"/> containing the converted tensor data.</returns>
        /// <exception cref="ArgumentNullException">Thrown when the <paramref name="tensor"/> is null.</exception>
        /// <exception cref="NotSupportedException">Thrown when conversion to <typeparamref name="T"/> is not supported.</exception>

        public static T[] ToArray<T>(this ITensor? tensor) where T : struct
        {
            if (tensor is null)
                throw new ArgumentNullException(nameof(tensor));

            float[] source = tensor.ToArray();

            if (typeof(T) == typeof(double))
            {
                var result = new double[source.Length];
                for (int i = 0; i < source.Length; i++)
                    result[i] = source[i];
                return (T[])(object)result;
            }
            else if (typeof(T) == typeof(float))
            {
                return (T[])(object)source;
            }
            else if (typeof(T) == typeof(Complex))
            {
                var result = new Complex[source.Length];
                for (int i = 0; i < source.Length; i++)
                    result[i] = new Complex(source[i], 0.0);
                return (T[])(object)result;
            }

            throw new NotSupportedException(
                $"Conversion from {tensor.GetType().Name} to {typeof(T).Name}[] is not supported.");
        }
        /// <summary>
        /// Converts the tensor data into a flat array of the specified struct type.
        /// </summary>
        /// <typeparam name="T">The target value type (supported types: <see cref="double"/>, <see cref="float"/>, <see cref="Complex"/>).</typeparam>
        /// <param name="tensor">The source tensor.</param>
        /// <returns>An array of type <typeparamref name="T"/> containing the converted tensor data.</returns>
        /// <exception cref="ArgumentNullException">Thrown when the <paramref name="tensor"/> is null.</exception>
        /// <exception cref="NotSupportedException">Thrown when conversion to <typeparamref name="T"/> is not supported.</exception>

        public static double[] ToArray(this ITensor? tensor)
        {
            return tensor.ToArray<double>();
        }
        /// <summary>
        /// Adds a single-precision floating-point scalar value element-wise to the tensor.
        /// </summary>
        /// <param name="t">The tensor to add to.</param>
        /// <param name="scalar">The scalar value to add.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise addition result.</returns>

        public static ITensor Add(this ITensor t, float scalar)
            => t.Add(Tensor.FromScalar(scalar, t.Device));
        /// <summary>
        /// Adds an integer scalar value element-wise to the tensor.
        /// </summary>
        /// <param name="t">The tensor to add to.</param>
        /// <param name="scalar">The scalar value to add.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise addition result.</returns>

        public static ITensor Add(this ITensor t, int scalar)
            => t.Add(Tensor.FromScalar(scalar, t.Device));
        /// <summary>
        /// Subtracts a single-precision floating-point scalar value element-wise from the tensor.
        /// </summary>
        /// <param name="t">The tensor to subtract from.</param>
        /// <param name="scalar">The scalar value to subtract.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise subtraction result.</returns>

        public static ITensor Subtract(this ITensor t, float scalar)
            => t.Subtract(Tensor.FromScalar(scalar, t.Device));
        /// <summary>
        /// Multiplies the tensor by a single-precision floating-point scalar value.
        /// </summary>
        /// <param name="t">The tensor to multiply.</param>
        /// <param name="scalar">The scalar multiplier.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise multiplication result.</returns>

        public static ITensor Multiply(this ITensor t, float scalar)
            => t.Multiply(Tensor.FromScalar(scalar, t.Device));
        /// <summary>
        /// Creates a new tensor with the same shape and on the same device as the specified tensor, filled with ones.
        /// </summary>
        /// <param name="t">The prototype tensor.</param>
        /// <returns>A new <see cref="ITensor"/> populated with ones.</returns>

        public static ITensor OnesLike(this ITensor t)
            => Tensor.Ones(t.Shape, t.Device);
        /// <summary>
        /// Creates a new tensor with the same shape and on the same device as the specified tensor, filled with zeros.
        /// </summary>
        /// <param name="t">The prototype tensor.</param>
        /// <returns>A new <see cref="ITensor"/> populated with zeros.</returns>

        public static ITensor ZerosLike(this ITensor t)
            => Tensor.Zeros(t.Shape, t.Device);
    }
}