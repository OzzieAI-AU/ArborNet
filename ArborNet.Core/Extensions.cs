// ****************************************************************************
// Project:     ArborNet
// Description: A C# Machine Learning Library implemented in .NET 10 with 
//              full CUDA support.
// Author:      OzzieAI - Chris Sykes
// License:     MIT License
// 
// Copyright (c) 2026 Chris Sykes (OzzieAI)
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in 
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
// THE SOFTWARE.
// ****************************************************************************
using ArborNet.Activations;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using System.Collections;
using System.Numerics;
using ArborNet.Core.Backends;

namespace ArborNet.Core
{
    /// <summary>
    /// Provides extension methods for <see cref="ITensor"/> to support scalar arithmetic operations,
    /// common neural network activation functions, and tensor data conversion utilities.
    /// </summary>
    /// <remarks>
    /// These methods enable fluent, expressive tensor manipulation commonly used in deep learning pipelines.
    /// </remarks>
    public static class TensorScalarExtensions
    {
        /// <summary>
        /// Multiplies each element of the tensor by the specified double scalar value.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <param name="scalar">The scalar value to multiply by.</param>
        /// <returns>A new tensor containing the result of the element-wise multiplication.</returns>
        public static ITensor Multiply(this ITensor t, double scalar)
            => t.Multiply((float)scalar);

        /// <summary>
        /// Divides each element of the tensor by the specified scalar value.
        /// Implemented as multiplication by the reciprocal for numerical efficiency.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <param name="scalar">The scalar value to divide by. Must not be zero.</param>
        /// <returns>A new tensor containing the result of the element-wise division.</returns>
        public static ITensor Divide(this ITensor t, float scalar)
            => t.Multiply(1f / scalar);

        /// <summary>
        /// Applies the hyperbolic tangent (tanh) activation function element-wise to the tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>A new tensor with tanh applied to each element.</returns>
        public static ITensor Tanh(this ITensor t)
            => new Tanh().Forward(t);

        /// <summary>
        /// Applies the Rectified Linear Unit (ReLU) activation function element-wise to the tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>A new tensor with ReLU applied to each element.</returns>
        public static ITensor Relu(this ITensor t)
            => new ReLU().Forward(t);

        /// <summary>
        /// Applies the Gaussian Error Linear Unit (GELU) activation function element-wise to the tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>A new tensor with GELU applied to each element.</returns>
        public static ITensor Gelu(this ITensor t)
            => new Gelu().Forward(t);

        /// <summary>
        /// Returns the tensor data as a one-dimensional float array.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>The tensor elements as a float array.</returns>
        public static float[] Data(this ITensor t) => t.ToArray();

        /// <summary>
        /// Converts the tensor to an array of the specified struct type <typeparamref name="T"/>.
        /// </summary>
        /// <typeparam name="T">The target element type. Must be a value type (struct).</typeparam>
        /// <param name="tensor">The tensor to convert to an array.</param>
        /// <returns>An array containing the tensor's data converted to type <typeparamref name="T"/>.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="tensor"/> is null.</exception>
        /// <exception cref="NotSupportedException">
        /// Thrown when the tensor backend or target type <typeparamref name="T"/> is not supported.
        /// Currently supports <see cref="CpuBackend"/> with <see cref="float"/>, <see cref="double"/>, 
        /// and <see cref="System.Numerics.Complex"/> target types.
        /// </exception>
        public static T[] ToArray<T>(this ITensor? tensor) where T : struct
        {
            if (tensor is null)
                throw new ArgumentNullException(nameof(tensor));

            if (tensor is CpuBackend cpuTensor)
            {
                float[] source = cpuTensor.ToArray();

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
            }

            throw new NotSupportedException(
                $"Conversion from {tensor.GetType().Name} to {typeof(T).Name}[] is not yet supported.");
        }

        /// <summary>
        /// Converts the tensor to a double-precision floating-point array.
        /// </summary>
        /// <param name="tensor">The tensor to convert.</param>
        /// <returns>An array of <see cref="double"/> containing the tensor data.</returns>
        public static double[] ToArray(this ITensor? tensor)
        {
            return tensor.ToArray<double>();
        }

        /// <summary>
        /// Adds a float scalar value to each element of the tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <param name="scalar">The scalar value to add to each element.</param>
        /// <returns>A new tensor containing the result of the element-wise addition.</returns>
        public static ITensor Add(this ITensor t, float scalar)
            => t.Add(Tensor.FromScalar(scalar, t.Device));

        /// <summary>
        /// Adds an integer scalar value to each element of the tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <param name="scalar">The scalar value to add to each element.</param>
        /// <returns>A new tensor containing the result of the element-wise addition.</returns>
        public static ITensor Add(this ITensor t, int scalar)
            => t.Add(Tensor.FromScalar(scalar, t.Device));

        /// <summary>
        /// Subtracts a float scalar value from each element of the tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <param name="scalar">The scalar value to subtract from each element.</param>
        /// <returns>A new tensor containing the result of the element-wise subtraction.</returns>
        public static ITensor Subtract(this ITensor t, float scalar)
            => t.Subtract(Tensor.FromScalar(scalar, t.Device));

        /// <summary>
        /// Multiplies each element of the tensor by the specified float scalar value.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <param name="scalar">The scalar value to multiply by.</param>
        /// <returns>A new tensor containing the result of the element-wise multiplication.</returns>
        public static ITensor Multiply(this ITensor t, float scalar)
            => t.Multiply(Tensor.FromScalar(scalar, t.Device));

        /// <summary>
        /// Creates a new tensor filled with ones that has the same shape and device as the input tensor.
        /// </summary>
        /// <param name="t">The tensor whose shape and device will be used.</param>
        /// <returns>A new tensor of ones with matching shape and device.</returns>
        public static ITensor OnesLike(this ITensor t)
            => Tensor.Ones(t.Shape, t.Device);

        /// <summary>
        /// Creates a new tensor filled with zeros that has the same shape and device as the input tensor.
        /// </summary>
        /// <param name="t">The tensor whose shape and device will be used.</param>
        /// <returns>A new tensor of zeros with matching shape and device.</returns>
        public static ITensor ZerosLike(this ITensor t)
            => Tensor.Zeros(t.Shape, t.Device);
    }
}