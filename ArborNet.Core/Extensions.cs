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
    // ****************************************************************************
    // Project:     ArborNet
    // Description: A C# Machine Learning Library implemented in .NET 10 with 
    //              full CUDA support.
    // Author:      OzzieAI - Chris Sykes
    // License:     MIT License
    // ****************************************************************************
    public static class TensorScalarExtensions
        {
            public static ITensor Multiply(this ITensor t, double scalar)
                => t.Multiply((float)scalar);

            public static ITensor Divide(this ITensor t, float scalar)
                => t.Multiply(1f / scalar);

            public static ITensor Tanh(this ITensor t)
                => new Tanh().Forward(t);

            public static ITensor Relu(this ITensor t)
                => new ReLU().Forward(t);

            public static ITensor Gelu(this ITensor t)
                => new Gelu().Forward(t);

            public static float[] Data(this ITensor t) => t.ToArray();

            /// <summary>
            /// Converts the tensor to an array of the specified struct type <typeparamref name="T"/>.
            /// Safely unwraps the tensor to extract the underlying CpuBackend.
            /// </summary>
            public static T[] ToArray<T>(this ITensor? tensor) where T : struct
            {
                if (tensor is null)
                    throw new ArgumentNullException(nameof(tensor));

                // Resolve the underlying backend node from wrappers (Tensor/Variable)
                ITensor unwrapped = Tensor.Unwrap(tensor);

                if (unwrapped is CpuBackend cpuTensor)
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
                else if (unwrapped is CudaBackend cudaTensor)
                {
                    // Pull data back to host CPU for structural conversion
                    var hostData = cudaTensor.ToArray();
                    var tempCpu = new CpuBackend(hostData, cudaTensor.Shape);
                    return tempCpu.ToArray<T>();
                }

                throw new NotSupportedException(
                    $"Conversion from {tensor.GetType().Name} to {typeof(T).Name}[] is not supported on the active backend.");
            }

            public static double[] ToArray(this ITensor? tensor)
            {
                return tensor.ToArray<double>();
            }

            public static ITensor Add(this ITensor t, float scalar)
                => t.Add(Tensor.FromScalar(scalar, t.Device));

            public static ITensor Add(this ITensor t, int scalar)
                => t.Add(Tensor.FromScalar(scalar, t.Device));

            public static ITensor Subtract(this ITensor t, float scalar)
                => t.Subtract(Tensor.FromScalar(scalar, t.Device));

            public static ITensor Multiply(this ITensor t, float scalar)
                => t.Multiply(Tensor.FromScalar(scalar, t.Device));

            public static ITensor OnesLike(this ITensor t)
                => Tensor.Ones(t.Shape, t.Device);

            public static ITensor ZerosLike(this ITensor t)
                => Tensor.Zeros(t.Shape, t.Device);
    }
}