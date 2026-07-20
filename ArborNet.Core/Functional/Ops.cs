// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Functional
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.Linq;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Core.Devices;
    using ArborNet.Activations;
    /// <summary>
    /// Provides a comprehensive set of static functional operations for tensor creation,
    /// mathematical computations, and neural network primitives.
    /// </summary>
    /// <remarks>
    /// This class follows a functional style similar to PyTorch's <c>torch</c> module.
    /// Most operations are thin wrappers that delegate to the underlying <see cref="ITensor"/>
    /// implementations or activation functions. All factory methods default to the CPU device
    /// when no device is explicitly provided.
    /// </remarks>

    #endregion

    public static class Ops
    {
        /// <summary>
        /// Creates a tensor filled with zeros.
        /// </summary>
        /// <param name="shape">The shape of the tensor to create.</param>
        /// <param name="device">The device to allocate the tensor on. If null, defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A new tensor of the specified shape filled with zeros.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="shape"/> is null.</exception>
        public static ITensor Zeros(TensorShape shape, Device device = null)
=> Tensor.Zeros(shape, device ?? Device.CPU);
        /// <summary>
        /// Creates a tensor filled with ones.
        /// </summary>
        /// <param name="shape">The shape of the tensor to create.</param>
        /// <param name="device">The device to allocate the tensor on. If null, defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A new tensor of the specified shape filled with ones.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="shape"/> is null.</exception>

        public static ITensor Ones(TensorShape shape, Device device = null)
    => Tensor.Ones(shape, device ?? Device.CPU);
        /// <summary>
        /// Creates a tensor filled with a specified scalar value.
        /// </summary>
        /// <param name="shape">The shape of the tensor to create.</param>
        /// <param name="value">The value to fill every element with.</param>
        /// <param name="device">The device to allocate the tensor on. If null, defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A new tensor of the specified shape filled with the given value.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="shape"/> is null.</exception>

        public static ITensor Full(TensorShape shape, float value, Device device = null)
    => Tensor.Full(shape, value, device ?? Device.CPU);
        /// <summary>
        /// Creates a scalar (0-dimensional) tensor from a single float value.
        /// </summary>
        /// <param name="value">The scalar value.</param>
        /// <param name="device">The device to allocate the tensor on. If null, defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A scalar tensor containing the provided value.</returns>

        public static ITensor FromScalar(float value, Device device = null)
    => Tensor.FromScalar(value, device ?? Device.CPU);
        /// <summary>
        /// Creates a tensor with values drawn from a uniform distribution over [0, 1).
        /// </summary>
        /// <param name="shape">The shape of the tensor to create.</param>
        /// <param name="device">The device to allocate the tensor on. If null, defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A tensor filled with random uniform values.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="shape"/> is null.</exception>

        public static ITensor Rand(TensorShape shape, Device device = null)
    => Tensor.Rand(shape, device ?? Device.CPU);
        /// <summary>
        /// Creates a tensor with values drawn from a standard normal distribution (mean = 0, std = 1).
        /// </summary>
        /// <param name="shape">The shape of the tensor to create.</param>
        /// <param name="device">The device to allocate the tensor on. If null, defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A tensor filled with random values from a normal distribution.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="shape"/> is null.</exception>

        public static ITensor Randn(TensorShape shape, Device device = null)
    => Tensor.Randn(shape, device ?? Device.CPU);
        /// <summary>
        /// Creates a tensor from a 1D array of data with the specified shape.
        /// </summary>
        /// <param name="data">The source array containing the tensor elements.</param>
        /// <param name="shape">The shape that the data represents.</param>
        /// <param name="device">The device to allocate the tensor on. If null, defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A new tensor initialized with the provided data.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="data"/> or <paramref name="shape"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the length of <paramref name="data"/> does not match the product of the dimensions in <paramref name="shape"/>.</exception>

        public static ITensor FromArray(float[] data, TensorShape shape, Device device = null)
    => Tensor.FromArray(data, shape, device ?? Device.CPU);
        /// <summary>
        /// Performs element-wise addition of two tensors.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>A new tensor containing the element-wise sum of <paramref name="a"/> and <paramref name="b"/>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> or <paramref name="b"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the shapes of <paramref name="a"/> and <paramref name="b"/> are incompatible for element-wise addition.</exception>

        // Arithmetic

        public static ITensor Add(ITensor a, ITensor b) => a.Add(b);
        /// <summary>
        /// Performs element-wise subtraction of two tensors.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>A new tensor containing the element-wise difference of <paramref name="a"/> and <paramref name="b"/>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> or <paramref name="b"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the shapes of <paramref name="a"/> and <paramref name="b"/> are incompatible for element-wise subtraction.</exception>

        public static ITensor Subtract(ITensor a, ITensor b) => a.Subtract(b);
        /// <summary>
        /// Performs element-wise multiplication of two tensors.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>A new tensor containing the element-wise product of <paramref name="a"/> and <paramref name="b"/>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> or <paramref name="b"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the shapes of <paramref name="a"/> and <paramref name="b"/> are incompatible for element-wise multiplication.</exception>

        public static ITensor Multiply(ITensor a, ITensor b) => a.Multiply(b);
        /// <summary>
        /// Performs element-wise division of two tensors.
        /// </summary>
        /// <param name="a">The first input tensor (numerator).</param>
        /// <param name="b">The second input tensor (denominator).</param>
        /// <returns>A new tensor containing the element-wise division of <paramref name="a"/> by <paramref name="b"/>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> or <paramref name="b"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the shapes of <paramref name="a"/> and <paramref name="b"/> are incompatible for element-wise division.</exception>

        public static ITensor Divide(ITensor a, ITensor b) => a.Divide(b);
        /// <summary>
        /// Multiplies a tensor by a scalar value.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <param name="scalar">The scalar multiplier.</param>
        /// <returns>A new tensor containing the scaled values.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null.</exception>

        public static ITensor Mul(ITensor a, float scalar) => a.Multiply(scalar);
        /// <summary>
        /// Performs matrix multiplication of two tensors.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>The result of the matrix multiplication.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> or <paramref name="b"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the dimensions of <paramref name="a"/> and <paramref name="b"/> are incompatible for matrix multiplication.</exception>

        public static ITensor MatMul(ITensor a, ITensor b) => a.MatMul(b);
        /// <summary>
        /// Reshapes the input tensor to the specified dimensions.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <param name="shape">The new shape dimensions.</param>
        /// <returns>A tensor with the new shape (may share memory when possible).</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> or <paramref name="shape"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the total number of elements in the new shape does not match the original tensor.</exception>

        public static ITensor Reshape(ITensor a, params int[] shape) => a.Reshape(shape);
        /// <summary>
        /// Permutes the dimensions of the tensor according to the provided permutation.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <param name="perm">The permutation of the dimensions.</param>
        /// <returns>The transposed tensor according to the permutation.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> or <paramref name="perm"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the length of <paramref name="perm"/> does not match the tensor's rank, or contains invalid dimension indices.</exception>

        public static ITensor Transpose(ITensor a, int[] perm) => a.Transpose(perm);
        /// <summary>
        /// Computes the sum of all elements in the tensor.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <returns>A scalar tensor containing the sum of all elements.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null.</exception>

        public static ITensor Sum(ITensor a) => a.Sum();
        /// <summary>
        /// Computes the mean of all elements in the tensor.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <returns>A scalar tensor containing the mean of all elements.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null.</exception>

        public static ITensor Mean(ITensor a) => a.Mean();
        /// <summary>
        /// Computes the maximum values along the specified axis.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <param name="axis">The axis to reduce over. A value of -1 reduces over all dimensions.</param>
        /// <returns>A tensor containing the maximum values.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="axis"/> is out of the valid range of dimensions.</exception>

        public static ITensor Max(ITensor a, int axis = -1) => a.Max(axis);
        /// <summary>
        /// Computes the minimum values along the specified axis.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <param name="axis">The axis to reduce over. A value of -1 reduces over all dimensions.</param>
        /// <returns>A tensor containing the minimum values.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="axis"/> is out of the valid range of dimensions.</exception>

        public static ITensor Min(ITensor a, int axis = -1) => a.Min(axis);
        /// <summary>
        /// Computes the exponential of each element in the input tensor.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <returns>A tensor with the exponential function applied element-wise.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null.</exception>

        public static ITensor Exp(ITensor a) => a.Exp();
        /// <summary>
        /// Computes the natural logarithm of each element in the input tensor.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <returns>A tensor with the logarithm function applied element-wise.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null.</exception>

        public static ITensor Log(ITensor a) => a.Log();
        /// <summary>
        /// Computes the square root of each element in the input tensor.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <returns>A tensor with the square root function applied element-wise.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null.</exception>

        public static ITensor Sqrt(ITensor a) => a.Sqrt();
        /// <summary>
        /// Computes the sine of each element in the input tensor.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <returns>A tensor with the sine function applied element-wise.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null.</exception>

        public static ITensor Sin(ITensor a) => a.Sin();
        /// <summary>
        /// Computes the cosine of each element in the input tensor.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <returns>A tensor with the cosine function applied element-wise.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null.</exception>

        public static ITensor Cos(ITensor a) => a.Cos();
        /// <summary>
        /// Raises each element of the tensor to the specified power.
        /// </summary>
        /// <param name="a">The base tensor.</param>
        /// <param name="exponent">The exponent (scalar).</param>
        /// <returns>A tensor with each element raised to the given power.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null.</exception>

        public static ITensor Pow(ITensor a, float exponent) => a.Pow(exponent);
        /// <summary>
        /// Raises each element of the first tensor to the power of the corresponding element in the second tensor.
        /// </summary>
        /// <param name="a">The base tensor.</param>
        /// <param name="exponent">The exponent tensor.</param>
        /// <returns>A tensor containing the element-wise power operation.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> or <paramref name="exponent"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the shapes of <paramref name="a"/> and <paramref name="exponent"/> are incompatible.</exception>

        public static ITensor Pow(ITensor a, ITensor exponent) => a.Pow(exponent);
        /// <summary>
        /// Concatenates a sequence of tensors along the specified axis.
        /// </summary>
        /// <param name="tensors">The list of tensors to concatenate.</param>
        /// <param name="axis">The axis along which the tensors are concatenated.</param>
        /// <returns>A single tensor resulting from the concatenation.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="tensors"/> is null.</exception>
        /// <exception cref="InvalidOperationException">Thrown when <paramref name="tensors"/> contains no elements.</exception>
        /// <exception cref="ArgumentException">Thrown when the tensors have mismatched dimensions along any axis other than the target <paramref name="axis"/>.</exception>

        public static ITensor Concat(IEnumerable<ITensor> tensors, int axis = 0)
    => tensors.First().Concat(tensors.Skip(1), axis);
        /// <summary>
        /// Extracts slices from the tensor using start, end, and step specifications per dimension.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <param name="slices">Array of slice tuples (start, end, step) for each dimension.</param>
        /// <returns>The sliced tensor.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> or <paramref name="slices"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the slice range or dimensions are invalid for the tensor's shape.</exception>

        public static ITensor Slice(ITensor a, params (int start, int end, int step)[] slices)
    => a.Slice(slices);
        /// <summary>
        /// Adds two tensors with automatic broadcasting support.
        /// </summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor (will be broadcast if necessary).</param>
        /// <returns>The result of the broadcasted addition.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> or <paramref name="b"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the shapes cannot be broadcasted together.</exception>

        public static ITensor BroadcastAdd(ITensor a, ITensor b) => a.BroadcastAdd(b);
        /// <summary>
        /// Performs element-wise greater-than comparison.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>A boolean tensor where each element is true if a &gt; b.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> or <paramref name="b"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when tensor shapes are incompatible for comparison.</exception>

        // Comparisons

        public static ITensor GreaterThan(ITensor a, ITensor b) => a.GreaterThan(b);
        /// <summary>
        /// Performs element-wise greater-than-or-equal comparison.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>A boolean tensor where each element is true if a &gt;= b.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> or <paramref name="b"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when tensor shapes are incompatible for comparison.</exception>

        public static ITensor GreaterThanOrEqual(ITensor a, ITensor b) => a.GreaterThanOrEqual(b);
        /// <summary>
        /// Performs element-wise less-than-or-equal comparison.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>A boolean tensor where each element is true if a &lt;= b.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> or <paramref name="b"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when tensor shapes are incompatible for comparison.</exception>

        public static ITensor LessEqual(ITensor a, ITensor b) => a.LessEqual(b);
        /// <summary>
        /// Selects values from <paramref name="trueValue"/> or <paramref name="falseValue"/> 
        /// based on the boolean condition tensor.
        /// </summary>
        /// <param name="condition">The condition tensor (non-zero values are treated as true).</param>
        /// <param name="trueValue">Values selected where condition is true.</param>
        /// <param name="falseValue">Values selected where condition is false.</param>
        /// <returns>A tensor containing values chosen according to the condition.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="condition"/>, <paramref name="trueValue"/>, or <paramref name="falseValue"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the shapes of the inputs are incompatible for broad-selection.</exception>

        public static ITensor Where(ITensor condition, ITensor trueValue, ITensor falseValue)
    => condition.Where(condition, trueValue, falseValue);
        /// <summary>
        /// Applies the hyperbolic tangent (tanh) activation function.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>The tensor after applying the tanh function element-wise.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="t"/> is null.</exception>

        // Activations (delegating to activation classes)

        public static ITensor Tanh(ITensor t) => new Tanh().Forward(t);
        /// <summary>
        /// Applies the Rectified Linear Unit (ReLU) activation function (max(0, x)).
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>The tensor after applying the ReLU function element-wise.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="t"/> is null.</exception>

        public static ITensor Relu(ITensor t) => new ReLU().Forward(t);
        /// <summary>
        /// Applies the sigmoid activation function (1 / (1 + exp(-x))).
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>The tensor after applying the sigmoid function element-wise.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="t"/> is null.</exception>

        public static ITensor Sigmoid(ITensor t) => new Sigmoid().Forward(t);
        /// <summary>
        /// Applies the softmax activation function along the specified axis.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <param name="axis">The axis to compute softmax over. Default is -1 (last dimension).</param>
        /// <returns>The tensor after applying the softmax function.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="t"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="axis"/> is out of the valid range of dimensions.</exception>

        public static ITensor Softmax(ITensor t, int axis = -1) => new Softmax(axis).Forward(t);
        /// <summary>
        /// Creates a deep copy of the specified tensor, duplicating its underlying data, computational graph status, and metadata.
        /// </summary>
        /// <param name="a">The source tensor to clone.</param>
        /// <returns>A new <see cref="ITensor"/> instance that is a copy of the original tensor.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null.</exception>

        public static ITensor Clone(ITensor a)
        {
            return a.Clone();
        }
        /// <summary>
        /// Computes the backward pass (backpropagation) of the computational graph starting from the current tensor.
        /// </summary>
        /// <remarks>
        /// This method executes reverse-mode automatic differentiation. It calculates gradients of the current 
        /// tensor with respect to the leaf nodes of the computational graph, accumulating the results 
        /// in the gradient attributes of those leaf tensors. This overload assumes a seed gradient of 1.0, 
        /// which is standard when starting backpropagation from a scalar loss function.
        /// </remarks>
        /// <param name="a">The tensor from which to start backpropagation (typically a scalar loss tensor).</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null.</exception>

        // Gradients and Autograd

        public static void Backward(ITensor a)
        {
            a.Backward();
        }
        /// <summary>
        /// Computes the backward pass (backpropagation) of the computational graph starting from the current tensor, 
        /// using an explicit upstream gradient tensor.
        /// </summary>
        /// <remarks>
        /// This method executes reverse-mode automatic differentiation. It is primarily used when backpropagating 
        /// from a non-scalar output tensor by providing an explicit external gradient of identical shape.
        /// </remarks>
        /// <param name="a">The tensor from which to start backpropagation.</param>
        /// <param name="gradient">The upstream gradient representing incoming sensitivities.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> or <paramref name="gradient"/> is null.</exception>

        public static void Backward(ITensor a, ITensor gradient)
        {
            a.Backward(gradient);
        }
        /// <summary>
        /// Copies or moves the tensor to the specified execution device (e.g., CPU, CUDA GPU).
        /// </summary>
        /// <remarks>
        /// If the tensor is already on the target device, this operation may return the original tensor or a reference-mapped copy 
        /// depending on the underlying device engine's architecture.
        /// </remarks>
        /// <param name="a">The source tensor.</param>
        /// <param name="device">The target <see cref="Device"/> where execution should occur.</param>
        /// <returns>A tensor with identical data residing on the specified device.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> or <paramref name="device"/> is null.</exception>

        // Device Management

        public static ITensor To(ITensor a, Device device)
        {
            return a.To(device);
        }
        /// <summary>
        /// Checks if the tensor's underlying data storage is currently allocated on the CPU.
        /// </summary>
        /// <param name="a">The tensor to inspect.</param>
        /// <returns><c>true</c> if the tensor resides on the CPU; otherwise, <c>false</c>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null.</exception>

        public static bool IsCpu(ITensor a)
        {
            return a.IsCpu();
        }
        /// <summary>
        /// Checks if the tensor's underlying data storage is currently allocated on a CUDA-capable GPU device.
        /// </summary>
        /// <param name="a">The tensor to inspect.</param>
        /// <returns><c>true</c> if the tensor resides on a CUDA GPU; otherwise, <c>false</c>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null.</exception>

        public static bool IsCuda(ITensor a)
        {
            return a.IsCuda();
        }
    }
}