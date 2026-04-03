using System;
using System.Collections.Generic;
using System.Linq;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Devices;
using ArborNet.Activations;

namespace ArborNet.Core.Functional
{
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
    public static class Ops
    {
        /// <summary>
        /// Creates a tensor filled with zeros.
        /// </summary>
        /// <param name="shape">The shape of the tensor to create.</param>
        /// <param name="device">The device to allocate the tensor on. If null, defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A new tensor of the specified shape filled with zeros.</returns>
        public static ITensor Zeros(TensorShape shape, Device device = null)
            => Tensor.Zeros(shape, device ?? Device.CPU);

        /// <summary>
        /// Creates a tensor filled with ones.
        /// </summary>
        /// <param name="shape">The shape of the tensor to create.</param>
        /// <param name="device">The device to allocate the tensor on. If null, defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A new tensor of the specified shape filled with ones.</returns>
        public static ITensor Ones(TensorShape shape, Device device = null)
            => Tensor.Ones(shape, device ?? Device.CPU);

        /// <summary>
        /// Creates a tensor filled with a specified scalar value.
        /// </summary>
        /// <param name="shape">The shape of the tensor to create.</param>
        /// <param name="value">The value to fill every element with.</param>
        /// <param name="device">The device to allocate the tensor on. If null, defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A new tensor of the specified shape filled with the given value.</returns>
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
        public static ITensor Rand(TensorShape shape, Device device = null)
            => Tensor.Rand(shape, device ?? Device.CPU);

        /// <summary>
        /// Creates a tensor with values drawn from a standard normal distribution (mean = 0, std = 1).
        /// </summary>
        /// <param name="shape">The shape of the tensor to create.</param>
        /// <param name="device">The device to allocate the tensor on. If null, defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A tensor filled with random values from a normal distribution.</returns>
        public static ITensor Randn(TensorShape shape, Device device = null)
            => Tensor.Randn(shape, device ?? Device.CPU);

        /// <summary>
        /// Creates a tensor from a 1D array of data with the specified shape.
        /// </summary>
        /// <param name="data">The source array containing the tensor elements.</param>
        /// <param name="shape">The shape that the data represents.</param>
        /// <param name="device">The device to allocate the tensor on. If null, defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A new tensor initialized with the provided data.</returns>
        public static ITensor FromArray(float[] data, TensorShape shape, Device device = null)
            => Tensor.FromArray(data, shape, device ?? Device.CPU);

        // Arithmetic

        /// <summary>
        /// Performs element-wise addition of two tensors.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>A new tensor containing the element-wise sum of <paramref name="a"/> and <paramref name="b"/>.</returns>
        public static ITensor Add(ITensor a, ITensor b) => a.Add(b);

        /// <summary>
        /// Performs element-wise subtraction of two tensors.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>A new tensor containing the element-wise difference of <paramref name="a"/> and <paramref name="b"/>.</returns>
        public static ITensor Subtract(ITensor a, ITensor b) => a.Subtract(b);

        /// <summary>
        /// Performs element-wise multiplication of two tensors.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>A new tensor containing the element-wise product of <paramref name="a"/> and <paramref name="b"/>.</returns>
        public static ITensor Multiply(ITensor a, ITensor b) => a.Multiply(b);

        /// <summary>
        /// Performs element-wise division of two tensors.
        /// </summary>
        /// <param name="a">The first input tensor (numerator).</param>
        /// <param name="b">The second input tensor (denominator).</param>
        /// <returns>A new tensor containing the element-wise division of <paramref name="a"/> by <paramref name="b"/>.</returns>
        public static ITensor Divide(ITensor a, ITensor b) => a.Divide(b);

        /// <summary>
        /// Multiplies a tensor by a scalar value.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <param name="scalar">The scalar multiplier.</param>
        /// <returns>A new tensor containing the scaled values.</returns>
        public static ITensor Mul(ITensor a, float scalar) => a.Multiply(scalar);

        /// <summary>
        /// Performs matrix multiplication of two tensors.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>The result of the matrix multiplication.</returns>
        public static ITensor MatMul(ITensor a, ITensor b) => a.MatMul(b);

        /// <summary>
        /// Reshapes the input tensor to the specified dimensions.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <param name="shape">The new shape dimensions.</param>
        /// <returns>A tensor with the new shape (may share memory when possible).</returns>
        public static ITensor Reshape(ITensor a, params int[] shape) => a.Reshape(shape);

        /// <summary>
        /// Permutes the dimensions of the tensor according to the provided permutation.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <param name="perm">The permutation of the dimensions.</param>
        /// <returns>The transposed tensor according to the permutation.</returns>
        public static ITensor Transpose(ITensor a, int[] perm) => a.Transpose(perm);

        /// <summary>
        /// Computes the sum of all elements in the tensor.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <returns>A scalar tensor containing the sum of all elements.</returns>
        public static ITensor Sum(ITensor a) => a.Sum();

        /// <summary>
        /// Computes the mean of all elements in the tensor.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <returns>A scalar tensor containing the mean of all elements.</returns>
        public static ITensor Mean(ITensor a) => a.Mean();

        /// <summary>
        /// Computes the maximum values along the specified axis.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <param name="axis">The axis to reduce over. A value of -1 reduces over all dimensions.</param>
        /// <returns>A tensor containing the maximum values.</returns>
        public static ITensor Max(ITensor a, int axis = -1) => a.Max(axis);

        /// <summary>
        /// Computes the minimum values along the specified axis.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <param name="axis">The axis to reduce over. A value of -1 reduces over all dimensions.</param>
        /// <returns>A tensor containing the minimum values.</returns>
        public static ITensor Min(ITensor a, int axis = -1) => a.Min(axis);

        /// <summary>
        /// Computes the exponential of each element in the input tensor.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <returns>A tensor with the exponential function applied element-wise.</returns>
        public static ITensor Exp(ITensor a) => a.Exp();

        /// <summary>
        /// Computes the natural logarithm of each element in the input tensor.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <returns>A tensor with the logarithm function applied element-wise.</returns>
        public static ITensor Log(ITensor a) => a.Log();

        /// <summary>
        /// Computes the square root of each element in the input tensor.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <returns>A tensor with the square root function applied element-wise.</returns>
        public static ITensor Sqrt(ITensor a) => a.Sqrt();

        /// <summary>
        /// Computes the sine of each element in the input tensor.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <returns>A tensor with the sine function applied element-wise.</returns>
        public static ITensor Sin(ITensor a) => a.Sin();

        /// <summary>
        /// Computes the cosine of each element in the input tensor.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <returns>A tensor with the cosine function applied element-wise.</returns>
        public static ITensor Cos(ITensor a) => a.Cos();

        /// <summary>
        /// Raises each element of the tensor to the specified power.
        /// </summary>
        /// <param name="a">The base tensor.</param>
        /// <param name="exponent">The exponent (scalar).</param>
        /// <returns>A tensor with each element raised to the given power.</returns>
        public static ITensor Pow(ITensor a, float exponent) => a.Pow(exponent);

        /// <summary>
        /// Raises each element of the first tensor to the power of the corresponding element in the second tensor.
        /// </summary>
        /// <param name="a">The base tensor.</param>
        /// <param name="exponent">The exponent tensor.</param>
        /// <returns>A tensor containing the element-wise power operation.</returns>
        public static ITensor Pow(ITensor a, ITensor exponent) => a.Pow(exponent);

        /// <summary>
        /// Concatenates a sequence of tensors along the specified axis.
        /// </summary>
        /// <param name="tensors">The list of tensors to concatenate.</param>
        /// <param name="axis">The axis along which the tensors are concatenated.</param>
        /// <returns>A single tensor resulting from the concatenation.</returns>
        public static ITensor Concat(IEnumerable<ITensor> tensors, int axis = 0)
            => tensors.First().Concat(tensors.Skip(1), axis);

        /// <summary>
        /// Extracts slices from the tensor using start, end, and step specifications per dimension.
        /// </summary>
        /// <param name="a">The input tensor.</param>
        /// <param name="slices">Array of slice tuples (start, end, step) for each dimension.</param>
        /// <returns>The sliced tensor.</returns>
        public static ITensor Slice(ITensor a, params (int start, int end, int step)[] slices)
            => a.Slice(slices);

        /// <summary>
        /// Adds two tensors with automatic broadcasting support.
        /// </summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor (will be broadcast if necessary).</param>
        /// <returns>The result of the broadcasted addition.</returns>
        public static ITensor BroadcastAdd(ITensor a, ITensor b) => a.BroadcastAdd(b);

        // Comparisons

        /// <summary>
        /// Performs element-wise greater-than comparison.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>A boolean tensor where each element is true if a &gt; b.</returns>
        public static ITensor GreaterThan(ITensor a, ITensor b) => a.GreaterThan(b);

        /// <summary>
        /// Performs element-wise greater-than-or-equal comparison.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>A boolean tensor where each element is true if a &gt;= b.</returns>
        public static ITensor GreaterThanOrEqual(ITensor a, ITensor b) => a.GreaterThanOrEqual(b);

        /// <summary>
        /// Performs element-wise less-than-or-equal comparison.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>A boolean tensor where each element is true if a &lt;= b.</returns>
        public static ITensor LessEqual(ITensor a, ITensor b) => a.LessEqual(b);

        /// <summary>
        /// Selects values from <paramref name="trueValue"/> or <paramref name="falseValue"/> 
        /// based on the boolean condition tensor.
        /// </summary>
        /// <param name="condition">The condition tensor (non-zero values are treated as true).</param>
        /// <param name="trueValue">Values selected where condition is true.</param>
        /// <param name="falseValue">Values selected where condition is false.</param>
        /// <returns>A tensor containing values chosen according to the condition.</returns>
        public static ITensor Where(ITensor condition, ITensor trueValue, ITensor falseValue)
            => condition.Where(condition, trueValue, falseValue);

        // Activations (delegating to activation classes)

        /// <summary>
        /// Applies the hyperbolic tangent (tanh) activation function.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>The tensor after applying the tanh function element-wise.</returns>
        public static ITensor Tanh(ITensor t) => new Tanh().Forward(t);

        /// <summary>
        /// Applies the Rectified Linear Unit (ReLU) activation function (max(0, x)).
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>The tensor after applying the ReLU function element-wise.</returns>
        public static ITensor Relu(ITensor t) => new ReLU().Forward(t);

        /// <summary>
        /// Applies the sigmoid activation function (1 / (1 + exp(-x))).
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>The tensor after applying the sigmoid function element-wise.</returns>
        public static ITensor Sigmoid(ITensor t) => new Sigmoid().Forward(t);

        /// <summary>
        /// Applies the softmax activation function along the specified axis.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <param name="axis">The axis to compute softmax over. Default is -1 (last dimension).</param>
        /// <returns>The tensor after applying the softmax function.</returns>
        public static ITensor Softmax(ITensor t, int axis = -1) => new Softmax(axis).Forward(t);
    

        /// <summary>
        /// Clones a tensor.
        /// </summary>
        /// <param name="a">The tensor.</param>
        /// <returns>The cloned tensor.</returns>
        public static ITensor Clone(ITensor a)
        {
            return a.Clone();
        }

        // Gradients and Autograd

        /// <summary>
        /// Computes backward pass for a tensor.
        /// </summary>
        /// <param name="a">The tensor.</param>
        public static void Backward(ITensor a)
        {
            a.Backward();
        }

        /// <summary>
        /// Computes backward pass with a gradient.
        /// </summary>
        /// <param name="a">The tensor.</param>
        /// <param name="gradient">The gradient.</param>
        public static void Backward(ITensor a, ITensor gradient)
        {
            a.Backward(gradient);
        }

        // Device Management

        /// <summary>
        /// Moves a tensor to a device.
        /// </summary>
        /// <param name="a">The tensor.</param>
        /// <param name="device">The target device.</param>
        /// <returns>The tensor on the new device.</returns>
        public static ITensor To(ITensor a, Device device)
        {
            return a.To(device);
        }

        /// <summary>
        /// Checks if tensor is on CPU.
        /// </summary>
        /// <param name="a">The tensor.</param>
        /// <returns>True if on CPU.</returns>
        public static bool IsCpu(ITensor a)
        {
            return a.IsCpu();
        }

        /// <summary>
        /// Checks if tensor is on CUDA.
        /// </summary>
        /// <param name="a">The tensor.</param>
        /// <returns>True if on CUDA.</returns>
        public static bool IsCuda(ITensor a)
        {
            return a.IsCuda();
        }
    }
}