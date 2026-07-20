// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Fluent
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.Linq;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Activations;
    using ArborNet.Layers;
    using ArborNet.Layers.Normalization;
    using ArborNet.Losses;
    using ArborNet.Core.Functional;
    /// <summary>
    /// The heart of ArborNet — a beautifully designed, fluent, and highly expressive API 
    /// for tensor operations, neural network construction, and on-the-fly execution.
    /// </summary>

    #endregion

    public sealed class X : IEquatable<X>
    {
        private readonly ITensor _tensor;
        /// <summary>
        /// Gets the underlying ITensor instance managed by this fluent wrapper.
        /// </summary>

        public ITensor Tensor => _tensor;
        /// <summary>
        /// Gets the shape of the underlying tensor.
        /// </summary>

        public TensorShape Shape => _tensor.Shape;
        /// <summary>
        /// Gets the device on which the underlying tensor is allocated.
        /// </summary>

        public Device Device => _tensor.Device;

        /// <summary>
        /// Initializes a new instance of the <see cref="X"/> class by wrapping any <see cref="ITensor"/>.
        /// </summary>
        /// <param name="tensor">The input tensor to wrap.</param>
        public X(ITensor tensor)
        {
            _tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
        }

        #region Conversions, Equality and Boolean Logic

        /// <summary>
        /// Implicitly converts an <see cref="X"/> fluent wrapper to its underlying <see cref="ITensor"/>.
        /// </summary>
        public static implicit operator Tensor(X x) => (Tensor)ArborNet.Core.Tensors.Tensor.Unwrap(x._tensor);

        /// <summary>
        /// Implicitly converts a native <see cref="Tensor"/> into a fluent <see cref="X"/> wrapper.
        /// </summary>
        public static implicit operator X(Tensor t) => new(t);

        /// <summary>
        /// Implicitly converts a <see cref="Variable"/> into a fluent <see cref="X"/> wrapper.
        /// </summary>
        public static implicit operator X(Variable v) => new(v);

        /// <summary>
        /// Explicitly converts an <see cref="X"/> fluent wrapper to a <see cref="Variable"/>.
        /// </summary>
        public static explicit operator Variable(X x) => new(x._tensor);

        /// <summary>
        /// Evaluates the scalar representation of this tensor as a boolean.
        /// </summary>
        public static implicit operator bool(X x) => x.ToScalar() != 0f;

        /// <summary>
        /// Operator true override to enable native C# logical execution.
        /// </summary>
        public static bool operator true(X x) => x.ToScalar() != 0f;

        /// <summary>
        /// Operator false override to enable native C# logical execution.
        /// </summary>
        public static bool operator false(X x) => x.ToScalar() == 0f;
        /// <summary>
        /// Determines whether the current <see cref="X"/> wrapper is equal to another <see cref="X"/> wrapper.
        /// </summary>
        /// <param name="other">An object to compare with this object.</param>
        /// <returns><c>true</c> if the current object is equal to the <paramref name="other"/> parameter; otherwise, <c>false</c>.</returns>

        public bool Equals(X? other)
        {
            if (other is null) return false;
            return ReferenceEquals(this, other) || _tensor.Equals(other._tensor);
        }
        /// <summary>
        /// Determines whether the specified object is equal to the current <see cref="X"/> wrapper.
        /// </summary>
        /// <param name="obj">The object to compare with the current object.</param>
        /// <returns><c>true</c> if the specified object is equal to the current object; otherwise, <c>false</c>.</returns>

        public override bool Equals(object? obj) => obj is X other && Equals(other);
        /// <summary>
        /// Serves as the default hash function for the underlying tensor.
        /// </summary>
        /// <returns>A hash code for the current object.</returns>

        public override int GetHashCode() => _tensor.GetHashCode();
        /// <summary>Creates an <see cref="X"/> wrapper around an existing tensor.</summary>
        /// <param name="tensor">The underlying tensor to wrap.</param>
        /// <returns>A new <see cref="X"/> instance.</returns>

        #endregion

        #region Static Factories

        public static X From(ITensor tensor) => new(tensor);
        /// <summary>Creates a tensor from a flat array with the specified shape dimensions.</summary>
        /// <param name="data">The source element array.</param>
        /// <param name="shape">The dimensions of the target tensor.</param>
        /// <returns>A new <see cref="X"/> instance initialized with the provided data.</returns>

        public static X From(float[] data, params int[] shape)
    => new(ArborNet.Core.Tensors.Tensor.FromArray(data, new TensorShape(shape)));
        /// <summary>Creates a tensor from a flat array with the specified shape dimensions.</summary>
        /// <param name="data">The source element array.</param>
        /// <param name="shape">The dimensions of the target tensor.</param>
        /// <returns>A new <see cref="X"/> instance initialized with the provided data.</returns>

        public static X FromArray(float[] data, params int[] shape)
    => new(ArborNet.Core.Tensors.Tensor.FromArray(data, new TensorShape(shape)));
        /// <summary>Creates a tensor from a flat array with the specified TensorShape structure.</summary>
        /// <param name="data">The source element array.</param>
        /// <param name="shape">The <see cref="TensorShape"/> describing the geometry of the target tensor.</param>
        /// <param name="device">The compute device on which the memory should be allocated.</param>
        /// <returns>A new <see cref="X"/> instance initialized with the provided data on the specified device.</returns>

        public static X FromArray(float[] data, TensorShape shape, Device? device = null)
    => new(ArborNet.Core.Tensors.Tensor.FromArray(data, shape, device));
        /// <summary>Creates a tensor filled with zeros.</summary>
        /// <param name="shape">The dimensions of the tensor.</param>
        /// <returns>A new <see cref="X"/> instance filled with zeros.</returns>

        public static X Zeros(params int[] shape)
    => new(ArborNet.Core.Tensors.Tensor.Zeros(new TensorShape(shape)));
        /// <summary>Creates a tensor filled with ones.</summary>
        /// <param name="shape">The dimensions of the tensor.</param>
        /// <returns>A new <see cref="X"/> instance filled with ones.</returns>

        public static X Ones(params int[] shape)
    => new(ArborNet.Core.Tensors.Tensor.Ones(new TensorShape(shape)));
        /// <summary>Creates a tensor with values drawn from a uniform distribution [0, 1).</summary>
        /// <param name="shape">The dimensions of the tensor.</param>
        /// <returns>A new <see cref="X"/> instance with uniformly distributed values.</returns>

        public static X Rand(params int[] shape)
    => new(ArborNet.Core.Tensors.Tensor.Rand(new TensorShape(shape)));
        /// <summary>Creates a tensor with values drawn from a standard normal distribution.</summary>
        /// <param name="shape">The dimensions of the tensor.</param>
        /// <returns>A new <see cref="X"/> instance with normally distributed values.</returns>

        public static X Randn(params int[] shape)
    => new(ArborNet.Core.Tensors.Tensor.Randn(new TensorShape(shape)));
        /// <summary>Elegant syntax: <c>X(tensor)</c>.</summary>
        /// <param name="tensor">The target tensor.</param>
        /// <returns>An <see cref="X"/> wrapper around the specified tensor.</returns>

        public static X Of(ITensor tensor) => new(tensor);
        /// <summary>
        /// Computes the numerical negation of the underlying tensor.
        /// </summary>
        /// <returns>A new <see cref="X"/> containing the negated values.</returns>

        #endregion

        #region Fluent Unary Math

        public X Negate() => new(_tensor.Negate());
        /// <summary>
        /// Computes the exponential (e^x) of each element in the underlying tensor.
        /// </summary>
        /// <returns>A new <see cref="X"/> containing the exponential values.</returns>
        public X Exp() => new(_tensor.Exp());
        /// <summary>
        /// Computes the natural logarithm of each element in the underlying tensor.
        /// </summary>
        /// <returns>A new <see cref="X"/> containing the logarithmic values.</returns>
        public X Log() => new(_tensor.Log());
        /// <summary>
        /// Computes the square root of each element in the underlying tensor.
        /// </summary>
        /// <returns>A new <see cref="X"/> containing the square roots.</returns>
        public X Sqrt() => new(_tensor.Sqrt());
        /// <summary>
        /// Computes the absolute value of each element in the underlying tensor.
        /// </summary>
        /// <returns>A new <see cref="X"/> containing the absolute values.</returns>
        public X Abs() => new(_tensor.Abs());
        /// <summary>
        /// Computes the trigonometric sine of each element in the underlying tensor.
        /// </summary>
        /// <returns>A new <see cref="X"/> containing the sine values.</returns>
        public X Sin() => new(_tensor.Sin());
        /// <summary>
        /// Computes the trigonometric cosine of each element in the underlying tensor.
        /// </summary>
        /// <returns>A new <see cref="X"/> containing the cosine values.</returns>
        public X Cos() => new(_tensor.Cos());
        /// <summary>
        /// Computes the element-wise sign representation (-1, 0, or 1) of the underlying tensor.
        /// </summary>
        /// <returns>A new <see cref="X"/> containing the sign values.</returns>
        public X Sign() => new(_tensor.Sign());
        /// <summary>
        /// Computes the element-wise logical NOT operation on the underlying tensor.
        /// </summary>
        /// <returns>A new <see cref="X"/> containing the logical NOT results.</returns>
        public X LogicalNot() => new(_tensor.LogicalNot());
        /// <summary>
        /// Clips each element of the tensor to a defined closed interval [min, max].
        /// </summary>
        /// <param name="min">The minimum allowable value boundary.</param>
        /// <param name="max">The maximum allowable value boundary.</param>
        /// <returns>A new <see cref="X"/> with values clipped within the interval.</returns>
        public X Clip(float min, float max) => new(_tensor.Clip(min, max));
        /// <summary>
        /// Performs element-wise addition between this tensor and another fluent wrapper tensor.
        /// </summary>
        /// <param name="other">The tensor wrapper to add.</param>
        /// <returns>A new <see cref="X"/> containing the sum.</returns>

        #endregion

        #region Fluent Binary Math

        public X Add(X other) => new(_tensor.Add(other._tensor));
        /// <summary>
        /// Performs element-wise addition between this tensor and a raw <see cref="ITensor"/>.
        /// </summary>
        /// <param name="other">The tensor to add.</param>
        /// <returns>A new <see cref="X"/> containing the sum.</returns>
        public X Add(ITensor other) => new(_tensor.Add(other));
        /// <summary>
        /// Adds a scalar constant to each element of this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to add.</param>
        /// <returns>A new <see cref="X"/> containing the summed values.</returns>
        public X Add(float scalar) => new(_tensor.Add(scalar));
        /// <summary>
        /// Performs element-wise subtraction between this tensor and another fluent wrapper tensor.
        /// </summary>
        /// <param name="other">The tensor wrapper to subtract.</param>
        /// <returns>A new <see cref="X"/> containing the difference.</returns>

        public X Subtract(X other) => new(_tensor.Subtract(other._tensor));
        /// <summary>
        /// Performs element-wise subtraction between this tensor and a raw <see cref="ITensor"/>.
        /// </summary>
        /// <param name="other">The tensor to subtract.</param>
        /// <returns>A new <see cref="X"/> containing the difference.</returns>
        public X Subtract(ITensor other) => new(_tensor.Subtract(other));
        /// <summary>
        /// Subtracts a scalar constant from each element of this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to subtract.</param>
        /// <returns>A new <see cref="X"/> containing the subtracted values.</returns>
        public X Subtract(float scalar) => new(_tensor.Subtract(scalar));
        /// <summary>
        /// Performs element-wise multiplication between this tensor and another fluent wrapper tensor.
        /// </summary>
        /// <param name="other">The multiplier tensor wrapper.</param>
        /// <returns>A new <see cref="X"/> containing the element-wise product.</returns>

        public X Multiply(X other) => new(_tensor.Multiply(other._tensor));
        /// <summary>
        /// Performs element-wise multiplication between this tensor and a raw <see cref="ITensor"/>.
        /// </summary>
        /// <param name="other">The multiplier tensor.</param>
        /// <returns>A new <see cref="X"/> containing the element-wise product.</returns>
        public X Multiply(ITensor other) => new(_tensor.Multiply(other));
        /// <summary>
        /// Multiplies each element of this tensor by a single-precision scalar.
        /// </summary>
        /// <param name="scalar">The single-precision scalar value.</param>
        /// <returns>A new <see cref="X"/> containing the scaled values.</returns>
        public X Multiply(float scalar) => new(_tensor.Multiply(scalar));
        /// <summary>
        /// Multiplies each element of this tensor by a double-precision scalar.
        /// </summary>
        /// <param name="scalar">The double-precision scalar value.</param>
        /// <returns>A new <see cref="X"/> containing the scaled values.</returns>
        public X Multiply(double scalar) => new(_tensor.Multiply((float)scalar));
        /// <summary>
        /// Performs element-wise division of this tensor by another fluent wrapper tensor.
        /// </summary>
        /// <param name="other">The divisor tensor wrapper.</param>
        /// <returns>A new <see cref="X"/> containing the quotient.</returns>

        public X Divide(X other) => new(_tensor.Divide(other._tensor));
        /// <summary>
        /// Performs element-wise division of this tensor by a raw <see cref="ITensor"/>.
        /// </summary>
        /// <param name="other">The divisor tensor.</param>
        /// <returns>A new <see cref="X"/> containing the quotient.</returns>
        public X Divide(ITensor other) => new(_tensor.Divide(other));
        /// <summary>
        /// Divides each element of this tensor by a single-precision scalar.
        /// </summary>
        /// <param name="scalar">The single-precision divisor value.</param>
        /// <returns>A new <see cref="X"/> containing the divided values.</returns>
        public X Divide(float scalar) => new(_tensor.Divide(scalar));
        /// <summary>
        /// Divides each element of this tensor by a double-precision scalar.
        /// </summary>
        /// <param name="scalar">The double-precision divisor value.</param>
        /// <returns>A new <see cref="X"/> containing the divided values.</returns>
        public X Divide(double scalar) => new(_tensor.Divide(scalar));
        /// <summary>
        /// Computes the element-wise exponentiation of this tensor to a given scalar power.
        /// </summary>
        /// <param name="exponent">The scalar power value.</param>
        /// <returns>A new <see cref="X"/> containing the exponentiated values.</returns>

        public X Pow(float exponent) => new(_tensor.Pow(exponent));
        /// <summary>
        /// Computes the element-wise exponentiation of this tensor to a given tensor power.
        /// </summary>
        /// <param name="exponent">The tensor wrapper containing the exponents.</param>
        /// <returns>A new <see cref="X"/> containing the exponentiated values.</returns>
        public X Pow(X exponent) => new(_tensor.Pow(exponent._tensor));
        /// <summary>
        /// Computes the matrix multiplication between this tensor and another fluent wrapper tensor.
        /// </summary>
        /// <param name="other">The right-hand side tensor wrapper.</param>
        /// <returns>A new <see cref="X"/> containing the matrix product.</returns>

        public X MatMul(X other) => new(_tensor.MatMul(other._tensor));
        /// <summary>
        /// Computes the matrix multiplication between this tensor and a raw <see cref="ITensor"/>.
        /// </summary>
        /// <param name="other">The right-hand side tensor.</param>
        /// <returns>A new <see cref="X"/> containing the matrix product.</returns>
        public X MatMul(ITensor other) => new(_tensor.MatMul(other));
        /// <summary>
        /// Computes the sum of elements along a specific axis or across the entire tensor.
        /// </summary>
        /// <param name="axis">The optional target index of the axis along which the sum is computed.</param>
        /// <param name="keepDims">Whether the reduced dimensions are retained with length 1.</param>
        /// <returns>A new <see cref="X"/> containing the sum values.</returns>

        #endregion

        #region Fluent Reductions & Shape Operations

        public X Sum(int? axis = null, bool keepDims = false) => new(_tensor.Sum(axis, keepDims));
        /// <summary>
        /// Computes the sum of elements along multiple specified axes.
        /// </summary>
        /// <param name="axes">The collection of target axes indexes.</param>
        /// <param name="keepDims">Whether the reduced dimensions are retained with length 1.</param>
        /// <returns>A new <see cref="X"/> containing the sum values.</returns>
        public X Sum(int[] axes, bool keepDims = false) => new(_tensor.Sum(axes, keepDims));
        /// <summary>
        /// Computes the arithmetic mean of elements along a specific axis or across the entire tensor.
        /// </summary>
        /// <param name="axis">The optional target index of the axis along which the mean is computed.</param>
        /// <param name="keepDims">Whether the reduced dimensions are retained with length 1.</param>
        /// <returns>A new <see cref="X"/> containing the calculated means.</returns>
        public X Mean(int? axis = null, bool keepDims = false) => new(_tensor.Mean(axis, keepDims));
        /// <summary>
        /// Computes the arithmetic mean of elements along multiple specified axes.
        /// </summary>
        /// <param name="axes">The collection of target axes indexes.</param>
        /// <param name="keepDims">Whether the reduced dimensions are retained with length 1.</param>
        /// <returns>A new <see cref="X"/> containing the calculated means.</returns>
        public X Mean(int[] axes, bool keepDims = false) => new(_tensor.Mean(axes, keepDims));
        /// <summary>
        /// Finds the maximum element along a specific axis.
        /// </summary>
        /// <param name="axis">The target axis index. Defaults to -1 (last axis).</param>
        /// <param name="keepDims">Whether the reduced dimensions are retained with length 1.</param>
        /// <returns>A new <see cref="X"/> containing the maximum values.</returns>
        public X Max(int axis = -1, bool keepDims = false) => new(_tensor.Max(axis, keepDims));
        /// <summary>
        /// Finds the minimum element along a specific axis.
        /// </summary>
        /// <param name="axis">The target axis index. Defaults to -1 (last axis).</param>
        /// <param name="keepDims">Whether the reduced dimensions are retained with length 1.</param>
        /// <returns>A new <see cref="X"/> containing the minimum values.</returns>
        public X Min(int axis = -1, bool keepDims = false) => new(_tensor.Min(axis, keepDims));
        /// <summary>
        /// Identifies the index of the minimum element along the specified axis.
        /// </summary>
        /// <param name="axis">The target axis index.</param>
        /// <returns>A new <see cref="X"/> containing indices of the minimum values.</returns>
        public X ArgMin(int axis) => new(_tensor.ArgMin(axis));
        /// <summary>
        /// Identifies the index of the maximum element along the specified axis.
        /// </summary>
        /// <param name="axis">The target axis index.</param>
        /// <returns>A new <see cref="X"/> containing indices of the maximum values.</returns>
        public X ArgMax(int axis) => new(_tensor.ArgMax(axis));
        /// <summary>
        /// Computes the cumulative sum of elements along a specified axis.
        /// </summary>
        /// <param name="axis">The target axis index.</param>
        /// <returns>A new <see cref="X"/> containing the cumulative sum sequence.</returns>
        public X CumSum(int axis) => new(_tensor.CumSum(axis));
        /// <summary>
        /// Adjusts the shape of the underlying tensor without altering its data.
        /// </summary>
        /// <param name="newShape">The desired structural dimensions.</param>
        /// <returns>A new <see cref="X"/> reshaped with the specified dimensions.</returns>

        public X Reshape(params int[] newShape) => new(_tensor.Reshape(newShape));
        /// <summary>
        /// Reorders the dimensions of this tensor according to a given permutation.
        /// </summary>
        /// <param name="perm">The list of dimension indexes indicating the new order.</param>
        /// <returns>A new transposed <see cref="X"/>.</returns>
        public X Transpose(params int[] perm) => new(_tensor.Transpose(perm));
        /// <summary>
        /// Slices the tensor along its dimensions using specified range configurations.
        /// </summary>
        /// <param name="slices">A list of tuples defining the (start, end, step) bounds for each dimension.</param>
        /// <returns>A new sliced <see cref="X"/>.</returns>
        public X Slice(params (int start, int end, int step)[] slices) => new(_tensor.Slice(slices));
        /// <summary>
        /// Broadcasts this tensor to a larger target shape compatibility layout.
        /// </summary>
        /// <param name="targetShape">The target destination shape layout.</param>
        /// <returns>A new <see cref="X"/> mapped to the broader shape.</returns>
        public X BroadcastTo(TensorShape targetShape) => new(_tensor.BroadcastTo(targetShape));
        /// <summary>
        /// Reshapes and broadcasts the tensor along a specified axis configuration.
        /// </summary>
        /// <param name="target">The target shape.</param>
        /// <param name="axis">The focal target axis.</param>
        /// <returns>A new reshaped and broadcasted <see cref="X"/>.</returns>
        public X ReshapeWithBroadcast(TensorShape target, int axis) => new(_tensor.ReshapeWithBroadcast(target, axis));
        /// <summary>
        /// Concatenates multiple fluent tensor wrappers together along a specified axis.
        /// </summary>
        /// <param name="others">An enumerable sequence of tensors to concatenate with this one.</param>
        /// <param name="axis">The axis along which to concatenate. Defaults to 0.</param>
        /// <returns>A new concatenated <see cref="X"/>.</returns>

        public X Concat(IEnumerable<X> others, int axis = 0)
            => new(_tensor.Concat(others.Select(o => o._tensor), axis));
        /// <summary>
        /// Flattens the tensor into a 2D matrix structure, preserving the batch size on the first axis.
        /// </summary>
        /// <returns>A new flattened <see cref="X"/> with shape [batchSize, -1].</returns>

        public X Flatten()
        {
            int batchSize = _tensor.Shape[0];
            return new(_tensor.Reshape(batchSize, -1));
        }
        /// <summary>
        /// Performs an element-wise "greater than" comparison against another tensor wrapper.
        /// </summary>
        /// <param name="other">The tensor wrapper to compare against.</param>
        /// <returns>A mask tensor where elements are 1 if true, 0 if false.</returns>

        #endregion

        #region Fluent Logical Comparisons

        public X GreaterThan(X other) => new(_tensor.GreaterThan(other._tensor));
        /// <summary>
        /// Performs an element-wise "greater than" comparison against a scalar constant.
        /// </summary>
        /// <param name="scalar">The scalar value to compare against.</param>
        /// <returns>A mask tensor where elements are 1 if true, 0 if false.</returns>
        public X GreaterThan(float scalar) => new(_tensor.GreaterThan(ArborNet.Core.Tensors.Tensor.FromScalar(scalar, _tensor.Device)));
        /// <summary>
        /// Performs an element-wise "greater than or equal to" comparison against another tensor wrapper.
        /// </summary>
        /// <param name="other">The tensor wrapper to compare against.</param>
        /// <returns>A mask tensor where elements are 1 if true, 0 if false.</returns>
        public X GreaterThanOrEqual(X other) => new(_tensor.GreaterThanOrEqual(other._tensor));
        /// <summary>
        /// Performs an element-wise "greater than or equal to" comparison against a scalar constant.
        /// </summary>
        /// <param name="scalar">The scalar value to compare against.</param>
        /// <returns>A mask tensor where elements are 1 if true, 0 if false.</returns>
        public X GreaterThanOrEqual(float scalar) => new(_tensor.GreaterThanOrEqual(ArborNet.Core.Tensors.Tensor.FromScalar(scalar, _tensor.Device)));
        /// <summary>
        /// Performs an element-wise "less than or equal to" comparison against another tensor wrapper.
        /// </summary>
        /// <param name="other">The tensor wrapper to compare against.</param>
        /// <returns>A mask tensor where elements are 1 if true, 0 if false.</returns>
        public X LessEqual(X other) => new(_tensor.LessEqual(other._tensor));
        /// <summary>
        /// Performs an element-wise "less than or equal to" comparison against a scalar constant.
        /// </summary>
        /// <param name="scalar">The scalar value to compare against.</param>
        /// <returns>A mask tensor where elements are 1 if true, 0 if false.</returns>
        public X LessEqual(float scalar) => new(_tensor.LessEqual(ArborNet.Core.Tensors.Tensor.FromScalar(scalar, _tensor.Device)));
        /// <summary>
        /// Performs an element-wise equality comparison against another tensor wrapper.
        /// </summary>
        /// <param name="other">The tensor wrapper to compare against.</param>
        /// <returns>A mask tensor where elements are 1 if equal, 0 otherwise.</returns>
        public X Equal(X other) => new(_tensor.Equal(other._tensor));
        /// <summary>
        /// Performs an element-wise equality comparison against a scalar constant.
        /// </summary>
        /// <param name="scalar">The scalar value to compare against.</param>
        /// <returns>A mask tensor where elements are 1 if equal, 0 otherwise.</returns>
        public X Equal(float scalar) => new(_tensor.Equal(ArborNet.Core.Tensors.Tensor.FromScalar(scalar, _tensor.Device)));
        /// <summary>
        /// Selects elements from two tensors based on a specified boolean condition mask.
        /// </summary>
        /// <param name="condition">The condition mask tensor determining selection.</param>
        /// <param name="trueValue">The value source used where the condition is true.</param>
        /// <param name="falseValue">The value source used where the condition is false.</param>
        /// <returns>A new <see cref="X"/> containing the elements selected by the condition.</returns>

        public X Where(X condition, X trueValue, X falseValue)
            => new(_tensor.Where(condition._tensor, trueValue._tensor, falseValue._tensor));
        /// <summary>
        /// Transfers this tensor to a target computing device.
        /// </summary>
        /// <param name="targetDevice">The device target.</param>
        /// <returns>A new <see cref="X"/> allocated on the target device.</returns>

        #endregion

        #region Fluent Device Routing

        public X To(Device targetDevice) => new(_tensor.To(targetDevice));
        /// <summary>
        /// Moves this tensor to host CPU memory.
        /// </summary>
        /// <returns>A new <see cref="X"/> allocated on the CPU.</returns>
        public X Cpu() => new(_tensor.To(Device.CPU));
        /// <summary>
        /// Moves this tensor to CUDA-compatible GPU memory.
        /// </summary>
        /// <param name="id">The device identifier index.</param>
        /// <returns>A new <see cref="X"/> allocated on the specified CUDA device.</returns>
        public X Cuda(int id = 0) => new(_tensor.To(Device.Cuda(id)));
        /// <summary>
        /// Moves this tensor to ROCm-compatible GPU memory.
        /// </summary>
        /// <param name="id">The device identifier index.</param>
        /// <returns>A new <see cref="X"/> allocated on the specified ROCm device.</returns>
        public X Rocm(int id = 0) => new(_tensor.To(Device.Rocm(id)));
        /// <summary>
        /// Applies the Rectified Linear Unit (ReLU) activation function element-wise.
        /// </summary>
        /// <returns>A new <see cref="X"/> after ReLU processing.</returns>

        #endregion

        #region Fluent Activations

        public X ReLU() => new(new ReLU().Forward(_tensor));
        /// <summary>
        /// Applies the Gaussian Error Linear Unit (GELU) activation function element-wise.
        /// </summary>
        /// <returns>A new <see cref="X"/> after GELU processing.</returns>
        public X GELU() => new(new Gelu().Forward(_tensor));
        /// <summary>
        /// Applies the Hyperbolic Tangent (Tanh) activation function element-wise.
        /// </summary>
        /// <returns>A new <see cref="X"/> after Tanh processing.</returns>
        public X Tanh() => new(new Tanh().Forward(_tensor));
        /// <summary>
        /// Applies the Sigmoid activation function element-wise.
        /// </summary>
        /// <returns>A new <see cref="X"/> after Sigmoid processing.</returns>
        public X Sigmoid() => new(new Sigmoid().Forward(_tensor));
        /// <summary>
        /// Applies the Softmax activation function along the specified axis.
        /// </summary>
        /// <param name="axis">The axis along which normalized probabilities are calculated. Defaults to -1.</param>
        /// <returns>A new <see cref="X"/> after Softmax normalization.</returns>
        public X Softmax(int axis = -1) => new(new Softmax(axis).Forward(_tensor));
        /// <summary>
        /// Applies the Exponential Linear Unit (ELU) activation function element-wise.
        /// </summary>
        /// <param name="alpha">The scalar scale value for negative inputs. Defaults to 1.0f.</param>
        /// <returns>A new <see cref="X"/> after ELU processing.</returns>
        public X ELU(float alpha = 1.0f) => new(new ELU(alpha).Forward(_tensor));
        /// <summary>
        /// Applies the Leaky Rectified Linear Unit (LeakyReLU) activation function element-wise.
        /// </summary>
        /// <param name="negativeSlope">The control parameter determining negative scale. Defaults to 0.01f.</param>
        /// <returns>A new <see cref="X"/> after LeakyReLU processing.</returns>
        public X LeakyReLU(float negativeSlope = 0.01f) => new(new LeakyReLU(negativeSlope).Forward(_tensor));
        /// <summary>
        /// Applies the Mish activation function element-wise.
        /// </summary>
        /// <returns>A new <see cref="X"/> after Mish processing.</returns>
        public X Mish() => new(new Mish().Forward(_tensor));
        /// <summary>
        /// Applies the Softplus activation function element-wise.
        /// </summary>
        /// <returns>A new <see cref="X"/> after Softplus processing.</returns>
        public X Softplus() => new(new Softplus().Forward(_tensor));
        /// <summary>
        /// Applies the Swish activation function element-wise.
        /// </summary>
        /// <returns>A new <see cref="X"/> after Swish processing.</returns>
        public X Swish() => new(new Swish().Forward(_tensor));
        /// <summary>
        /// Applies the Sigmoid Linear Unit (SiLU) activation function element-wise.
        /// </summary>
        /// <returns>A new <see cref="X"/> after SiLU processing.</returns>
        public X SiLU() => new(new SiLU().Forward(_tensor));
        /// <summary>
        /// Applies the Gated Linear Unit (GLU) activation function.
        /// </summary>
        /// <returns>A new <see cref="X"/> after GLU processing.</returns>
        public X GLU() => new(new GLU().Forward(_tensor));
        /// <summary>
        /// Applies the SwiGLU activation function.
        /// </summary>
        /// <returns>A new <see cref="X"/> after SwiGLU processing.</returns>
        public X SwiGLU() => new(new SwiGLU().Forward(_tensor));
        /// <summary>
        /// Feeds this tensor through a neural network layer.
        /// </summary>
        /// <param name="layer">The network layer implementation.</param>
        /// <returns>A new <see cref="X"/> wrapping the computed layer output.</returns>

        #endregion

        #region Fluent Neural Network Builders

        public X Apply(ILayer layer) => new(layer.Forward(_tensor));
        /// <summary>
        /// Applies a fully connected Linear transformation layer.
        /// </summary>
        /// <param name="outFeatures">Number of output feature dimensions.</param>
        /// <param name="bias">Determines whether to compute a learnable bias offset. Defaults to <c>true</c>.</param>
        /// <returns>A new <see cref="X"/> containing the linearly projected values.</returns>

        public X Linear(int outFeatures, bool bias = true)
        {
            int inFeatures = _tensor.Shape[^1];
            var layer = new Linear(inFeatures, outFeatures, _tensor.Device);
            return Apply(layer);
        }
        /// <summary>
        /// Applies a 1-dimensional convolutional layer.
        /// </summary>
        /// <param name="outChannels">The number of output channel filters.</param>
        /// <param name="kernelSize">The size of the convolution kernel window.</param>
        /// <param name="stride">The sliding movement stride of the kernel. Defaults to 1.</param>
        /// <param name="padding">The zero-padding width. Defaults to 0.</param>
        /// <param name="useBias">Determines whether to use a learnable bias. Defaults to <c>true</c>.</param>
        /// <returns>A new <see cref="X"/> output after 1D convolution processing.</returns>

        public X Conv1D(int outChannels, int kernelSize, int stride = 1, int padding = 0, bool useBias = true)
        {
            int inChannels = _tensor.Shape[1];
            var layer = new Conv1D(inChannels, outChannels, kernelSize, stride, padding, useBias, _tensor.Device);
            return Apply(layer);
        }
        /// <summary>
        /// Applies a 2-dimensional convolutional layer.
        /// </summary>
        /// <param name="outChannels">The number of output channel filters.</param>
        /// <param name="kernelSize">The size of the convolution kernel window.</param>
        /// <param name="stride">The sliding movement stride of the kernel. Defaults to 1.</param>
        /// <param name="padding">The zero-padding dimensions. Defaults to 0.</param>
        /// <param name="useBias">Determines whether to use a learnable bias. Defaults to <c>true</c>.</param>
        /// <returns>A new <see cref="X"/> output after 2D convolution processing.</returns>

        public X Conv2D(int outChannels, int kernelSize, int stride = 1, int padding = 0, bool useBias = true)
        {
            int inChannels = _tensor.Shape[1];
            var layer = new Conv2D(inChannels, outChannels, kernelSize, stride, padding, useBias, _tensor.Device);
            return Apply(layer);
        }
        /// <summary>
        /// Applies a 3-dimensional convolutional layer.
        /// </summary>
        /// <param name="outChannels">The number of output channel filters.</param>
        /// <param name="kernelDepth">The depth dimension of the convolution kernel.</param>
        /// <param name="kernelHeight">The height dimension of the convolution kernel.</param>
        /// <param name="kernelWidth">The width dimension of the convolution kernel.</param>
        /// <param name="hasBias">Determines whether to use a learnable bias. Defaults to <c>true</c>.</param>
        /// <param name="stride">The sliding movement stride of the kernel. Defaults to 1.</param>
        /// <param name="padding">The zero-padding dimensions. Defaults to 0.</param>
        /// <returns>A new <see cref="X"/> output after 3D convolution processing.</returns>

        public X Conv3D(int outChannels, int kernelDepth, int kernelHeight, int kernelWidth, bool hasBias = true, int stride = 1, int padding = 0)
        {
            int inChannels = _tensor.Shape[1];
            var layer = new Conv3D(inChannels, outChannels, kernelDepth, kernelHeight, kernelWidth, hasBias, stride, padding);
            return Apply(layer);
        }
        /// <summary>
        /// Applies Batch Normalization.
        /// </summary>
        /// <param name="numFeatures">The dimension size of the features.</param>
        /// <param name="eps">Small constant added to denominator for numerical stability. Defaults to 1e-5f.</param>
        /// <param name="momentum">The factor used for computing running stats. Defaults to 0.1f.</param>
        /// <param name="useAffine">Indicates whether to utilize learnable scale/shift. Defaults to <c>true</c>.</param>
        /// <returns>A new <see cref="X"/> after batch normalization.</returns>

        public X BatchNorm(int numFeatures, float eps = 1e-5f, float momentum = 0.1f, bool useAffine = true)
        {
            var layer = new BatchNorm(numFeatures, eps, momentum, useAffine);
            layer.To(_tensor.Device);
            return Apply(layer);
        }
        /// <summary>
        /// Applies Layer Normalization over the last dimension of this tensor.
        /// </summary>
        /// <returns>A new <see cref="X"/> after layer normalization.</returns>

        public X LayerNorm()
        {
            var layer = new ArborNet.Layers.LayerNorm(new[] { _tensor.Shape[^1] });
            layer.To(_tensor.Device);
            return Apply(layer);
        }
        /// <summary>
        /// Applies Layer Normalization with target structural configuration shape parameters.
        /// </summary>
        /// <param name="normalizedShape">The structural dimensions normalized by the layer.</param>
        /// <param name="eps">Small stability constant. Defaults to 1e-5f.</param>
        /// <param name="useAffine">Indicates whether to utilize learnable scale/shift. Defaults to <c>true</c>.</param>
        /// <returns>A new <see cref="X"/> after layer normalization.</returns>

        public X LayerNorm(int[] normalizedShape, float eps = 1e-5f, bool useAffine = true)
        {
            var layer = new ArborNet.Layers.Normalization.LayerNorm(normalizedShape, eps, useAffine);
            layer.To(_tensor.Device);
            return Apply(layer);
        }
        /// <summary>
        /// Applies Root Mean Square (RMS) Normalization.
        /// </summary>
        /// <param name="numFeatures">The structural size of features normalized.</param>
        /// <param name="eps">Small stability constant. Defaults to 1e-6f.</param>
        /// <param name="useAffine">Indicates whether to utilize learnable scale parameters. Defaults to <c>true</c>.</param>
        /// <returns>A new <see cref="X"/> after RMS normalization.</returns>

        public X RMSNorm(int numFeatures, float eps = 1e-6f, bool useAffine = true)
        {
            var layer = new RMSNorm(numFeatures, eps, useAffine);
            layer.To(_tensor.Device);
            return Apply(layer);
        }
        /// <summary>
        /// Applies Group Normalization.
        /// </summary>
        /// <param name="numChannels">The total channels present within the input.</param>
        /// <param name="numGroups">The target partitions for dividing channels.</param>
        /// <param name="eps">Small stability constant. Defaults to 1e-5f.</param>
        /// <param name="useAffine">Indicates whether to utilize learnable scale/shift. Defaults to <c>true</c>.</param>
        /// <returns>A new <see cref="X"/> after group normalization.</returns>

        public X GroupNorm(int numChannels, int numGroups, float eps = 1e-5f, bool useAffine = true)
        {
            var layer = new GroupNorm(numChannels, numGroups, eps, useAffine);
            layer.To(_tensor.Device);
            return Apply(layer);
        }
        /// <summary>
        /// Applies Instance Normalization.
        /// </summary>
        /// <param name="numChannels">The total count of input channels.</param>
        /// <param name="eps">Small stability constant. Defaults to 1e-5f.</param>
        /// <param name="useAffine">Indicates whether to utilize learnable scale/shift. Defaults to <c>true</c>.</param>
        /// <returns>A new <see cref="X"/> after instance normalization.</returns>

        public X InstanceNorm(int numChannels, float eps = 1e-5f, bool useAffine = true)
        {
            var layer = new InstanceNorm(numChannels, eps, useAffine);
            layer.To(_tensor.Device);
            return Apply(layer);
        }
        /// <summary>
        /// Applies Dropout regularizing logic with a specified rate.
        /// </summary>
        /// <param name="p">The dropout rate probability. Defaults to 0.5f.</param>
        /// <returns>A new <see cref="X"/> with dropped-out nodes.</returns>

        public X Dropout(float p = 0.5f) => Apply(new Dropout(p));
        /// <summary>
        /// Applies a scaled dot-product attention layer.
        /// </summary>
        /// <param name="embedDim">The total structural dimension size of embeddings.</param>
        /// <param name="numHeads">The parallel execution head count.</param>
        /// <param name="useBias">Indicates whether learning bias projections should be utilized. Defaults to <c>true</c>.</param>
        /// <returns>A new <see cref="X"/> output after applying attention.</returns>

        public X Attention(int embedDim, int numHeads, bool useBias = true)
            => Apply(new Attention(embedDim, numHeads, useBias));
        /// <summary>
        /// Applies a multi-head attention layer.
        /// </summary>
        /// <param name="dModel">The model embedding projection dimension.</param>
        /// <param name="numHeads">The parallel execution head count.</param>
        /// <param name="useBias">Indicates whether learning bias projections should be utilized. Defaults to <c>true</c>.</param>
        /// <returns>A new <see cref="X"/> output after multi-head attention.</returns>

        public X MultiHeadAttention(int dModel, int numHeads, bool useBias = true)
            => Apply(new MultiHeadAttention(dModel, numHeads, useBias));
        /// <summary>
        /// Applies a standard Transformer block.
        /// </summary>
        /// <param name="dModel">The model embedding projection dimension.</param>
        /// <param name="numHeads">The parallel execution head count.</param>
        /// <param name="ffDim">The hidden feed-forward dimension size. If 0, defaults to 4 * dModel.</param>
        /// <returns>A new <see cref="X"/> output after the transformer block operation.</returns>

        public X TransformerBlock(int dModel, int numHeads, int ffDim = 0)
            => Apply(new TransformerBlock(dModel, numHeads, ffDim));
        /// <summary>
        /// Applies an Embedding lookup layer.
        /// </summary>
        /// <param name="numEmbeddings">The size of the lookup vocabulary.</param>
        /// <param name="embeddingDim">The structural dimension size of individual embeddings.</param>
        /// <returns>A new <see cref="X"/> containing retrieved embedding vectors.</returns>

        public X Embedding(int numEmbeddings, int embeddingDim)
            => Apply(new Embedding(numEmbeddings, embeddingDim));
        /// <summary>
        /// Calculates the Mean Squared Error (MSE) loss between this tensor and targeted ground truths.
        /// </summary>
        /// <param name="targets">The ground-truth comparison target values.</param>
        /// <param name="reduction">The type of reduction logic ("mean", "sum", "none"). Defaults to "mean".</param>
        /// <returns>A new loss <see cref="X"/>.</returns>

        #endregion

        #region Fluent Loss Chaining

        public X MseLoss(X targets, string reduction = "mean")
            => new(new MSE().Forward(_tensor, targets._tensor, reduction));
        /// <summary>
        /// Calculates the Categorical Cross-Entropy loss between this tensor and targeted ground truths.
        /// </summary>
        /// <param name="targets">The ground-truth comparison target labels.</param>
        /// <param name="reduction">The type of reduction logic ("mean", "sum", "none"). Defaults to "mean".</param>
        /// <returns>A new loss <see cref="X"/>.</returns>

        public X CrossEntropyLoss(X targets, string reduction = "mean")
            => new(new CrossEntropy().Forward(_tensor, targets._tensor, reduction));
        /// <summary>
        /// Calculates the Binary Cross-Entropy loss between this tensor and targeted ground truths.
        /// </summary>
        /// <param name="targets">The ground-truth comparison target labels.</param>
        /// <param name="reduction">The type of reduction logic ("mean", "sum", "none"). Defaults to "mean".</param>
        /// <returns>A new loss <see cref="X"/>.</returns>

        public X BinaryCrossEntropyLoss(X targets, string reduction = "mean")
            => new(new BinaryCrossEntropy().Forward(_tensor, targets._tensor, reduction));
        /// <summary>
        /// Calculates the Hinge loss between this tensor and targeted ground truths.
        /// </summary>
        /// <param name="targets">The ground-truth comparison target labels.</param>
        /// <param name="reduction">The type of reduction logic ("mean", "sum", "none"). Defaults to "mean".</param>
        /// <returns>A new loss <see cref="X"/>.</returns>

        public X HingeLoss(X targets, string reduction = "mean")
            => new(new Hinge().Forward(_tensor, targets._tensor, reduction));
        /// <summary>
        /// Calculates the Huber loss between this tensor and targeted ground truths.
        /// </summary>
        /// <param name="targets">The ground-truth comparison target labels.</param>
        /// <param name="delta">The threshold parameter defining quadratic versus linear calculation boundary. Defaults to 1.0f.</param>
        /// <param name="reduction">The type of reduction logic ("mean", "sum", "none"). Defaults to "mean".</param>
        /// <returns>A new loss <see cref="X"/>.</returns>

        public X HuberLoss(X targets, float delta = 1.0f, string reduction = "mean")
            => new(new Huber(delta).Forward(_tensor, targets._tensor, reduction));
        /// <summary>
        /// Calculates the Kullback-Leibler divergence (KL Div) loss between this tensor and targeted ground truths.
        /// </summary>
        /// <param name="targets">The ground-truth target distributions.</param>
        /// <param name="reduction">The type of reduction logic ("mean", "sum", "batchmean", "none"). Defaults to "mean".</param>
        /// <returns>A new loss <see cref="X"/>.</returns>

        public X KLDivLoss(X targets, string reduction = "mean")
            => new(new KLDiv().Forward(_tensor, targets._tensor, reduction));
        /// <summary>
        /// Copies and returns the underlying tensor's data into a flat array.
        /// </summary>
        /// <returns>A flat <see cref="T:float[]"/> containing the elements.</returns>

        #endregion

        #region Autograd & Execution Accessors

        public float[] ToArray() => _tensor.ToArray();
        /// <summary>
        /// Evaluates and returns the single scalar value of the underlying tensor.
        /// </summary>
        /// <returns>The primary single-precision floating-point value.</returns>
        public float ToScalar() => _tensor.ToScalar();
        /// <summary>
        /// Triggers autograd to execute backpropagation backwards from this node.
        /// </summary>
        public void Backward() => _tensor.Backward();
        /// <summary>
        /// Clears the computed backpropagation gradients associated with this tensor.
        /// </summary>
        public void ClearGrad() => _tensor.ClearGrad();

        #endregion

        #region Rigorous Operator Overloading

        public static X operator +(X a, X b) => a.Add(b);
        public static X operator +(X a, float b) => a.Add(b);
        public static X operator +(float a, X b) => b.Add(a);

        public static X operator -(X a, X b) => a.Subtract(b);
        public static X operator -(X a, float b) => a.Subtract(b);
        public static X operator -(float a, X b) => new X(ArborNet.Core.Tensors.Tensor.FromScalar(a, b.Device)).Subtract(b);
        public static X operator -(X a) => a.Negate();

        public static X operator *(X a, X b) => a.Multiply(b);
        public static X operator *(X a, float b) => a.Multiply(b);
        public static X operator *(float a, X b) => b.Multiply(a);

        public static X operator /(X a, X b) => a.Divide(b);
        public static X operator /(X a, float b) => a.Divide(b);
        public static X operator /(float a, X b) => new X(ArborNet.Core.Tensors.Tensor.FromScalar(a, b.Device)).Divide(b);

        public static X operator >(X a, X b) => a.GreaterThan(b);
        public static X operator >(X a, float b) => a.GreaterThan(b);
        public static X operator <(X a, X b) => b.GreaterThan(a);
        public static X operator <(X a, float b) => new X(ArborNet.Core.Tensors.Tensor.FromScalar(b, a.Device)).GreaterThan(a);

        public static X operator >=(X a, X b) => a.GreaterThanOrEqual(b);
        public static X operator >=(X a, float b) => a.GreaterThanOrEqual(b);
        public static X operator <=(X a, X b) => b.GreaterThanOrEqual(a);
        public static X operator <=(X a, float b) => new X(ArborNet.Core.Tensors.Tensor.FromScalar(b, a.Device)).GreaterThanOrEqual(a);

        public static X operator ==(X a, X b) => a.Equal(b);
        public static X operator ==(X a, float b) => a.Equal(b);
        public static X operator !=(X a, X b) => a.Equal(b).LogicalNot();
        public static X operator !=(X a, float b) => a.Equal(b).LogicalNot();

        #endregion
    }
}
