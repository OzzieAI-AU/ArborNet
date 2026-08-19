// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Tensors
{

    #region Using Statements:

    using ArborNet.Activations;
    using ArborNet.Core.Backends;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    /// <summary>
    /// Thread-safe variable wrapper that prevents race conditions and memory corruption
    /// during concurrent backpropagation using atomic synchronization. Wraps an underlying <see cref="ITensor"/>
    /// and handles gradient calculation states.
    /// </summary>

    #endregion

    public class Variable : ITensor
    {

        public uint Version => _inner.Version;

        internal readonly ITensor _inner;
        private readonly object _gradLock = new();
        
        /// <summary>
        /// Gets or sets the input tensors that generated this tensor in the computational graph.
        /// </summary>
        public ITensor[] Inputs { get => _inner.Inputs; set => _inner.Inputs = value; }
        
        /// <summary>
        /// Gets the shape (dimensions) of the underlying tensor.
        /// </summary>
        public TensorShape Shape => _inner.Shape;
        
        /// <summary>
        /// Gets the execution device (e.g., CPU, CUDA) where this tensor resides.
        /// </summary>
        public Device Device => _inner.Device;
        /// <summary>
        /// Gets or sets a value indicating whether gradients should be computed for this tensor during backward passes.
        /// </summary>
        public bool RequiresGrad { get => _inner.RequiresGrad; set => _inner.RequiresGrad = value; }
        /// <summary>
        /// Gets or sets the accumulated gradient tensor for this variable in a thread-safe manner.
        /// </summary>

        public ITensor? Grad
        {
            get { lock (_gradLock) { return _inner.Grad; } }
            set { lock (_gradLock) { _inner.Grad = value; } }
        }
        /// <summary>
        /// Gets or sets the backward function (gradient computation function) associated with this tensor.
        /// </summary>

        public Func<ITensor, ITensor>? GradFn { get => _inner.GradFn; set => _inner.GradFn = value; }
        /// <summary>
        /// Gets the underlying flat array data representation of the tensor.
        /// </summary>
        public float[] Data => _inner.ToArray();

        public Variable(ITensor inner, bool requiresGrad = false)
        {
            _inner = inner ?? throw new ArgumentNullException(nameof(inner));
            RequiresGrad = requiresGrad || inner.RequiresGrad;
        }

        public ITensor Cast(string dtype) => new Variable(_inner.Cast(dtype), RequiresGrad);

        public ITensor Squeeze(int? axis) => new Variable(_inner.Squeeze(axis), RequiresGrad);

        public ITensor Unsqueeze(int axis) => new Variable(_inner.Unsqueeze(axis), RequiresGrad);

        public (ITensor values, ITensor indices) TopK(int k, int axis = -1)
        {
            var (v, i) = _inner.TopK(k, axis);
            return (new Variable(v, RequiresGrad), new Variable(i, false));
        }

        public string DType => "float32";

        /// <summary>
        /// Sets the underlying tensor data with the specified float array.
        /// </summary>
        /// <param name="floats">The array of floating-point numbers containing the new data.</param>

        public void SetData(float[] floats) => _inner.SetData(floats);
        /// <summary>
        /// Retrieves the tensor data as a flat array of floating-point values.
        /// </summary>
        /// <returns>A flat array of the tensor's values.</returns>
        public float[] ToArray() => _inner.ToArray();
        /// <summary>
        /// Converts a single-element tensor to a scalar floating-point value.
        /// </summary>
        /// <returns>The scalar value of the tensor.</returns>
        public float ToScalar() => _inner.ToScalar();
        /// <summary>
        /// Creates a deep copy of the current variable.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> that is a copy of this instance.</returns>
        public ITensor Clone() => new Variable(_inner.Clone(), RequiresGrad);
        /// <summary>
        /// Copies the tensor to the specified execution device (e.g., CPU or CUDA GPU).
        /// </summary>
        /// <param name="device">The target device.</param>
        /// <returns>A new <see cref="ITensor"/> residing on the target device.</returns>
        public ITensor To(Device device) => new Variable(_inner.To(device), RequiresGrad);
        /// <summary>
        /// Determines whether this tensor resides on the CPU.
        /// </summary>
        /// <returns><c>true</c> if the tensor is on the CPU; otherwise, <c>false</c>.</returns>
        public bool IsCpu() => _inner.IsCpu();
        /// <summary>
        /// Determines whether this tensor resides on a CUDA GPU.
        /// </summary>
        /// <returns><c>true</c> if the tensor is on CUDA; otherwise, <c>false</c>.</returns>
        public bool IsCuda() => _inner.IsCuda();
        /// <summary>
        /// Returns all trainable parameters associated with this tensor and its inputs.
        /// </summary>
        /// <returns>An enumerable collection of trainable parameters (tensors).</returns>
        public IEnumerable<ITensor> Parameters() => _inner.Parameters();
        /// <summary>
        /// Adds another tensor to this tensor in-place, modifying this instance's underlying data.
        /// </summary>
        /// <param name="other">The tensor to add.</param>

        public void AddInPlace(ITensor other)
        {
            lock (_gradLock) { _inner.AddInPlace(other); }
        }
        /// <summary>
        /// Adds a scalar value to this tensor in-place, modifying this instance's underlying data.
        /// </summary>
        /// <param name="scalar">The scalar value to add.</param>

        public void AddInPlace(float scalar)
        {
            lock (_gradLock) { _inner.AddInPlace(scalar); }
        }
        /// <summary>
        /// Subtracts another tensor from this tensor in-place, modifying this instance's underlying data.
        /// </summary>
        /// <param name="other">The tensor to subtract.</param>

        public void SubtractInPlace(ITensor other)
        {
            lock (_gradLock) { _inner.SubtractInPlace(other); }
        }
        /// <summary>
        /// Subtracts a scalar value from this tensor in-place, modifying this instance's underlying data.
        /// </summary>
        /// <param name="scalar">The scalar value to subtract.</param>

        public void SubtractInPlace(float scalar)
        {
            lock (_gradLock) { _inner.SubtractInPlace(scalar); }
        }
        /// <summary>
        /// Multiplies this tensor by another tensor in-place, modifying this instance's underlying data.
        /// </summary>
        /// <param name="other">The tensor multiplier.</param>

        public void MultiplyInPlace(ITensor other)
        {
            lock (_gradLock) { _inner.MultiplyInPlace(other); }
        }
        /// <summary>
        /// Multiplies this tensor by a scalar value in-place, modifying this instance's underlying data.
        /// </summary>
        /// <param name="scalar">The scalar multiplier.</param>

        public void MultiplyInPlace(float scalar)
        {
            lock (_gradLock) { _inner.MultiplyInPlace(scalar); }
        }
        /// <summary>
        /// Gathers values along an axis specified by the given index tensor.
        /// </summary>
        /// <param name="axis">The axis along which to gather values.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <returns>A new <see cref="ITensor"/> containing the gathered values.</returns>

        public ITensor Gather(int axis, ITensor indices)
            => new Variable(_inner.Gather(axis, indices), RequiresGrad || indices.RequiresGrad);
        /// <summary>
        /// Accumulates (adds) a gradient tensor to this variable's existing gradient in a thread-safe manner.
        /// </summary>
        /// <param name="delta">The incoming gradient tensor to accumulate.</param>

        public void AccumulateGrad(ITensor delta)
        {
            if (delta == null) return;
            lock (_gradLock)
            {
                _inner.AccumulateGrad(delta);
            }
        }
        /// <summary>
        /// Performs element-wise addition between this tensor and another tensor.
        /// </summary>
        /// <param name="other">The tensor to add.</param>
        /// <returns>A new <see cref="ITensor"/> representing the sum.</returns>

        public ITensor Add(ITensor other) => new Variable(_inner.Add(other), RequiresGrad || other.RequiresGrad);
        /// <summary>
        /// Performs element-wise subtraction of another tensor from this tensor.
        /// </summary>
        /// <param name="other">The tensor to subtract.</param>
        /// <returns>A new <see cref="ITensor"/> representing the difference.</returns>
        public ITensor Subtract(ITensor other) => new Variable(_inner.Subtract(other), RequiresGrad || other.RequiresGrad);
        /// <summary>
        /// Performs element-wise multiplication between this tensor and another tensor.
        /// </summary>
        /// <param name="other">The tensor multiplier.</param>
        /// <returns>A new <see cref="ITensor"/> representing the product.</returns>
        public ITensor Multiply(ITensor other) => new Variable(_inner.Multiply(other), RequiresGrad || other.RequiresGrad);
        /// <summary>
        /// Performs element-wise division of this tensor by another tensor.
        /// </summary>
        /// <param name="other">The divisor tensor.</param>
        /// <returns>A new <see cref="ITensor"/> representing the quotient.</returns>
        public ITensor Divide(ITensor other) => new Variable(_inner.Divide(other), RequiresGrad || other.RequiresGrad);
        /// <summary>
        /// Performs element-wise addition of a scalar to this tensor.
        /// </summary>
        /// <param name="scalar">The scalar to add.</param>
        /// <returns>A new <see cref="ITensor"/> representing the sum.</returns>

        public ITensor Add(float scalar) => new Variable(_inner.Add(scalar), RequiresGrad);
        /// <summary>
        /// Performs element-wise subtraction of a scalar from this tensor.
        /// </summary>
        /// <param name="scalar">The scalar to subtract.</param>
        /// <returns>A new <see cref="ITensor"/> representing the difference.</returns>
        public ITensor Subtract(float scalar) => new Variable(_inner.Subtract(scalar), RequiresGrad);
        /// <summary>
        /// Performs element-wise multiplication of this tensor by a scalar value.
        /// </summary>
        /// <param name="scalar">The scalar multiplier.</param>
        /// <returns>A new <see cref="ITensor"/> representing the product.</returns>
        public ITensor Multiply(float scalar) => new Variable(_inner.Multiply(scalar), RequiresGrad);
        /// <summary>
        /// Performs element-wise division of this tensor by a scalar value.
        /// </summary>
        /// <param name="scalar">The scalar divisor.</param>
        /// <returns>A new <see cref="ITensor"/> representing the quotient.</returns>
        public ITensor Divide(float scalar) => new Variable(_inner.Divide(scalar), RequiresGrad);
        /// <summary>
        /// Performs element-wise subtraction of an integer value from this tensor.
        /// </summary>
        /// <param name="other">The integer value to subtract.</param>
        /// <returns>A new <see cref="ITensor"/> representing the difference.</returns>

        public ITensor Subtract(int other) => Subtract((float)other);
        /// <summary>
        /// Performs element-wise multiplication of this tensor by a double-precision scalar value.
        /// </summary>
        /// <param name="scalar">The double-precision scalar multiplier.</param>
        /// <returns>A new <see cref="ITensor"/> representing the product.</returns>
        public ITensor Multiply(double scalar) => Multiply((float)scalar);
        /// <summary>
        /// Performs element-wise division of this tensor by a double-precision scalar value.
        /// </summary>
        /// <param name="scalar">The double-precision scalar divisor.</param>
        /// <returns>A new <see cref="ITensor"/> representing the quotient.</returns>
        public ITensor Divide(double scalar) => Multiply(1.0 / scalar);
        /// <summary>
        /// Computes the element-wise negation of this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> with negated values.</returns>

        public ITensor Negate() => new Variable(_inner.Negate(), RequiresGrad);
        /// <summary>
        /// Computes the element-wise exponential (e^x) of this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> with exponential values.</returns>
        public ITensor Exp() => new Variable(_inner.Exp(), RequiresGrad);
        /// <summary>
        /// Computes the element-wise natural logarithm (ln) of this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> with logarithmic values.</returns>
        public ITensor Log() => new Variable(_inner.Log(), RequiresGrad);
        /// <summary>
        /// Computes the element-wise square root of this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> with square root values.</returns>
        public ITensor Sqrt() => new Variable(_inner.Sqrt(), RequiresGrad);
        /// <summary>
        /// Computes the element-wise absolute value of this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> with absolute values.</returns>
        public ITensor Abs() => new Variable(_inner.Abs(), RequiresGrad);
        /// <summary>
        /// Computes the element-wise sine of this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> containing the sine of each element.</returns>
        public ITensor Sin() => new Variable(_inner.Sin(), RequiresGrad);
        /// <summary>
        /// Computes the element-wise cosine of this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> containing the cosine of each element.</returns>
        public ITensor Cos() => new Variable(_inner.Cos(), RequiresGrad);
        /// <summary>
        /// Computes the element-wise sign (-1, 0, or 1) of this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> containing the signs of the elements.</returns>
        public ITensor Sign() => new Variable(_inner.Sign(), false);
        /// <summary>
        /// Computes the element-wise base tensor raised to the power of the exponent tensor.
        /// </summary>
        /// <param name="exponent">The exponent tensor.</param>
        /// <returns>A new <see cref="ITensor"/> containing the results.</returns>

        public ITensor Pow(ITensor exponent) => new Variable(_inner.Pow(exponent), RequiresGrad || exponent.RequiresGrad);
        /// <summary>
        /// Computes the element-wise base tensor raised to the power of a scalar exponent.
        /// </summary>
        /// <param name="exponent">The scalar exponent.</param>
        /// <returns>A new <see cref="ITensor"/> containing the results.</returns>
        public ITensor Pow(float exponent) => new Variable(_inner.Pow(exponent), RequiresGrad);
        /// <summary>
        /// Performs matrix multiplication between this tensor and another tensor.
        /// </summary>
        /// <param name="other">The right-hand side tensor.</param>
        /// <returns>A new <see cref="ITensor"/> representing the matrix product.</returns>
        public ITensor MatMul(ITensor other) => new Variable(_inner.MatMul(other), RequiresGrad || other.RequiresGrad);
        /// <summary>
        /// Transposes the dimensions of this tensor according to the specified permutation.
        /// </summary>
        /// <param name="perm">The permutation order of dimensions.</param>
        /// <returns>A new transposed <see cref="ITensor"/>.</returns>
        public ITensor Transpose(int[] perm) => new Variable(_inner.Transpose(perm), RequiresGrad);
        /// <summary>
        /// Reshapes this tensor to the target dimensions without modifying its underlying data.
        /// </summary>
        /// <param name="newShape">The desired shape dimensions.</param>
        /// <returns>A reshaped <see cref="ITensor"/>.</returns>
        public ITensor Reshape(params int[] newShape) => new Variable(_inner.Reshape(newShape), RequiresGrad);
        /// <summary>
        /// Slices the tensor along specified indices, start, end, and stride steps.
        /// </summary>
        /// <param name="slices">An array of slice tuples defining range and step for each dimension.</param>
        /// <returns>A new sliced <see cref="ITensor"/>.</returns>
        public ITensor Slice(params (int start, int end, int step)[] slices) => new Variable(_inner.Slice(slices), RequiresGrad);
        /// <summary>
        /// Concatenates this tensor with a sequence of other tensors along the specified axis.
        /// </summary>
        /// <param name="others">The sequence of tensors to concatenate.</param>
        /// <param name="axis">The dimension along which the tensors will be joined.</param>
        /// <returns>A concatenated <see cref="ITensor"/>.</returns>
        public ITensor Concat(IEnumerable<ITensor> others, int axis = 0) => new Variable(_inner.Concat(others, axis), RequiresGrad);
        /// <summary>
        /// Broadcasts the current tensor to a new target shape.
        /// </summary>
        /// <param name="targetShape">The target shape to broadcast to.</param>
        /// <returns>A new broadcasted <see cref="ITensor"/>.</returns>
        public ITensor BroadcastTo(TensorShape targetShape) => new Variable(_inner.BroadcastTo(targetShape), RequiresGrad);
        /// <summary>
        /// Reshapes and broadcasts this tensor relative to a target shape and specific alignment axis.
        /// </summary>
        /// <param name="target">The target shape.</param>
        /// <param name="axis">The alignment dimension.</param>
        /// <returns>A new reshaped and broadcasted <see cref="ITensor"/>.</returns>
        public ITensor ReshapeWithBroadcast(TensorShape target, int axis) => new Variable(_inner.ReshapeWithBroadcast(target, axis), RequiresGrad);
        /// <summary>
        /// Computes the sum of elements along the specified axis.
        /// </summary>
        /// <param name="axis">The axis along which to sum. If <c>null</c>, sums all elements.</param>
        /// <param name="keepDims">If <c>true</c>, retains reduced dimensions with length 1.</param>
        /// <returns>A new <see cref="ITensor"/> containing the sum.</returns>

        public ITensor Sum(int? axis = null, bool keepDims = false) => new Variable(_inner.Sum(axis, keepDims), RequiresGrad);
        /// <summary>
        /// Computes the sum of elements along multiple specified axes.
        /// </summary>
        /// <param name="axes">An array of dimensions to sum across.</param>
        /// <param name="keepDims">If <c>true</c>, retains reduced dimensions with length 1.</param>
        /// <returns>A new <see cref="ITensor"/> containing the sum.</returns>
        public ITensor Sum(int[] axes, bool keepDims = false) => new Variable(_inner.Sum(axes, keepDims), RequiresGrad);
        /// <summary>
        /// Computes the mean of elements along the specified axis.
        /// </summary>
        /// <param name="axis">The axis along which to compute the mean. If <c>null</c>, computes the mean over all elements.</param>
        /// <param name="keepDims">If <c>true</c>, retains reduced dimensions with length 1.</param>
        /// <returns>A new <see cref="ITensor"/> containing the computed mean.</returns>
        public ITensor Mean(int? axis = null, bool keepDims = false) => new Variable(_inner.Mean(axis, keepDims), RequiresGrad);
        /// <summary>
        /// Computes the mean of elements along multiple specified axes.
        /// </summary>
        /// <param name="axes">An array of dimensions to average across.</param>
        /// <param name="keepDims">If <c>true</c>, retains reduced dimensions with length 1.</param>
        /// <returns>A new <see cref="ITensor"/> containing the computed mean.</returns>
        public ITensor Mean(int[] axes, bool keepDims = false) => new Variable(_inner.Mean(axes, keepDims), RequiresGrad);
        /// <summary>
        /// Finds the maximum value along the specified axis.
        /// </summary>
        /// <param name="axis">The axis to scan. Default is -1 (the last axis).</param>
        /// <param name="keepDims">If <c>true</c>, retains reduced dimensions with length 1.</param>
        /// <returns>A new <see cref="ITensor"/> containing the maximum values.</returns>

        public ITensor Max(int axis = -1, bool keepDims = false) => new Variable(_inner.Max(axis, keepDims), RequiresGrad);
        /// <summary>
        /// Finds the minimum value along the specified axis.
        /// </summary>
        /// <param name="axis">The axis to scan. Default is -1 (the last axis).</param>
        /// <param name="keepDims">If <c>true</c>, retains reduced dimensions with length 1.</param>
        /// <returns>A new <see cref="ITensor"/> containing the minimum values.</returns>
        public ITensor Min(int axis = -1, bool keepDims = false) => new Variable(_inner.Min(axis, keepDims), RequiresGrad);
        /// <summary>
        /// Computes the cumulative sum of elements along a specified axis.
        /// </summary>
        /// <param name="axis">The axis along which to compute the cumulative sum.</param>
        /// <returns>A new <see cref="ITensor"/> representing the cumulative sum.</returns>
        public ITensor CumSum(int axis) => new Variable(_inner.CumSum(axis), RequiresGrad);
        /// <summary>
        /// Compares this tensor against another element-wise for a greater-than relation.
        /// </summary>
        /// <param name="other">The comparison tensor.</param>
        /// <returns>A boolean/numeric mask <see cref="ITensor"/> indicating where the relationship is true.</returns>

        public ITensor GreaterThan(ITensor other) => new Variable(_inner.GreaterThan(other), false);
        /// <summary>
        /// Compares this tensor against another element-wise for a greater-than-or-equal relation.
        /// </summary>
        /// <param name="other">The comparison tensor.</param>
        /// <returns>A boolean/numeric mask <see cref="ITensor"/> indicating where the relationship is true.</returns>
        public ITensor GreaterThanOrEqual(ITensor other) => new Variable(_inner.GreaterThanOrEqual(other), false);
        /// <summary>
        /// Compares this tensor against another element-wise for a less-than-or-equal relation.
        /// </summary>
        /// <param name="other">The comparison tensor.</param>
        /// <returns>A boolean/numeric mask <see cref="ITensor"/> indicating where the relationship is true.</returns>
        public ITensor LessEqual(ITensor other) => new Variable(_inner.LessEqual(other), false);
        /// <summary>
        /// Compares this tensor against another element-wise for equality.
        /// </summary>
        /// <param name="other">The comparison tensor.</param>
        /// <returns>A boolean/numeric mask <see cref="ITensor"/> indicating where elements are equal.</returns>
        public ITensor Equal(ITensor other) => new Variable(_inner.Equal(other), false);
        /// <summary>
        /// Selects elements from either the true or false value tensor based on a conditional mask tensor.
        /// </summary>
        /// <param name="condition">The condition tensor acting as a mask.</param>
        /// <param name="trueValue">The values selected where the condition is true.</param>
        /// <param name="falseValue">The values selected where the condition is false.</param>
        /// <returns>A new blended <see cref="ITensor"/> based on the mask.</returns>
        public ITensor Where(ITensor condition, ITensor trueValue, ITensor falseValue)
            => new Variable(_inner.Where(condition, trueValue, falseValue), false);
        /// <summary>
        /// Computes the hyperbolic tangent activation function of this tensor.
        /// </summary>
        /// <returns>A new activation-applied <see cref="ITensor"/>.</returns>

        public ITensor Tanh() => new Variable(new Tanh().Forward(_inner), RequiresGrad);
        /// <summary>
        /// Computes the Rectified Linear Unit (ReLU) activation function of this tensor.
        /// </summary>
        /// <returns>A new activation-applied <see cref="ITensor"/>.</returns>
        public ITensor Relu() => new Variable(new ReLU().Forward(_inner), RequiresGrad);
        /// <summary>
        /// Computes the Sigmoid activation function of this tensor.
        /// </summary>
        /// <returns>A new activation-applied <see cref="ITensor"/>.</returns>
        public ITensor Sigmoid() => new Variable(new Sigmoid().Forward(_inner), RequiresGrad);
        /// <summary>
        /// Computes the Softmax normalization function of this tensor along a specified axis.
        /// </summary>
        /// <param name="axis">The axis along which to apply Softmax. Default is -1.</param>
        /// <returns>A new normalized <see cref="ITensor"/>.</returns>
        public ITensor Softmax(int axis = -1) => new Variable(new Softmax(axis).Forward(_inner), RequiresGrad);
        /// <summary>
        /// Performs automatic backpropagation starting from this tensor, computing gradients for all prerequisite nodes in the graph.
        /// </summary>
        /// <param name="gradient">The starting external gradient. If <c>null</c>, assumes a default scalar gradient of 1.0.</param>

        public void Backward(ITensor? gradient = null) => _inner.Backward(gradient);
        /// <summary>
        /// Clears the accumulated gradient for this variable in a thread-safe manner.
        /// </summary>
        public void ClearGrad() { lock (_gradLock) { _inner.ClearGrad(); } }
        /// <summary>
        /// Finds the indices of the minimum values along a specified axis.
        /// </summary>
        /// <param name="axis">The target dimension to check.</param>
        /// <returns>A new <see cref="ITensor"/> containing index coordinates.</returns>

        public ITensor ArgMin(int axis) => new Variable(_inner.ArgMin(axis), false);
        /// <summary>
        /// Finds the indices of the maximum values along a specified axis.
        /// </summary>
        /// <param name="axis">The target dimension to check.</param>
        /// <returns>A new <see cref="ITensor"/> containing index coordinates.</returns>
        public ITensor ArgMax(int axis) => new Variable(_inner.ArgMax(axis), false);
        /// <summary>
        /// Adds another tensor to this tensor using implicit shape broadcasting.
        /// </summary>
        /// <param name="other">The tensor to add.</param>
        /// <returns>A new broadcasted-sum <see cref="ITensor"/>.</returns>
        public ITensor BroadcastAdd(ITensor other) => Add(other);
        /// <summary>
        /// Performs an element-wise logical NOT operation.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> representing logically negated values.</returns>
        public ITensor LogicalNot() => new Variable(_inner.LogicalNot(), RequiresGrad);
        /// <summary>
        /// Clips (clamps) the elements of this tensor within a specified range defined by minimum and maximum bounds.
        /// </summary>
        /// <param name="v1">The minimum bounding value.</param>
        /// <param name="v2">The maximum bounding value.</param>
        /// <returns>A new clipped <see cref="ITensor"/>.</returns>
        public ITensor Clip(float v1, float v2) => new Variable(_inner.Clip(v1, v2), RequiresGrad);
    }
}