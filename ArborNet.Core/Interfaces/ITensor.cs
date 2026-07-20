// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Interfaces
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Tensors;

    #endregion

    /// <summary>
    /// Core tensor interface for ArborNet - the foundation of all operations, autograd, and device abstraction.
    /// All backends (CPU/CUDA) and wrappers must implement this exactly.
    /// </summary>
    public interface ITensor
    {
        /// <summary>
        /// Gets the shape (dimensions) of the tensor.
        /// </summary>
        /// <value>A <see cref="TensorShape"/> representing the dimensional structure of this tensor.</value>
        /// <remarks>
        /// The shape is immutable. Any modification of structure requires a view-based change 
        /// such as <see cref="Reshape(int[])"/> or <see cref="Transpose(int[])"/>.
        /// </remarks>
        TensorShape Shape { get; }
        /// <summary>
        /// Gets the execution/storage device (e.g., CPU, GPU) where this tensor resides.
        /// </summary>
        /// <value>The backend <see cref="Device"/> where this tensor's memory is allocated.</value>
        /// <remarks>
        /// Operations between multiple tensors are only valid if they share the same execution device.
        /// Use <see cref="To(Device)"/> to migrate tensors across devices.
        /// </remarks>

        Device Device { get; }
        /// <summary>
        /// Gets or sets the list of input tensors that produced this tensor in the computation graph.
        /// Used primarily by the autograd engine during the backward pass.
        /// </summary>
        /// <value>An array of parent <see cref="ITensor"/> instances, or <c>null</c> if this is a leaf node.</value>
        /// <remarks>
        /// This collection represents the incoming directed edges of the Directed Acyclic Graph (DAG) 
        /// created dynamically during forward-pass computations.
        /// </remarks>

        ITensor[] Inputs { get; set; }
        /// <summary>
        /// Gets or sets a value indicating whether gradients need to be computed and accumulated for this tensor during backward propagation.
        /// </summary>
        /// <value><c>true</c> if the computational graph should track operations to calculate gradients; otherwise, <c>false</c>.</value>
        /// <remarks>
        /// Setting this to <c>false</c> disables subgraph tracking for operations on this tensor,
        /// which dramatically reduces memory usage and improves performance during evaluation/inference.
        /// </remarks>

        bool RequiresGrad { get; set; }
        /// <summary>
        /// Gets or sets the accumulated gradient of this tensor.
        /// </summary>
        /// <value>A gradient <see cref="ITensor"/> of identical shape, or <c>null</c> if no gradient has been calculated.</value>
        /// <remarks>
        /// Gradients are calculated and populated incrementally by the <see cref="Backward(ITensor)"/> call stack.
        /// Call <see cref="ClearGrad"/> to reset this property prior to a new backward pass.
        /// </remarks>

        ITensor? Grad { get; set; }
        /// <summary>
        /// Gets or sets the backward/gradient function used to compute gradients for this tensor's inputs.
        /// </summary>
        /// <value>A delegate mapping the output gradient of this node to the input gradient, or <c>null</c>.</value>
        /// <remarks>
        /// The derivative operations are stored as delegates representing the vector-Jacobian product 
        /// of the operation that generated this tensor.
        /// </remarks>

        Func<ITensor, ITensor>? GradFn { get; set; }
        /// <summary>
        /// Gets the raw underlying flat array of 32-bit floating-point data representing the tensor.
        /// </summary>
        /// <value>A contiguous array containing the localized flat data elements.</value>
        /// <remarks>
        /// Accessing this property on accelerator-backed (CUDA) tensors may trigger a blocking, 
        /// synchronous host-to-device memory transfer. For safer/cleaner extractions, use <see cref="ToArray"/>.
        /// </remarks>

        float[] Data { get; }
        /// <summary>
        /// Accumulates a gradient delta into the current tensor's <see cref="Grad"/> property.
        /// </summary>
        /// <param name="delta">The gradient tensor containing the delta values to accumulate.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="delta"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if shapes or devices of the tensors do not match.</exception>

        void AccumulateGrad(ITensor delta);
        /// <summary>
        /// Gathers values along an axis specified by the given indices tensor.
        /// </summary>
        /// <param name="axis">The axis along which to index.</param>
        /// <param name="indices">The indices tensor containing the coordinate mapping.</param>
        /// <returns>A new <see cref="ITensor"/> containing the elements gathered from the specified axis.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="indices"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="axis"/> is invalid for this tensor's shape.</exception>

        ITensor Gather(int axis, ITensor indices);
        /// <summary>
        /// Materializes and returns the flat underlying data as an array.
        /// </summary>
        /// <returns>A copy or reference to the raw float data array.</returns>
        /// <remarks>
        /// For hardware accelerated backends, this forces synchronization and copies data back to the host RAM.
        /// </remarks>

        float[] ToArray();
        /// <summary>
        /// Extracts the single scalar value from a 1-element tensor.
        /// </summary>
        /// <returns>The scalar representation of the single-element tensor.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the tensor does not contain exactly one element.</exception>

        float ToScalar();
        /// <summary>
        /// Creates a deep copy of the tensor, duplicating both metadata and underlying data.
        /// </summary>
        /// <returns>A new instance of <see cref="ITensor"/> containing identical data.</returns>
        /// <remarks>
        /// The cloned tensor does not preserve structural graph properties such as <see cref="Inputs"/> or <see cref="GradFn"/>.
        /// </remarks>

        ITensor Clone();
        /// <summary>
        /// Transfers or copies the tensor to the specified execution device (e.g., CPU, CUDA).
        /// </summary>
        /// <param name="device">The destination device.</param>
        /// <returns>A tensor represented on the targeted device.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="device"/> is <c>null</c>.</exception>
        /// <remarks>
        /// If the tensor is already stored on the requested <paramref name="device"/>, this may return the same instance or a shallow copy.
        /// </remarks>

        ITensor To(Device device);
        /// <summary>
        /// Determines whether the tensor is allocated on host CPU memory.
        /// </summary>
        /// <returns><c>true</c> if the tensor resides on the CPU; otherwise, <c>false</c>.</returns>

        bool IsCpu();
        /// <summary>
        /// Determines whether the tensor is allocated on a CUDA-capable GPU device.
        /// </summary>
        /// <returns><c>true</c> if the tensor resides on CUDA; otherwise, <c>false</c>.</returns>

        bool IsCuda();
        /// <summary>
        /// Retrieves an enumerable collection of all trainable sub-tensors/parameters associated with this tensor.
        /// </summary>
        /// <returns>An enumeration of tensors representing the model parameters.</returns>
        /// <remarks>
        /// This is primarily utilized in deep learning modules to aggregate weights and biases recursively.
        /// </remarks>

        IEnumerable<ITensor> Parameters();
        /// <summary>
        /// Adds another tensor to this tensor in-place, modifying the underlying data directly.
        /// </summary>
        /// <param name="other">The tensor to add.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if shapes or devices do not match.</exception>

        void AddInPlace(ITensor other);
        /// <summary>
        /// Adds a scalar float value to this tensor in-place, modifying the underlying data directly.
        /// </summary>
        /// <param name="scalar">The scalar value to add.</param>

        void AddInPlace(float scalar);
        /// <summary>
        /// Subtracts another tensor from this tensor in-place, modifying the underlying data directly.
        /// </summary>
        /// <param name="other">The tensor to subtract.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if shapes or devices do not match.</exception>

        void SubtractInPlace(ITensor other);
        /// <summary>
        /// Subtracts a scalar float value from this tensor in-place, modifying the underlying data directly.
        /// </summary>
        /// <param name="scalar">The scalar value to subtract.</param>

        void SubtractInPlace(float scalar);
        /// <summary>
        /// Multiplies this tensor by another tensor element-wise in-place, modifying the underlying data directly.
        /// </summary>
        /// <param name="other">The tensor to multiply by.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if shapes or devices do not match.</exception>

        void MultiplyInPlace(ITensor other);
        /// <summary>
        /// Multiplies this tensor by a scalar float value in-place, modifying the underlying data directly.
        /// </summary>
        /// <param name="scalar">The scalar value to multiply by.</param>

        void MultiplyInPlace(float scalar);
        /// <summary>
        /// Performs element-wise addition between this tensor and another tensor.
        /// </summary>
        /// <param name="other">The tensor to add.</param>
        /// <returns>A new tensor representing the element-wise sum.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if devices do not match or shapes are incompatible.</exception>

        ITensor Add(ITensor other);
        /// <summary>
        /// Performs element-wise subtraction between this tensor and another tensor.
        /// </summary>
        /// <param name="other">The tensor to subtract.</param>
        /// <returns>A new tensor representing the element-wise difference.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if devices do not match or shapes are incompatible.</exception>

        ITensor Subtract(ITensor other);
        /// <summary>
        /// Performs element-wise multiplication between this tensor and another tensor.
        /// </summary>
        /// <param name="other">The tensor to multiply by.</param>
        /// <returns>A new tensor representing the element-wise product.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if devices do not match or shapes are incompatible.</exception>

        ITensor Multiply(ITensor other);
        /// <summary>
        /// Performs element-wise division between this tensor and another tensor.
        /// </summary>
        /// <param name="other">The divisor tensor.</param>
        /// <returns>A new tensor representing the element-wise quotient.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if devices do not match or shapes are incompatible.</exception>

        ITensor Divide(ITensor other);
        /// <summary>
        /// Performs addition of a scalar float to each element in this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to add.</param>
        /// <returns>A new tensor with the scalar added to all elements.</returns>

        ITensor Add(float scalar);
        /// <summary>
        /// Performs subtraction of a scalar float from each element in this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to subtract.</param>
        /// <returns>A new tensor with the scalar subtracted from all elements.</returns>

        ITensor Subtract(float scalar);
        /// <summary>
        /// Performs multiplication of each element in this tensor by a scalar float.
        /// </summary>
        /// <param name="scalar">The scalar multiplier.</param>
        /// <returns>A new tensor with all elements scaled.</returns>

        ITensor Multiply(float scalar);
        /// <summary>
        /// Performs division of each element in this tensor by a scalar float.
        /// </summary>
        /// <param name="scalar">The scalar divisor.</param>
        /// <returns>A new tensor with all elements divided by the scalar.</returns>

        ITensor Divide(float scalar);
        /// <summary>
        /// Performs subtraction of an integer value from each element in this tensor.
        /// </summary>
        /// <param name="other">The integer value to subtract.</param>
        /// <returns>A new tensor with the integer subtracted from all elements.</returns>

        ITensor Subtract(int other);
        /// <summary>
        /// Performs multiplication of each element in this tensor by a double-precision scalar.
        /// </summary>
        /// <param name="scalar">The double-precision scalar multiplier.</param>
        /// <returns>A new tensor with all elements scaled.</returns>

        ITensor Multiply(double scalar);
        /// <summary>
        /// Performs division of each element in this tensor by a double-precision scalar.
        /// </summary>
        /// <param name="scalar">The double-precision scalar divisor.</param>
        /// <returns>A new tensor with all elements divided by the scalar.</returns>

        ITensor Divide(double scalar);
        /// <summary>
        /// Negates the values of the tensor elements (computes element-wise negative).
        /// </summary>
        /// <returns>A new tensor with all values negated.</returns>

        ITensor Negate();
        /// <summary>
        /// Computes the exponential (e^x) of each element in the tensor.
        /// </summary>
        /// <returns>A new tensor containing the exponential of each element.</returns>

        ITensor Exp();
        /// <summary>
        /// Computes the natural logarithm (ln) of each element in the tensor.
        /// </summary>
        /// <returns>A new tensor containing the natural log of each element.</returns>

        ITensor Log();
        /// <summary>
        /// Computes the square root of each element in the tensor.
        /// </summary>
        /// <returns>A new tensor containing the square root of each element.</returns>

        ITensor Sqrt();
        /// <summary>
        /// Computes the absolute value of each element in the tensor.
        /// </summary>
        /// <returns>A new tensor containing the absolute values.</returns>

        ITensor Abs();
        /// <summary>
        /// Computes the sine of each element in the tensor.
        /// </summary>
        /// <returns>A new tensor containing the sine of each element.</returns>

        ITensor Sin();
        /// <summary>
        /// Computes the cosine of each element in the tensor.
        /// </summary>
        /// <returns>A new tensor containing the cosine of each element.</returns>

        ITensor Cos();
        /// <summary>
        /// Computes the power of each element in the tensor raised to a float exponent.
        /// </summary>
        /// <param name="exponent">The float exponent.</param>
        /// <returns>A new tensor containing the elements raised to the specified exponent.</returns>

        ITensor Pow(float exponent);
        /// <summary>
        /// Computes the power of each element in the tensor raised to the elements of another tensor.
        /// </summary>
        /// <param name="exponent">The tensor containing exponents.</param>
        /// <returns>A new tensor containing the element-wise power results.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="exponent"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if devices do not match or shapes are incompatible.</exception>

        ITensor Pow(ITensor exponent);
        /// <summary>
        /// Performs matrix multiplication (dot product) of two tensors.
        /// </summary>
        /// <param name="other">The right-hand matrix/tensor in the multiplication.</param>
        /// <returns>A new tensor representing the matrix multiplication result.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if devices do not match or matrix dimensions are incompatible.</exception>

        ITensor MatMul(ITensor other);
        /// <summary>
        /// Permutes the dimensions of the tensor according to the specified permutation array.
        /// </summary>
        /// <param name="perm">An array of integers representing the desired axis order.</param>
        /// <returns>A new transposed view or tensor.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="perm"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">Thrown when permutation array does not match the rank of the tensor.</exception>

        ITensor Transpose(int[] perm);
        /// <summary>
        /// Reshapes the tensor to a new set of dimensions without changing its underlying data.
        /// </summary>
        /// <param name="newShape">The new dimensions of the tensor.</param>
        /// <returns>A new reshaped view or tensor.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="newShape"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">Thrown if the total number of elements in <paramref name="newShape"/> does not match the current total element count.</exception>

        ITensor Reshape(params int[] newShape);
        /// <summary>
        /// Slices the tensor along its dimensions using specified start, end, and step parameters.
        /// </summary>
        /// <param name="slices">A collection of tuples defining (start, end, step) for each dimension.</param>
        /// <returns>A new tensor representing the sliced subsection.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="slices"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">Thrown if the number of slices exceeds the tensor's rank.</exception>

        ITensor Slice(params (int start, int end, int step)[] slices);
        /// <summary>
        /// Concatenates this tensor with a collection of other tensors along a specified axis.
        /// </summary>
        /// <param name="others">The tensors to concatenate with this tensor.</param>
        /// <param name="axis">The axis along which to concatenate. Defaults to 0.</param>
        /// <returns>A new combined tensor.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="others"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="axis"/> is outside the valid range of dimensions.</exception>
        /// <exception cref="InvalidOperationException">Thrown if tensors have mismatched shapes along non-concatenating axes or reside on different devices.</exception>

        ITensor Concat(IEnumerable<ITensor> others, int axis = 0);
        /// <summary>
        /// Broadcasts this tensor to a target shape, expanding singleton dimensions.
        /// </summary>
        /// <param name="targetShape">The desired target shape.</param>
        /// <returns>A new broadcasted view or copy of the tensor.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="targetShape"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the tensor cannot be broadcasted to the target shape.</exception>

        ITensor BroadcastTo(TensorShape targetShape);
        /// <summary>
        /// Performs an addition with automatic broadcasting of the operand.
        /// </summary>
        /// <param name="other">The tensor to add, which may require broadcasting.</param>
        /// <returns>A new tensor containing the addition result.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the shapes cannot be broadcasted together or devices do not match.</exception>

        ITensor BroadcastAdd(ITensor other);
        /// <summary>
        /// Reshapes the tensor while applying broadcasting rules along a specified axis.
        /// </summary>
        /// <param name="target">The target shape.</param>
        /// <param name="axis">The broadcast-anchoring axis index.</param>
        /// <returns>A new reshaped and broadcasted tensor.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="target"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="axis"/> is out of bounds.</exception>

        ITensor ReshapeWithBroadcast(TensorShape target, int axis);
        /// <summary>
        /// Sums the elements of the tensor along the specified axis or across all elements.
        /// </summary>
        /// <param name="axis">The axis along which to sum; if null, sums the entire tensor.</param>
        /// <param name="keepDims">If true, the reduced dimensions are retained with length 1; otherwise, they are removed.</param>
        /// <returns>A new tensor containing the sum.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="axis"/> is out of bounds for the tensor's dimensions.</exception>

        ITensor Sum(int? axis = null, bool keepDims = false);
        /// <summary>
        /// Sums the elements of the tensor across multiple specified axes.
        /// </summary>
        /// <param name="axes">An array of dimensions along which to sum.</param>
        /// <param name="keepDims">If true, the reduced dimensions are retained with length 1; otherwise, they are removed.</param>
        /// <returns>A new tensor containing the sum.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="axes"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when any axis in <paramref name="axes"/> is out of bounds.</exception>

        ITensor Sum(int[] axes, bool keepDims = false);
        /// <summary>
        /// Computes the mean of the elements of the tensor along the specified axis or across all elements.
        /// </summary>
        /// <param name="axis">The axis along which to compute mean; if null, computes mean of the entire tensor.</param>
        /// <param name="keepDims">If true, the reduced dimensions are retained with length 1; otherwise, they are removed.</param>
        /// <returns>A new tensor containing the mean values.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="axis"/> is out of bounds.</exception>

        ITensor Mean(int? axis = null, bool keepDims = false);
        /// <summary>
        /// Computes the mean of the elements of the tensor across multiple specified axes.
        /// </summary>
        /// <param name="axes">An array of dimensions along which to compute the mean.</param>
        /// <param name="keepDims">If true, the reduced dimensions are retained with length 1; otherwise, they are removed.</param>
        /// <returns>A new tensor containing the mean values.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="axes"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when any axis in <paramref name="axes"/> is out of bounds.</exception>

        ITensor Mean(int[] axes, bool keepDims = false);
        /// <summary>
        /// Computes the maximum value along the specified axis.
        /// </summary>
        /// <param name="axis">The axis to scan for maximum values. Defaults to -1 (the last axis).</param>
        /// <param name="keepDims">If true, the reduced dimensions are retained with length 1; otherwise, they are removed.</param>
        /// <returns>A new tensor containing the maximum values.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="axis"/> is out of bounds.</exception>

        ITensor Max(int axis = -1, bool keepDims = false);
        /// <summary>
        /// Computes the minimum value along the specified axis.
        /// </summary>
        /// <param name="axis">The axis to scan for minimum values. Defaults to -1 (the last axis).</param>
        /// <param name="keepDims">If true, the reduced dimensions are retained with length 1; otherwise, they are removed.</param>
        /// <returns>A new tensor containing the minimum values.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="axis"/> is out of bounds.</exception>

        ITensor Min(int axis = -1, bool keepDims = false);
        /// <summary>
        /// Computes the indices of the minimum values along a specified axis.
        /// </summary>
        /// <param name="axis">The axis along which to find minimum values.</param>
        /// <returns>A new tensor containing the indices of the minimum values.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="axis"/> is out of bounds.</exception>

        ITensor ArgMin(int axis);
        /// <summary>
        /// Computes the indices of the maximum values along a specified axis.
        /// </summary>
        /// <param name="axis">The axis along which to find maximum values.</param>
        /// <returns>A new tensor containing the indices of the maximum values.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="axis"/> is out of bounds.</exception>

        ITensor ArgMax(int axis);
        /// <summary>
        /// Computes the cumulative sum of elements along a specified axis.
        /// </summary>
        /// <param name="axis">The axis along which the cumulative sum is calculated.</param>
        /// <returns>A new tensor containing the cumulative sum.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="axis"/> is out of bounds.</exception>

        ITensor CumSum(int axis);
        /// <summary>
        /// Performs an element-wise "greater than" comparison against another tensor.
        /// </summary>
        /// <param name="other">The tensor to compare against.</param>
        /// <returns>A binary/mask tensor with 1.0 where true, and 0.0 where false.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if devices do not match or shapes are incompatible.</exception>

        ITensor GreaterThan(ITensor other);
        /// <summary>
        /// Performs an element-wise "greater than or equal to" comparison against another tensor.
        /// </summary>
        /// <param name="other">The tensor to compare against.</param>
        /// <returns>A binary/mask tensor with 1.0 where true, and 0.0 where false.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if devices do not match or shapes are incompatible.</exception>

        ITensor GreaterThanOrEqual(ITensor other);
        /// <summary>
        /// Performs an element-wise "less than or equal to" comparison against another tensor.
        /// </summary>
        /// <param name="other">The tensor to compare against.</param>
        /// <returns>A binary/mask tensor with 1.0 where true, and 0.0 where false.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if devices do not match or shapes are incompatible.</exception>

        ITensor LessEqual(ITensor other);
        /// <summary>
        /// Performs an element-wise equality comparison against another tensor.
        /// </summary>
        /// <param name="other">The tensor to compare against.</param>
        /// <returns>A binary/mask tensor with 1.0 where true, and 0.0 where false.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if devices do not match or shapes are incompatible.</exception>

        ITensor Equal(ITensor other);
        /// <summary>
        /// Selects elements from either <paramref name="trueValue"/> or <paramref name="falseValue"/> depending on the condition.
        /// </summary>
        /// <param name="condition">The condition tensor indicating which source to choose from.</param>
        /// <param name="trueValue">The tensor to choose from when the condition is true (non-zero).</param>
        /// <param name="falseValue">The tensor to choose from when the condition is false (zero).</param>
        /// <returns>A new merged tensor.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="condition"/>, <paramref name="trueValue"/>, or <paramref name="falseValue"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if devices do not match or shapes are incompatible.</exception>

        ITensor Where(ITensor condition, ITensor trueValue, ITensor falseValue);
        /// <summary>
        /// Computes the sign of each element in the tensor (-1 for negative, 0 for zero, 1 for positive).
        /// </summary>
        /// <returns>A new tensor representing the signs of elements.</returns>

        ITensor Sign();
        /// <summary>
        /// Applies the hyperbolic tangent (tanh) activation function element-wise.
        /// </summary>
        /// <returns>A new tensor containing the hyperbolic tangent of each element.</returns>

        ITensor Tanh();
        /// <summary>
        /// Applies the Rectified Linear Unit (ReLU) activation function element-wise.
        /// </summary>
        /// <returns>A new tensor containing max(0, x) for each element.</returns>

        ITensor Relu();
        /// <summary>
        /// Applies the Sigmoid activation function element-wise.
        /// </summary>
        /// <returns>A new tensor containing 1 / (1 + e^-x) for each element.</returns>

        ITensor Sigmoid();
        /// <summary>
        /// Applies the Softmax function along the specified axis.
        /// </summary>
        /// <param name="axis">The axis along which to apply the softmax normalization. Defaults to -1.</param>
        /// <returns>A new tensor representing the probability distribution.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="axis"/> is out of bounds.</exception>

        ITensor Softmax(int axis = -1);
        /// <summary>
        /// Starts the backward propagation of gradients from this tensor down the computation graph.
        /// </summary>
        /// <param name="gradient">The external gradient to start backpropagation with; defaults to a 1.0 scalar if null.</param>
        /// <remarks>
        /// This computes gradients recursively via reverse-mode automatic differentiation.
        /// It will navigate back through nodes associated with <see cref="Inputs"/> and apply <see cref="GradFn"/>.
        /// </remarks>

        void Backward(ITensor? gradient = null);
        /// <summary>
        /// Clears/nullifies the accumulated gradient on this tensor.
        /// </summary>

        void ClearGrad();
        /// <summary>
        /// Overwrites the underlying values of this tensor with the provided float array.
        /// </summary>
        /// <param name="floats">The array of new data values.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="floats"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">Thrown if the length of <paramref name="floats"/> does not match the tensor's capacity.</exception>

        void SetData(float[] floats);
        /// <summary>
        /// Applies an element-wise logical NOT operation to the tensor (reverses boolean states).
        /// </summary>
        /// <returns>A new tensor representing logical negation.</returns>

        ITensor LogicalNot();
        /// <summary>
        /// Clips/clamps the elements of the tensor to be within a designated range.
        /// </summary>
        /// <param name="v1">The lower bound clamp value.</param>
        /// <param name="v2">The upper bound clamp value.</param>
        /// <returns>A new tensor with elements clipped to the range [v1, v2].</returns>

        ITensor Clip(float v1, float v2);
    }
}