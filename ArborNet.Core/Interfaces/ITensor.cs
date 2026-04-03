using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using ArborNet.Core.Devices;
using ArborNet.Core.Tensors;

namespace ArborNet.Core.Interfaces
{
    /// <summary>
    /// Core tensor interface for ArborNet - the foundation of all operations, autograd, and device abstraction.
    /// All backends (CPU/CUDA) and layers must implement this exactly.
    /// Supports immutable functional style with full ND broadcasting and autograd.
    /// </summary>
    public interface ITensor
    {
        /// <summary>
        /// Gets the shape of the tensor, describing its dimensions and size in each axis.
        /// </summary>
        /// <value>The tensor's shape.</value>
        TensorShape Shape { get; }

        /// <summary>
        /// Gets the device on which the tensor's data is allocated (CPU or CUDA).
        /// </summary>
        /// <value>The device the tensor resides on.</value>
        Device Device { get; }

        /// <summary>
        /// Gets or sets a value indicating whether gradients should be tracked for this tensor
        /// during the backward pass.
        /// </summary>
        /// <value><c>true</c> if gradients should be computed for this tensor; otherwise, <c>false</c>.</value>
        bool RequiresGrad { get; set; }

        /// <summary>
        /// Gets or sets the gradient tensor accumulated for this tensor during autograd.
        /// </summary>
        /// <value>The gradient with respect to this tensor, or <c>null</c> if not computed.</value>
        ITensor? Grad { get; set; }

        /// <summary>
        /// Gets or sets the gradient function (backward function) used by the autograd engine
        /// to compute gradients with respect to the inputs of the operation that produced this tensor.
        /// </summary>
        /// <value>The gradient function, or <c>null</c> for leaf tensors.</value>
        Func<ITensor, ITensor>? GradFn { get; set; }

        /// <summary>
        /// Gets the raw underlying data buffer as a float array in row-major (C-style) order.
        /// </summary>
        /// <value>The raw data buffer. Modifications should be avoided as the tensor may share storage.</value>
        float[] Data { get; }

        /// <summary>
        /// Returns a copy of the tensor's data as a new one-dimensional float array.
        /// </summary>
        /// <returns>A new array containing all elements of the tensor in row-major order.</returns>
        float[] ToArray();

        /// <summary>
        /// Converts the tensor to a scalar float value. The tensor must contain exactly one element.
        /// </summary>
        /// <returns>The scalar value contained in the tensor.</returns>
        float ToScalar();

        /// <summary>
        /// Creates a deep copy of the tensor, including data and metadata.
        /// The cloned tensor is independent of the original.
        /// </summary>
        /// <returns>A new tensor that is an exact copy of this tensor.</returns>
        ITensor Clone();

        /// <summary>
        /// Moves or copies the tensor to the specified device.
        /// </summary>
        /// <param name="device">The target device (CPU or CUDA).</param>
        /// <returns>A tensor residing on the target device with the same data and shape.</returns>
        ITensor To(Device device);

        /// <summary>
        /// Determines whether this tensor is allocated on the CPU device.
        /// </summary>
        /// <returns><c>true</c> if the tensor is on CPU; otherwise, <c>false</c>.</returns>
        bool IsCpu();

        /// <summary>
        /// Determines whether this tensor is allocated on a CUDA device.
        /// </summary>
        /// <returns><c>true</c> if the tensor is on CUDA; otherwise, <c>false</c>.</returns>
        bool IsCuda();

        /// <summary>
        /// Returns an enumerable containing tensors that should be treated as parameters
        /// (typically used by modules to collect trainable parameters).
        /// </summary>
        /// <returns>An enumerable of parameter tensors.</returns>
        IEnumerable<ITensor> Parameters();

        ITensor ArgMin(int axis);


        ITensor ArgMax(int axis);


        /// <summary>
        /// Performs element-wise addition between this tensor and another tensor,
        /// with automatic broadcasting applied as needed.
        /// </summary>
        /// <param name="other">The tensor to add to this tensor.</param>
        /// <returns>A new tensor containing the result of the addition.</returns>
        ITensor Add(ITensor other);

        /// <summary>
        /// Performs element-wise subtraction between this tensor and another tensor,
        /// with automatic broadcasting applied as needed.
        /// </summary>
        /// <param name="other">The tensor to subtract from this tensor.</param>
        /// <returns>A new tensor containing the result of the subtraction.</returns>
        ITensor Subtract(ITensor other);

        /// <summary>
        /// Performs element-wise multiplication between this tensor and another tensor,
        /// with automatic broadcasting applied as needed.
        /// </summary>
        /// <param name="other">The tensor to multiply with this tensor.</param>
        /// <returns>A new tensor containing the result of the multiplication.</returns>
        ITensor Multiply(ITensor other);

        /// <summary>
        /// Performs element-wise division between this tensor and another tensor,
        /// with automatic broadcasting applied as needed.
        /// </summary>
        /// <param name="other">The tensor to divide this tensor by.</param>
        /// <returns>A new tensor containing the result of the division.</returns>
        ITensor Divide(ITensor other);

        /// <summary>
        /// Adds a scalar value to each element of the tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to add.</param>
        /// <returns>A new tensor containing the result of the addition.</returns>
        ITensor Add(float scalar);

        /// <summary>
        /// Subtracts a scalar value from each element of the tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to subtract.</param>
        /// <returns>A new tensor containing the result of the subtraction.</returns>
        ITensor Subtract(float scalar);

        /// <summary>
        /// Multiplies each element of the tensor by a scalar value.
        /// </summary>
        /// <param name="scalar">The scalar value to multiply by.</param>
        /// <returns>A new tensor containing the result of the multiplication.</returns>
        ITensor Multiply(float scalar);

        /// <summary>
        /// Divides each element of the tensor by a scalar value.
        /// </summary>
        /// <param name="scalar">The scalar value to divide by.</param>
        /// <returns>A new tensor containing the result of the division.</returns>
        ITensor Divide(float scalar);

        /// <summary>
        /// Subtracts an integer value from each element of the tensor.
        /// </summary>
        /// <param name="other">The integer value to subtract.</param>
        /// <returns>A new tensor containing the result of the subtraction.</returns>
        ITensor Subtract(int other);

        /// <summary>
        /// Multiplies each element of the tensor by a double-precision scalar value.
        /// </summary>
        /// <param name="scalar">The scalar value to multiply by.</param>
        /// <returns>A new tensor containing the result of the multiplication.</returns>
        ITensor Multiply(double scalar);

        /// <summary>
        /// Divides each element of the tensor by a double-precision scalar value.
        /// </summary>
        /// <param name="scalar">The scalar value to divide by.</param>
        /// <returns>A new tensor containing the result of the division.</returns>
        ITensor Divide(double scalar);

        /// <summary>
        /// Returns a new tensor with the negation of each element (multiplication by -1).
        /// </summary>
        /// <returns>A new tensor containing the negated values.</returns>
        ITensor Negate();

        /// <summary>
        /// Applies the exponential function to each element of the tensor.
        /// </summary>
        /// <returns>A new tensor with the exponential applied element-wise.</returns>
        ITensor Exp();

        /// <summary>
        /// Applies the natural logarithm to each element of the tensor.
        /// </summary>
        /// <returns>A new tensor with the logarithm applied element-wise.</returns>
        ITensor Log();

        /// <summary>
        /// Computes the square root of each element of the tensor.
        /// </summary>
        /// <returns>A new tensor with the square root applied element-wise.</returns>
        ITensor Sqrt();

        /// <summary>
        /// Computes the absolute value of each element of the tensor.
        /// </summary>
        /// <returns>A new tensor with the absolute value applied element-wise.</returns>
        ITensor Abs();

        /// <summary>
        /// Computes the sine of each element of the tensor (in radians).
        /// </summary>
        /// <returns>A new tensor with the sine applied element-wise.</returns>
        ITensor Sin();

        /// <summary>
        /// Computes the cosine of each element of the tensor (in radians).
        /// </summary>
        /// <returns>A new tensor with the cosine applied element-wise.</returns>
        ITensor Cos();

        /// <summary>
        /// Raises each element of the tensor to the specified power.
        /// </summary>
        /// <param name="exponent">The exponent to raise each element to.</param>
        /// <returns>A new tensor with the power operation applied element-wise.</returns>
        ITensor Pow(float exponent);

        /// <summary>
        /// Performs element-wise exponentiation using another tensor as the exponents,
        /// with broadcasting support.
        /// </summary>
        /// <param name="exponent">The tensor containing the exponents.</param>
        /// <returns>A new tensor containing the result of the power operation.</returns>
        ITensor Pow(ITensor exponent);

        /// <summary>
        /// Performs a matrix multiplication (or batched matrix multiplication) with another tensor.
        /// </summary>
        /// <param name="other">The tensor to multiply with.</param>
        /// <returns>A new tensor containing the result of the matrix multiplication.</returns>
        ITensor MatMul(ITensor other);

        /// <summary>
        /// Permutes the dimensions of the tensor according to the provided permutation array.
        /// </summary>
        /// <param name="perm">An array describing the new order of dimensions.</param>
        /// <returns>A new tensor with dimensions transposed according to the permutation.</returns>
        ITensor Transpose(int[] perm);

        /// <summary>
        /// Returns a new tensor with the same data but a different shape.
        /// The total number of elements must remain the same.
        /// </summary>
        /// <param name="newShape">The desired new shape.</param>
        /// <returns>A new tensor with the specified shape.</returns>
        ITensor Reshape(params int[] newShape);

        /// <summary>
        /// Computes the sum of elements along the given axis (or all elements if axis is null).
        /// </summary>
        /// <param name="axis">The axis to reduce over. If <c>null</c>, reduces all dimensions.</param>
        /// <returns>A new tensor containing the summed values.</returns>
        ITensor Sum(int? axis = null);

        /// <summary>
        /// Computes the mean of elements along the given axis (or all elements if axis is null).
        /// </summary>
        /// <param name="axis">The axis to reduce over. If <c>null</c>, reduces all dimensions.</param>
        /// <returns>A new tensor containing the mean values.</returns>
        ITensor Mean(int? axis = null);

        /// <summary>
        /// Computes the mean of elements along multiple axes.
        /// </summary>
        /// <param name="axes">The axes to reduce over.</param>
        /// <returns>A new tensor containing the mean values.</returns>
        ITensor Mean(int[] axes);

        /// <summary>
        /// Computes the maximum value along the specified axis.
        /// </summary>
        /// <param name="axis">The axis to reduce over. Default is the last axis (-1).</param>
        /// <returns>A new tensor containing the maximum values.</returns>
        ITensor Max(int axis = -1);

        /// <summary>
        /// Computes the minimum value along the specified axis.
        /// </summary>
        /// <param name="axis">The axis to reduce over. Default is the last axis (-1).</param>
        /// <returns>A new tensor containing the minimum values.</returns>
        ITensor Min(int axis = -1);

        /// <summary>
        /// Applies the hyperbolic tangent (tanh) activation function element-wise.
        /// </summary>
        /// <returns>A new tensor with tanh applied to each element.</returns>
        ITensor Tanh();

        /// <summary>
        /// Applies the Rectified Linear Unit (ReLU) activation function element-wise.
        /// </summary>
        /// <returns>A new tensor with ReLU applied to each element.</returns>
        ITensor Relu();

        /// <summary>
        /// Applies the sigmoid activation function element-wise.
        /// </summary>
        /// <returns>A new tensor with sigmoid applied to each element.</returns>
        ITensor Sigmoid();

        /// <summary>
        /// Applies the softmax function along the specified axis.
        /// The softmax is computed as exp(x) / sum(exp(x)) along the axis.
        /// </summary>
        /// <param name="axis">The axis to apply softmax along. Default is the last axis (-1).</param>
        /// <returns>A new tensor with softmax probabilities.</returns>
        ITensor Softmax(int axis = -1);

        /// <summary>
        /// Extracts slices from the tensor using start, stop, and step specifications for each dimension.
        /// </summary>
        /// <param name="slices">An array of slice tuples (start, end, step) for each dimension.</param>
        /// <returns>A new tensor containing the extracted slice.</returns>
        ITensor Slice(params (int start, int end, int step)[] slices);

        /// <summary>
        /// Concatenates this tensor with one or more other tensors along the specified axis.
        /// All tensors must have compatible shapes except in the concatenation dimension.
        /// </summary>
        /// <param name="others">The tensors to concatenate with this tensor.</param>
        /// <param name="axis">The axis along which to concatenate. Default is 0.</param>
        /// <returns>A new tensor containing the concatenated result.</returns>
        ITensor Concat(IEnumerable<ITensor> others, int axis = 0);

        /// <summary>
        /// Broadcasts this tensor to the target shape without copying data when possible.
        /// </summary>
        /// <param name="targetShape">The target shape to broadcast to.</param>
        /// <returns>A new tensor (view when possible) with the broadcasted shape.</returns>
        ITensor BroadcastTo(TensorShape targetShape);

        /// <summary>
        /// Performs an optimized addition with broadcasting, avoiding unnecessary intermediate tensors.
        /// </summary>
        /// <param name="other">The tensor to add to this tensor.</param>
        /// <returns>A new tensor containing the result of the broadcasted addition.</returns>
        ITensor BroadcastAdd(ITensor other);

        /// <summary>
        /// Reshapes the tensor and broadcasts it along a specific axis to match a target shape.
        /// Used internally to prepare tensors for element-wise operations.
        /// </summary>
        /// <param name="target">The target shape to match.</param>
        /// <param name="axis">The axis used for broadcasting alignment.</param>
        /// <returns>A new tensor that has been reshaped and broadcasted.</returns>
        ITensor ReshapeWithBroadcast(TensorShape target, int axis);

        /// <summary>
        /// Performs an element-wise greater-than comparison with another tensor.
        /// Returns a float tensor containing 1.0 where true and 0.0 where false.
        /// </summary>
        /// <param name="other">The tensor to compare against.</param>
        /// <returns>A new tensor containing the comparison results as floats.</returns>
        ITensor GreaterThan(ITensor other);

        /// <summary>
        /// Performs an element-wise greater-than-or-equal comparison with another tensor.
        /// Returns a float tensor containing 1.0 where true and 0.0 where false.
        /// </summary>
        /// <param name="other">The tensor to compare against.</param>
        /// <returns>A new tensor containing the comparison results as floats.</returns>
        ITensor GreaterThanOrEqual(ITensor other);

        /// <summary>
        /// Performs an element-wise less-than-or-equal comparison with another tensor.
        /// Returns a float tensor containing 1.0 where true and 0.0 where false.
        /// </summary>
        /// <param name="other">The tensor to compare against.</param>
        /// <returns>A new tensor containing the comparison results as floats.</returns>
        ITensor LessEqual(ITensor other);

        /// <summary>
        /// Returns elements from <paramref name="trueValue"/> where <paramref name="condition"/> is true,
        /// and elements from <paramref name="falseValue"/> where it is false.
        /// </summary>
        /// <param name="condition">The condition tensor (non-zero is considered true).</param>
        /// <param name="trueValue">Values selected when condition is true.</param>
        /// <param name="falseValue">Values selected when condition is false.</param>
        /// <returns>A new tensor containing values chosen according to the condition.</returns>
        ITensor Where(ITensor condition, ITensor trueValue, ITensor falseValue);

        /// <summary>
        /// Returns the sign of each element: 1 for positive, -1 for negative, and 0 for zero.
        /// </summary>
        /// <returns>A new tensor containing the sign of each element.</returns>
        ITensor Sign();

        /// <summary>
        /// Executes the backward pass from this tensor, computing gradients for all
        /// tensors that require gradients in the computation graph.
        /// </summary>
        /// <param name="gradient">Optional incoming gradient. If null, a scalar 1.0 is used.</param>
        void Backward(ITensor? gradient = null);

        /// <summary>
        /// Clears the gradient and recursively clears gradients of all tensors
        /// reachable through the computation graph.
        /// </summary>
        void ClearGrad();
        
        /// <summary>
        /// Replaces the internal data storage of this tensor with the provided values.
        /// </summary>
        /// <param name="floats">The new float array to set as the tensor's data.
        /// Must have exactly the same length as <see cref="Shape.TotalElements"/>.</param>
        /// <remarks>
        /// <para>
        /// This method is primarily used by optimizers to write updated parameter values 
        /// back into the tensor after gradient steps. It bypasses normal tensor operations 
        /// and directly mutates the underlying storage.
        /// </para>
        /// <para>
        /// <strong>Warning:</strong> This is a low-level mutation method. Use with caution 
        /// as it does not trigger autograd or shape validation beyond length checking in 
        /// most backends.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="floats"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the length of <paramref name="floats"/> 
        /// does not match the total number of elements in the tensor's shape.</exception>
        void SetData(float[] floats);

        /// <summary>
        /// Performs a logical NOT operation on the tensor, treating non-zero values as True 
        /// and zero as False.
        /// </summary>
        /// <returns>A new tensor of the same shape containing 1.0 where the original value was 0, 
        /// and 0.0 where the original value was non-zero (element-wise logical negation).</returns>
        /// <remarks>
        /// This operation is commonly used in masking, conditional logic, and implementing 
        /// activation functions such as ReLU, LeakyReLU, and HardSigmoid derivatives.
        /// The result is always a float tensor with values strictly 0.0 or 1.0.
        /// </remarks>
        ITensor LogicalNot();

        /// <summary>
        /// Clips all elements of the tensor to be within the specified range [min, max].
        /// </summary>
        /// <param name="v1">The minimum value. Any element smaller than this will be set to <paramref name="v1"/>.</param>
        /// <param name="v2">The maximum value. Any element larger than this will be set to <paramref name="v2"/>.</param>
        /// <returns>A new tensor of the same shape with all values clamped to the range [<paramref name="v1"/>, <paramref name="v2"/>].</returns>
        /// <remarks>
        /// <para>
        /// This is equivalent to <c>max(v1, min(v2, x))</c> applied element-wise.
        /// Commonly used in HardTanh, HardSigmoid, gradient clipping, and numerical stability.
        /// </para>
        /// <para>
        /// If <paramref name="v1"/> &gt; <paramref name="v2"/>, the behavior is undefined 
        /// (most implementations will swap them internally for robustness).
        /// </para>
        /// </remarks>
        ITensor Clip(float v1, float v2);

        /// <summary>
        /// Computes the cumulative sum of elements along the specified axis.
        /// </summary>
        /// <param name="axis">The axis along which to compute the cumulative sum.
        /// Negative values are supported and count from the end (e.g., -1 is the last axis).</param>
        /// <returns>A new tensor of the same shape containing the cumulative sums along the given axis.</returns>
        /// <remarks>
        /// <para>
        /// For each position along the specified axis, the value is the sum of all preceding elements 
        /// (inclusive) in that direction. This is also known as a "prefix sum" or "running total".
        /// </para>
        /// <para>
        /// Example (axis = 1 on a 2D tensor):
        /// <br/>Input:  [[1, 2, 3],
        /// <br/>          [4, 5, 6]]
        /// <br/>Output: [[1, 3, 6],
        /// <br/>          [4, 9, 15]]
        /// </para>
        /// <para>
        /// When <see cref="ITensor.RequiresGrad"/> is <c>true</c>, a custom gradient function is 
        /// automatically registered to support full backpropagation through the cumulative sum.
        /// The gradient of CumSum is equivalent to a reverse cumulative sum (suffix sum).
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown when the specified axis is out of range for the tensor's rank.
        /// </exception>
        ITensor CumSum(int axis);

        /// <summary>
        /// Element-wise equality comparison with another tensor.
        /// </summary>
        ITensor Equal(ITensor other);
    }
}