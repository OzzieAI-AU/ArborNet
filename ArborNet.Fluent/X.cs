using System;
using System.Collections.Generic;
using System.Linq;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Activations;
using ArborNet.Layers;
using ArborNet.Core.Functional;

namespace ArborNet.Fluent
{
    /// <summary>
    /// The heart of ArborNet — a beautifully designed, fluent, and expressive API 
    /// for tensor operations and neural network construction.
    /// 
    /// <para>
    /// Provides two complementary usage patterns:
    /// <list type="bullet">
    ///   <item><b>Fluent scripting style</b> — perfect for research, notebooks, and rapid prototyping.</item>
    ///   <item><b>Module-based style</b> — production-ready with reusable, stateful layers.</item>
    /// </list>
    /// </para>
    /// 
    /// <example>
    /// <b>Fluent scripting:</b>
    /// <code>
    /// var output = X(input)
    ///     .Linear(512)
    ///     .ReLU()
    ///     .Dropout(0.1f)
    ///     .Linear(10)
    ///     .Softmax();
    /// </code>
    /// </example>
    /// </summary>
    public sealed class X
    {


        /// <summary>
        /// The underlying <see cref="Tensor"/> instance managed by this fluent wrapper.
        /// </summary>
        private readonly Tensor _tensor;

        
        /// <summary>
        /// Gets the underlying tensor wrapped by this fluent API instance.
        /// </summary>
        public Tensor Tensor => _tensor;

        /// <summary>
        /// Gets the underlying tensor wrapped by this fluent API instance.
        /// </summary>
        public TensorShape Shape => _tensor.Shape;



        /// <summary>
        /// Initializes a new instance of the <see cref="X"/> class by wrapping the specified tensor.
        /// </summary>
        /// <param name="tensor">The input tensor to wrap. Must not be <see langword="null"/>.</param>
        /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is <see langword="null"/>.</exception>
        public X(ITensor tensor)
        {
            _tensor = (Tensor)tensor ?? throw new ArgumentNullException(nameof(tensor));
        }


        #region Static Factories

        /// <summary>
        /// Creates an <see cref="X"/> wrapper around an existing tensor.
        /// </summary>
        public static X From(ITensor tensor) => new X(tensor);

        /// <summary>
        /// Creates a tensor from a flat array with the specified shape.
        /// </summary>
        public static X From(float[] data, params int[] shape)
            => new X(Tensor.FromArray(data, new TensorShape(shape)));

        /// <summary>
        /// Creates a tensor filled with zeros.
        /// </summary>
        public static X Zeros(params int[] shape)
            => new X(Tensor.Zeros(new TensorShape(shape)));

        /// <summary>
        /// Creates a tensor filled with ones.
        /// </summary>
        public static X Ones(params int[] shape)
            => new X(Tensor.Ones(new TensorShape(shape)));

        /// <summary>
        /// Creates a tensor with values drawn from a uniform distribution [0, 1).
        /// </summary>
        public static X Rand(params int[] shape)
            => new X(Tensor.Rand(new TensorShape(shape)));

        /// <summary>
        /// Creates a tensor with values drawn from a standard normal distribution (mean=0, std=1).
        /// </summary>
        public static X Randn(params int[] shape)
            => new X(Tensor.Randn(new TensorShape(shape)));

        /// <summary>
        /// Elegant syntax: <c>X(tensor)</c>.
        /// </summary>
        public static X Of(ITensor tensor) => new X(tensor);

        #endregion

        #region Static Tensor Utilities

        /// <summary>
        /// Creates a new tensor filled with zeros of the specified shape on the given device.
        /// </summary>
        /// <param name="shape">The shape of the tensor to create.</param>
        /// <param name="device">The target device. Defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A new zero-filled <see cref="ITensor"/>.</returns>
        public static ITensor Zeros(TensorShape shape, Device device = null)
            => Tensor.Zeros(shape, device ?? Device.CPU);

        /// <summary>
        /// Creates a new tensor filled with ones of the specified shape on the given device.
        /// </summary>
        /// <param name="shape">The shape of the tensor to create.</param>
        /// <param name="device">The target device. Defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A new one-filled <see cref="ITensor"/>.</returns>
        public static ITensor Ones(TensorShape shape, Device device = null)
            => Tensor.Ones(shape, device ?? Device.CPU);

        /// <summary>
        /// Creates a scalar tensor containing the specified value on the given device.
        /// </summary>
        /// <param name="value">The scalar value to store in the tensor.</param>
        /// <param name="device">The target device. Defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A new scalar <see cref="ITensor"/>.</returns>
        public static ITensor FromScalar(float value, Device device = null)
            => Tensor.FromScalar(value, device ?? Device.CPU);

        /// <summary>
        /// Creates a new tensor with random values from a uniform distribution [0, 1) of the specified shape on the given device.
        /// </summary>
        /// <param name="shape">The shape of the tensor to create.</param>
        /// <param name="device">The target device. Defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A new random <see cref="ITensor"/>.</returns>
        public static ITensor Rand(TensorShape shape, Device device = null)
            => Tensor.Rand(shape, device ?? Device.CPU);

        /// <summary>
        /// Creates a new tensor with random values from a standard normal distribution (mean=0, std=1) of the specified shape on the given device.
        /// </summary>
        /// <param name="shape">The shape of the tensor to create.</param>
        /// <param name="device">The target device. Defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A new random <see cref="ITensor"/>.</returns>
        public static ITensor Randn(TensorShape shape, Device device = null)
            => Tensor.Randn(shape, device ?? Device.CPU);

        /// <summary>
        /// Creates a tensor from a flat data array with the specified shape on the given device.
        /// </summary>
        /// <param name="data">The flat array containing the tensor data.</param>
        /// <param name="shape">The shape to apply to the data.</param>
        /// <param name="device">The target device. Defaults to <see cref="Device.CPU"/>.</param>
        /// <returns>A new <see cref="ITensor"/> populated from the data.</returns>
        public static ITensor FromArray(float[] data, TensorShape shape, Device device = null)
            => Tensor.FromArray(data, shape, device ?? Device.CPU);

        /// <summary>
        /// Creates a tensor from a flat data array with the specified dimensions.
        /// </summary>
        /// <param name="data">The flat array containing the tensor data.</param>
        /// <param name="dimensions">The dimensions defining the tensor shape.</param>
        /// <returns>A new <see cref="ITensor"/> populated from the data.</returns>
        public static ITensor FromArray(float[] data, params int[] dimensions)
            => Tensor.FromArray(data, new TensorShape(dimensions));

        /// <summary>
        /// Computes the square root of each element in the tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>A new tensor containing the square roots.</returns>
        public static ITensor Sqrt(ITensor t) => t.Sqrt();
        /// <summary>
        /// Computes the natural logarithm of each element in the tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>A new tensor containing the logarithms.</returns>
        public static ITensor Log(ITensor t) => t.Log();
        /// <summary>
        /// Computes the exponential (base-e) of each element in the tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>A new tensor containing the exponentials.</returns>
        public static ITensor Exp(ITensor t) => t.Exp();
        /// <summary>
        /// Raises each element of the tensor to the specified power.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <param name="exponent">The power to raise elements to.</param>
        /// <returns>A new tensor containing the powered values.</returns>
        public static ITensor Pow(ITensor t, float exponent) => t.Pow(exponent);

        //public static ITensor Add(float b) => Add(b);
        //public static ITensor Add(double b) => Add(b);
        /// <summary>
        /// Adds two tensors element-wise.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>A new tensor containing the element-wise sum.</returns>
        public static ITensor Add(ITensor a, ITensor b) => a.Add(b);
        /// <summary>
        /// Subtracts the second tensor from the first element-wise.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The tensor to subtract.</param>
        /// <returns>A new tensor containing the element-wise difference.</returns>
        public static ITensor Subtract(ITensor a, ITensor b) => a.Subtract(b);
        /// <summary>
        /// Multiplies two tensors element-wise.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>A new tensor containing the element-wise product.</returns>
        public static ITensor Multiply(ITensor a, ITensor b) => a.Multiply(b);
        /// <summary>
        /// Divides the first tensor by the second element-wise.
        /// </summary>
        /// <param name="a">The dividend tensor.</param>
        /// <param name="b">The divisor tensor.</param>
        /// <returns>A new tensor containing the element-wise quotient.</returns>
        public static ITensor Divide(ITensor a, ITensor b) => a.Divide(b);

        /// <summary>
        /// Computes the mean (average) of the tensor elements along the specified axis or globally.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <param name="axis">The axis to compute the mean along. <see langword="null"/> for global mean.</param>
        /// <returns>A new reduced tensor containing the means.</returns>
        public static ITensor Mean(ITensor t, int? axis = null) => t.Mean(axis);
        /// <summary>
        /// Computes the sum of the tensor elements along the specified axis or globally.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <param name="axis">The axis to sum along. <see langword="null"/> for global sum.</param>
        /// <returns>A new reduced tensor containing the sums.</returns>
        public static ITensor Sum(ITensor t, int? axis = null) => t.Sum(axis);
        /// <summary>
        /// Computes the maximum value along the specified axis.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <param name="axis">The axis to compute max along. Defaults to the last dimension (-1).</param>
        /// <returns>A new reduced tensor containing the maximums.</returns>
        public static ITensor Max(ITensor t, int axis = -1) => t.Max(axis);

        /// <summary>
        /// Concatenates a sequence of tensors along the specified axis.
        /// </summary>
        /// <param name="tensors">An enumerable of tensors to concatenate. Must not be empty.</param>
        /// <param name="axis">The dimension along which to concatenate. Defaults to 0 (batch dimension).</param>
        /// <returns>A new <see cref="ITensor"/> containing the concatenated result.</returns>
        public static ITensor Concat(IEnumerable<ITensor> tensors, int axis = 0)
            => tensors.First().Concat(tensors.Skip(1), axis);

        #endregion

        #region Static Activation Helpers

        /// <summary>
        /// Applies the ReLU (Rectified Linear Unit) activation function to the input tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>A new <see cref="X"/> wrapping the activated tensor.</returns>
        public static X Relu(ITensor t) => new X(new ReLU().Forward(t));
        /// <summary>
        /// Applies the GELU (Gaussian Error Linear Unit) activation function to the input tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>A new <see cref="X"/> wrapping the activated tensor.</returns>
        public static X Gelu(ITensor t) => new X(new Gelu().Forward(t));
        /// <summary>
        /// Applies the hyperbolic tangent (Tanh) activation function to the input tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>A new <see cref="X"/> wrapping the activated tensor.</returns>
        public static X Tanh(ITensor t) => new X(new Tanh().Forward(t));
        /// <summary>
        /// Applies the sigmoid activation function to the input tensor.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <returns>A new <see cref="X"/> wrapping the activated tensor.</returns>
        public static X Sigmoid(ITensor t) => new X(new Sigmoid().Forward(t));
        /// <summary>
        /// Applies the softmax activation function to the input tensor along the specified axis.
        /// </summary>
        /// <param name="t">The input tensor.</param>
        /// <param name="axis">The axis along which to apply softmax. Defaults to -1 (last dimension).</param>
        /// <returns>A new <see cref="X"/> wrapping the activated tensor.</returns>
        public static X Softmax(ITensor t, int axis = -1) => new X(new Softmax(axis).Forward(t));

        #endregion

        #region Fluent Instance API

        /// <summary>
        /// Applies an <see cref="Embedding"/> layer. Useful for token sequences.
        /// </summary>
        public X Embedding(int vocabSize, int dim)
            => new X(new ArborNet.Layers.Embedding(vocabSize, dim).Forward(_tensor));

        /// <summary>
        /// Creates and applies a new <see cref="Linear"/> layer (convenient for scripting).
        /// </summary>
        public X Linear(int outFeatures, bool bias = true)
        {
            int inFeatures = _tensor.Shape[_tensor.Shape.Rank - 1];
            return new X(new Linear(inFeatures, outFeatures).Forward(_tensor));
        }

        /// <summary>
        /// Applies a pre-defined layer (the recommended pattern for Modules).
        /// </summary>
        public X Apply(ILayer layer) => new X(layer.Forward(_tensor));

        /// <summary>Applies ReLU activation.</summary>
        public X ReLU() => new X(new ReLU().Forward(_tensor));

        /// <summary>Applies GELU activation.</summary>
        public X GELU() => new X(new Gelu().Forward(_tensor));

        /// <summary>Applies Tanh activation.</summary>
        public X Tanh() => new X(new Tanh().Forward(_tensor));

        /// <summary>Applies Sigmoid activation.</summary>
        public X Sigmoid() => new X(new Sigmoid().Forward(_tensor));

        /// <summary>Applies Softmax activation.</summary>
        public X Softmax(int axis = -1) => new X(new Softmax(axis).Forward(_tensor));

        /// <summary>Applies Dropout regularization.</summary>
        public X Dropout(float p = 0.5f) => new X(new Dropout(p).Forward(_tensor));

        /// <summary>Applies Layer Normalization.</summary>
        public X LayerNorm() => new X(new LayerNorm(new[] { _tensor.Shape[_tensor.Shape.Rank - 1] }).Forward(_tensor));

        /// <summary>Adds a scalar value (element-wise).</summary>
        public X Add(float scalar) => new X(_tensor.Add(scalar));

        /// <summary>Adds a scalar value (element-wise).</summary>
        public X Add(double scalar) => new X(_tensor.Add((float)scalar));

        /// <summary>Multiplies by a scalar value (element-wise).</summary>
        public X Multiply(float scalar) => new X(_tensor.Multiply(scalar));

        /// <summary>Multiplies by a scalar value (element-wise).</summary>
        public X Multiply(double scalar) => new X(_tensor.Multiply((float)scalar));

        /// <summary>Adds another tensor (element-wise).</summary>
        public X Add(X other) => new X(_tensor.Add(other._tensor));

        /// <summary>Adds another tensor (element-wise).</summary>
        public X Add(ITensor other) => new X(_tensor.Add(other));

        /// <summary>Multiplies with another tensor (element-wise).</summary>
        public X Multiply(X other) => new X(_tensor.Multiply(other._tensor));

        /// <summary>Computes mean across specified axes.</summary>
        public X Mean(int[] axes) => new X(_tensor.Mean(axes));

        /// <summary>Flattens all dimensions after the batch dimension.</summary>
        public X Flatten()
        {
            int[] newShape = new int[_tensor.Shape.Rank];
            newShape[0] = _tensor.Shape[0];
            newShape[^1] = -1;
            return new X(_tensor.Reshape(newShape));
        }

        /// <summary>Converts the tensor to a 1D float array.</summary>
        public float[] ToArray() => _tensor.ToArray();

        /// <summary>Triggers backpropagation from this tensor.</summary>
        public void Backward() => _tensor.Backward();

        #endregion

        #region Operator Overloads

        /// <summary>
        /// Overloaded addition operator for element-wise addition of two <see cref="X"/> instances.
        /// </summary>
        /// <param name="a">The first <see cref="X"/> operand.</param>
        /// <param name="b">The second <see cref="X"/> operand.</param>
        /// <returns>A new <see cref="X"/> containing the element-wise sum.</returns>
        public static X operator +(X a, X b) => a.Add(b);
        /// <summary>
        /// Overloaded multiplication operator for element-wise multiplication of two <see cref="X"/> instances.
        /// </summary>
        /// <param name="a">The first <see cref="X"/> operand.</param>
        /// <param name="b">The second <see cref="X"/> operand.</param>
        /// <returns>A new <see cref="X"/> containing the element-wise product.</returns>
        public static X operator *(X a, X b) => a.Multiply(b);

        #endregion
    }
}