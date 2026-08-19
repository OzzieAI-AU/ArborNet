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

    using ArborNet.Activations;
    using ArborNet.Core;
    using ArborNet.Core.Activations;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Functional;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;
    using ArborNet.Layers.Fractal;
    using ArborNet.Layers.Normalization;
    using ArborNet.Losses;
    using ArborNet.Models;

    /// <summary>
    /// The heart of ArborNet — a beautifully designed, fluent, and highly expressive API 
    /// for tensor operations, neural network construction, and on-the-fly execution.
    /// </summary>

    #endregion


    /// <summary>
    /// The heart of ArborNet — a beautifully designed, fluent, and highly expressive API 
    /// for tensor operations, neural network construction, and on-the-fly execution.
    /// Provides chained accessibility to every single component in the ArborNet ecosystem.
    /// </summary>
    public sealed class X : IEquatable<X>
    {
        private readonly ITensor _tensor;

        /// <summary>Gets the underlying ITensor instance managed by this fluent wrapper.</summary>
        public ITensor Tensor => _tensor;

        /// <summary>Gets the shape of the underlying tensor.</summary>
        public TensorShape Shape => _tensor.Shape;

        /// <summary>Gets the device on which the underlying tensor is allocated.</summary>
        public Device Device => _tensor.Device;

        /// <summary>
        /// Initializes a new instance of the <see cref="X"/> class by wrapping any <see cref="ITensor"/>.
        /// </summary>
        public X(ITensor tensor)
        {
            _tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
        }

        // =========================================================================================
        #region Conversions, Equality, and Boolean Logic
        // =========================================================================================

        public static implicit operator Tensor(X x) => (Tensor)ArborNet.Core.Tensors.Tensor.Unwrap(x._tensor);
        public static implicit operator X(Tensor t) => new(t);
        public static implicit operator X(Variable v) => new(v);
        public static explicit operator Variable(X x) => new(x._tensor);
        public static implicit operator bool(X x) => x.ToScalar() != 0f;
        public static bool operator true(X x) => x.ToScalar() != 0f;
        public static bool operator false(X x) => x.ToScalar() == 0f;

        public bool Equals(X? other)
        {
            if (other is null) return false;
            return ReferenceEquals(this, other) || _tensor.Equals(other._tensor);
        }
        public override bool Equals(object? obj) => obj is X other && Equals(other);
        public override int GetHashCode() => _tensor.GetHashCode();

        #endregion

        // =========================================================================================
        #region Static Factories
        // =========================================================================================

        public static X From(ITensor tensor) => new(tensor);
        public static X From(float[] data, params int[] shape) => new(ArborNet.Core.Tensors.Tensor.FromArray(data, new TensorShape(shape)));
        public static X FromArray(float[] data, params int[] shape) => new(ArborNet.Core.Tensors.Tensor.FromArray(data, new TensorShape(shape)));
        public static X FromArray(float[] data, TensorShape shape, Device? device = null) => new(ArborNet.Core.Tensors.Tensor.FromArray(data, shape, device));
        public static X Zeros(params int[] shape) => new(ArborNet.Core.Tensors.Tensor.Zeros(new TensorShape(shape)));
        public static X Ones(params int[] shape) => new(ArborNet.Core.Tensors.Tensor.Ones(new TensorShape(shape)));
        public static X Rand(params int[] shape) => new(ArborNet.Core.Tensors.Tensor.Rand(new TensorShape(shape)));
        public static X Randn(params int[] shape) => new(ArborNet.Core.Tensors.Tensor.Randn(new TensorShape(shape)));
        public static X Eye(int size, Device? device = null) => new(ArborNet.Core.Tensors.Tensor.Eye(size, device));
        public static X Of(ITensor tensor) => new(tensor);

        #endregion

        // =========================================================================================
        #region Fluent Unary Math
        // =========================================================================================

        public X Negate() => new(_tensor.Negate());
        public X Exp() => new(_tensor.Exp());
        public X Log() => new(_tensor.Log());
        public X Sqrt() => new(_tensor.Sqrt());
        public X Abs() => new(_tensor.Abs());
        public X Sin() => new(_tensor.Sin());
        public X Cos() => new(_tensor.Cos());
        public X Sign() => new(_tensor.Sign());
        public X LogicalNot() => new(_tensor.LogicalNot());
        public X Clip(float min, float max) => new(_tensor.Clip(min, max));

        #endregion

        // =========================================================================================
        #region Fluent Binary Math
        // =========================================================================================

        public X Add(X other) => new(_tensor.Add(other._tensor));
        public X Add(ITensor other) => new(_tensor.Add(other));
        public X Add(float scalar) => new(_tensor.Add(scalar));

        public X Subtract(X other) => new(_tensor.Subtract(other._tensor));
        public X Subtract(ITensor other) => new(_tensor.Subtract(other));
        public X Subtract(float scalar) => new(_tensor.Subtract(scalar));

        public X Multiply(X other) => new(_tensor.Multiply(other._tensor));
        public X Multiply(ITensor other) => new(_tensor.Multiply(other));
        public X Multiply(float scalar) => new(_tensor.Multiply(scalar));
        public X Multiply(double scalar) => new(_tensor.Multiply((float)scalar));

        public X Divide(X other) => new(_tensor.Divide(other._tensor));
        public X Divide(ITensor other) => new(_tensor.Divide(other));
        public X Divide(float scalar) => new(_tensor.Divide(scalar));
        public X Divide(double scalar) => new(_tensor.Divide((float)scalar));

        public X Pow(float exponent) => new(_tensor.Pow(exponent));
        public X Pow(X exponent) => new(_tensor.Pow(exponent._tensor));

        public X MatMul(X other) => new(_tensor.MatMul(other._tensor));
        public X MatMul(ITensor other) => new(_tensor.MatMul(other));

        #endregion

        // =========================================================================================
        #region Fluent Reductions & Shape Operations
        // =========================================================================================

        public X Sum(int? axis = null, bool keepDims = false) => new(_tensor.Sum(axis, keepDims));
        public X Sum(int[] axes, bool keepDims = false) => new(_tensor.Sum(axes, keepDims));
        public X Mean(int? axis = null, bool keepDims = false) => new(_tensor.Mean(axis, keepDims));
        public X Mean(int[] axes, bool keepDims = false) => new(_tensor.Mean(axes, keepDims));
        public X Max(int axis = -1, bool keepDims = false) => new(_tensor.Max(axis, keepDims));
        public X Min(int axis = -1, bool keepDims = false) => new(_tensor.Min(axis, keepDims));
        public X ArgMin(int axis) => new(_tensor.ArgMin(axis));
        public X ArgMax(int axis) => new(_tensor.ArgMax(axis));
        public X CumSum(int axis) => new(_tensor.CumSum(axis));

        public X Reshape(params int[] newShape) => new(_tensor.Reshape(newShape));
        public X Transpose(params int[] perm) => new(_tensor.Transpose(perm));
        public X Slice(params (int start, int end, int step)[] slices) => new(_tensor.Slice(slices));
        public X BroadcastTo(TensorShape targetShape) => new(_tensor.BroadcastTo(targetShape));
        public X ReshapeWithBroadcast(TensorShape target, int axis) => new(_tensor.ReshapeWithBroadcast(target, axis));
        public X Concat(IEnumerable<X> others, int axis = 0) => new(_tensor.Concat(others.Select(o => o._tensor), axis));

        public X Flatten()
        {
            int batchSize = _tensor.Shape[0];
            return new(_tensor.Reshape(batchSize, -1));
        }

        #endregion

        // =========================================================================================
        #region Fluent Logical Comparisons
        // =========================================================================================

        public X GreaterThan(X other) => new(_tensor.GreaterThan(other._tensor));
        public X GreaterThan(float scalar) => new(_tensor.GreaterThan(ArborNet.Core.Tensors.Tensor.FromScalar(scalar, _tensor.Device)));
        public X GreaterThanOrEqual(X other) => new(_tensor.GreaterThanOrEqual(other._tensor));
        public X GreaterThanOrEqual(float scalar) => new(_tensor.GreaterThanOrEqual(ArborNet.Core.Tensors.Tensor.FromScalar(scalar, _tensor.Device)));
        public X LessEqual(X other) => new(_tensor.LessEqual(other._tensor));
        public X LessEqual(float scalar) => new(_tensor.LessEqual(ArborNet.Core.Tensors.Tensor.FromScalar(scalar, _tensor.Device)));
        public X Equal(X other) => new(_tensor.Equal(other._tensor));
        public X Equal(float scalar) => new(_tensor.Equal(ArborNet.Core.Tensors.Tensor.FromScalar(scalar, _tensor.Device)));
        public X Where(X condition, X trueValue, X falseValue) => new(_tensor.Where(condition._tensor, trueValue._tensor, falseValue._tensor));

        #endregion

        // =========================================================================================
        #region Fluent Device Routing
        // =========================================================================================

        public X To(Device targetDevice) => new(_tensor.To(targetDevice));
        public X Cpu() => new(_tensor.To(Device.CPU));
        public X Cuda(int id = 0) => new(_tensor.To(Device.Cuda(id)));
        public X Rocm(int id = 0) => new(_tensor.To(Device.Rocm(id)));

        #endregion

        // =========================================================================================
        #region Fluent Activations
        // =========================================================================================

        public X ReLU() => new(new ReLU().Forward(_tensor));
        public X GELU() => new(new Gelu().Forward(_tensor));
        public X Tanh() => new(new Tanh().Forward(_tensor));
        public X Sigmoid() => new(new Sigmoid().Forward(_tensor));
        public X Softmax(int axis = -1) => new(new Softmax(axis).Forward(_tensor));
        public X ELU(float alpha = 1.0f) => new(new ELU(alpha).Forward(_tensor));
        public X LeakyReLU(float negativeSlope = 0.01f) => new(new LeakyReLU(negativeSlope).Forward(_tensor));
        public X Mish() => new(new Mish().Forward(_tensor));
        public X Softplus() => new(new Softplus().Forward(_tensor));
        public X Swish() => new(new Swish().Forward(_tensor));
        public X SiLU() => new(new SiLU().Forward(_tensor));
        public X GLU() => new(new GLU().Forward(_tensor));
        public X SwiGLU() => new(new SwiGLU().Forward(_tensor));
        public X HardSigmoid() => new(new HardSigmoid().Forward(_tensor));
        public X HardTanh() => new(new HardTanh().Forward(_tensor));
        public X TanhShrink() => new(new TanhShrink().Forward(_tensor));
        public X Softsign() => new(new Softsign().Forward(_tensor));
        public X SiTU() => new(new SiTU().Forward(_tensor));
        public X SELU() => new(new SELU().Forward(_tensor));

        #endregion

        // =========================================================================================
        #region Fluent Neural Network Builders & Layers
        // =========================================================================================

        /// <summary>Base method to apply any implementation of <see cref="ILayer"/>.</summary>
        public X Apply(ILayer layer) => new(layer.Forward(_tensor));

        /// <summary>Base method to apply any implementation of <see cref="IModel"/>.</summary>
        public X Apply(IModel model) => new(model.Forward(_tensor));

        // -------------------------
        // Core Layers
        // -------------------------
        public X Linear(int outFeatures, bool bias = true) => Apply(new Linear(_tensor.Shape[^1], outFeatures, _tensor.Device));
        public X Conv1D(int outChannels, int kernelSize, int stride = 1, int padding = 0, bool useBias = true)
            => Apply(new Conv1D(_tensor.Shape[1], outChannels, kernelSize, stride, padding, useBias, _tensor.Device));
        public X Conv2D(int outChannels, int kernelSize, int stride = 1, int padding = 0, bool useBias = true)
            => Apply(new Conv2D(_tensor.Shape[1], outChannels, kernelSize, stride, padding, useBias, _tensor.Device));
        public X Conv3D(int outChannels, int kernelDepth, int kernelHeight, int kernelWidth, bool hasBias = true, int stride = 1, int padding = 0)
            => Apply(new Conv3D(_tensor.Shape[1], outChannels, kernelDepth, kernelHeight, kernelWidth, hasBias, stride, padding));

        // -------------------------
        // Recurrent Layers
        // -------------------------
        public X GRU(int inputSize, int hiddenSize) => Apply(new GRU(inputSize, hiddenSize));
        public X LSTM(int inputSize, int hiddenSize, Device? device = null) => Apply(new LSTM(inputSize, hiddenSize, device ?? _tensor.Device));

        // -------------------------
        // Normalizations
        // -------------------------
        public X BatchNorm(int numFeatures, float eps = 1e-5f, float momentum = 0.1f, bool useAffine = true)
        { var l = new BatchNorm(numFeatures, eps, momentum, useAffine); l.To(_tensor.Device); return Apply(l); }
        public X LayerNorm()
        { var l = new ArborNet.Layers.Normalization.LayerNorm(new[] { _tensor.Shape[^1] }); l.To(_tensor.Device); return Apply(l); }
        public X LayerNorm(int[] normalizedShape, float eps = 1e-5f, bool useAffine = true)
        { var l = new ArborNet.Layers.Normalization.LayerNorm(normalizedShape, eps, useAffine); l.To(_tensor.Device); return Apply(l); }
        public X RMSNorm(int numFeatures, float eps = 1e-6f, bool useAffine = true)
        { var l = new RMSNorm(numFeatures, eps, useAffine); l.To(_tensor.Device); return Apply(l); }
        public X GroupNorm(int numChannels, int numGroups, float eps = 1e-5f, bool useAffine = true)
        { var l = new GroupNorm(numChannels, numGroups, eps, useAffine); l.To(_tensor.Device); return Apply(l); }
        public X InstanceNorm(int numChannels, float eps = 1e-5f, bool useAffine = true)
        { var l = new InstanceNorm(numChannels, eps, useAffine); l.To(_tensor.Device); return Apply(l); }
        public X LayerScale(int numFeatures, float initScale = 1e-2f) => Apply(new LayerScale(numFeatures, initScale));

        // -------------------------
        // Pooling
        // -------------------------
        public X MaxPool2D(int kernelSize = 2, int stride = 2, int padding = 0) => Apply(new MaxPool2D(kernelSize, stride, padding));
        public X AdaptiveAvgPool2D(int outputSize = 1) => Apply(new AdaptiveAvgPool2D(outputSize));

        // -------------------------
        // Attention & Embeddings
        // -------------------------
        public X Dropout(float p = 0.5f) => Apply(new Dropout(p));
        public X Attention(int embedDim, int numHeads, bool useBias = true) => Apply(new Attention(embedDim, numHeads, useBias));
        public X MultiHeadAttention(int dModel, int numHeads, bool useBias = true) => Apply(new MultiHeadAttention(dModel, numHeads, useBias));
        public X DeltaAttention(int dModel, int nHeads) => Apply(new DeltacAttention(dModel, nHeads, _tensor.Device));
        public X Embedding(int numEmbeddings, int embeddingDim) => Apply(new Embedding(numEmbeddings, embeddingDim));
        public X PositionalEncoding(int dModel, int maxLen = 512) => Apply(new PositionalEncoding(dModel, maxLen, _tensor.Device));

        // -------------------------
        // Advanced Model Blocks & MoE
        // -------------------------
        public X TransformerBlock(int dModel, int numHeads, int ffDim = 0) => Apply(new TransformerBlock(dModel, numHeads, ffDim));
        public X MistralBlock(int hiddenDim, int numHeads, int kvHeads, int slidingWindow) => Apply(new MistralBlock(hiddenDim, numHeads, kvHeads, slidingWindow));
        public X ConvNeXtBlock(int dim) => Apply(new ConvNeXtBlock(dim));
        public X BasicBlock(int inChannels, int planes, int stride, int expansion) => Apply(new BasicBlock(inChannels, planes, stride, expansion, _tensor.Device));
        public X BottleneckBlock(int inChannels, int planes, int stride, int expansion) => Apply(new BottleneckBlock(inChannels, planes, stride, expansion, _tensor.Device));
        public X StableLatentMoE(int dModel, int numExperts, int activeExperts, int expertCapacity)
            => Apply(new StableLatentcMoE(dModel, numExperts, activeExperts, expertCapacity, _tensor.Device));

        // -------------------------
        // Fractal & Subquadratic
        // -------------------------
        public X FractalLinear(int inFeatures, int outFeatures, FractalType initType, bool useBias = true)
            => Apply(new FractalLinear(inFeatures, outFeatures, initType, useBias));
        public X SubquadraticAttention(int dModel, int headCount, FractalType initType)
            => Apply(new SubquadraticAttention(dModel, headCount, initType));
        public X FractalTransformerBlock(int dModel, int dFF, int headCount, FractalType initType)
            => Apply(new FractalTransformerBlock(dModel, dFF, headCount, initType));

        #endregion




        // =========================================================================================
        #region Fluent Loss Chaining
        // =========================================================================================

        public X MseLoss(X targets, string reduction = "mean") => new(new MSE().Forward(_tensor, targets._tensor, reduction));
        public X CrossEntropyLoss(X targets, string reduction = "mean") => new(new CrossEntropy().Forward(_tensor, targets._tensor, reduction));
        public X BinaryCrossEntropyLoss(X targets, string reduction = "mean") => new(new BinaryCrossEntropy().Forward(_tensor, targets._tensor, reduction));
        public X HingeLoss(X targets, string reduction = "mean") => new(new Hinge().Forward(_tensor, targets._tensor, reduction));
        public X HuberLoss(X targets, float delta = 1.0f, string reduction = "mean") => new(new Huber(delta).Forward(_tensor, targets._tensor, reduction));
        public X KLDivLoss(X targets, string reduction = "mean") => new(new KLDiv().Forward(_tensor, targets._tensor, reduction));
        public X NLLLoss(X targets, string reduction = "mean") => new(new NLL().Forward(_tensor, targets._tensor, reduction));

        #endregion

        // =========================================================================================
        #region Machine Learning Pipeline Wrappers
        // =========================================================================================

        /// <summary>Applies a trained KMeans model classification assignment via Predict.</summary>
        public X KMeansPredict(KMeans model) => new(model.Predict(_tensor));

        #endregion

        // =========================================================================================
        #region Autograd & Execution Accessors
        // =========================================================================================

        public float[] ToArray() => _tensor.ToArray();
        public float ToScalar() => _tensor.ToScalar();
        public void Backward() => _tensor.Backward();
        public void ClearGrad() => _tensor.ClearGrad();

        #endregion

        // =========================================================================================
        #region Rigorous Operator Overloading
        // =========================================================================================

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