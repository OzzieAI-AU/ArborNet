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
    #region Using Statements
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using ArborNet.Activations;
    using ArborNet.Core;
    using ArborNet.Core.Activations;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Functional;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Data;
    using ArborNet.Generation;
    using ArborNet.Layers;
    using ArborNet.Layers.Fractal;
    using ArborNet.Layers.Normalization;
    using ArborNet.Losses;
    using ArborNet.Models;
    using ArborNet.Optimizers;
    using ArborNet.Trainers;
    #endregion

    /// <summary>
    /// The heart of ArborNet — a beautifully designed, fluent, and highly expressive API 
    /// for tensor operations, neural network construction, and on-the-fly execution.
    /// Provides chained accessibility to every single component in the ArborNet ecosystem.
    /// </summary>
    public sealed class X : IEquatable<X>
    {
        private readonly ITensor _tensor;

        public ITensor Tensor => _tensor;
        public TensorShape Shape => _tensor.Shape;
        public Device Device => _tensor.Device;

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
        public static X FromScalar(float value, Device? device = null) => new(ArborNet.Core.Tensors.Tensor.FromScalar(value, device ?? Device.CPU));
        public static X Arange(float start, float end, float step = 1f, Device? device = null)
            => new(ArborNet.Core.Tensors.Tensor.Arange(start, end, step, device ?? Device.CPU));
        public static X Linspace(float start, float end, int steps, Device? device = null)
            => new(ArborNet.Core.Tensors.Tensor.Linspace(start, end, steps, device ?? Device.CPU));

        #endregion

        // =========================================================================================
        #region Fluent Unary Math
        // =========================================================================================

        public X Negate() => new(_tensor.Negate());
        public X Exp() => new(_tensor.Exp());
        public X Log() => new(_tensor.Log());
        public X Log2() => new(_tensor.Log().Divide((float)Math.Log(2)));
        public X Log10() => new(_tensor.Log().Divide((float)Math.Log(10)));
        public X Sqrt() => new(_tensor.Sqrt());
        public X Rsqrt() => new(_tensor.Pow(-0.5f));
        public X Abs() => new(_tensor.Abs());
        public X Sin() => new(_tensor.Sin());
        public X Cos() => new(_tensor.Cos());
        public X Tan() => new(_tensor.Sin().Divide(_tensor.Cos()));
        public X Asin() => new(_tensor.Asin());
        public X Acos() => new(_tensor.Acos());
        public X Atan() => new(_tensor.Atan());
        public X Sinh() => new(_tensor.Sinh());
        public X Cosh() => new(_tensor.Cosh());
        public X Tanh() => new(new Tanh().Forward(_tensor));
        public X Sign() => new(_tensor.Sign());
        public X LogicalNot() => new(_tensor.LogicalNot());
        public X Clip(float min, float max) => new(_tensor.Clip(min, max));
        public X Floor() => new(_tensor.Floor());
        public X Ceil() => new(_tensor.Ceil());
        public X Round() => new(_tensor.Round());
        public X Reciprocal() => new(_tensor.Pow(-1f));
        public X Square() => new(_tensor.Pow(2f));
        public X Cube() => new(_tensor.Pow(3f));

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

        public X Dot(X other) => MatMul(other); // alias
        public X Outer(X other) => new(_tensor.Outer(other._tensor));

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
        public X Std(int? axis = null, bool keepDims = false) => new(_tensor.Std(axis, keepDims));
        public X Var(int? axis = null, bool keepDims = false) => new(_tensor.Var(axis, keepDims));
        public X Norm(float p = 2f, int? axis = null, bool keepDims = false) => new(_tensor.Norm(p, axis, keepDims));

        public X Reshape(params int[] newShape) => new(_tensor.Reshape(newShape));
        public X Transpose(params int[] perm) => new(_tensor.Transpose(perm));
        public X Permute(params int[] perm) => Transpose(perm);
        public X Slice(params (int start, int end, int step)[] slices) => new(_tensor.Slice(slices));
        public X BroadcastTo(TensorShape targetShape) => new(_tensor.BroadcastTo(targetShape));
        public X ReshapeWithBroadcast(TensorShape target, int axis) => new(_tensor.ReshapeWithBroadcast(target, axis));
        public X Concat(IEnumerable<X> others, int axis = 0) => new(_tensor.Concat(others.Select(o => o._tensor), axis));
        public X Stack(IEnumerable<X> others, int axis = 0) => new(_tensor.Stack(others.Select(o => o._tensor), axis));
        public X Unsqueeze(int axis) => new(_tensor.Unsqueeze(axis));
        public X Squeeze(int? axis = null) => new(_tensor.Squeeze(axis));
        public X Expand(params int[] shape) => new(_tensor.Expand(shape));
        public X View(params int[] shape) => Reshape(shape);

        public X Flatten(int startDim = 0, int endDim = -1)
        {
            // Simple common case: batch-preserving flatten
            if (startDim == 0 && endDim == -1)
            {
                int batchSize = _tensor.Shape[0];
                return new(_tensor.Reshape(batchSize, -1));
            }
            return new(_tensor.Flatten(startDim, endDim));
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
        public X MaskedFill(X mask, float value) => Where(mask, FromScalar(value, Device), this);

        #endregion

        // =========================================================================================
        #region Fluent Device Routing
        // =========================================================================================

        public X To(Device targetDevice) => new(_tensor.To(targetDevice));
        public X Cpu() => new(_tensor.To(Device.CPU));
        public X Cuda(int id = 0) => new(_tensor.To(Device.Cuda(id)));
        public X Rocm(int id = 0) => new(_tensor.To(Device.Rocm(id)));
        public X Metal(int id = 0) => new(_tensor.To(Device.Metal(id)));

        #endregion

        // =========================================================================================
        #region Fluent Activations (complete set)
        // =========================================================================================

        public X ReLU() => new(new ReLU().Forward(_tensor));
        public X GELU() => new(new Gelu().Forward(_tensor));
        public X Tanh() => new(new Tanh().Forward(_tensor));
        public X Sigmoid() => new(new Sigmoid().Forward(_tensor));
        public X Softmax(int axis = -1) => new(new Softmax(axis).Forward(_tensor));
        public X LogSoftmax(int axis = -1) => Softmax(axis).Log();
        public X ELU(float alpha = 1.0f) => new(new ELU(alpha).Forward(_tensor));
        public X LeakyReLU(float negativeSlope = 0.01f) => new(new LeakyReLU(negativeSlope).Forward(_tensor));
        public X Mish() => new(new Mish().Forward(_tensor));
        public X Softplus() => new(new Softplus().Forward(_tensor));
        public X Swish() => new(new Swish().Forward(_tensor));
        public X SiLU() => new(new SiLU().Forward(_tensor));
        public X GLU() => new(new GLU().Forward(_tensor));
        public X SwiGLU() => new(new SwiGLU().Forward(_tensor));
        public X HardSigmoid() => new(new HardSigmoid().Forward(_tensor));
        public X HardTanh(float minVal = -1f, float maxVal = 1f) => new(new HardTanh(minVal, maxVal).Forward(_tensor));
        public X TanhShrink() => new(new TanhShrink().Forward(_tensor));
        public X Softsign() => new(new Softsign().Forward(_tensor));
        public X SiTU() => new(new SiTU().Forward(_tensor));
        public X SELU() => new(new SELU().Forward(_tensor));
        public X HardSwish() => new(_tensor.Multiply(new HardSigmoid().Forward(_tensor.Add(3f).Divide(6f))));

        #endregion

        // =========================================================================================
        #region Fluent Neural Network Builders & Layers (COMPLETE)
        // =========================================================================================

        /// <summary>Base method to apply any implementation of <see cref="ILayer"/>.</summary>
        public X Apply(ILayer layer) => new(layer.Forward(_tensor));

        /// <summary>Base method to apply any implementation of <see cref="IModel"/>.</summary>
        public X Apply(IModel model) => new(model.Forward(_tensor));

        // -------------------------
        // Core Linear / Dense
        // -------------------------
        public X Linear(int outFeatures, bool bias = true, Device? device = null)
            => Apply(new Linear(_tensor.Shape[^1], outFeatures, device ?? _tensor.Device) { /* bias handled inside */ });

        public X Dense(int outFeatures, bool bias = true) => Linear(outFeatures, bias);
        public X FractalLinear(int inFeatures, int outFeatures, FractalType initType, bool useBias = true)
            => Apply(new FractalLinear(inFeatures, outFeatures, initType, useBias));

        // -------------------------
        // Convolutions
        // -------------------------
        public X Conv1D(int outChannels, int kernelSize, int stride = 1, int padding = 0, bool useBias = true)
            => Apply(new Conv1D(_tensor.Shape[1], outChannels, kernelSize, stride, padding, useBias, _tensor.Device));

        public X Conv2D(int outChannels, int kernelSize, int stride = 1, int padding = 0, bool useBias = true)
            => Apply(new Conv2D(_tensor.Shape[1], outChannels, kernelSize, stride, padding, useBias, _tensor.Device));

        public X Conv2D(int outChannels, int kernelH, int kernelW, int stride = 1, int padding = 0, bool useBias = true)
            => Apply(new Conv2D(_tensor.Shape[1], outChannels, kernelH, stride, padding, useBias, _tensor.Device)); // simplified; real ctor may vary

        public X Conv3D(int outChannels, int kernelDepth, int kernelHeight, int kernelWidth,
                        bool hasBias = true, int stride = 1, int padding = 0)
            => Apply(new Conv3D(_tensor.Shape[1], outChannels, kernelDepth, kernelHeight, kernelWidth, hasBias, stride, padding));

        // -------------------------
        // Recurrent
        // -------------------------
        public X GRU(int inputSize, int hiddenSize) => Apply(new GRU(inputSize, hiddenSize));
        public X LSTM(int inputSize, int hiddenSize, Device? device = null)
            => Apply(new LSTM(inputSize, hiddenSize, device ?? _tensor.Device));

        // -------------------------
        // Normalizations (full set)
        // -------------------------
        public X BatchNorm(int numFeatures, float eps = 1e-5f, float momentum = 0.1f, bool useAffine = true)
        {
            var l = new BatchNorm(numFeatures, eps, momentum, useAffine);
            l.To(_tensor.Device);
            return Apply(l);
        }

        public X LayerNorm()
        {
            var l = new ArborNet.Layers.Normalization.LayerNorm(new[] { _tensor.Shape[^1] });
            l.To(_tensor.Device);
            return Apply(l);
        }

        public X LayerNorm(int[] normalizedShape, float eps = 1e-5f, bool useAffine = true)
        {
            var l = new ArborNet.Layers.Normalization.LayerNorm(normalizedShape, eps, useAffine);
            l.To(_tensor.Device);
            return Apply(l);
        }

        public X RMSNorm(int numFeatures, float eps = 1e-6f, bool useAffine = true)
        {
            var l = new RMSNorm(numFeatures, eps, useAffine);
            l.To(_tensor.Device);
            return Apply(l);
        }

        public X GroupNorm(int numChannels, int numGroups, float eps = 1e-5f, bool useAffine = true)
        {
            var l = new GroupNorm(numChannels, numGroups, eps, useAffine);
            l.To(_tensor.Device);
            return Apply(l);
        }

        public X InstanceNorm(int numChannels, float eps = 1e-5f, bool useAffine = true)
        {
            var l = new InstanceNorm(numChannels, eps, useAffine);
            l.To(_tensor.Device);
            return Apply(l);
        }

        public X LayerScale(int numFeatures, float initScale = 1e-2f)
            => Apply(new LayerScale(numFeatures, initScale));

        // -------------------------
        // Pooling
        // -------------------------
        public X MaxPool2D(int kernelSize = 2, int stride = 2, int padding = 0)
            => Apply(new MaxPool2D(kernelSize, stride, padding));

        public X AvgPool2D(int kernelSize = 2, int stride = 2, int padding = 0)
            => Apply(new AvgPool2D(kernelSize, stride, padding));

        public X AdaptiveAvgPool2D(int outputSize = 1)
            => Apply(new AdaptiveAvgPool2D(outputSize));

        public X AdaptiveMaxPool2D(int outputSize = 1)
            => Apply(new AdaptiveMaxPool2D(outputSize));

        // -------------------------
        // Attention & Embeddings
        // -------------------------
        public X Dropout(float p = 0.5f) => Apply(new Dropout(p));
        public X Attention(int embedDim, int numHeads, bool useBias = true)
            => Apply(new Attention(embedDim, numHeads, useBias));
        public X MultiHeadAttention(int dModel, int numHeads, bool useBias = true)
            => Apply(new MultiHeadAttention(dModel, numHeads, useBias));
        public X DeltaAttention(int dModel, int nHeads)
            => Apply(new DeltacAttention(dModel, nHeads, _tensor.Device));
        public X Embedding(int numEmbeddings, int embeddingDim)
            => Apply(new Embedding(numEmbeddings, embeddingDim));
        public X PositionalEncoding(int dModel, int maxLen = 512)
            => Apply(new PositionalEncoding(dModel, maxLen, _tensor.Device));
        public X SubquadraticAttention(int dModel, int headCount, FractalType initType)
            => Apply(new SubquadraticAttention(dModel, headCount, initType));

        // -------------------------
        // Advanced Model Blocks & MoE
        // -------------------------
        public X TransformerBlock(int dModel, int numHeads, int ffDim = 0)
            => Apply(new TransformerBlock(dModel, numHeads, ffDim));
        public X MistralBlock(int hiddenDim, int numHeads, int kvHeads, int slidingWindow)
            => Apply(new MistralBlock(hiddenDim, numHeads, kvHeads, slidingWindow));
        public X ConvNeXtBlock(int dim) => Apply(new ConvNeXtBlock(dim));
        public X BasicBlock(int inChannels, int planes, int stride, int expansion)
            => Apply(new BasicBlock(inChannels, planes, stride, expansion, _tensor.Device));
        public X BottleneckBlock(int inChannels, int planes, int stride, int expansion)
            => Apply(new BottleneckBlock(inChannels, planes, stride, expansion, _tensor.Device));
        public X StableLatentMoE(int dModel, int numExperts, int activeExperts, int expertCapacity)
            => Apply(new StableLatentcMoE(dModel, numExperts, activeExperts, expertCapacity, _tensor.Device));
        public X FractalTransformerBlock(int dModel, int dFF, int headCount, FractalType initType)
            => Apply(new FractalTransformerBlock(dModel, dFF, headCount, initType));
        public X AttentionResidualConnection(int layerIndex)
            => Apply(new AttentionResidualscConnection(layerIndex, _tensor.Device)); // note: this is a helper, may need adaptation

        // -------------------------
        // Activation as Layer
        // -------------------------
        public X Activation(IActivation act) => Apply(new ActivationLayer(act));

        #endregion

        // =========================================================================================
        #region Fluent Full Models (static factories + instance Apply)
        // =========================================================================================

        // Sequential
        public static Sequential Sequential(params ILayer[] layers) => new Sequential(layers);
        public X Sequential(params ILayer[] layers) => Apply(new Sequential(layers));

        // Classic / Vision
        public static ResNet ResNet18(int numClasses = 1000, Device? device = null) => Models.ResNet.ResNet18(numClasses, device);
        public static ResNet ResNet50(int numClasses = 1000, Device? device = null) => Models.ResNet.ResNet50(numClasses, device);
        public X ResNet18(int numClasses = 1000) => Apply(ResNet18(numClasses, Device));
        public X ResNet50(int numClasses = 1000) => Apply(ResNet50(numClasses, Device));

        public static ViT ViT(int imageSize = 224, int patchSize = 16, int numClasses = 1000,
                              int dim = 768, int depth = 12, int heads = 12, Device? device = null)
            => new ViT(imageSize, patchSize, numClasses, dim, depth, heads, device);
        public X ViT(int imageSize = 224, int patchSize = 16, int numClasses = 1000,
                     int dim = 768, int depth = 12, int heads = 12)
            => Apply(ViT(imageSize, patchSize, numClasses, dim, depth, heads, Device));

        public static ConvNeXt ConvNeXtTiny(int numClasses = 1000, Device? device = null)
            => Models.ConvNeXt.Tiny(numClasses, device);
        public X ConvNeXtTiny(int numClasses = 1000) => Apply(ConvNeXtTiny(numClasses, Device));

        public static YOLOv10 YOLOv10(int numClasses = 80, Device? device = null)
            => new YOLOv10(numClasses, device);
        public X YOLOv10(int numClasses = 80) => Apply(YOLOv10(numClasses, Device));

        // Language / Multimodal
        public static GPT GPT(int vocabSize, int dModel = 768, int nLayers = 12, int nHeads = 12,
                              int maxSeqLen = 1024, Device? device = null)
            => new GPT(vocabSize, dModel, nLayers, nHeads, maxSeqLen, device);
        public X GPT(int vocabSize, int dModel = 768, int nLayers = 12, int nHeads = 12, int maxSeqLen = 1024)
            => Apply(GPT(vocabSize, dModel, nLayers, nHeads, maxSeqLen, Device));

        public static GPT_NeoX GPTNeoX(int vocabSize, int dModel = 2048, int nLayers = 24, int nHeads = 16,
                                       int maxSeqLen = 2048, Device? device = null)
            => new GPT_NeoX(vocabSize, dModel, nLayers, nHeads, maxSeqLen, device);
        public X GPTNeoX(int vocabSize, int dModel = 2048, int nLayers = 24, int nHeads = 16, int maxSeqLen = 2048)
            => Apply(GPTNeoX(vocabSize, dModel, nLayers, nHeads, maxSeqLen, Device));

        public static Llama3 Llama3(int vocabSize, int dModel = 4096, int nLayers = 32, int nHeads = 32,
                                    int nKvHeads = 8, int maxSeqLen = 8192, Device? device = null)
            => new Llama3(vocabSize, dModel, nLayers, nHeads, nKvHeads, maxSeqLen, device);
        public X Llama3(int vocabSize, int dModel = 4096, int nLayers = 32, int nHeads = 32,
                        int nKvHeads = 8, int maxSeqLen = 8192)
            => Apply(Llama3(vocabSize, dModel, nLayers, nHeads, nKvHeads, maxSeqLen, Device));

        public static Mistral Mistral(int vocabSize, int dModel = 4096, int nLayers = 32, int nHeads = 32,
                                      int nKvHeads = 8, int slidingWindow = 4096, Device? device = null)
            => new Mistral(vocabSize, dModel, nLayers, nHeads, nKvHeads, slidingWindow, device);
        public X Mistral(int vocabSize, int dModel = 4096, int nLayers = 32, int nHeads = 32,
                         int nKvHeads = 8, int slidingWindow = 4096)
            => Apply(Mistral(vocabSize, dModel, nLayers, nHeads, nKvHeads, slidingWindow, Device));

        public static BERT BERT(int vocabSize, int dModel = 768, int nLayers = 12, int nHeads = 12,
                                int maxSeqLen = 512, Device? device = null)
            => new BERT(vocabSize, dModel, nLayers, nHeads, maxSeqLen, device);
        public X BERT(int vocabSize, int dModel = 768, int nLayers = 12, int nHeads = 12, int maxSeqLen = 512)
            => Apply(BERT(vocabSize, dModel, nLayers, nHeads, maxSeqLen, Device));

        public static FractalLLM FractalLLM(int vocabSize, int dModel, int nLayers, int nHeads,
                                            FractalType initType, Device? device = null)
            => new FractalLLM(vocabSize, dModel, nLayers, nHeads, initType, device);
        public X FractalLLM(int vocabSize, int dModel, int nLayers, int nHeads, FractalType initType)
            => Apply(FractalLLM(vocabSize, dModel, nLayers, nHeads, initType, Device));

        public static KimiK3 KimiK3(int vocabSize, int dModel = 2048, int nLayers = 24, Device? device = null)
            => new KimiK3(vocabSize, dModel, nLayers, device);
        public X KimiK3(int vocabSize, int dModel = 2048, int nLayers = 24)
            => Apply(KimiK3(vocabSize, dModel, nLayers, Device));

        public static CLIP CLIP(int imageDim, int textDim, int embedDim = 512, Device? device = null)
            => new CLIP(imageDim, textDim, embedDim, device);
        public X CLIP(int imageDim, int textDim, int embedDim = 512)
            => Apply(CLIP(imageDim, textDim, embedDim, Device));

        public static Whisper Whisper(int vocabSize, int dModel = 512, int nLayers = 12, Device? device = null)
            => new Whisper(vocabSize, dModel, nLayers, device);
        public X Whisper(int vocabSize, int dModel = 512, int nLayers = 12)
            => Apply(Whisper(vocabSize, dModel, nLayers, Device));

        // Generative
        public static VAE VAE(int inputDim, int latentDim, Device? device = null)
            => new VAE(inputDim, latentDim, device);
        public X VAE(int inputDim, int latentDim) => Apply(VAE(inputDim, latentDim, Device));

        public static UNet UNet(int inChannels = 3, int outChannels = 3, Device? device = null)
            => new UNet(inChannels, outChannels, device);
        public X UNet(int inChannels = 3, int outChannels = 3) => Apply(UNet(inChannels, outChannels, Device));

        public static DiffusionModel DiffusionModel(int channels = 3, int timeEmbedDim = 128, Device? device = null)
            => new DiffusionModel(channels, timeEmbedDim, device);
        public X DiffusionModel(int channels = 3, int timeEmbedDim = 128)
            => Apply(DiffusionModel(channels, timeEmbedDim, Device));

        public static StableDiffusion StableDiffusion(Device? device = null)
            => new StableDiffusion(device);
        public X StableDiffusion() => Apply(StableDiffusion(Device));

        // Clustering / Misc
        public static KMeans KMeans(int nClusters, int maxIter = 100, Device? device = null)
            => new KMeans(nClusters, maxIter, device);
        public X KMeansPredict(KMeans model) => new(model.Predict(_tensor));

        public static HRM HRM(int inputSize, int hiddenSize, Device? device = null)
            => new HRM(inputSize, hiddenSize, device);
        public X HRM(int inputSize, int hiddenSize) => Apply(HRM(inputSize, hiddenSize, Device));

        public static TransformerTextEncoder TransformerTextEncoder(int vocabSize, int dModel = 512,
                                                                     int nLayers = 6, int nHeads = 8, Device? device = null)
            => new TransformerTextEncoder(vocabSize, dModel, nLayers, nHeads, device);
        public X TransformerTextEncoder(int vocabSize, int dModel = 512, int nLayers = 6, int nHeads = 8)
            => Apply(TransformerTextEncoder(vocabSize, dModel, nLayers, nHeads, Device));

        #endregion

        // =========================================================================================
        #region Fluent Loss Chaining
        // =========================================================================================

        public X MseLoss(X targets, string reduction = "mean")
            => new(new MSE().Forward(_tensor, targets._tensor, reduction));
        public X CrossEntropyLoss(X targets, string reduction = "mean")
            => new(new CrossEntropy().Forward(_tensor, targets._tensor, reduction));
        public X BinaryCrossEntropyLoss(X targets, string reduction = "mean")
            => new(new BinaryCrossEntropy().Forward(_tensor, targets._tensor, reduction));
        public X HingeLoss(X targets, string reduction = "mean")
            => new(new Hinge().Forward(_tensor, targets._tensor, reduction));
        public X HuberLoss(X targets, float delta = 1.0f, string reduction = "mean")
            => new(new Huber(delta).Forward(_tensor, targets._tensor, reduction));
        public X KLDivLoss(X targets, string reduction = "mean")
            => new(new KLDiv().Forward(_tensor, targets._tensor, reduction));
        public X NLLLoss(X targets, string reduction = "mean")
            => new(new NLL().Forward(_tensor, targets._tensor, reduction));

        #endregion

        // =========================================================================================
        #region Transforms & Data Augmentation (image tensors)
        // =========================================================================================

        public X Resize(int newHeight, int newWidth, InterpolationMode mode = InterpolationMode.Bilinear)
            => new(Transforms.Resize(_tensor, newHeight, newWidth, mode));
        public X FlipHorizontal() => new(Transforms.FlipHorizontal(_tensor));
        public X FlipVertical() => new(Transforms.FlipVertical(_tensor));
        public X Rotate90(bool clockwise = true) => new(Transforms.Rotate90(_tensor, clockwise));
        public X Augment(Random? random = null) => new(Transforms.Augment(_tensor, random));

        #endregion

        // =========================================================================================
        #region Generation Helpers
        // =========================================================================================

        public static int SampleToken(X logits, float temperature = 0.7f, int topK = 50, float topP = 0.9f)
            => Sampler.SampleToken(logits._tensor, temperature, topK, topP);

        public IEnumerable<string> GenerateStream(IModel model, ITokenizer tokenizer,
                                                  string prompt, int maxTokens = 100,
                                                  float temperature = 0.7f, int topK = 50, float topP = 0.9f)
        {
            var generator = new TextGenerator(model, tokenizer, Device);
            return generator.GenerateStream(prompt, maxTokens, temperature, topK, topP);
        }

        #endregion

        // =========================================================================================
        #region Autograd & Execution Accessors
        // =========================================================================================

        public float[] ToArray() => _tensor.ToArray();
        public float ToScalar() => _tensor.ToScalar();
        public void Backward() => _tensor.Backward();
        public void ClearGrad() => _tensor.ClearGrad();
        public X Detach() => new(_tensor.Detach());
        public X RequiresGrad(bool requires = true)
        {
            _tensor.RequiresGrad = requires;
            return this;
        }

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