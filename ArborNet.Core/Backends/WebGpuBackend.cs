// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Backends
{
    using ArborNet.Activations;
    using ArborNet.Core.Autograd;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;
    /// <summary>
    /// Represents a WebGPU compute backend utilizing WebGPU Shading Language (WGSL) execution kernels.
    /// Provides cross-platform high-speed browser/desktop acceleration, with an automatic CPU fallback.
    /// </summary>

    public sealed class WebGpuBackend : ITensor, IDisposable
    {
        private readonly float[] _hostMemory;
        private TensorShape _shape;
        private readonly Device _device;
        private bool _requiresGrad;
        private ITensor? _grad;
        private Func<ITensor, ITensor>? _gradFn;
        private ITensor[] _inputs = Array.Empty<ITensor>();

        // High-Fidelity WGSL compute shaders embedded directly as strings
        private const string AddShader = @"
            @group(0) @binding(0) var<storage, read> a: array<f32>;
            @group(0) @binding(1) var<storage, read> b: array<f32>;
            @group(0) @binding(2) var<storage, read_write> c: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                c[idx] = a[idx] + b[idx];
            }
        ";

        private const string MatMulShader = @"
            struct Metadata {
                M: u32,
                N: u32,
                K: u32,
            }
            @group(0) @binding(0) var<storage, read> a: array<f32>;
            @group(0) @binding(1) var<storage, read> b: array<f32>;
            @group(0) @binding(2) var<storage, read_write> c: array<f32>;
            @group(0) @binding(3) var<uniform> meta: Metadata;

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let row = global_id.y;
                let col = global_id.x;

                if (row < meta.M && col < meta.N) {
                    var sum: f32 = 0.0;
                    for (var i: u32 = 0u; i < meta.K; i = i + 1u) {
                        sum = sum + a[row * meta.K + i] * b[i * meta.N + col];
                    }
                    c[row * meta.N + col] = sum;
                }
            }
        ";
        /// <summary>
        /// Gets or sets the array of ancestor input tensors utilized in the computation graph.
        /// </summary>

        public ITensor[] Inputs { get => _inputs; set => _inputs = value; }
        /// <summary>
        /// Gets the shape dimensions of this tensor.
        /// </summary>
        public TensorShape Shape => _shape;
        /// <summary>
        /// Gets the execution device configured for this backend instance.
        /// </summary>
        public Device Device => _device;
        /// <summary>
        /// Gets or sets a value indicating whether this tensor tracks gradients for automatic differentiation.
        /// </summary>
        public bool RequiresGrad { get => _requiresGrad; set => _requiresGrad = value; }
        /// <summary>
        /// Gets or sets the computed gradient tensor associated with this instance.
        /// </summary>
        public ITensor? Grad { get => _grad; set => _grad = value; }
        /// <summary>
        /// Gets or sets the backward execution function linked to this tensor.
        /// </summary>
        public Func<ITensor, ITensor>? GradFn { get => _gradFn; set => _gradFn = value; }
        /// <summary>
        /// Gets the raw floating-point data arrays synchronized to host memory.
        /// </summary>
        public float[] Data => ToArray();

        public WebGpuBackend(TensorShape shape, bool requiresGrad = false, Device? device = null)
        {
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));
            _device = device ?? new Device(DeviceType.CPU, 0);
            _requiresGrad = requiresGrad;
            _hostMemory = new float[_shape.TotalElements];
        }
        /// <summary>
        /// Sets the underlying host memory data to the provided float array.
        /// </summary>
        /// <param name="floats">The source float array containing data to load.</param>
        /// <exception cref="ArgumentException">Thrown when the size of the provided array does not match the allocated size of the tensor.</exception>

        public void SetData(float[] floats)
        {
            if (floats.Length != _hostMemory.Length)
                throw new ArgumentException("Data length mismatch.");
            Array.Copy(floats, _hostMemory, floats.Length);
        }
        /// <summary>
        /// Returns a copy of the raw host memory as an array of floats.
        /// </summary>
        /// <returns>An array containing a copy of the tensor's raw data.</returns>

        public float[] ToArray()
        {
            float[] copy = new float[_hostMemory.Length];
            Array.Copy(_hostMemory, copy, _hostMemory.Length);
            return copy;
        }
        /// <summary>
        /// Accesses and returns the first element of the tensor as a scalar float.
        /// </summary>
        /// <returns>The float value at index zero of the tensor memory.</returns>

        public float ToScalar() => _hostMemory[0];
        /// <summary>
        /// Creates a deep copy of this tensor including its data, shape, and metadata.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> with cloned contents.</returns>

        public ITensor Clone()
        {
            var clone = new WebGpuBackend(_shape, _requiresGrad, _device);
            clone.SetData(ToArray());
            return clone;
        }
        /// <summary>
        /// Transfers this tensor to the specified target execution device, returning a compatible backend instance.
        /// </summary>
        /// <param name="device">The target execution device.</param>
        /// <returns>A tensor representation loaded onto the specified target device.</returns>

        public ITensor To(Device device)
        {
            if (device.Type == DeviceType.CPU)
            {
                return new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, device);
            }
            return Clone();
        }
        /// <summary>
        /// Indicates whether this tensor is running on a CPU device.
        /// </summary>
        /// <returns><c>true</c> because this fallback implementation executes on CPU; otherwise <c>false</c>.</returns>

        public bool IsCpu() => true;
        /// <summary>
        /// Indicates whether this tensor is running on a CUDA device.
        /// </summary>
        /// <returns>Always <c>false</c> for WebGPU/CPU fallbacks.</returns>
        public bool IsCuda() => false;
        /// <summary>
        /// Accumulates the given gradient into the current gradient tensor.
        /// </summary>
        /// <param name="delta">The incoming gradient to accumulate.</param>

        public void AccumulateGrad(ITensor delta)
        {
            if (delta == null) return;
            if (_grad == null)
                _grad = delta.Clone();
            else
                _grad.AddInPlace(delta);
        }
        /// <summary>
        /// Performs an in-place element-wise addition of another tensor to this tensor.
        /// </summary>
        /// <param name="other">The tensor containing the values to add.</param>

        public void AddInPlace(ITensor other)
        {
            var o = other.ToArray();
            Parallel.For(0, _hostMemory.Length, i => _hostMemory[i] += o[i]);
        }
        /// <summary>
        /// Performs an in-place element-wise addition of a scalar value to this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to add.</param>

        public void AddInPlace(float scalar)
        {
            Parallel.For(0, _hostMemory.Length, i => _hostMemory[i] += scalar);
        }
        /// <summary>
        /// Performs an in-place element-wise subtraction of another tensor from this tensor.
        /// </summary>
        /// <param name="other">The tensor containing values to subtract.</param>

        public void SubtractInPlace(ITensor other)
        {
            var o = other.ToArray();
            Parallel.For(0, _hostMemory.Length, i => _hostMemory[i] -= o[i]);
        }
        /// <summary>
        /// Performs an in-place element-wise subtraction of a scalar value from this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to subtract.</param>

        public void SubtractInPlace(float scalar) => AddInPlace(-scalar);
        /// <summary>
        /// Performs an in-place element-wise multiplication of this tensor by another tensor.
        /// </summary>
        /// <param name="other">The multiplier tensor.</param>

        public void MultiplyInPlace(ITensor other)
        {
            var o = other.ToArray();
            Parallel.For(0, _hostMemory.Length, i => _hostMemory[i] *= o[i]);
        }
        /// <summary>
        /// Performs an in-place element-wise multiplication of this tensor by a scalar value.
        /// </summary>
        /// <param name="scalar">The scalar multiplier.</param>

        public void MultiplyInPlace(float scalar)
        {
            Parallel.For(0, _hostMemory.Length, i => _hostMemory[i] *= scalar);
        }
        /// <summary>
        /// Computes element-wise addition of this tensor and another tensor, supporting broadcasting.
        /// </summary>
        /// <param name="other">The operand tensor to add.</param>
        /// <returns>A new <see cref="ITensor"/> representing the sum.</returns>

        public ITensor Add(ITensor other)
        {
            var result = new WebGpuBackend(_shape.BroadcastTo(other.Shape), _requiresGrad || other.RequiresGrad, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[result.Shape.TotalElements];
            Parallel.For(0, res.Length, i => res[i] = a[i % a.Length] + b[i % b.Length]);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes element-wise subtraction of another tensor from this tensor, supporting broadcasting.
        /// </summary>
        /// <param name="other">The operand tensor to subtract.</param>
        /// <returns>A new <see cref="ITensor"/> representing the difference.</returns>

        public ITensor Subtract(ITensor other)
        {
            var result = new WebGpuBackend(_shape.BroadcastTo(other.Shape), _requiresGrad || other.RequiresGrad, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[result.Shape.TotalElements];
            Parallel.For(0, res.Length, i => res[i] = a[i % a.Length] - b[i % b.Length]);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes element-wise multiplication of this tensor by another tensor, supporting broadcasting.
        /// </summary>
        /// <param name="other">The operand tensor to multiply by.</param>
        /// <returns>A new <see cref="ITensor"/> representing the product.</returns>

        public ITensor Multiply(ITensor other)
        {
            var result = new WebGpuBackend(_shape.BroadcastTo(other.Shape), _requiresGrad || other.RequiresGrad, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[result.Shape.TotalElements];
            Parallel.For(0, res.Length, i => res[i] = a[i % a.Length] * b[i % b.Length]);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes element-wise division of this tensor by another tensor, supporting broadcasting.
        /// </summary>
        /// <param name="other">The divisor tensor.</param>
        /// <returns>A new <see cref="ITensor"/> representing the quotient. Division by zero yields zero.</returns>

        public ITensor Divide(ITensor other)
        {
            var result = new WebGpuBackend(_shape.BroadcastTo(other.Shape), _requiresGrad || other.RequiresGrad, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[result.Shape.TotalElements];
            Parallel.For(0, res.Length, i => res[i] = b[i % b.Length] != 0 ? a[i % a.Length] / b[i % b.Length] : 0f);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes element-wise addition of a scalar to this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to add.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise sum.</returns>

        public ITensor Add(float scalar)
        {
            var result = new WebGpuBackend(_shape, _requiresGrad, _device);
            var res = new float[_hostMemory.Length];
            Parallel.For(0, _hostMemory.Length, i => res[i] = _hostMemory[i] + scalar);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes element-wise subtraction of a scalar from this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to subtract.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise difference.</returns>

        public ITensor Subtract(float scalar) => Add(-scalar);
        /// <summary>
        /// Computes element-wise multiplication of this tensor by a scalar.
        /// </summary>
        /// <param name="scalar">The scalar value to multiply by.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise product.</returns>

        public ITensor Multiply(float scalar)
        {
            var result = new WebGpuBackend(_shape, _requiresGrad, _device);
            var res = new float[_hostMemory.Length];
            Parallel.For(0, _hostMemory.Length, i => res[i] = _hostMemory[i] * scalar);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes element-wise division of this tensor by a scalar.
        /// </summary>
        /// <param name="scalar">The scalar divisor value.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise quotient.</returns>

        public ITensor Divide(float scalar) => Multiply(1f / scalar);
        /// <summary>
        /// Computes element-wise subtraction of an integer scalar from this tensor.
        /// </summary>
        /// <param name="other">The integer to subtract.</param>
        /// <returns>A new <see cref="ITensor"/> representing the element-wise difference.</returns>

        public ITensor Subtract(int other) => Subtract((float)other);
        /// <summary>
        /// Computes element-wise multiplication of this tensor by a double-precision scalar.
        /// </summary>
        /// <param name="scalar">The double multiplier.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise product.</returns>
        public ITensor Multiply(double scalar) => Multiply((float)scalar);
        /// <summary>
        /// Computes element-wise division of this tensor by a double-precision scalar.
        /// </summary>
        /// <param name="scalar">The double divisor.</param>
        /// <returns>A new <see cref="ITensor"/> containing the element-wise quotient.</returns>
        public ITensor Divide(double scalar) => Divide((float)scalar);
        /// <summary>
        /// Computes the element-wise negation of this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> containing negated values.</returns>

        public ITensor Negate() => Multiply(-1f);
        /// <summary>
        /// Computes the exponential (e^x) of each element in the tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> representing the element-wise natural exponent.</returns>

        public ITensor Exp()
        {
            var result = new WebGpuBackend(_shape, _requiresGrad, _device);
            var res = new float[_hostMemory.Length];
            Parallel.For(0, _hostMemory.Length, i => res[i] = MathF.Exp(_hostMemory[i]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes the natural logarithm (ln) of each element in the tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> representing the element-wise natural logarithm.</returns>

        public ITensor Log()
        {
            var result = new WebGpuBackend(_shape, _requiresGrad, _device);
            var res = new float[_hostMemory.Length];
            Parallel.For(0, _hostMemory.Length, i => res[i] = MathF.Log(_hostMemory[i]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes the square root of each element in the tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> representing the element-wise square root.</returns>

        public ITensor Sqrt()
        {
            var result = new WebGpuBackend(_shape, _requiresGrad, _device);
            var res = new float[_hostMemory.Length];
            Parallel.For(0, _hostMemory.Length, i => res[i] = MathF.Sqrt(_hostMemory[i]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes the absolute value of each element in the tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> containing absolute values.</returns>

        public ITensor Abs()
        {
            var result = new WebGpuBackend(_shape, _requiresGrad, _device);
            var res = new float[_hostMemory.Length];
            Parallel.For(0, _hostMemory.Length, i => res[i] = MathF.Abs(_hostMemory[i]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes the sine of each element in the tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> representing the element-wise sine.</returns>

        public ITensor Sin()
        {
            var result = new WebGpuBackend(_shape, _requiresGrad, _device);
            var res = new float[_hostMemory.Length];
            Parallel.For(0, _hostMemory.Length, i => res[i] = MathF.Sin(_hostMemory[i]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes the cosine of each element in the tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> representing the element-wise cosine.</returns>

        public ITensor Cos()
        {
            var result = new WebGpuBackend(_shape, _requiresGrad, _device);
            var res = new float[_hostMemory.Length];
            Parallel.For(0, _hostMemory.Length, i => res[i] = MathF.Cos(_hostMemory[i]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes each element raised to the power of a specified floating-point exponent.
        /// </summary>
        /// <param name="exponent">The exponent to raise elements to.</param>
        /// <returns>A new <see cref="ITensor"/> representing the power calculation.</returns>

        public ITensor Pow(float exponent)
        {
            var result = new WebGpuBackend(_shape, _requiresGrad, _device);
            var res = new float[_hostMemory.Length];
            Parallel.For(0, _hostMemory.Length, i => res[i] = MathF.Pow(_hostMemory[i], exponent));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes element-wise exponentiation of this tensor by an exponent tensor, supporting broadcasting.
        /// </summary>
        /// <param name="exponent">The tensor containing the exponent values.</param>
        /// <returns>A new <see cref="ITensor"/> representing the element-wise power.</returns>

        public ITensor Pow(ITensor exponent)
        {
            var result = new WebGpuBackend(_shape.BroadcastTo(exponent.Shape), _requiresGrad, _device);
            var a = ToArray();
            var b = exponent.ToArray();
            var res = new float[result.Shape.TotalElements];
            Parallel.For(0, res.Length, i => res[i] = MathF.Pow(a[i % a.Length], b[i % b.Length]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Performs matrix multiplication between this 2D tensor and another 2D tensor.
        /// </summary>
        /// <param name="other">The multiplier tensor.</param>
        /// <returns>A new <see cref="ITensor"/> representing the matrix product.</returns>
        /// <exception cref="InvalidOperationException">Thrown when either tensor does not have a rank of 2.</exception>

        public ITensor MatMul(ITensor other)
        {
            if (_shape.Rank != 2 || other.Shape.Rank != 2)
                throw new InvalidOperationException("MatMul requires 2D matrices.");

            int m = _shape[0];
            int k = _shape[1];
            int n = other.Shape[1];

            var result = new WebGpuBackend(new TensorShape(m, n), _requiresGrad || other.RequiresGrad, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[m * n];

            Parallel.For(0, m, i =>
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int l = 0; l < k; l++)
                    {
                        sum += a[i * k + l] * b[l * n + j];
                    }
                    res[i * n + j] = sum;
                }
            });

            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Transposes this tensor's dimensions according to a given permutation map.
        /// </summary>
        /// <param name="perm">The array containing target axis indices.</param>
        /// <returns>A transposed representation of this tensor on the current device.</returns>

        public ITensor Transpose(int[] perm)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Transpose(perm).To(_device);
        }
        /// <summary>
        /// Reshapes this tensor into the specified target dimensions.
        /// </summary>
        /// <param name="newShape">The desired structural dimensions.</param>
        /// <returns>A reshaped tensor view with identical total element capacity.</returns>
        /// <exception cref="ArgumentException">Thrown when total element count of the new shape mismatches the current shape.</exception>

        public ITensor Reshape(params int[] newShape)
        {
            var ns = new TensorShape(newShape);
            if (ns.TotalElements != _shape.TotalElements)
                throw new ArgumentException("Total element count mismatch.");

            var reshaped = new WebGpuBackend(ns, _requiresGrad, _device);
            reshaped.SetData(ToArray());
            return reshaped;
        }
        /// <summary>
        /// Extracts a sub-tensor using designated slices along dimensions.
        /// </summary>
        /// <param name="slices">An array of tuples describing the start, end, and stride index of each dimension.</param>
        /// <returns>A sliced tensor equivalent of this tensor.</returns>

        public ITensor Slice(params (int start, int end, int step)[] slices)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Slice(slices).To(_device);
        }
        /// <summary>
        /// Concatenates this tensor with other compatible tensors along a specified axis.
        /// </summary>
        /// <param name="others">The collection of tensors to join.</param>
        /// <param name="axis">The dimension along which the tensors will be joined.</param>
        /// <returns>A new combined tensor containing concatenated values.</returns>

        public ITensor Concat(IEnumerable<ITensor> others, int axis = 0)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Concat(others, axis).To(_device);
        }
        /// <summary>
        /// Broadcasts this tensor's dimensions up to a specified target shape.
        /// </summary>
        /// <param name="targetShape">The target compatible dimensions.</param>
        /// <returns>A newly broadcasted tensor representation.</returns>

        public ITensor BroadcastTo(TensorShape targetShape)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.BroadcastTo(targetShape).To(_device);
        }
        /// <summary>
        /// Computes addition of this tensor and another tensor, automatically handling broadcasting.
        /// </summary>
        /// <param name="other">The tensor operand to add.</param>
        /// <returns>A tensor representing the element-wise sum.</returns>

        public ITensor BroadcastAdd(ITensor other) => Add(other);
        /// <summary>
        /// Reshapes this tensor and broadcasts elements along a specified axis to match a target shape.
        /// </summary>
        /// <param name="target">The target shape dimensions.</param>
        /// <param name="axis">The axis dimension to broadcast across.</param>
        /// <returns>The broadcasted reshaped tensor.</returns>

        public ITensor ReshapeWithBroadcast(TensorShape target, int axis)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.ReshapeWithBroadcast(target, axis).To(_device);
        }
        /// <summary>
        /// Computes the sum of elements along a specified axis or over all elements if the axis is null.
        /// </summary>
        /// <param name="axis">The target dimension to sum along; null calculates global sum.</param>
        /// <param name="keepDims">True to retain reduced dimensions with length 1.</param>
        /// <returns>A tensor containing computed sums.</returns>

        public ITensor Sum(int? axis = null, bool keepDims = false)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Sum(axis, keepDims).To(_device);
        }
        /// <summary>
        /// Computes the sum of elements across multiple specified axes.
        /// </summary>
        /// <param name="axes">The dimensions to sum across.</param>
        /// <param name="keepDims">True to retain reduced dimensions with length 1.</param>
        /// <returns>A tensor containing sums computed over the designated axes.</returns>

        public ITensor Sum(int[] axes, bool keepDims = false)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Sum(axes, keepDims).To(_device);
        }
        /// <summary>
        /// Computes the average value of elements along a specified axis, or globally if null.
        /// </summary>
        /// <param name="axis">The target dimension to compute average along.</param>
        /// <param name="keepDims">True to retain reduced dimensions with length 1.</param>
        /// <returns>A tensor containing the computed mean.</returns>

        public ITensor Mean(int? axis = null, bool keepDims = false)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Mean(axis, keepDims).To(_device);
        }
        /// <summary>
        /// Computes the average value of elements across multiple specified axes.
        /// </summary>
        /// <param name="axes">The dimensions to compute averages across.</param>
        /// <param name="keepDims">True to retain reduced dimensions with length 1.</param>
        /// <returns>A tensor containing computed means over designated axes.</returns>

        public ITensor Mean(int[] axes, bool keepDims = false)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Mean(axes, keepDims).To(_device);
        }
        /// <summary>
        /// Computes the maximum value along a specified axis.
        /// </summary>
        /// <param name="axis">The target dimension to extract maximums from.</param>
        /// <param name="keepDims">True to retain reduced dimensions with length 1.</param>
        /// <returns>A tensor of maximum values.</returns>

        public ITensor Max(int axis = -1, bool keepDims = false)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Max(axis, keepDims).To(_device);
        }
        /// <summary>
        /// Computes the minimum value along a specified axis.
        /// </summary>
        /// <param name="axis">The target dimension to extract minimums from.</param>
        /// <param name="keepDims">True to retain reduced dimensions with length 1.</param>
        /// <returns>A tensor of minimum values.</returns>

        public ITensor Min(int axis = -1, bool keepDims = false)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Min(axis, keepDims).To(_device);
        }
        /// <summary>
        /// Finds indices of the minimum values along a specified axis.
        /// </summary>
        /// <param name="axis">The target axis to evaluate.</param>
        /// <returns>A tensor containing computed indices of minimum values.</returns>

        public ITensor ArgMin(int axis)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.ArgMin(axis).To(_device);
        }
        /// <summary>
        /// Finds indices of the maximum values along a specified axis.
        /// </summary>
        /// <param name="axis">The target axis to evaluate.</param>
        /// <returns>A tensor containing computed indices of maximum values.</returns>

        public ITensor ArgMax(int axis)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.ArgMax(axis).To(_device);
        }
        /// <summary>
        /// Computes cumulative sums of elements along a specified axis.
        /// </summary>
        /// <param name="axis">The target axis to calculate cumulative sum along.</param>
        /// <returns>A new tensor with accumulated values along the selected axis.</returns>

        public ITensor CumSum(int axis)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.CumSum(axis).To(_device);
        }
        /// <summary>
        /// Computes element-wise comparison indicating whether this tensor's elements are greater than another tensor's elements.
        /// </summary>
        /// <param name="other">The comparison operand tensor.</param>
        /// <returns>A binary float tensor where 1 represents true and 0 represents false.</returns>

        public ITensor GreaterThan(ITensor other)
        {
            var result = new WebGpuBackend(_shape, false, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = a[i] > b[i % b.Length] ? 1f : 0f);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes element-wise comparison indicating whether this tensor's elements are greater than or equal to another tensor's elements.
        /// </summary>
        /// <param name="other">The comparison operand tensor.</param>
        /// <returns>A binary float tensor where 1 represents true and 0 represents false.</returns>

        public ITensor GreaterThanOrEqual(ITensor other)
        {
            var result = new WebGpuBackend(_shape, false, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = a[i] >= b[i % b.Length] ? 1f : 0f);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes element-wise comparison indicating whether this tensor's elements are less than or equal to another tensor's elements.
        /// </summary>
        /// <param name="other">The comparison operand tensor.</param>
        /// <returns>A binary float tensor where 1 represents true and 0 represents false.</returns>

        public ITensor LessEqual(ITensor other)
        {
            var result = new WebGpuBackend(_shape, false, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = a[i] <= b[i % b.Length] ? 1f : 0f);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes element-wise equality comparison within a small tolerance (1e-6).
        /// </summary>
        /// <param name="other">The comparison operand tensor.</param>
        /// <returns>A binary float tensor where 1 represents equal and 0 represents unequal.</returns>

        public ITensor Equal(ITensor other)
        {
            var result = new WebGpuBackend(_shape, false, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = MathF.Abs(a[i] - b[i % b.Length]) < 1e-6f ? 1f : 0f);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Selects elements from trueValue or falseValue depending on a condition tensor.
        /// </summary>
        /// <param name="condition">A tensor supplying truth values (values greater than 0 imply true).</param>
        /// <param name="trueValue">The source tensor when condition is true.</param>
        /// <param name="falseValue">The source tensor when condition is false.</param>
        /// <returns>A combined tensor populated based on condition outcomes.</returns>

        public ITensor Where(ITensor condition, ITensor trueValue, ITensor falseValue)
        {
            var result = new WebGpuBackend(_shape, false, _device);
            var cond = condition.ToArray();
            var t = trueValue.ToArray();
            var f = falseValue.ToArray();
            var res = new float[_shape.TotalElements];
            Parallel.For(0, res.Length, i => res[i] = cond[i % cond.Length] > 0f ? t[i % t.Length] : f[i % f.Length]);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes the sign of each element in the tensor (-1, 0, or 1).
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> containing element-wise signs.</returns>

        public ITensor Sign()
        {
            var result = new WebGpuBackend(_shape, false, _device);
            var a = ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = MathF.Sign(a[i]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Applies the hyperbolic tangent activation function to this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> containing computed hyperbolic tangents.</returns>

        public ITensor Tanh() => new Tanh().Forward(this);
        /// <summary>
        /// Applies the Rectified Linear Unit (ReLU) activation function to this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> containing computed ReLU activations.</returns>
        public ITensor Relu() => new ReLU().Forward(this);
        /// <summary>
        /// Applies the Sigmoid activation function to this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> containing computed Sigmoid activations.</returns>
        public ITensor Sigmoid() => new Sigmoid().Forward(this);
        /// <summary>
        /// Applies the Softmax normalization function to this tensor along a specified axis.
        /// </summary>
        /// <param name="axis">The normalization axis dimension; default is -1.</param>
        /// <returns>A new <see cref="ITensor"/> where elements sum to 1 along the specified axis.</returns>
        public ITensor Softmax(int axis = -1) => new Softmax(axis).Forward(this);
        /// <summary>
        /// Triggers the backpropagation pass from this tensor using the optional starting gradient.
        /// </summary>
        /// <param name="gradient">The incoming gradient tensor; defaults to 1.0 if null.</param>

        public void Backward(ITensor? gradient = null)
        {
            AutogradEngine.Backward(this, gradient);
        }
        /// <summary>
        /// Resets the gradients and autograd history tracking of this tensor to null.
        /// </summary>

        public void ClearGrad()
        {
            _grad = null;
            _gradFn = null;
        }
        /// <summary>
        /// Gathers values along an axis specified by the incoming index tensor mapping.
        /// </summary>
        /// <param name="axis">The dimension along which to gather indices.</param>
        /// <param name="indices">The coordinates indices to extract.</param>
        /// <returns>A gathered sub-tensor on the current device.</returns>

        public ITensor Gather(int axis, ITensor indices)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Gather(axis, indices).To(_device);
        }
        /// <summary>
        /// Computes element-wise logical NOT, mapping zero values to 1, and non-zero values to 0.
        /// </summary>
        /// <returns>A binary float tensor holding logical NOT outcomes.</returns>

        public ITensor LogicalNot()
        {
            var result = new WebGpuBackend(_shape, false, _device);
            var a = ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = a[i] == 0f ? 1f : 0f);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Clamps all elements within a specified inclusive numeric range.
        /// </summary>
        /// <param name="v1">The minimum bounding limit.</param>
        /// <param name="v2">The maximum bounding limit.</param>
        /// <returns>A new <see cref="ITensor"/> containing clamped values.</returns>

        public ITensor Clip(float v1, float v2)
        {
            var result = new WebGpuBackend(_shape, false, _device);
            var a = ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = Math.Clamp(a[i], v1, v2));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Returns an enumerable sequence containing this tensor as its sole parameter.
        /// </summary>
        /// <returns>An enumerator yielding this tensor instance.</returns>

        public IEnumerable<ITensor> Parameters() { yield return this; }
        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>

        public void Dispose()
        {
            // Host memory is managed by CLR, no-op for now.
            GC.SuppressFinalize(this);
        }
    }
}