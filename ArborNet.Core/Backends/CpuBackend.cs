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

    #region Using Statements:

    using ArborNet.Activations;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using System;
    using System.Buffers;
    using System.Collections.Generic;
    using System.Linq;
    using System.Numerics;
    using System.Runtime.CompilerServices;
    /// <summary>
    /// Represents a CPU-based tensor backend implementation that supports multidimensional operations, 
    /// SIMD hardware acceleration, automatic differentiation (autograd), and lazy resource management.
    /// </summary>

    #endregion

    public sealed class CpuBackend : ITensor, IDisposable
    {
        /// <summary>
        /// The underlying flat array containing the tensor elements.
        /// </summary>
        private float[] _data;

        /// <summary>
        /// The total number of elements in the tensor.
        /// </summary>
        private readonly int _length;

        /// <summary>
        /// Indicates whether the underlying data array was rented from the shared array pool.
        /// </summary>
        private bool _isRented;

        /// <summary>
        /// The shape and dimensional structure of the tensor.
        /// </summary>
        private TensorShape _shape;

        /// <summary>
        /// The execution device assigned to the tensor (defaults to CPU).
        /// </summary>
        private Device _device;

        /// <summary>
        /// A value indicating whether gradients should be computed and tracked for this tensor.
        /// </summary>
        private bool _requiresGrad;

        /// <summary>
        /// The accumulated gradient tensor for backpropagation.
        /// </summary>
        private ITensor? _grad;

        /// <summary>
        /// The backward gradient function used to propagate gradients during autograd execution.
        /// </summary>
        private Func<ITensor, ITensor>? _gradFn;

        /// <summary>
        /// The source tensors that produced this tensor in the computational graph.
        /// </summary>
        private ITensor[] _inputs = Array.Empty<ITensor>();

        /// <summary>
        /// Lock object used to ensure thread safety during state modifications and numerical accumulations.
        /// </summary>
        private readonly object _lock = new();
        /// <summary>
        /// Gets or sets the parent input tensors associated with this node in the execution graph.
        /// </summary>

        public ITensor[] Inputs { get => _inputs; set => _inputs = value; }
        /// <summary>
        /// Gets the shape structure of the tensor.
        /// </summary>

        public TensorShape Shape => _shape;
        /// <summary>
        /// Gets the physical hardware device associated with this backend.
        /// </summary>

        public Device Device => _device;
        /// <summary>
        /// Gets or sets a value indicating whether this tensor tracks and accumulates gradients.
        /// </summary>

        public bool RequiresGrad { get => _requiresGrad; set => _requiresGrad = value; }
        /// <summary>
        /// Gets or sets the gradient associated with this tensor.
        /// </summary>

        public ITensor? Grad { get => _grad; set => _grad = value; }
        /// <summary>
        /// Gets or sets the backwards function delegate used for autograd computations.
        /// </summary>

        public Func<ITensor, ITensor>? GradFn { get => _gradFn; set => _gradFn = value; }
        /// <summary>
        /// Gets a copy of the underlying tensor data as a flat float array.
        /// </summary>

        public float[] Data => ToArray();

        /// <summary>
        /// Initializes a new instance of the <see cref="CpuBackend"/> class with a specified shape, 
        /// renting memory from the shared array pool.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="requiresGrad">Whether the tensor tracks gradients.</param>
        /// <param name="device">The targeted device. Defaults to CPU.</param>
        /// <exception cref="ArgumentNullException">Thrown if the provided shape is null.</exception>
        public CpuBackend(TensorShape shape, bool requiresGrad = false, Device? device = null)
        {
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));
            _length = shape.TotalElements;
            _data = ArrayPool<float>.Shared.Rent(_length);
            _isRented = true;
            Array.Clear(_data, 0, _length);
            _requiresGrad = requiresGrad;
            _device = device ?? Device.CPU;
            TensorScope.Register(this);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="CpuBackend"/> class wrapping an existing, managed float array.
        /// </summary>
        /// <param name="data">The pre-allocated flat array representing the tensor's elements.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="requiresGrad">Whether the tensor tracks gradients.</param>
        /// <param name="device">The targeted device. Defaults to CPU.</param>
        /// <exception cref="ArgumentNullException">Thrown if shape or data is null.</exception>
        public CpuBackend(float[] data, TensorShape shape, bool requiresGrad = false, Device? device = null)
        {
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));
            _length = shape.TotalElements;
            _data = data ?? throw new ArgumentNullException(nameof(data));
            _isRented = false;
            _requiresGrad = requiresGrad;
            _device = device ?? Device.CPU;
        }
        /// <summary>
        /// Accumulates the given gradient delta into this tensor's gradient state, 
        /// automatically handling dimension reductions due to broadcasting.
        /// </summary>
        /// <param name="delta">The incoming gradient tensor to accumulate.</param>

        public void AccumulateGrad(ITensor delta)
        {
            if (delta == null) return;
            var reducedDelta = ReduceGradientToTarget(delta, _shape);
            lock (_lock)
            {
                if (_grad == null)
                {
                    _grad = reducedDelta.Clone();
                }
                else
                {
                    _grad.AddInPlace(reducedDelta);
                }
            }
        }
        /// <summary>
        /// Adds another tensor to this tensor in-place.
        /// </summary>
        /// <param name="other">The tensor containing the values to add.</param>

        public void AddInPlace(ITensor other)
        {
            var otherRaw = Unwrap(other);
            lock (_lock)
            {
                int len = Math.Min(_length, otherRaw._length);
                ArborNet.Core.Native.SIMD.Accelerate.Add(_data, _data, otherRaw._data, len);
            }
        }
        /// <summary>
        /// Adds a scalar value to every element of this tensor in-place.
        /// </summary>
        /// <param name="scalar">The scalar value to add.</param>

        public void AddInPlace(float scalar)
        {
            lock (_lock)
            {
                for (int i = 0; i < _length; i++) _data[i] += scalar;
            }
        }
        /// <summary>
        /// Subtracts another tensor from this tensor in-place.
        /// </summary>
        /// <param name="other">The tensor containing the values to subtract.</param>

        public void SubtractInPlace(ITensor other)
        {
            var otherRaw = Unwrap(other);
            lock (_lock)
            {
                int len = Math.Min(_length, otherRaw._length);
                ArborNet.Core.Native.SIMD.Accelerate.Subtract(_data, _data, otherRaw._data, len);
            }
        }
        /// <summary>
        /// Subtracts a scalar value from every element of this tensor in-place.
        /// </summary>
        /// <param name="scalar">The scalar value to subtract.</param>

        public void SubtractInPlace(float scalar)
        {
            lock (_lock)
            {
                for (int i = 0; i < _length; i++) _data[i] -= scalar;
            }
        }
        /// <summary>
        /// Multiplies this tensor by another tensor element-wise in-place.
        /// </summary>
        /// <param name="other">The tensor containing the values to multiply by.</param>

        public void MultiplyInPlace(ITensor other)
        {
            var otherRaw = Unwrap(other);
            lock (_lock)
            {
                int len = Math.Min(_length, otherRaw._length);
                ArborNet.Core.Native.SIMD.Accelerate.Multiply(_data, _data, otherRaw._data, len);
            }
        }
        /// <summary>
        /// Multiplies every element of this tensor by a scalar value in-place.
        /// </summary>
        /// <param name="scalar">The scalar factor.</param>

        public void MultiplyInPlace(float scalar)
        {
            lock (_lock)
            {
                for (int i = 0; i < _length; i++) _data[i] *= scalar;
            }
        }
        /// <summary>
        /// Gathers values along a specified axis using 1D indices. Currently optimized for 2D inputs [Batch, Classes].
        /// </summary>
        /// <param name="axis">The dimension along which to gather elements.</param>
        /// <param name="indices">The indices containing the elements to extract.</param>
        /// <returns>A new <see cref="ITensor"/> containing the gathered elements.</returns>
        /// <exception cref="NotSupportedException">Thrown if the tensor rank is not 2.</exception>
        /// <exception cref="IndexOutOfRangeException">Thrown if an index is outside the bounds of the specified axis size.</exception>

        public ITensor Gather(int axis, ITensor indices)
        {
            var idxRaw = Unwrap(indices);
            float[] idxData = idxRaw.ToArray();

            if (_shape.Rank != 2)
                throw new NotSupportedException("Gather is currently optimized for 2D inputs [Batch, Classes].");

            int batch = _shape[0];
            int classes = _shape[1];
            var outShape = new TensorShape(batch);
            float[] outData = new float[batch];

            for (int i = 0; i < batch; i++)
            {
                int classIdx = (int)idxData[i];
                if (classIdx < 0 || classIdx >= classes)
                    throw new IndexOutOfRangeException($"Index {classIdx} is out of bounds for axis with size {classes}.");
                outData[i] = _data[i * classes + classIdx];
            }

            var result = new CpuBackend(outData, outShape, _requiresGrad, _device);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                var capturedIndices = idxRaw;
                result.GradFn = gradOutput =>
                {
                    float[] goData = gradOutput.ToArray();
                    float[] gradInputData = new float[capturedSelf._shape.TotalElements];

                    for (int i = 0; i < batch; i++)
                    {
                        int classIdx = (int)capturedIndices._data[i];
                        gradInputData[i * classes + classIdx] = goData[i];
                    }

                    var gradInput = new CpuBackend(gradInputData, capturedSelf._shape, false, capturedSelf._device);
                    capturedSelf.AccumulateGrad(gradInput);
                    return gradOutput;
                };
            }

            return new Tensor(result);
        }
        /// <summary>
        /// Returns a safe copy of the underlying flat tensor elements.
        /// </summary>
        /// <returns>A new flat float array copying the tensor data.</returns>

        public float[] ToArray()
        {
            var copy = new float[_length];
            lock (_lock)
            {
                Array.Copy(_data, 0, copy, 0, _length);
            }
            return copy;
        }
        /// <summary>
        /// Extracts the single scalar value from a 1-element tensor.
        /// </summary>
        /// <returns>The scalar value of the tensor.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the tensor does not contain exactly one element.</exception>

        public float ToScalar()
        {
            if (_length != 1) throw new InvalidOperationException("Tensor is not a scalar.");
            return _data[0];
        }
        /// <summary>
        /// Creates a deep copy of the tensor and its configuration.
        /// </summary>
        /// <returns>A new cloned instance of <see cref="ITensor"/>.</returns>

        public ITensor Clone() => new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
        /// <summary>
        /// Transfers this tensor to another specified hardware execution device.
        /// </summary>
        /// <param name="device">The target execution device.</param>
        /// <returns>An <see cref="ITensor"/> backed by the targeted device.</returns>
        /// <exception cref="NotSupportedException">Thrown if the requested device conversion is unsupported.</exception>

        public ITensor To(Device device)
        {
            if (device.Type == DeviceType.CPU) return Clone();
            if (device.Type == DeviceType.CUDA) return new CudaBackend(ToArray(), _shape.Clone(), _requiresGrad, device);
            throw new NotSupportedException($"Transfer to {device.Type} is not supported.");
        }
        /// <summary>
        /// Updates the elements of the tensor with the contents of the given float array.
        /// </summary>
        /// <param name="floats">The source array containing the new elements.</param>
        /// <exception cref="ArgumentNullException">Thrown if the input array is null.</exception>
        /// <exception cref="ArgumentException">Thrown if the input array length does not match the tensor total elements.</exception>

        public void SetData(float[] floats)
        {
            if (floats == null) throw new ArgumentNullException(nameof(floats));
            if (floats.Length != _shape.TotalElements)
                throw new ArgumentException("Data size mismatch.");

            lock (_lock)
            {
                if (_isRented && _data != null)
                {
                    ArrayPool<float>.Shared.Return(_data);
                    _isRented = false;
                }
                _data = (float[])floats.Clone();
            }
        }
        /// <summary>
        /// Gets a value indicating whether this tensor runs on CPU hardware.
        /// </summary>
        /// <returns>True.</returns>

        public bool IsCpu() => true;
        /// <summary>
        /// Gets a value indicating whether this tensor runs on CUDA hardware.
        /// </summary>
        /// <returns>False.</returns>

        public bool IsCuda() => false;
        /// <summary>
        /// Yields this backend as a trackable optimization parameter.
        /// </summary>
        /// <returns>An enumerable containing this tensor.</returns>

        public IEnumerable<ITensor> Parameters() { yield return this; }
        /// <summary>
        /// Computes the broadcasted output shape and necessary source strides for binary elementwise broadcasting operations.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>A tuple containing the broadcasted target shape, the stride offsets for tensor A, and the stride offsets for tensor B.</returns>

        private static (TensorShape shape, long[] strideA, long[] strideB) GetBroadcastShapeAndStrides(ITensor a, ITensor b)
        {
            var ca = Unwrap(a);
            var cb = Unwrap(b);
            var resultShape = ca._shape.BroadcastTo(cb._shape);
            var strideA = ComputeBroadcastStrides(ca._shape.Dimensions, resultShape.Dimensions);
            var strideB = ComputeBroadcastStrides(cb._shape.Dimensions, resultShape.Dimensions);
            return (resultShape, strideA, strideB);
        }
        /// <summary>
        /// Calculates indexing strides based on the original shape mapping to a broadcasted target shape.
        /// </summary>
        /// <param name="original">The dimensions of the original pre-broadcasted shape.</param>
        /// <param name="target">The dimensions of the target broadcasted shape.</param>
        /// <returns>An array of stride values corresponding to the target dimensions.</returns>

        private static long[] ComputeBroadcastStrides(int[] original, int[] target)
        {
            long[] strides = new long[target.Length];
            int offset = target.Length - original.Length;

            for (int i = 0; i < target.Length; i++)
            {
                int origIdx = i - offset;
                if (origIdx >= 0 && original[origIdx] != 1)
                {
                    long stride = 1;
                    for (int j = origIdx + 1; j < original.Length; j++)
                        stride *= original[j];
                    strides[i] = stride;
                }
                else
                {
                    strides[i] = 0;
                }
            }
            return strides;
        }
        /// <summary>
        /// Map a 1D flat output index back to the flat index of a source tensor using its broadcast strides.
        /// </summary>
        /// <param name="flatIdx">The 1D flat index in the target broadcasted tensor.</param>
        /// <param name="strides">The stride mapping array calculated for the source tensor.</param>
        /// <param name="shape">The target broadcasted shape dimensions.</param>
        /// <returns>The corresponding 1D flat index in the original source tensor.</returns>

        private static int GetBroadcastIndex(int flatIdx, long[] strides, int[] shape)
        {
            int idx = 0;
            int remaining = flatIdx;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                int dimSize = shape[i];
                int coord = remaining % dimSize;
                idx += (int)(coord * strides[i]);
                remaining /= dimSize;
            }
            return idx;
        }
        /// <summary>
        /// Broadcasts the current tensor to a compatible target shape.
        /// </summary>
        /// <param name="targetShape">The desired target shape.</param>
        /// <returns>A new <see cref="ITensor"/> broadcasted to the target shape.</returns>

        public ITensor BroadcastTo(TensorShape targetShape)
        {
            if (_shape.Equals(targetShape)) return new Tensor(this);
            var result = new CpuBackend(targetShape, _requiresGrad, _device);
            var strides = ComputeBroadcastStrides(_shape.Dimensions, targetShape.Dimensions);

            for (int i = 0; i < result._data.Length; i++)
            {
                int srcIdx = GetBroadcastIndex(i, strides, targetShape.Dimensions);
                result._data[i] = _data[srcIdx % _data.Length];
            }
            return new Tensor(result);
        }
        /// <summary>
        /// Performs elementwise addition between this tensor and another tensor, supporting broadcasting.
        /// </summary>
        /// <param name="other">The tensor to add.</param>
        /// <returns>A new <see cref="ITensor"/> representing the sum.</returns>

        public ITensor Add(ITensor other)
        {
            var otherRaw = Unwrap(other);
            var result = new CpuBackend(_shape.BroadcastTo(otherRaw._shape), _requiresGrad || other.RequiresGrad, _device);

            if (_shape.Equals(otherRaw._shape))
            {
                ArborNet.Core.Native.SIMD.Accelerate.Add(result._data, _data, otherRaw._data, _length);
                RegisterBinaryAutograd(result, this, other, (g, _, _) => (g, g));
                return new Tensor(result);
            }

            if (otherRaw._length == 1)
            {
                float scalar = otherRaw._data[0];
                VectorizedScalarAdd(result._data, _data, scalar, _length);
                RegisterBinaryAutograd(result, this, other, (g, _, _) => (g, g));
                return new Tensor(result);
            }

            return Elementwise(other, (a, b) => a + b, (g, _, _) => (g, g));
        }
        /// <summary>
        /// Performs vectorized hardware scalar addition utilizing SIMD instructions.
        /// </summary>
        /// <param name="dest">The destination flat array where results will be written.</param>
        /// <param name="src">The source flat array containing input elements.</param>
        /// <param name="scalar">The scalar value to add to each element.</param>
        /// <param name="length">The number of elements to process.</param>

        private static unsafe void VectorizedScalarAdd(float[] dest, float[] src, float scalar, int length)
        {
            int i = 0;
            int vectorCount = Vector<float>.Count;
            var scalarVec = new Vector<float>(scalar);

            fixed (float* pSrc = src, pDest = dest)
            {
                for (; i <= length - vectorCount; i += vectorCount)
                {
                    var vec = Unsafe.Read<Vector<float>>(pSrc + i);
                    var res = vec + scalarVec;
                    Unsafe.Write(pDest + i, res);
                }
            }
            for (; i < length; i++)
            {
                dest[i] = src[i] + scalar;
            }
        }
        /// <summary>
        /// Subtracts another tensor from this tensor, supporting broadcasting.
        /// </summary>
        /// <param name="other">The tensor to subtract.</param>
        /// <returns>A new <see cref="ITensor"/> representing the difference.</returns>

        public ITensor Subtract(ITensor other)
        {
            var otherRaw = Unwrap(other);
            if (_shape.Equals(otherRaw._shape))
            {
                var resultData = new float[_data.Length];
                ArborNet.Core.Native.SIMD.Accelerate.Subtract(resultData, _data, otherRaw._data, _data.Length);

                var rawResult = new CpuBackend(resultData, _shape.Clone(), _requiresGrad || other.RequiresGrad, _device)
                {
                    Inputs = new[] { this, other }
                };

                RegisterBinaryAutograd(rawResult, this, other, (g, _, _) => (g, g.Negate()));
                return new Tensor(rawResult);
            }

            return Elementwise(other, (a, b) => a - b, (g, _, _) => (g, g.Negate()));
        }
        /// <summary>
        /// Multiplies this tensor by another tensor element-wise, supporting broadcasting.
        /// </summary>
        /// <param name="other">The tensor to multiply by.</param>
        /// <returns>A new <see cref="ITensor"/> representing the product.</returns>

        public ITensor Multiply(ITensor other)
        {
            var otherRaw = Unwrap(other);
            if (_shape.Equals(otherRaw._shape))
            {
                var resultData = new float[_data.Length];
                ArborNet.Core.Native.SIMD.Accelerate.Multiply(resultData, _data, otherRaw._data, _data.Length);

                var rawResult = new CpuBackend(resultData, _shape.Clone(), _requiresGrad || other.RequiresGrad, _device)
                {
                    Inputs = new[] { this, other }
                };

                RegisterBinaryAutograd(rawResult, this, other, (g, a, b) => (g.Multiply(b), g.Multiply(a)));
                return new Tensor(rawResult);
            }

            return Elementwise(other, (a, b) => a * b, (g, a, b) => (g.Multiply(b), g.Multiply(a)));
        }
        /// <summary>
        /// Divides this tensor by another tensor element-wise, supporting broadcasting.
        /// </summary>
        /// <param name="other">The divisor tensor.</param>
        /// <returns>A new <see cref="ITensor"/> representing the quotient.</returns>

        public ITensor Divide(ITensor other)
        {
            return Elementwise(other, (a, b) => b != 0 ? a / b : 0f,
                (g, a, b) => (g.Divide(b), g.Multiply(a.Negate()).Divide(b.Multiply(b))));
        }
        /// <summary>
        /// Computes a binary element-wise operation over two tensors, managing broadcasting and autograd tracking.
        /// </summary>
        /// <param name="other">The other tensor operand.</param>
        /// <param name="op">The delegate representing the binary scalar operation.</param>
        /// <param name="gradFn">The gradient backward function delegate returning a tuple of gradients for (self, other).</param>
        /// <returns>A new <see cref="ITensor"/> containing the computed results.</returns>

        private ITensor Elementwise(ITensor other, Func<float, float, float> op, Func<ITensor, ITensor, ITensor, (ITensor, ITensor)> gradFn)
        {
            var selfRaw = Unwrap(this);
            var otherRaw = Unwrap(other);

            var (resultShape, strideA, strideB) = GetBroadcastShapeAndStrides(selfRaw, otherRaw);
            var resultData = new float[resultShape.TotalElements];
            for (int i = 0; i < resultData.Length; i++)
            {
                int idxA = GetBroadcastIndex(i, strideA, resultShape.Dimensions);
                int idxB = GetBroadcastIndex(i, strideB, resultShape.Dimensions);
                resultData[i] = op(selfRaw._data[idxA % selfRaw._data.Length],
                                   otherRaw._data[idxB % otherRaw._data.Length]);
            }

            bool requiresGrad = this.RequiresGrad || other.RequiresGrad;
            var rawResult = new CpuBackend(resultData, resultShape, requiresGrad, _device)
            {
                Inputs = new[] { this, other }
            };

            RegisterBinaryAutograd(rawResult, this, other, gradFn);
            return new Tensor(rawResult);
        }
        /// <summary>
        /// Computes an element-wise unary operation over the current tensor with custom gradient propagation.
        /// </summary>
        /// <param name="op">The delegate representing the unary scalar operation.</param>
        /// <param name="gradFn">The gradient backward function delegate returning the gradient for self.</param>
        /// <returns>A new <see cref="ITensor"/> containing the computed results.</returns>

        private ITensor ElementwiseScalar(Func<float, float> op, Func<ITensor, ITensor> gradFn)
        {
            var resultData = new float[_data.Length];
            for (int i = 0; i < _data.Length; i++)
                resultData[i] = op(_data[i]);

            var rawResult = new CpuBackend(resultData, _shape, _requiresGrad, _device)
            {
                Inputs = new[] { this }
            };

            if (_requiresGrad)
            {
                var capturedSelf = this;
                rawResult.GradFn = gradOutput =>
                {
                    var gSelf = gradFn(gradOutput);
                    capturedSelf.AccumulateGrad(gSelf);
                    return gradOutput;
                };
            }

            return new Tensor(rawResult);
        }
        /// <summary>
        /// Helper method to register custom gradient backward actions for binary computational nodes.
        /// </summary>
        /// <param name="result">The resulting backend tensor of the operation.</param>
        /// <param name="self">The left operand tensor.</param>
        /// <param name="other">The right operand tensor.</param>
        /// <param name="gradFn">The gradient computation delegate returning a tuple of gradients for (self, other).</param>

        private static void RegisterBinaryAutograd(CpuBackend result, ITensor self, ITensor other, Func<ITensor, ITensor, ITensor, (ITensor, ITensor)> gradFn)
        {
            if (result.RequiresGrad)
            {
                result.GradFn = gradOutput =>
                {
                    var (gSelf, gOther) = gradFn(gradOutput, self, other);
                    if (self.RequiresGrad) self.AccumulateGrad(gSelf);
                    if (other.RequiresGrad) other.AccumulateGrad(gOther);
                    return gradOutput;
                };
            }
        }
        /// <summary>
        /// Reduces a gradient delta tensor back to a target shape to account for automatic broadcasting during forward computation.
        /// </summary>
        /// <param name="delta">The gradient tensor to reduce.</param>
        /// <param name="targetShape">The original pre-broadcast shape target.</param>
        /// <returns>A reduced gradient <see cref="ITensor"/> matching the target shape.</returns>

        public static ITensor ReduceGradientToTarget(ITensor delta, TensorShape targetShape)
        {
            var current = delta;
            var sDelta = current.Shape;

            if (sDelta.Equals(targetShape)) return current;
            if (targetShape.TotalElements == 1) return Tensor.FromScalar(current.Sum().ToScalar(), current.Device);

            int rankDelta = sDelta.Rank;
            int rankTarget = targetShape.Rank;

            if (rankDelta > rankTarget)
            {
                int dimsToSum = rankDelta - rankTarget;
                for (int i = 0; i < dimsToSum; i++) current = current.Sum(0, keepDims: false);
                sDelta = current.Shape;
                rankDelta = sDelta.Rank;
            }

            for (int i = 0; i < rankDelta; i++)
            {
                if (targetShape.Dimensions[i] == 1 && sDelta.Dimensions[i] > 1)
                {
                    current = current.Sum(i, keepDims: true);
                    sDelta = current.Shape;
                }
            }
            return current;
        }
        /// <summary>
        /// Adds a scalar float value to this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to add.</param>
        /// <returns>A new sum <see cref="ITensor"/>.</returns>

        public ITensor Add(float scalar) => ElementwiseScalar(x => x + scalar, g => g);
        /// <summary>
        /// Subtracts a scalar float value from this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to subtract.</param>
        /// <returns>A new difference <see cref="ITensor"/>.</returns>

        public ITensor Subtract(float scalar) => ElementwiseScalar(x => x - scalar, g => g);
        /// <summary>
        /// Multiplies this tensor by a scalar float value.
        /// </summary>
        /// <param name="scalar">The scalar multiplier.</param>
        /// <returns>A new product <see cref="ITensor"/>.</returns>

        public ITensor Multiply(float scalar) => ElementwiseScalar(x => x * scalar, g => g.Multiply(scalar));
        /// <summary>
        /// Divides this tensor by a scalar float value.
        /// </summary>
        /// <param name="scalar">The scalar divisor.</param>
        /// <returns>A new quotient <see cref="ITensor"/>.</returns>

        public ITensor Divide(float scalar) => Multiply(1f / scalar);
        /// <summary>
        /// Subtracts a scalar integer value from this tensor.
        /// </summary>
        /// <param name="other">The scalar integer value to subtract.</param>
        /// <returns>A new difference <see cref="ITensor"/>.</returns>

        public ITensor Subtract(int other) => Subtract((float)other);
        /// <summary>
        /// Multiplies this tensor by a scalar double value.
        /// </summary>
        /// <param name="scalar">The scalar multiplier.</param>
        /// <returns>A new product <see cref="ITensor"/>.</returns>

        public ITensor Multiply(double scalar) => Multiply((float)scalar);
        /// <summary>
        /// Divides this tensor by a scalar double value.
        /// </summary>
        /// <param name="scalar">The scalar divisor.</param>
        /// <returns>A new quotient <see cref="ITensor"/>.</returns>

        public ITensor Divide(double scalar) => Multiply(1.0 / scalar);
        /// <summary>
        /// Negates every element of this tensor.
        /// </summary>
        /// <returns>A negated copy of this <see cref="ITensor"/>.</returns>

        public ITensor Negate() => ElementwiseScalar(x => -x, g => g.Negate());
        /// <summary>
        /// Computes the exponential (e^x) of each element.
        /// </summary>
        /// <returns>An element-wise exponential <see cref="ITensor"/>.</returns>

        public ITensor Exp() => ElementwiseScalar(MathF.Exp, g => g.Multiply(this.Exp()));
        /// <summary>
        /// Computes the natural logarithm of each element.
        /// </summary>
        /// <returns>An element-wise natural log <see cref="ITensor"/>.</returns>

        public ITensor Log() => ElementwiseScalar(MathF.Log, g => g.Divide(this));
        /// <summary>
        /// Computes the square root of each element.
        /// </summary>
        /// <returns>An element-wise square root <see cref="ITensor"/>.</returns>

        public ITensor Sqrt() => ElementwiseScalar(MathF.Sqrt, g => g.Divide(this.Sqrt().Multiply(2)));
        /// <summary>
        /// Computes the absolute value of each element.
        /// </summary>
        /// <returns>An element-wise absolute value <see cref="ITensor"/>.</returns>

        public ITensor Abs() => ElementwiseScalar(MathF.Abs, g => g.Multiply(this.GreaterThan(Tensor.Zeros(_shape)).Multiply(2).Subtract(1)));
        /// <summary>
        /// Computes the sine of each element.
        /// </summary>
        /// <returns>An element-wise sine <see cref="ITensor"/>.</returns>

        public ITensor Sin() => ElementwiseScalar(MathF.Sin, g => g.Multiply(this.Cos()));
        /// <summary>
        /// Computes the cosine of each element.
        /// </summary>
        /// <returns>An element-wise cosine <see cref="ITensor"/>.</returns>

        public ITensor Cos() => ElementwiseScalar(MathF.Cos, g => g.Multiply(this.Sin().Negate()));
        /// <summary>
        /// Computes the numerical sign indicator (-1, 0, 1) of each element.
        /// </summary>
        /// <returns>An element-wise sign representation <see cref="ITensor"/>.</returns>

        public ITensor Sign() => ElementwiseScalar(x => MathF.Sign(x), g => Tensor.Zeros(g.Shape, g.Device));
        /// <summary>
        /// Computes the mathematical power (x^exponent) of each element using a scalar float exponent.
        /// </summary>
        /// <param name="exponent">The numeric power exponent.</param>
        /// <returns>A new <see cref="ITensor"/> containing the elementwise exponentiation results.</returns>

        public ITensor Pow(float exponent)
        {
            return ElementwiseScalar(x => MathF.Pow(x, exponent),
                g => g.Multiply(Tensor.FromScalar(exponent)).Multiply(this.Pow(exponent - 1)));
        }
        /// <summary>
        /// Computes the element-wise mathematical power using exponents specified in another tensor.
        /// </summary>
        /// <param name="exponent">The exponent tensor.</param>
        /// <returns>A new <see cref="ITensor"/> representing base raised to exponent power.</returns>

        public ITensor Pow(ITensor exponent) => Elementwise(exponent, (a, b) => MathF.Pow(a, b),
    (g, a, b) => (
        g.Multiply(b.Multiply(a.Pow(b.Subtract(Tensor.FromScalar(1f))))),
        g.Multiply(a.Pow(b).Multiply(a.Log()))
    ));
        /// <summary>
        /// Adds another tensor to this tensor supporting automatic broadcasting.
        /// </summary>
        /// <param name="other">The tensor to add.</param>
        /// <returns>The broadcasted sum <see cref="ITensor"/>.</returns>

        public ITensor BroadcastAdd(ITensor other) => Add(other);
        /// <summary>
        /// Reshapes this tensor and broadcasts it to a target shape along a specific axis index.
        /// </summary>
        /// <param name="target">The target shape.</param>
        /// <param name="axis">The alignment dimension index.</param>
        /// <returns>A reshaped and broadcasted <see cref="ITensor"/>.</returns>
        /// <exception cref="ArgumentNullException">Thrown if the target shape is null.</exception>

        public ITensor ReshapeWithBroadcast(TensorShape target, int axis = -1)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));
            int targetRank = target.Rank;
            if (axis < 0) axis = targetRank + axis;

            var viewDims = Enumerable.Repeat(1, targetRank).ToArray();
            int origIdx = 0;
            for (int i = axis; i < targetRank && origIdx < _shape.Rank; i++)
            {
                viewDims[i] = _shape.Dimensions[origIdx++];
            }

            return this.Reshape(viewDims).BroadcastTo(target);
        }
        /// <summary>
        /// Performs matrix multiplication on two 2D tensors.
        /// </summary>
        /// <param name="other">The multiplier matrix.</param>
        /// <returns>A new 2D <see cref="ITensor"/> representing the matrix product.</returns>
        /// <exception cref="InvalidOperationException">Thrown if either tensor is not 2D.</exception>

        public ITensor MatMul(ITensor other)
        {
            var selfRaw = Unwrap(this);
            var otherRaw = Unwrap(other);

            if (selfRaw._shape.Rank != 2 || otherRaw._shape.Rank != 2)
                throw new InvalidOperationException("MatMul requires 2D tensors.");

            int m = selfRaw._shape[0], k = selfRaw._shape[1], n = otherRaw._shape[1];
            var resultData = new float[m * n];

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float sum = 0;
                    for (int l = 0; l < k; l++)
                        sum += selfRaw._data[i * k + l] * otherRaw._data[l * n + j];
                    resultData[i * n + j] = sum;
                }

            bool requiresGrad = this.RequiresGrad || other.RequiresGrad;
            var rawResult = new CpuBackend(resultData, new TensorShape(m, n), requiresGrad, _device)
            {
                Inputs = new[] { this, other }
            };

            if (requiresGrad)
            {
                var capturedSelf = this;
                var capturedOther = other;

                rawResult.GradFn = gradOutput =>
                {
                    if (capturedSelf.RequiresGrad)
                    {
                        var gradSelf = gradOutput.MatMul(capturedOther.Transpose(new[] { 1, 0 }));
                        capturedSelf.AccumulateGrad(gradSelf);
                    }
                    if (capturedOther.RequiresGrad)
                    {
                        var gradOther = capturedSelf.Transpose(new[] { 1, 0 }).MatMul(gradOutput);
                        capturedOther.AccumulateGrad(gradOther);
                    }
                    return gradOutput;
                };
            }

            return new Tensor(rawResult);
        }
        /// <summary>
        /// Transposes this tensor according to a dimension permutation layout.
        /// </summary>
        /// <param name="perm">An array of integers representing the dimension permutation sequence.</param>
        /// <returns>A transposed representation of the tensor.</returns>
        /// <exception cref="ArgumentException">Thrown if the permutation array size does not match the tensor rank.</exception>

        public ITensor Transpose(int[] perm)
        {
            if (perm == null || perm.Length != _shape.Rank)
                throw new ArgumentException("Permutation rank mismatch.");

            var newShape = perm.Select(p => _shape.Dimensions[p]).ToArray();
            var resultData = new float[_data.Length];
            var indices = new int[_shape.Rank];

            for (int i = 0; i < _data.Length; i++)
            {
                int temp = i;
                for (int d = _shape.Rank - 1; d >= 0; d--)
                {
                    indices[d] = temp % _shape.Dimensions[d];
                    temp /= _shape.Dimensions[d];
                }

                int newIdx = 0, stride = 1;
                for (int d = _shape.Rank - 1; d >= 0; d--)
                {
                    newIdx += indices[perm[d]] * stride;
                    stride *= newShape[d];
                }
                resultData[newIdx] = _data[i];
            }

            var rawResult = new CpuBackend(resultData, new TensorShape(newShape), _requiresGrad, _device);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                var capturedPerm = (int[])perm.Clone();
                rawResult.GradFn = gradOutput =>
                {
                    var invPerm = InvertPerm(capturedPerm);
                    var gradSelf = gradOutput.Transpose(invPerm);
                    capturedSelf.AccumulateGrad(gradSelf);
                    return gradOutput;
                };
            }

            return new Tensor(rawResult);
        }
        /// <summary>
        /// Inverts a dimension permutation sequence.
        /// </summary>
        /// <param name="perm">The dimension permutation layout array to invert.</param>
        /// <returns>The inverted dimension permutation layout array.</returns>

        private static int[] InvertPerm(int[] perm)
        {
            int[] inv = new int[perm.Length];
            for (int i = 0; i < perm.Length; i++) inv[perm[i]] = i;
            return inv;
        }
        /// <summary>
        /// Reshapes this tensor into a target dimensional structure.
        /// </summary>
        /// <param name="newShape">The targeted dimensions.</param>
        /// <returns>A new <see cref="ITensor"/> with the reshaped layout.</returns>
        /// <exception cref="ArgumentException">Thrown if total element count differs between shapes.</exception>

        public ITensor Reshape(params int[] newShape)
        {
            var ns = new TensorShape(newShape);
            if (ns.TotalElements != _shape.TotalElements)
                throw new ArgumentException("Total element mismatch in Reshape.");

            var rawResult = new CpuBackend(_data, ns, _requiresGrad, _device);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                rawResult.GradFn = gradOutput =>
                {
                    var reshapedGrad = gradOutput.Reshape(_shape.Dimensions);
                    capturedSelf.AccumulateGrad(reshapedGrad);
                    return gradOutput;
                };
            }

            return new Tensor(rawResult);
        }
        /// <summary>
        /// Sums all elements of the tensor, optionally across a targeted axis.
        /// </summary>
        /// <param name="axis">The reduction axis index.</param>
        /// <param name="keepDims">Whether to preserve the reduced dimension with size 1.</param>
        /// <returns>The summed <see cref="ITensor"/> output.</returns>

        public ITensor Sum(int? axis = null, bool keepDims = false)
        {
            if (!axis.HasValue)
            {
                var scalarValue = _data.Sum();
                var newShape = keepDims ? new TensorShape(Enumerable.Repeat(1, _shape.Rank).ToArray()) : new TensorShape(1);
                var rawResult = new CpuBackend(new[] { scalarValue }, newShape, _requiresGrad, _device);

                if (_requiresGrad)
                {
                    var capturedSelf = this;
                    rawResult.GradFn = gradOutput =>
                    {
                        var gradSelf = Tensor.Ones(_shape, _device);
                        capturedSelf.AccumulateGrad(gradSelf);
                        return gradOutput;
                    };
                }

                return new Tensor(rawResult);
            }

            return ReduceAlongAxis(axis.Value, false, keepDims);
        }
        /// <summary>
        /// Sums elements across multiple dimensions.
        /// </summary>
        /// <param name="axes">An array of target reduction dimensions.</param>
        /// <param name="keepDims">Whether to preserve reduced dimensions.</param>
        /// <returns>A summed multidimensional <see cref="ITensor"/>.</returns>

        public ITensor Sum(int[] axes, bool keepDims = false)
        {
            if (axes == null || axes.Length == 0) return Sum((int?)null, keepDims);
            int rank = _shape.Rank;
            var normalizedAxes = axes.Select(a => a < 0 ? a + rank : a).Distinct().ToList();
            normalizedAxes.Sort((a, b) => b.CompareTo(a));

            ITensor result = this;
            foreach (int axis in normalizedAxes)
                result = result.Sum(axis, keepDims);

            return result;
        }
        /// <summary>
        /// Computes the arithmetic mean value of the elements, optionally across a targeted axis.
        /// </summary>
        /// <param name="axis">The reduction axis index.</param>
        /// <param name="keepDims">Whether to preserve the reduced dimension with size 1.</param>
        /// <returns>The mean value <see cref="ITensor"/>.</returns>

        public ITensor Mean(int? axis = null, bool keepDims = false)
        {
            if (!axis.HasValue)
            {
                var scalarValue = _data.Average();
                var newShape = keepDims ? new TensorShape(Enumerable.Repeat(1, _shape.Rank).ToArray()) : new TensorShape(1);
                var rawResult = new CpuBackend(new[] { scalarValue }, newShape, _requiresGrad, _device);

                if (_requiresGrad)
                {
                    var capturedSelf = this;
                    rawResult.GradFn = gradOutput =>
                    {
                        var gradSelf = Tensor.Ones(_shape, _device).Divide(_data.Length);
                        capturedSelf.AccumulateGrad(gradSelf);
                        return gradOutput;
                    };
                }

                return new Tensor(rawResult);
            }
            return ReduceAlongAxis(axis.Value, true, keepDims);
        }
        /// <summary>
        /// Computes the arithmetic mean values across multiple specified dimensions.
        /// </summary>
        /// <param name="axes">An array of target reduction dimensions.</param>
        /// <param name="keepDims">Whether to preserve reduced dimensions.</param>
        /// <returns>The mean multidimensional <see cref="ITensor"/>.</returns>

        public ITensor Mean(int[] axes, bool keepDims = false)
        {
            if (axes == null || axes.Length == 0) return Mean((int?)null, keepDims);
            int rank = _shape.Rank;
            var normalizedAxes = axes.Select(a => a < 0 ? a + rank : a).Distinct().ToList();
            normalizedAxes.Sort((a, b) => b.CompareTo(a));

            ITensor result = this;
            foreach (int axis in normalizedAxes)
                result = result.Mean(axis, keepDims);

            return result;
        }
        /// <summary>
        /// Collapses and reduces the tensor dimensions along a target axis utilizing summation or averaging logic.
        /// </summary>
        /// <param name="axis">The targeted dimension axis index to reduce.</param>
        /// <param name="isMean">True to calculate the arithmetic mean; false to calculate the sum.</param>
        /// <param name="keepDims">True to retain the reduced dimension with size 1; false to squeeze it.</param>
        /// <returns>A new reduced <see cref="ITensor"/>.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if the specified axis is out of the dimensions range.</exception>

        private ITensor ReduceAlongAxis(int axis, bool isMean, bool keepDims = false)
        {
            if (axis < 0) axis = _shape.Rank + axis;
            if (axis < 0 || axis >= _shape.Rank) throw new ArgumentOutOfRangeException(nameof(axis));

            var dims = _shape.Dimensions;
            int reducedSize = dims[axis];
            int outer = 1; for (int i = 0; i < axis; i++) outer *= dims[i];
            int inner = 1; for (int i = axis + 1; i < dims.Length; i++) inner *= dims[i];

            int[] outDims = keepDims
                ? dims.Select((d, i) => i == axis ? 1 : d).ToArray()
                : dims.Where((_, i) => i != axis).ToArray();

            var output = new float[outer * inner];

            for (int o = 0; o < outer; o++)
                for (int i = 0; i < inner; i++)
                {
                    float acc = 0f;
                    int baseIdx = o * reducedSize * inner + i;
                    for (int r = 0; r < reducedSize; r++)
                        acc += _data[baseIdx + r * inner];
                    output[o * inner + i] = isMean ? acc / reducedSize : acc;
                }

            var rawResult = new CpuBackend(output, new TensorShape(outDims), _requiresGrad, _device);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                rawResult.GradFn = grad =>
                {
                    var expanded = grad.BroadcastTo(_shape);
                    var finalGrad = isMean ? expanded.Divide(reducedSize) : expanded;
                    capturedSelf.AccumulateGrad(finalGrad);
                    return finalGrad;
                };
            }

            return new Tensor(rawResult);
        }
        /// <summary>
        /// Finds the maximum elements along a specified tensor axis.
        /// </summary>
        /// <param name="axis">The axis dimension to scan.</param>
        /// <param name="keepDims">Whether to preserve the reduced dimension size of 1.</param>
        /// <returns>An <see cref="ITensor"/> containing the maximum values.</returns>

        public ITensor Max(int axis = -1, bool keepDims = false) => ReduceAlongAxis(axis < 0 ? _shape.Rank - 1 : axis, false, keepDims);
        /// <summary>
        /// Finds the minimum elements along a specified tensor axis.
        /// </summary>
        /// <param name="axis">The axis dimension to scan.</param>
        /// <param name="keepDims">Whether to preserve the reduced dimension size of 1.</param>
        /// <returns>An <see cref="ITensor"/> containing the minimum values.</returns>

        public ITensor Min(int axis = -1, bool keepDims = false) => ReduceAlongAxis(axis < 0 ? _shape.Rank - 1 : axis, false, keepDims);
        /// <summary>
        /// Performs an element-wise logical NOT operation (returns 1 for elements that equal 0; otherwise 0).
        /// </summary>
        /// <returns>A new logical boolean indicator <see cref="ITensor"/>.</returns>

        public ITensor LogicalNot()
        {
            var resultData = new float[_data.Length];
            for (int i = 0; i < _data.Length; i++)
            {
                resultData[i] = _data[i] == 0f ? 1f : 0f;
            }
            return new Tensor(new CpuBackend(resultData, _shape.Clone(), false, _device));
        }
        /// <summary>
        /// Clips tensor values to fall between specified minimum and maximum scalar bounds.
        /// </summary>
        /// <param name="v1">The first boundary limit.</param>
        /// <param name="v2">The second boundary limit.</param>
        /// <returns>A clipped copy of the original <see cref="ITensor"/>.</returns>

        public ITensor Clip(float v1, float v2)
        {
            if (v1 > v2) (v1, v2) = (v2, v1);
            var resultData = new float[_data.Length];
            for (int i = 0; i < _data.Length; i++)
            {
                float x = _data[i];
                resultData[i] = x < v1 ? v1 : (x > v2 ? v2 : x);
            }
            var rawResult = new CpuBackend(resultData, _shape.Clone(), _requiresGrad, _device);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                rawResult.GradFn = gradOutput =>
                {
                    var mask = new float[capturedSelf._data.Length];
                    for (int i = 0; i < capturedSelf._data.Length; i++)
                    {
                        float x = capturedSelf._data[i];
                        mask[i] = (x >= v1 && x <= v2) ? 1f : 0f;
                    }
                    var finalGrad = gradOutput.Multiply(new Tensor(new CpuBackend(mask, _shape.Clone(), false, _device)));
                    capturedSelf.AccumulateGrad(finalGrad);
                    return finalGrad;
                };
            }
            return new Tensor(rawResult);
        }
        /// <summary>
        /// Performs a generic element-wise comparison between two tensors.
        /// </summary>
        /// <param name="other">The other tensor operand to compare against.</param>
        /// <param name="cmp">The comparison delegate taking two floats and returning 1f for true or 0f for false.</param>
        /// <returns>A binary indicator <see cref="ITensor"/> where elements are 1f or 0f.</returns>

        private ITensor Comparison(ITensor other, Func<float, float, float> cmp)
        {
            var left = Unwrap(this);
            var right = Unwrap(other);

            var (resultShape, strideA, strideB) = GetBroadcastShapeAndStrides(left, right);
            var resultData = new float[resultShape.TotalElements];

            for (int i = 0; i < resultData.Length; i++)
            {
                int idxA = GetBroadcastIndex(i, strideA, resultShape.Dimensions);
                int idxB = GetBroadcastIndex(i, strideB, resultShape.Dimensions);
                resultData[i] = cmp(left._data[idxA % left._data.Length], right._data[idxB % right._data.Length]);
            }
            return new Tensor(new CpuBackend(resultData, resultShape, false, _device));
        }
        /// <summary>
        /// Tests element-wise equality between this tensor and another tensor.
        /// </summary>
        /// <param name="other">The comparison target tensor.</param>
        /// <returns>A binary indicator <see cref="ITensor"/>.</returns>

        public ITensor Equal(ITensor other) => Comparison(other, (a, b) => Math.Abs(a - b) < 1e-6f ? 1f : 0f);
        /// <summary>
        /// Tests element-wise greater than condition against another tensor.
        /// </summary>
        /// <param name="other">The comparison target tensor.</param>
        /// <returns>A binary indicator <see cref="ITensor"/>.</returns>

        public ITensor GreaterThan(ITensor other) => Comparison(other, (a, b) => a > b ? 1f : 0f);
        /// <summary>
        /// Tests element-wise greater than or equal to condition against another tensor.
        /// </summary>
        /// <param name="other">The comparison target tensor.</param>
        /// <returns>A binary indicator <see cref="ITensor"/>.</returns>

        public ITensor GreaterThanOrEqual(ITensor other) => Comparison(other, (a, b) => a >= b ? 1f : 0f);
        /// <summary>
        /// Tests element-wise less than or equal to condition against another tensor.
        /// </summary>
        /// <param name="other">The comparison target tensor.</param>
        /// <returns>A binary indicator <see cref="ITensor"/>.</returns>

        public ITensor LessEqual(ITensor other) => Comparison(other, (a, b) => a <= b ? 1f : 0f);
        /// <summary>
        /// Selects elements from trueValue or falseValue depending on evaluation condition of condition tensor.
        /// </summary>
        /// <param name="condition">The selection binary mask tensor.</param>
        /// <param name="trueValue">The values selected where condition evaluated to 1.</param>
        /// <param name="falseValue">The values selected where condition evaluated to 0.</param>
        /// <returns>A merged conditional output <see cref="ITensor"/>.</returns>

        public ITensor Where(ITensor condition, ITensor trueValue, ITensor falseValue)
        {
            var c = Unwrap(condition);
            var tv = Unwrap(trueValue);
            var fv = Unwrap(falseValue);

            var (resultShape, strideA, strideB) = GetBroadcastShapeAndStrides(this, condition);
            var resultData = new float[resultShape.TotalElements];

            for (int i = 0; i < resultData.Length; i++)
            {
                int idx = GetBroadcastIndex(i, strideA, resultShape.Dimensions);
                resultData[i] = c._data[idx % c._data.Length] > 0
                    ? tv._data[idx % tv._data.Length]
                    : fv._data[idx % fv._data.Length];
            }
            return new Tensor(new CpuBackend(resultData, resultShape, false, _device));
        }
        /// <summary>
        /// Applies hyperbolic tangent activation function to every element.
        /// </summary>
        /// <returns>An activated <see cref="ITensor"/>.</returns>

        public ITensor Tanh() => new Tanh().Forward(this);
        /// <summary>
        /// Applies Rectified Linear Unit activation function to every element.
        /// </summary>
        /// <returns>An activated <see cref="ITensor"/>.</returns>

        public ITensor Relu() => new ReLU().Forward(this);
        /// <summary>
        /// Applies standard sigmoid activation function to every element.
        /// </summary>
        /// <returns>An activated <see cref="ITensor"/>.</returns>

        public ITensor Sigmoid() => new Sigmoid().Forward(this);
        /// <summary>
        /// Applies softmax normalization activation across the targeted dimension axis.
        /// </summary>
        /// <param name="axis">The dimension along which normalization is calculated.</param>
        /// <returns>A normalized probability distribution <see cref="ITensor"/>.</returns>

        public ITensor Softmax(int axis = -1) => new Softmax(axis).Forward(this);
        /// <summary>
        /// Slices elements out of multidimensional space utilizing start, end and step constraints.
        /// </summary>
        /// <param name="slices">A parameters tuple array indicating start, end and step for each dimension.</param>
        /// <returns>A sliced <see cref="ITensor"/> subset.</returns>
        /// <exception cref="ArgumentException">Thrown if input slices length does not match tensor rank.</exception>

        public ITensor Slice(params (int start, int end, int step)[] slices)
        {
            if (slices.Length != _shape.Rank) throw new ArgumentException("Slice rank mismatch.");

            var starts = new int[_shape.Rank];
            var ends = new int[_shape.Rank];
            var steps = new int[_shape.Rank];
            var newShapeList = new List<int>();

            for (int i = 0; i < _shape.Rank; i++)
            {
                starts[i] = slices[i].start;
                ends[i] = slices[i].end == -1 ? _shape.Dimensions[i] : slices[i].end;
                steps[i] = slices[i].step == 0 ? 1 : slices[i].step;
                newShapeList.Add(((ends[i] - starts[i] - 1) / steps[i]) + 1);
            }

            int total = newShapeList.Aggregate(1, (a, b) => a * b);
            var result = new float[total];
            int resultIdx = 0;

            void Recurse(int dim, int flatIdx)
            {
                if (dim == _shape.Rank)
                {
                    result[resultIdx++] = _data[flatIdx];
                    return;
                }
                for (int i = starts[dim]; i < ends[dim]; i += steps[dim])
                    Recurse(dim + 1, flatIdx + i * GetStride(dim));
            }

            Recurse(0, 0);
            var output = new CpuBackend(result, new TensorShape(newShapeList.ToArray()), _requiresGrad, _device)
            {
                Inputs = new[] { this }
            };

            if (_requiresGrad)
            {
                var capturedStarts = (int[])starts.Clone();
                var capturedSteps = (int[])steps.Clone();
                var originalShape = _shape;
                var newShape = newShapeList.ToArray();
                var capturedSelf = this;

                output.GradFn = grad =>
                {
                    var scattered = new CpuBackend(originalShape, false, _device);
                    int gradIdx = 0;
                    float[] gradData = grad.ToArray();

                    void ScatterRecurse(int dim, int flatDst)
                    {
                        if (dim == originalShape.Rank)
                        {
                            scattered._data[flatDst] = gradData[gradIdx++];
                            return;
                        }
                        for (int i = capturedStarts[dim]; i < capturedStarts[dim] + newShape[dim] * capturedSteps[dim]; i += capturedSteps[dim])
                            ScatterRecurse(dim + 1, flatDst + i * GetStride(dim));
                    }

                    ScatterRecurse(0, 0);
                    capturedSelf.AccumulateGrad(scattered);
                    return scattered;
                };
            }
            return new Tensor(output);
        }
        /// <summary>
        /// Computes layout stride offset size for a given dimension index.
        /// </summary>
        /// <param name="dim">The targeted dimension index.</param>
        /// <returns>The stride offset value for the specified dimension.</returns>

        private int GetStride(int dim)
        {
            int stride = 1;
            for (int i = dim + 1; i < _shape.Rank; i++) stride *= _shape.Dimensions[i];
            return stride;
        }
        /// <summary>
        /// Concatenates this tensor with other CPU-based tensors along a specified axis.
        /// </summary>
        /// <param name="others">An enumerable list of tensors to join.</param>
        /// <param name="axis">The structural dimension axis along which data will be concatenated.</param>
        /// <returns>A unified, concatenated <see cref="ITensor"/>.</returns>
        /// <exception cref="ArgumentNullException">Thrown if others is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if the concatenation axis is out of bounds.</exception>
        /// <exception cref="ArgumentException">Thrown if tensor dimensions do not match except along the concatenation axis.</exception>

        public ITensor Concat(IEnumerable<ITensor> others, int axis = 0)
        {
            if (others == null) throw new ArgumentNullException(nameof(others));

            var all = new List<CpuBackend> { this };
            foreach (var o in others)
            {
                all.Add(Unwrap(o));
            }

            int rank = _shape.Rank;
            int actualAxis = axis < 0 ? rank + axis : axis;
            if (actualAxis < 0 || actualAxis >= rank)
                throw new ArgumentOutOfRangeException(nameof(axis), "Concatenation axis is out of range.");

            // Verify shapes match on all dimensions except the concatenation axis
            for (int i = 0; i < rank; i++)
            {
                if (i == actualAxis) continue;
                foreach (var t in all)
                {
                    if (t.Shape.Rank != rank || t.Shape[i] != _shape[i])
                        throw new ArgumentException($"All tensors must have matching dimensions except along the concatenation axis. Mismatch found on axis {i}.");
                }
            }

            // Calculate output shape dimensions
            int[] newDims = _shape.Dimensions.ToArray();
            newDims[actualAxis] = all.Sum(t => t.Shape[actualAxis]);
            var outShape = new TensorShape(newDims);

            float[] resultData = new float[outShape.TotalElements];

            // Compute dimensions before, on, and after the target concatenation axis
            int outerSize = 1;
            for (int i = 0; i < actualAxis; i++) outerSize *= newDims[i];

            int innerSize = 1;
            for (int i = actualAxis + 1; i < rank; i++) innerSize *= newDims[i];

            int outAxisStride = innerSize;
            int outOuterStride = newDims[actualAxis] * innerSize;

            // Copy slices from source tensors into the target layout
            for (int o = 0; o < outerSize; o++)
            {
                int currentAxisOffset = 0;
                foreach (var t in all)
                {
                    int tAxisSize = t.Shape[actualAxis];
                    float[] tData = t.ToArray(); // Safe array copy (respects ArrayPool bounds)

                    int tAxisStride = innerSize;
                    int tOuterStride = tAxisSize * innerSize;

                    for (int a = 0; a < tAxisSize; a++)
                    {
                        int srcOffset = o * tOuterStride + a * tAxisStride;
                        int dstOffset = o * outOuterStride + (currentAxisOffset + a) * outAxisStride;
                        Array.Copy(tData, srcOffset, resultData, dstOffset, innerSize);
                    }
                    currentAxisOffset += tAxisSize;
                }
            }

            var rawResult = new CpuBackend(resultData, outShape, _requiresGrad || all.Any(t => t.RequiresGrad), _device);

            // Register autograd backward pass for Concatenation
            if (rawResult.RequiresGrad)
            {
                var capturedAll = all.ToList();
                int capturedAxis = actualAxis;

                rawResult.GradFn = gradOutput =>
                {
                    float[] goData = gradOutput.ToArray();
                    int currentAxisOffset = 0;

                    foreach (var t in capturedAll)
                    {
                        int tAxisSize = t.Shape[capturedAxis];
                        if (!t.RequiresGrad)
                        {
                            currentAxisOffset += tAxisSize;
                            continue;
                        }

                        float[] giData = new float[t.Shape.TotalElements];
                        int tAxisStride = innerSize;
                        int tOuterStride = tAxisSize * innerSize;

                        for (int o = 0; o < outerSize; o++)
                        {
                            for (int a = 0; a < tAxisSize; a++)
                            {
                                int srcOffset = o * outOuterStride + (currentAxisOffset + a) * outAxisStride;
                                int dstOffset = o * tOuterStride + a * tAxisStride;
                                Array.Copy(goData, srcOffset, giData, dstOffset, innerSize);
                            }
                        }

                        var gradInput = new CpuBackend(giData, t.Shape, false, t._device);
                        t.AccumulateGrad(gradInput);
                        currentAxisOffset += tAxisSize;
                    }

                    return gradOutput;
                };
            }

            return new Tensor(rawResult);
        }
        /// <summary>
        /// Unwraps high level wrapper class variants to expose their core inner <see cref="CpuBackend"/> implementation.
        /// </summary>
        /// <param name="tensor">The high level interface wrapper tensor instance.</param>
        /// <returns>An extracted underlying <see cref="CpuBackend"/> instance.</returns>

        public static CpuBackend Unwrap(ITensor tensor)
        {
            ITensor current = tensor;
            while (true)
            {
                if (current is Tensor t) { current = t._backend; continue; }
                if (current is Variable v) { current = v._inner; continue; }
                break;
            }
            return (CpuBackend)current;
        }
        /// <summary>
        /// Triggers autograd backward pass starting from this node, calculating and accumulating gradients of parameters.
        /// </summary>
        /// <param name="gradient">The incoming backpropagated gradient tensor (defaults to a scalar 1.0 tensor if null).</param>

        public void Backward(ITensor? gradient = null)
        {
            ArborNet.Core.Autograd.AutogradEngine.Backward(this, gradient);
        }
        /// <summary>
        /// Clears the accumulated gradients and autograd tracking states for this node in the graph.
        /// </summary>

        public void ClearGrad()
        {
            _grad = null;
            _gradFn = null;
        }
        /// <summary>
        /// Creates a zero-initialized tensor.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="device">The targeted device (defaults to CPU).</param>
        /// <returns>A zero-filled <see cref="ITensor"/>.</returns>

        public static ITensor Zeros(TensorShape shape, Device? device = null) => new CpuBackend(shape, false, device);
        /// <summary>
        /// Creates a one-initialized tensor.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="device">The targeted device (defaults to CPU).</param>
        /// <returns>A one-filled <see cref="ITensor"/>.</returns>

        public static ITensor Ones(TensorShape shape, Device? device = null)
        {
            var data = new float[shape.TotalElements];
            Array.Fill(data, 1f);
            return new CpuBackend(data, shape, false, device);
        }
        /// <summary>
        /// Creates a 1D, single element scalar tensor wrapper containing a specified float value.
        /// </summary>
        /// <param name="value">The scalar numeric element.</param>
        /// <param name="device">The targeted device (defaults to CPU).</param>
        /// <returns>A scalar <see cref="ITensor"/>.</returns>

        public static ITensor FromScalar(float value, Device? device = null) => new CpuBackend(new[] { value }, new TensorShape(1), false, device);
        /// <summary>
        /// Creates a CPU tensor by wrapping an existing flat array of floats.
        /// </summary>
        /// <param name="data">The preallocated float elements array.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="device">The targeted device (defaults to CPU).</param>
        /// <returns>A new <see cref="ITensor"/>.</returns>

        public static ITensor FromArray(float[] data, TensorShape shape, Device? device = null) => new CpuBackend(data, shape, false, device);
        /// <summary>
        /// Creates a random tensor initialized with uniform distribution values between [0, 1).
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="device">The targeted device (defaults to CPU).</param>
        /// <returns>A random uniform <see cref="ITensor"/>.</returns>

        public static ITensor Rand(TensorShape shape, Device? device = null)
        {
            var r = new Random();
            return new CpuBackend(Enumerable.Range(0, shape.TotalElements).Select(_ => (float)r.NextDouble()).ToArray(), shape, false, device);
        }
        /// <summary>
        /// Creates a random tensor initialized with values between [-1, 1).
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="device">The targeted device (defaults to CPU).</param>
        /// <returns>A random <see cref="ITensor"/>.</returns>

        public static ITensor Randn(TensorShape shape, Device? device = null)
        {
            var r = new Random();
            return new CpuBackend(Enumerable.Range(0, shape.TotalElements).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray(), shape, false, device);
        }
        /// <summary>
        /// Creates a 2D identity matrix tensor containing 1s along the diagonal.
        /// </summary>
        /// <param name="size">The dimensional width and height of the matrix.</param>
        /// <param name="device">The targeted device (defaults to CPU).</param>
        /// <returns>An identity matrix <see cref="ITensor"/>.</returns>

        public static ITensor Eye(int size, Device? device = null)
        {
            var data = new float[size * size];
            for (int i = 0; i < size; i++) data[i * size + i] = 1f;
            return new CpuBackend(data, new TensorShape(size, size), false, device);
        }
        /// <summary>
        /// Finds index coordinates for the minimum values encountered along a target axis.
        /// </summary>
        /// <param name="axis">The dimension axis to scan.</param>
        /// <returns>An index-based <see cref="ITensor"/>.</returns>

        public ITensor ArgMin(int axis) => ReduceIndices(axis, false);
        /// <summary>
        /// Finds index coordinates for the maximum values encountered along a target axis.
        /// </summary>
        /// <param name="axis">The dimension axis to scan.</param>
        /// <returns>An index-based <see cref="ITensor"/>.</returns>

        public ITensor ArgMax(int axis) => ReduceIndices(axis, true);
        /// <summary>
        /// Helper reduction utility finding index locations of minimum or maximum values along an axis.
        /// </summary>
        /// <param name="axis">The targeted dimension axis index.</param>
        /// <param name="findMax">True to locate the maximum value index; false to locate the minimum value index.</param>
        /// <returns>An index-based <see cref="ITensor"/> containing the extreme index coordinates.</returns>

        private ITensor ReduceIndices(int axis, bool findMax)
        {
            if (axis < 0) axis = _shape.Rank + axis;
            var dims = _shape.Dimensions;
            int axisSize = dims[axis];
            int outer = 1; for (int i = 0; i < axis; i++) outer *= dims[i];
            int inner = 1; for (int i = axis + 1; i < dims.Length; i++) inner *= dims[i];

            var output = new float[outer * inner];
            for (int o = 0; o < outer; o++)
                for (int i = 0; i < inner; i++)
                {
                    int baseIdx = o * axisSize * inner + i;
                    float targetVal = _data[baseIdx];
                    int targetIdx = 0;
                    for (int r = 1; r < axisSize; r++)
                    {
                        float val = _data[baseIdx + r * inner];
                        if ((findMax && val > targetVal) || (!findMax && val < targetVal))
                        {
                            targetVal = val;
                            targetIdx = r;
                        }
                    }
                    output[o * inner + i] = targetIdx;
                }
            return new Tensor(new CpuBackend(output, new TensorShape(dims.Where((_, idx) => idx != axis).ToArray()), false, _device));
        }
        /// <summary>
        /// Calculates the cumulative sum of elements along a target dimensional axis.
        /// </summary>
        /// <param name="axis">The dimensional axis along which elements are accumulated.</param>
        /// <returns>A cumulative sum <see cref="ITensor"/>.</returns>

        public ITensor CumSum(int axis)
        {
            if (axis < 0) axis = _shape.Rank + axis;
            var dims = _shape.Dimensions;
            int axisSize = dims[axis];
            int outer = 1; for (int i = 0; i < axis; i++) outer *= dims[i];
            int inner = 1; for (int i = axis + 1; i < dims.Length; i++) inner *= dims[i];

            float[] output = new float[_data.Length];
            for (int o = 0; o < outer; o++)
                for (int i = 0; i < inner; i++)
                {
                    int baseIdx = o * axisSize * inner + i;
                    float runningSum = 0;
                    for (int r = 0; r < axisSize; r++)
                    {
                        int currentIdx = baseIdx + r * inner;
                        runningSum += _data[currentIdx];
                        output[currentIdx] = runningSum;
                    }
                }

            var result = new CpuBackend(output, _shape.Clone(), _requiresGrad, _device);
            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = gradOutput =>
                {
                    float[] goData = gradOutput.ToArray();
                    float[] giData = new float[goData.Length];
                    for (int o = 0; o < outer; o++)
                        for (int i = 0; i < inner; i++)
                        {
                            int baseIdx = o * axisSize * inner + i;
                            float runningGradSum = 0;
                            for (int r = axisSize - 1; r >= 0; r--)
                            {
                                int currentIdx = baseIdx + r * inner;
                                runningGradSum += goData[currentIdx];
                                giData[currentIdx] = runningGradSum;
                            }
                        }
                    var gradInput = new CpuBackend(giData, _shape.Clone(), false, _device);
                    capturedSelf.AccumulateGrad(gradInput);
                    return gradOutput;
                };
            }
            return new Tensor(result);
        }
        /// <summary>
        /// Disposes of the tensor, returning memory back to the shared array pool if it was rented.
        /// </summary>

        public void Dispose()
        {
            if (_isRented && _data != null)
            {
                ArrayPool<float>.Shared.Return(_data);
                _data = null!;
                _isRented = false;
            }
        }

        /// <summary>
        /// Finalizes an instance of the <see cref="CpuBackend"/> class, ensuring rented resources are released.
        /// </summary>
        ~CpuBackend() => Dispose();
    }
}