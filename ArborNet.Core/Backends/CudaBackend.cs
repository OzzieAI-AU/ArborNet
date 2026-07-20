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
    using ArborNet.Core;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Native;
    using ArborNet.Core.Native.PInvoke;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Runtime.InteropServices;
    using static ArborNet.Core.Native.PInvoke.CUDA;
    /// <summary>
    /// Manages an active block of memory allocated on a CUDA device.
    /// Utilizes a reference counting mechanism to safely share allocations across multiple tensor views or slices.
    /// </summary>

    #endregion


    internal sealed class CudaAllocation : IDisposable
    {
        private IntPtr _ptr;
        private readonly ulong _bytes;
        private int _refCount;
        private readonly object _lock = new();
        /// <summary>
        /// Gets the raw pointer pointing to the allocated memory block on the CUDA device.
        /// </summary>

        public IntPtr Ptr => _ptr;

        public CudaAllocation(ulong bytes)
        {
            _bytes = bytes;
            _ptr = CudaMemoryPool.Instance.Allocate(bytes);
            CudaMemset(_ptr, 0, bytes);
            _refCount = 1;
            GC.AddMemoryPressure((long)bytes);
        }
        /// <summary>
        /// Increments the reference count of this memory allocation in a thread-safe manner.
        /// </summary>

        public void AddRef()
        {
            lock (_lock)
            {
                _refCount++;
            }
        }
        /// <summary>
        /// Decrements the reference count of this memory allocation.
        /// If the reference count drops to zero, the memory is returned to the CUDA pool and GC memory pressure is removed.
        /// </summary>

        public void Release()
        {
            lock (_lock)
            {
                _refCount--;
                if (_refCount == 0)
                {
                    if (_ptr != IntPtr.Zero)
                    {
                        CudaMemoryPool.Instance.Free(_ptr, _bytes);
                        GC.RemoveMemoryPressure((long)_bytes);
                        _ptr = IntPtr.Zero;
                    }
                }
            }
        }
        /// <summary>
        /// Disposes of the allocation by releasing the current reference and suppressing finalization.
        /// </summary>

        public void Dispose()
        {
            Release();
            GC.SuppressFinalize(this);
        }

        ~CudaAllocation()
        {
            Release();
        }
    }
    /// <summary>
    /// Implements a high-performance tensor backed by CUDA device memory.
    /// Supports autograd backpropagation, lazy evaluation, and element-wise/matrix-multiplication operations.
    /// </summary>

    public sealed class CudaBackend : ITensor, IDisposable
    {
        private readonly CudaAllocation _allocation;
        private TensorShape _shape;
        private readonly Device _device;
        private bool _requiresGrad;
        private ITensor? _grad;
        private Func<ITensor, ITensor>? _gradFn;
        private bool _disposed;
        private readonly object _lock = new();
        private ITensor[] _inputs = Array.Empty<ITensor>();
        /// <summary>
        /// Gets or sets the input tensors that generated this tensor.
        /// Primarily used by the autograd engine to trace backward dependencies.
        /// </summary>

        public ITensor[] Inputs
        {
            get => _inputs;
            set => _inputs = value ?? Array.Empty<ITensor>();
        }
        /// <summary>
        /// Gets the structural shape of the tensor.
        /// </summary>

        public TensorShape Shape => _shape;
        /// <summary>
        /// Gets the hardware device on which this tensor's data resides.
        /// </summary>
        public Device Device => _device;
        /// <summary>
        /// Gets or sets a value indicating whether this tensor tracks operations to compute gradients during backpropagation.
        /// </summary>
        public bool RequiresGrad { get => _requiresGrad; set => _requiresGrad = value; }
        /// <summary>
        /// Gets or sets the accumulated gradient for this tensor.
        /// </summary>
        public ITensor? Grad { get => _grad; set => _grad = value; }
        /// <summary>
        /// Gets or sets the backward step function used to calculate gradients for the input variables of this tensor.
        /// </summary>
        public Func<ITensor, ITensor>? GradFn { get => _gradFn; set => _gradFn = value; }
        /// <summary>
        /// Gets the tensor's underlying data downloaded to the host CPU as a flat array of single-precision floating point numbers.
        /// </summary>
        public float[] Data => ToArray();

        public CudaBackend(TensorShape shape, bool requiresGrad = false, Device? device = null)
        {
            _shape = shape?.Clone() ?? throw new ArgumentNullException(nameof(shape));
            _device = device ?? Device.CUDA;
            _requiresGrad = requiresGrad;

            ulong bytes = (ulong)_shape.TotalElements * sizeof(float);
            _allocation = new CudaAllocation(bytes);
        }

        public CudaBackend(float[] hostData, TensorShape shape, bool requiresGrad = false, Device? device = null)
        {
            _shape = shape?.Clone() ?? throw new ArgumentNullException(nameof(shape));
            _device = device ?? Device.CUDA;
            _requiresGrad = requiresGrad;

            ulong bytes = (ulong)_shape.TotalElements * sizeof(float);
            _allocation = new CudaAllocation(bytes);
            CopyHostToDevice(hostData, _allocation.Ptr, _shape.TotalElements);
        }

        private CudaBackend(TensorShape shape, CudaAllocation allocation, bool requiresGrad, Device device)
        {
            _shape = shape.Clone();
            _allocation = allocation;
            _allocation.AddRef();
            _requiresGrad = requiresGrad;
            _device = device;
        }
        /// <summary>
        /// Accumulates gradients to this tensor. Performs automatic on-device dimension reduction
        /// and broadcasting matching if the incoming gradient shape differs from this tensor's shape.
        /// </summary>
        /// <param name="delta">The incoming gradient to accumulate.</param>

        public void AccumulateGrad(ITensor delta)
        {
            if (delta == null) return;

            ITensor reduced = delta;
            if (!delta.Shape.Equals(_shape))
            {
                // ELIMINATE host-transfer roundtrip: Execute shape reduction directly on-device
                reduced = new CudaBackend(_shape, false, _device);

                // Dynamically compute axes to reduce
                var deltaUnwrapped = Tensor.Unwrap(delta) as CudaBackend;
                int rankDiff = delta.Shape.Rank - _shape.Rank;

                if (rankDiff > 0)
                {
                    // Perform dimension collapsing using native GPU sum-reduction
                    int[] axesToReduce = Enumerable.Range(0, rankDiff).ToArray();
                    reduced = delta.Sum(axesToReduce, keepDims: false);
                }
                else
                {
                    // Reduce mismatched broadcast dimensions in place on-device
                    for (int i = 0; i < _shape.Rank; i++)
                    {
                        if (_shape[i] == 1 && delta.Shape[i] > 1)
                        {
                            reduced = reduced.Sum(i, keepDims: true);
                        }
                    }
                }
            }

            lock (_lock)
            {
                if (_grad == null)
                {
                    _grad = reduced.Clone();
                }
                else
                {
                    _grad.AddInPlace(reduced);
                }
            }
        }
        /// <summary>
        /// Performs an in-place addition of another CUDA tensor to this tensor.
        /// </summary>
        /// <param name="other">The tensor to add.</param>
        /// <exception cref="InvalidOperationException">Thrown when the other tensor is not allocated on CUDA.</exception>

        public void AddInPlace(ITensor other)
        {
            if (other is not CudaBackend o) throw new InvalidOperationException("Operand must reside on CUDA.");
            lock (_lock)
            {
                NativeAdd(_allocation.Ptr, o._allocation.Ptr, _allocation.Ptr, _shape.TotalElements);
            }
        }
        /// <summary>
        /// Performs an in-place addition of a scalar value to all elements of this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to add.</param>

        public void AddInPlace(float scalar)
        {
            using var scalarTensor = (CudaBackend)FromScalar(scalar, _device);
            lock (_lock)
            {
                NativeAdd(_allocation.Ptr, scalarTensor._allocation.Ptr, _allocation.Ptr, _shape.TotalElements);
            }
        }
        /// <summary>
        /// Performs an in-place subtraction of another CUDA tensor from this tensor.
        /// </summary>
        /// <param name="other">The tensor to subtract.</param>
        /// <exception cref="InvalidOperationException">Thrown when the other tensor is not allocated on CUDA.</exception>

        public void SubtractInPlace(ITensor other)
        {
            if (other is not CudaBackend o) throw new InvalidOperationException("Operand must reside on CUDA.");
            lock (_lock)
            {
                NativeSubtract(_allocation.Ptr, o._allocation.Ptr, _allocation.Ptr, _shape.TotalElements);
            }
        }
        /// <summary>
        /// Performs an in-place subtraction of a scalar value from all elements of this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to subtract.</param>

        public void SubtractInPlace(float scalar)
        {
            using var scalarTensor = (CudaBackend)FromScalar(scalar, _device);
            lock (_lock)
            {
                NativeSubtract(_allocation.Ptr, scalarTensor._allocation.Ptr, _allocation.Ptr, _shape.TotalElements);
            }
        }
        /// <summary>
        /// Performs an in-place multiplication of this tensor by another CUDA tensor.
        /// </summary>
        /// <param name="other">The tensor to multiply by.</param>
        /// <exception cref="InvalidOperationException">Thrown when the other tensor is not allocated on CUDA.</exception>

        public void MultiplyInPlace(ITensor other)
        {
            if (other is not CudaBackend o) throw new InvalidOperationException("Operand must reside on CUDA.");
            lock (_lock)
            {
                NativeMultiply(_allocation.Ptr, o._allocation.Ptr, _allocation.Ptr, _shape.TotalElements);
            }
        }
        /// <summary>
        /// Performs an in-place multiplication of this tensor by a scalar value.
        /// </summary>
        /// <param name="scalar">The scalar value to multiply by.</param>

        public void MultiplyInPlace(float scalar)
        {
            using var scalarTensor = (CudaBackend)FromScalar(scalar, _device);
            lock (_lock)
            {
                NativeMultiply(_allocation.Ptr, scalarTensor._allocation.Ptr, _allocation.Ptr, _shape.TotalElements);
            }
        }
        /// <summary>
        /// Gathers values along a specified axis using an index mapping tensor.
        /// </summary>
        /// <param name="axis">The axis along which to gather. Currently optimized for axis=1.</param>
        /// <param name="indices">The indexing tensor on the CUDA GPU.</param>
        /// <returns>A new <see cref="ITensor"/> containing the gathered elements.</returns>
        /// <exception cref="InvalidOperationException">Thrown when indices do not reside on CUDA GPU.</exception>
        /// <exception cref="NotSupportedException">Thrown when the operation is attempted on non-2D tensors or along axes other than 1.</exception>

        public ITensor Gather(int axis, ITensor indices)
        {
            var o = Tensor.Unwrap(indices) as CudaBackend;
            if (o == null)
                throw new InvalidOperationException("Indices must reside on CUDA GPU.");

            if (axis != 1 || _shape.Rank != 2)
                throw new NotSupportedException("GPU Gather currently optimized for 2D tensors along axis=1.");

            int batch = _shape[0];
            int classes = _shape[1];

            var result = new CudaBackend(new TensorShape(batch), _requiresGrad, _device);
            CUDA.NativeGather(_allocation.Ptr, o._allocation.Ptr, result._allocation.Ptr, batch, classes);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                var capturedIndices = o;

                result.GradFn = gradOutput =>
                {
                    var gradOutCuda = Tensor.Unwrap(gradOutput) as CudaBackend;
                    if (gradOutCuda == null)
                        throw new InvalidOperationException("Upstream gradient must reside on CUDA GPU.");

                    var gradInput = new CudaBackend(capturedSelf._shape, false, capturedSelf._device);
                    CUDA.CudaMemset(gradInput._allocation.Ptr, 0, (ulong)capturedSelf._shape.TotalElements * sizeof(float));

                    CUDA.GatherGrad(gradOutCuda._allocation.Ptr, capturedIndices._allocation.Ptr, gradInput._allocation.Ptr, batch, classes);

                    capturedSelf.AccumulateGrad(gradInput);
                    return gradOutput;
                };
            }

            return result;
        }
        /// <summary>
        /// Checks element-wise if this tensor is strictly greater than another tensor.
        /// </summary>
        /// <param name="other">The tensor to compare against.</param>
        /// <returns>A binary boolean tensor (containing values 1.0f or 0.0f) representing the comparative result.</returns>

        public ITensor GreaterThan(ITensor other) => ElementwiseBinary(other, CUDA.GreaterThan);
        /// <summary>
        /// Checks element-wise if this tensor is strictly less than another tensor.
        /// </summary>
        /// <param name="other">The tensor to compare against.</param>
        /// <returns>A binary boolean tensor representing the comparative result.</returns>
        public ITensor LessThan(ITensor other) => ElementwiseBinary(other, CUDA.LessThan);
        /// <summary>
        /// Checks element-wise if this tensor is greater than or equal to another tensor.
        /// </summary>
        /// <param name="other">The tensor to compare against.</param>
        /// <returns>A binary boolean tensor representing the comparative result.</returns>
        public ITensor GreaterThanOrEqual(ITensor other) => LessThan(other).LogicalNot();
        /// <summary>
        /// Checks element-wise if this tensor is less than or equal to another tensor.
        /// </summary>
        /// <param name="other">The tensor to compare against.</param>
        /// <returns>A binary boolean tensor representing the comparative result.</returns>
        public ITensor LessEqual(ITensor other) => GreaterThan(other).LogicalNot();
        /// <summary>
        /// Returns elements selected from either <paramref name="trueValue"/> or <paramref name="falseValue"/>, depending on <paramref name="condition"/>.
        /// </summary>
        /// <param name="condition">Condition tensor.</param>
        /// <param name="trueValue">Values to choose if condition evaluates to true.</param>
        /// <param name="falseValue">Values to choose if condition evaluates to false.</param>
        /// <returns>A combined tensor containing elements matching the specified condition.</returns>
        /// <exception cref="InvalidOperationException">Thrown when any of the operand tensors are not on CUDA.</exception>

        public ITensor Where(ITensor condition, ITensor trueValue, ITensor falseValue)
        {
            if (condition is not CudaBackend c || trueValue is not CudaBackend tv || falseValue is not CudaBackend fv)
                throw new InvalidOperationException("All tensors must be on CUDA for Where.");

            var result = new CudaBackend(_shape, false, _device);
            CUDA.Where(c._allocation.Ptr, tv._allocation.Ptr, fv._allocation.Ptr, result._allocation.Ptr, _shape.TotalElements);
            return result;
        }
        /// <summary>
        /// Finds the indices of the maximum values along an axis.
        /// </summary>
        /// <param name="axis">The reduction axis.</param>
        /// <returns>A tensor containing indices of the maximum values.</returns>

        public ITensor ArgMax(int axis)
        {
            int rank = _shape.Rank;
            int actualAxis = axis < 0 ? axis + rank : axis;
            int dim = _shape[actualAxis];

            int outer = 1; for (int i = 0; i < actualAxis; i++) outer *= _shape[i];
            int inner = 1; for (int i = actualAxis + 1; i < rank; i++) inner *= _shape[i];

            int[] outDims = _shape.Dimensions.Where((_, i) => i != actualAxis).ToArray();
            var result = new CudaBackend(new TensorShape(outDims), false, _device);

            CUDA.ArgMax(_allocation.Ptr, result._allocation.Ptr, outer, dim, inner);
            return result;
        }
        /// <summary>
        /// Finds the indices of the minimum values along an axis.
        /// </summary>
        /// <param name="axis">The reduction axis.</param>
        /// <returns>A tensor containing indices of the minimum values.</returns>

        public ITensor ArgMin(int axis)
        {
            int rank = _shape.Rank;
            int actualAxis = axis < 0 ? axis + rank : axis;
            int dim = _shape[actualAxis];

            int outer = 1; for (int i = 0; i < actualAxis; i++) outer *= _shape[i];
            int inner = 1; for (int i = actualAxis + 1; i < rank; i++) inner *= _shape[i];

            int[] outDims = _shape.Dimensions.Where((_, i) => i != actualAxis).ToArray();
            var result = new CudaBackend(new TensorShape(outDims), false, _device);

            CUDA.ArgMin(_allocation.Ptr, result._allocation.Ptr, outer, dim, inner);
            return result;
        }
        /// <summary>
        /// Computes the cumulative sum of elements along a given axis.
        /// </summary>
        /// <param name="axis">The axis along which the cumulative sum is computed.</param>
        /// <returns>A tensor containing the cumulative sums.</returns>

        public ITensor CumSum(int axis)
        {
            int rank = _shape.Rank;
            int actualAxis = axis < 0 ? axis + rank : axis;
            int dim = _shape[actualAxis];

            int outer = 1; for (int i = 0; i < actualAxis; i++) outer *= _shape[i];
            int inner = 1; for (int i = actualAxis + 1; i < rank; i++) inner *= _shape[i];

            var result = new CudaBackend(_shape, false, _device);
            CUDA.CumSum(_allocation.Ptr, result._allocation.Ptr, outer, dim, inner);
            return result;
        }
        /// <summary>
        /// Downloads the current tensor data from the CUDA device to the CPU host.
        /// </summary>
        /// <returns>A host-allocated single-precision floating point array representing the tensor's content.</returns>

        public float[] ToArray()
        {
            // Forces native device execution queue to synchronize before downloading back to host RAM
            CUDA.Synchronize();
            var host = new float[_shape.TotalElements];
            CopyDeviceToHost(_allocation.Ptr, host, _shape.TotalElements);
            return host;
        }
        /// <summary>
        /// Extracts the value of a single-element scalar tensor and returns it as a float.
        /// </summary>
        /// <returns>The scalar value representing the tensor contents.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the tensor is not a scalar (total elements != 1).</exception>

        public float ToScalar()
        {
            if (_shape.TotalElements != 1)
                throw new InvalidOperationException("Tensor is not a scalar.");
            CUDA.Synchronize();
            var host = new float[1];
            CopyDeviceToHost(_allocation.Ptr, host, 1);
            return host[0];
        }
        /// <summary>
        /// Creates a deep copy of the tensor on the same CUDA device.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/> with identical values and structure.</returns>

        public ITensor Clone()
        {
            ulong bytes = (ulong)_shape.TotalElements * sizeof(float);
            var cloneAlloc = new CudaAllocation(bytes);
            CudaMemcpy(cloneAlloc.Ptr, _allocation.Ptr, bytes, cudaMemcpyKind.cudaMemcpyDeviceToDevice);
            return new CudaBackend(_shape, cloneAlloc, _requiresGrad, _device);
        }
        /// <summary>
        /// Copies or moves the current tensor to a targeted CPU or GPU device.
        /// </summary>
        /// <param name="targetDevice">The target device destination.</param>
        /// <returns>A cloned tensor matching the desired target device backend.</returns>
        /// <exception cref="NotSupportedException">Thrown when target device type is unsupported.</exception>

        public ITensor To(Device targetDevice)
        {
            if (targetDevice.Type == DeviceType.CUDA) return Clone();
            if (targetDevice.Type == DeviceType.CPU)
                return new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, targetDevice);
            throw new NotSupportedException("Only CPU and CUDA transfers are supported.");
        }
        /// <summary>
        /// Creates a zero-filled CUDA tensor of the specified shape.
        /// </summary>
        /// <param name="shape">The desired shape of the tensor.</param>
        /// <param name="device">The device instance.</param>
        /// <returns>A zero-initialized <see cref="ITensor"/>.</returns>

        public static ITensor Zeros(TensorShape shape, Device device = null)
            => new CudaBackend(shape, false, device ?? Device.CUDA);
        /// <summary>
        /// Creates a one-filled CUDA tensor of the specified shape.
        /// </summary>
        /// <param name="shape">The desired shape of the tensor.</param>
        /// <param name="device">The device instance.</param>
        /// <returns>A one-initialized <see cref="ITensor"/>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="shape"/> is null.</exception>

        public static ITensor Ones(TensorShape shape, Device device = null)
        {
            if (shape == null) throw new ArgumentNullException(nameof(shape));
            var t = new CudaBackend(shape, false, device ?? Device.CUDA);
            NativeOnes(t._allocation.Ptr, shape.TotalElements);
            return t;
        }
        /// <summary>
        /// Creates a single-element scalar tensor initialized to a given value.
        /// </summary>
        /// <param name="value">The scalar value.</param>
        /// <param name="device">The targeted CUDA device.</param>
        /// <returns>A scalar <see cref="ITensor"/>.</returns>

        public static ITensor FromScalar(float value, Device device = null)
        {
            var t = new CudaBackend(new TensorShape(1), false, device ?? Device.CUDA);
            SetScalar(t._allocation.Ptr, value, 1);
            return t;
        }
        /// <summary>
        /// Creates a CUDA tensor initialized with the provided host data.
        /// </summary>
        /// <param name="data">The source CPU floating point array.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="device">The targeted CUDA device.</param>
        /// <returns>An initialized CUDA tensor.</returns>

        public static ITensor FromArray(float[] data, TensorShape shape, Device device = null)
            => new CudaBackend(data, shape, false, device);
        /// <summary>
        /// Creates a tensor of the specified shape populated with uniformly distributed random numbers in range [0.0, 1.0).
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="device">The targeted device.</param>
        /// <returns>A randomly initialized <see cref="ITensor"/>.</returns>

        public static ITensor Rand(TensorShape shape, Device device = null)
        {
            var cpu = CpuBackend.Rand(shape);
            return FromArray(cpu.ToArray(), shape, device);
        }
        /// <summary>
        /// Creates a tensor of the specified shape populated with normally distributed random numbers with mean 0 and variance 1.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="device">The targeted device.</param>
        /// <returns>A normally-distributed random <see cref="ITensor"/>.</returns>

        public static ITensor Randn(TensorShape shape, Device device = null)
        {
            var cpu = CpuBackend.Randn(shape);
            return FromArray(cpu.ToArray(), shape, device);
        }
        /// <summary>
        /// Creates a 2D identity tensor of the specified dimension size on the CUDA GPU.
        /// </summary>
        /// <param name="size">The dimension size of the identity matrix.</param>
        /// <param name="device">The targeted device.</param>
        /// <returns>An identity matrix <see cref="ITensor"/>.</returns>

        public static ITensor Eye(int size, Device device = null)
        {
            var cpu = CpuBackend.Eye(size);
            return FromArray(cpu.ToArray(), new TensorShape(size, size), device);
        }
        /// <summary>
        /// Adds another tensor to this tensor element-wise. Supports broadcasting.
        /// </summary>
        /// <param name="other">The tensor to add.</param>
        /// <returns>A new <see cref="ITensor"/> containing the sum.</returns>

        public ITensor Add(ITensor other) => ElementwiseBinary(other, NativeAdd);
        /// <summary>
        /// Subtracts another tensor from this tensor element-wise. Supports broadcasting.
        /// </summary>
        /// <param name="other">The tensor to subtract.</param>
        /// <returns>A new <see cref="ITensor"/> containing the difference.</returns>
        public ITensor Subtract(ITensor other) => ElementwiseBinary(other, NativeSubtract);
        /// <summary>
        /// Multiplies this tensor by another tensor element-wise. Supports broadcasting.
        /// </summary>
        /// <param name="other">The tensor to multiply.</param>
        /// <returns>A new <see cref="ITensor"/> containing the product.</returns>
        public ITensor Multiply(ITensor other) => ElementwiseBinary(other, NativeMultiply);
        /// <summary>
        /// Divides this tensor by another tensor element-wise. Supports broadcasting.
        /// </summary>
        /// <param name="other">The divisor tensor.</param>
        /// <returns>A new <see cref="ITensor"/> containing the quotient.</returns>
        public ITensor Divide(ITensor other) => ElementwiseBinary(other, NativeDivide);
        /// <summary>
        /// Private helper to perform element-wise binary operations with automatic broadcasting support,
        /// handling autograd gradient graph mapping.
        /// </summary>

        private ITensor ElementwiseBinary(ITensor other, Action<IntPtr, IntPtr, IntPtr, int> kernel)
        {
            if (other is not CudaBackend o)
                throw new InvalidOperationException("Both tensors must be on CUDA.");

            var resultShape = _shape.BroadcastTo(o.Shape);
            var result = new CudaBackend(resultShape, false, _device);

            kernel(_allocation.Ptr, o._allocation.Ptr, result._allocation.Ptr, resultShape.TotalElements);

            var resultTensor = new CudaBackend(resultShape, result._allocation, _requiresGrad || o.RequiresGrad, _device);
            resultTensor.Inputs = new[] { this, other };

            if (resultTensor.RequiresGrad)
            {
                var capturedSelf = this;
                var capturedOther = o;

                resultTensor.GradFn = grad =>
                {
                    if (capturedSelf.RequiresGrad)
                        capturedSelf.AccumulateGrad(grad.BroadcastTo(capturedSelf._shape));
                    if (capturedOther.RequiresGrad)
                        capturedOther.AccumulateGrad(grad.BroadcastTo(capturedOther.Shape));
                    return grad;
                };
            }
            return resultTensor;
        }
        /// <summary>
        /// Computes the power of this tensor to a scalar exponent element-wise.
        /// </summary>
        /// <param name="exponent">The exponent.</param>
        /// <returns>A new <see cref="ITensor"/> containing result values.</returns>

        public ITensor Pow(float exponent)
        {
            var result = new CudaBackend(_shape, false, _device);
            PowScalar(_allocation.Ptr, result._allocation.Ptr, _shape.TotalElements, exponent);
            return result;
        }
        /// <summary>
        /// Performs an element-wise logical NOT operation.
        /// </summary>
        /// <returns>A binary boolean tensor representation.</returns>

        public ITensor LogicalNot()
        {
            var result = new CudaBackend(_shape, false, _device);
            NativeLogicalNot(_allocation.Ptr, result._allocation.Ptr, _shape.TotalElements);
            return result;
        }
        /// <summary>
        /// Clips the values of this tensor to a specified range in place of a new tensor, supporting gradient calculation.
        /// </summary>
        /// <param name="min">The minimum allowable value.</param>
        /// <param name="max">The maximum allowable value.</param>
        /// <returns>A clipped <see cref="ITensor"/>.</returns>

        public ITensor Clip(float min, float max)
        {
            if (min > max) (min, max) = (max, min);
            var result = new CudaBackend(_shape, _requiresGrad, _device);
            NativeClip(_allocation.Ptr, result._allocation.Ptr, _shape.TotalElements, min, max);

            if (_requiresGrad)
            {
                var self = this;
                result.GradFn = grad =>
                {
                    var mask = new CudaBackend(_shape, false, _device);
                    NativeClipMask(self._allocation.Ptr, mask._allocation.Ptr, _shape.TotalElements, min, max);
                    var finalGrad = grad.Multiply(mask);
                    self.AccumulateGrad(finalGrad);
                    return finalGrad;
                };
            }
            return result;
        }
        /// <summary>
        /// Computes the element-wise negation of this tensor.
        /// </summary>
        /// <returns>A negated <see cref="ITensor"/>.</returns>

        public ITensor Negate() => ElementwiseUnary(NativeNegate);
        /// <summary>
        /// Computes the element-wise base-e exponential function of this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/>.</returns>
        public ITensor Exp() => ElementwiseUnary(NativeExp);
        /// <summary>
        /// Computes the element-wise natural logarithm of this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/>.</returns>
        public ITensor Log() => ElementwiseUnary(NativeLog);
        /// <summary>
        /// Computes the element-wise square root of this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/>.</returns>
        public ITensor Sqrt() => ElementwiseUnary(NativeSqrt);
        /// <summary>
        /// Computes the element-wise absolute value of this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/>.</returns>
        public ITensor Abs() => ElementwiseUnary(NativeAbs);
        /// <summary>
        /// Computes the element-wise trigonometric sine of this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/>.</returns>
        public ITensor Sin() => ElementwiseUnary(NativeSin);
        /// <summary>
        /// Computes the element-wise trigonometric cosine of this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/>.</returns>
        public ITensor Cos() => ElementwiseUnary(NativeCos);
        /// <summary>
        /// Computes the element-wise sign indicator of this tensor (-1, 0, or 1).
        /// </summary>
        /// <returns>A new <see cref="ITensor"/>.</returns>
        public ITensor Sign() => ElementwiseUnary(NativeSign);
        /// <summary>
        /// Applies the hyperbolic tangent activation function to this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/>.</returns>
        public ITensor Tanh() => new Tanh().Forward(this);
        /// <summary>
        /// Applies the Rectified Linear Unit activation function to this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/>.</returns>
        public ITensor Relu() => new ReLU().Forward(this);
        /// <summary>
        /// Applies the Sigmoid activation function to this tensor.
        /// </summary>
        /// <returns>A new <see cref="ITensor"/>.</returns>
        public ITensor Sigmoid() => new Sigmoid().Forward(this);
        /// <summary>
        /// Applies the Softmax activation function to this tensor along the specified axis.
        /// </summary>
        /// <param name="axis">The target axis. Defaults to the last dimension (-1).</param>
        /// <returns>A new <see cref="ITensor"/> normalized via softmax.</returns>
        public ITensor Softmax(int axis = -1) => new Softmax(axis).Forward(this);
        /// <summary>
        /// Executes a basic element-wise unary kernel on the GPU.
        /// </summary>

        private ITensor ElementwiseUnary(Action<IntPtr, IntPtr, int> kernel)
        {
            var result = new CudaBackend(_shape, false, _device);
            kernel(_allocation.Ptr, result._allocation.Ptr, _shape.TotalElements);
            return result;
        }
        /// <summary>
        /// Multiplies this 2D tensor by another 2D tensor using matrix multiplication.
        /// </summary>
        /// <param name="other">The right-hand matrix operand.</param>
        /// <returns>The resulting matrix multiplied <see cref="ITensor"/>.</returns>
        /// <exception cref="InvalidOperationException">Thrown when tensors are not 2D or reside on different devices.</exception>

        public ITensor MatMul(ITensor other)
        {
            if (other is not CudaBackend o || _shape.Rank != 2 || o.Shape.Rank != 2)
                throw new InvalidOperationException("MatMul requires 2D CUDA tensors.");

            int m = _shape[0], k = _shape[1], n = o.Shape[1];
            var result = new CudaBackend(new TensorShape(m, n), false, _device);
            NativeMatMul(_allocation.Ptr, o._allocation.Ptr, result._allocation.Ptr, m, n, k);
            return result;
        }
        /// <summary>
        /// Permutes the dimensions of this tensor according to the specified mapping.
        /// </summary>
        /// <param name="perm">An array of integers containing the permutation indices.</param>
        /// <returns>A transposed tensor view/copy.</returns>

        public ITensor Transpose(int[] perm)
        {
            if (perm.Length == 2 && perm[0] == 1 && perm[1] == 0)
            {
                var result = new CudaBackend(new TensorShape(_shape[1], _shape[0]), false, _device);
                NativeTranspose(_allocation.Ptr, result._allocation.Ptr, _shape[0], _shape[1]);
                return result;
            }

            var newShape = new TensorShape(_shape.Dimensions.Select((d, i) => _shape.Dimensions[perm[i]]).ToArray());
            var resultGen = new CudaBackend(newShape, false, _device);
            NativeGeneralTranspose(_allocation.Ptr, resultGen._allocation.Ptr, _shape.Dimensions, perm, perm.Length);
            return resultGen;
        }
        /// <summary>
        /// Changes the organizational shape of the tensor without changing its underlying values.
        /// </summary>
        /// <param name="newShape">The targeted new dimensions.</param>
        /// <returns>A reshaped tensor sharing the same native allocation reference.</returns>
        /// <exception cref="ArgumentException">Thrown when total elements differ from the original shape.</exception>

        public ITensor Reshape(params int[] newShape)
        {
            var ns = new TensorShape(newShape);
            if (ns.TotalElements != _shape.TotalElements)
                throw new ArgumentException("Cannot reshape to a different number of elements.");

            return new CudaBackend(ns, _allocation, _requiresGrad, _device);
        }
        /// <summary>
        /// Broadcasts this tensor to a compatible larger target shape.
        /// </summary>
        /// <param name="targetShape">The targeted broadcast dimensions.</param>
        /// <returns>A broadcasted tensor containing gradient step mapping for backward accumulation.</returns>

        public ITensor BroadcastTo(TensorShape targetShape)
        {
            if (_shape.Equals(targetShape)) return Clone();

            int[] alignedInDims = Enumerable.Repeat(1, targetShape.Rank).ToArray();
            int offset = targetShape.Rank - _shape.Rank;
            for (int i = 0; i < _shape.Rank; i++)
            {
                alignedInDims[i + offset] = _shape.Dimensions[i];
            }

            var alignedInputShape = new TensorShape(alignedInDims);
            var result = new CudaBackend(targetShape, false, _device);

            CUDA.Broadcast(_allocation.Ptr, result._allocation.Ptr, alignedInputShape.Dimensions, targetShape.Dimensions);

            var resultTensor = new CudaBackend(targetShape, result._allocation, _requiresGrad, _device);
            resultTensor.Inputs = new[] { this };

            if (_requiresGrad)
            {
                var capturedSelf = this;
                resultTensor.GradFn = grad =>
                {
                    if (capturedSelf.RequiresGrad)
                        capturedSelf.AccumulateGrad(grad.BroadcastTo(capturedSelf._shape));
                    return grad;
                };
            }

            return resultTensor;
        }
        /// <summary>
        /// Reshapes and automatically broadcasts the tensor along a specified axis alignment.
        /// </summary>
        /// <param name="target">The target shape.</param>
        /// <param name="axis">The starting axis for alignment.</param>
        /// <returns>A reshaped and broadcasted tensor.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="target"/> is null.</exception>

        public ITensor ReshapeWithBroadcast(TensorShape target, int axis)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));

            int targetRank = target.Rank;
            int actualAxis = axis < 0 ? targetRank + axis : axis;

            var viewDims = Enumerable.Repeat(1, targetRank).ToArray();
            int origIdx = 0;
            for (int i = actualAxis; i < targetRank && origIdx < _shape.Rank; i++)
            {
                viewDims[i] = _shape.Dimensions[origIdx++];
            }

            var reshapedView = this.Reshape(viewDims);
            return reshapedView.BroadcastTo(target);
        }
        /// <summary>
        /// Computes the sum of elements of this tensor, optionally along a specified axis.
        /// </summary>
        /// <param name="axis">The axis along which to sum, or null to reduce the entire tensor.</param>
        /// <param name="keepDims">True to retain reduced dimensions with size 1.</param>
        /// <returns>A summed tensor.</returns>

        public ITensor Sum(int? axis = null, bool keepDims = false)
        {
            if (axis is null)
            {
                var newShape = keepDims ? new TensorShape(Enumerable.Repeat(1, _shape.Rank).ToArray()) : new TensorShape(1);
                var result = new CudaBackend(newShape, false, _device);
                NativeSumAll(_allocation.Ptr, result._allocation.Ptr, _shape.TotalElements);
                return result;
            }

            int actualAxis = axis.Value < 0 ? axis.Value + _shape.Rank : axis.Value;
            int dimSize = _shape[actualAxis];

            var mean = Mean(axis.Value, keepDims);
            return mean.Multiply((float)dimSize);
        }
        /// <summary>
        /// Computes the sum of elements of this tensor along multiple axes.
        /// </summary>
        /// <param name="axes">The dimensions to collapse.</param>
        /// <param name="keepDims">True to retain reduced dimensions with size 1.</param>
        /// <returns>A summed tensor.</returns>

        public ITensor Sum(int[] axes, bool keepDims = false)
        {
            if (axes == null || axes.Length == 0) return Sum((int?)null, keepDims);
            int rank = _shape.Rank;
            var normalizedAxes = axes.Select(a => a < 0 ? a + rank : a).Distinct().ToList();
            normalizedAxes.Sort((a, b) => b.CompareTo(a));

            ITensor current = this;
            foreach (int axis in normalizedAxes)
            {
                current = current.Sum(axis, keepDims);
            }
            return current;
        }
        /// <summary>
        /// Computes the arithmetic mean of elements of this tensor along multiple axes.
        /// </summary>
        /// <param name="axes">The dimensions to collapse.</param>
        /// <param name="keepDims">True to retain reduced dimensions with size 1.</param>
        /// <returns>A mean-reduced tensor.</returns>

        public ITensor Mean(int[] axes, bool keepDims = false)
        {
            if (axes == null || axes.Length == 0)
            {
                var result = new CudaBackend(keepDims ? new TensorShape(Enumerable.Repeat(1, _shape.Rank).ToArray()) : new TensorShape(1), false, _device);
                CUDA.MeanAll(_allocation.Ptr, result._allocation.Ptr, _shape.TotalElements);
                return result;
            }

            ITensor current = this;
            var sortedAxes = axes.Select(a => a < 0 ? a + _shape.Rank : a).OrderByDescending(a => a).ToArray();

            foreach (int axis in sortedAxes)
            {
                current = current.Mean(axis, keepDims);
            }

            return current;
        }
        /// <summary>
        /// Computes the arithmetic mean of elements of this tensor, optionally along a specified axis.
        /// </summary>
        /// <param name="axis">The axis along which to calculate mean, or null to reduce the entire tensor.</param>
        /// <param name="keepDims">True to retain reduced dimensions with size 1.</param>
        /// <returns>A mean-reduced tensor.</returns>

        public ITensor Mean(int? axis = null, bool keepDims = false)
        {
            if (axis is null)
            {
                var newShape = keepDims ? new TensorShape(Enumerable.Repeat(1, _shape.Rank).ToArray()) : new TensorShape(1);
                var result = new CudaBackend(newShape, false, _device);
                CUDA.MeanAll(_allocation.Ptr, result._allocation.Ptr, _shape.TotalElements);
                return result;
            }

            int actualAxis = axis.Value < 0 ? axis.Value + _shape.Rank : axis.Value;
            int dim = _shape[actualAxis];
            int outer = 1; for (int i = 0; i < actualAxis; i++) outer *= _shape[i];
            int inner = 1; for (int i = actualAxis + 1; i < _shape.Rank; i++) inner *= _shape[i];

            int[] outDims = keepDims
                ? _shape.Dimensions.Select((d, i) => i == actualAxis ? 1 : d).ToArray()
                : _shape.Dimensions.Where((_, i) => i != actualAxis).ToArray();

            var resultTensor = new CudaBackend(new TensorShape(outDims), false, _device);

            CUDA.MeanAxis(_allocation.Ptr, resultTensor._allocation.Ptr, outer, dim, inner);
            return resultTensor;
        }
        /// <summary>
        /// Finds the maximum values along a specified axis.
        /// </summary>
        /// <param name="axis">The axis along which to search.</param>
        /// <param name="keepDims">True to retain reduced dimensions with size 1.</param>
        /// <returns>A tensor containing the maximum values.</returns>

        public ITensor Max(int axis = -1, bool keepDims = false)
        {
            int actualAxis = axis < 0 ? axis + _shape.Rank : axis;
            int dim = _shape[actualAxis];
            int outer = 1; for (int i = 0; i < actualAxis; i++) outer *= _shape[i];
            int inner = 1; for (int i = actualAxis + 1; i < _shape.Rank; i++) inner *= _shape[i];

            int[] outDims = keepDims
                ? _shape.Dimensions.Select((d, i) => i == actualAxis ? 1 : d).ToArray()
                : _shape.Dimensions.Where((_, i) => i != actualAxis).ToArray();

            var result = new CudaBackend(new TensorShape(outDims), false, _device);
            CUDA.MaxAxis(_allocation.Ptr, result._allocation.Ptr, outer, dim, inner);
            return result;
        }
        /// <summary>
        /// Finds the minimum values along a specified axis.
        /// </summary>
        /// <param name="axis">The axis along which to search.</param>
        /// <param name="keepDims">True to retain reduced dimensions with size 1.</param>
        /// <returns>A tensor containing the minimum values.</returns>

        public ITensor Min(int axis = -1, bool keepDims = false)
        {
            int actualAxis = axis < 0 ? axis + _shape.Rank : axis;
            int dim = _shape[actualAxis];
            int outer = 1; for (int i = 0; i < actualAxis; i++) outer *= _shape[i];
            int inner = 1; for (int i = actualAxis + 1; i < _shape.Rank; i++) inner *= _shape[i];

            int[] outDims = keepDims
                ? _shape.Dimensions.Select((d, i) => i == actualAxis ? 1 : d).ToArray()
                : _shape.Dimensions.Where((_, i) => i != actualAxis).ToArray();

            var result = new CudaBackend(new TensorShape(outDims), false, _device);
            CUDA.MinAxis(_allocation.Ptr, result._allocation.Ptr, outer, dim, inner);
            return result;
        }
        /// <summary>
        /// Extracts a slice from this tensor according to specified coordinate boundaries.
        /// </summary>
        /// <param name="slices">Array of tuples indicating start index, end index, and step size for each dimension.</param>
        /// <returns>A new sliced <see cref="ITensor"/> with dynamic backward mapping for autograd tracking.</returns>
        /// <exception cref="ArgumentException">Thrown when slice count does not match the tensor rank.</exception>

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

            var outShape = new TensorShape(newShapeList.ToArray());
            var result = new CudaBackend(outShape, _requiresGrad, _device);

            CUDA.Slice(_allocation.Ptr, result._allocation.Ptr, _shape.Dimensions, outShape.Dimensions, starts, steps, _shape.Rank);

            if (_requiresGrad)
            {
                var capturedStarts = (int[])starts.Clone();
                var capturedSteps = (int[])steps.Clone();
                var originalShape = _shape.Clone();
                var newShapeArr = newShapeList.ToArray();
                var capturedSelf = this;

                result.GradFn = grad =>
                {
                    var gradInput = new CudaBackend(originalShape, false, _device);
                    CUDA.CudaMemset(gradInput._allocation.Ptr, 0, (ulong)originalShape.TotalElements * sizeof(float));

                    var gradUnwrapped = Tensor.Unwrap(grad) as CudaBackend;
                    CUDA.SliceGrad(gradUnwrapped._allocation.Ptr, gradInput._allocation.Ptr, originalShape.Dimensions, newShapeArr, capturedStarts, capturedSteps, originalShape.Rank);

                    capturedSelf.AccumulateGrad(gradInput);
                    return gradInput;
                };
            }

            return result;
        }
        /// <summary>
        /// Concatenates this tensor with other tensors along a specified axis.
        /// </summary>
        /// <param name="others">An enumerable sequence of tensors to concatenate.</param>
        /// <param name="axis">The dimension along which tensors will be joined.</param>
        /// <returns>A new combined <see cref="ITensor"/>.</returns>
        /// <exception cref="InvalidOperationException">Thrown when any input tensor does not reside on CUDA.</exception>

        public ITensor Concat(IEnumerable<ITensor> others, int axis = 0)
        {
            var all = new List<CudaBackend> { this };
            foreach (var o in others)
            {
                var unwrapped = Tensor.Unwrap(o) as CudaBackend;
                if (unwrapped == null) throw new InvalidOperationException("All tensors must be on CUDA.");
                all.Add(unwrapped);
            }

            int rank = _shape.Rank;
            int actualAxis = axis < 0 ? rank + axis : axis;

            int[] newDims = _shape.Dimensions.ToArray();
            newDims[actualAxis] = all.Sum(t => t.Shape[actualAxis]);
            var outShape = new TensorShape(newDims);

            var result = new CudaBackend(outShape, _requiresGrad || all.Any(t => t.RequiresGrad), _device);

            int outerSize = 1;
            for (int i = 0; i < actualAxis; i++) outerSize *= newDims[i];
            int innerSize = 1;
            for (int i = actualAxis + 1; i < rank; i++) innerSize *= newDims[i];

            IntPtr[] inputPtrs = all.Select(t => t._allocation.Ptr).ToArray();
            int[] concatSizes = all.Select(t => t.Shape[actualAxis]).ToArray();

            CUDA.Concat(inputPtrs, result._allocation.Ptr, all.Count, outerSize, concatSizes, innerSize);

            if (result.RequiresGrad)
            {
                var capturedAll = all.ToList();
                int capturedAxis = actualAxis;

                result.GradFn = gradOutput =>
                {
                    int currentOffset = 0;
                    foreach (var t in capturedAll)
                    {
                        int tAxisSize = t.Shape[capturedAxis];
                        if (t.RequiresGrad)
                        {
                            var slices = new (int, int, int)[rank];
                            for (int i = 0; i < rank; i++)
                            {
                                if (i == capturedAxis)
                                    slices[i] = (currentOffset, currentOffset + tAxisSize, 1);
                                else
                                    slices[i] = (0, t.Shape[i], 1);
                            }

                            var gradInput = gradOutput.Slice(slices);
                            t.AccumulateGrad(gradInput);
                        }
                        currentOffset += tAxisSize;
                    }
                    return gradOutput;
                };
            }

            return result;
        }
        /// <summary>Adds a scalar value to each element of this tensor.</summary>

        public ITensor Add(float scalar) => Add(FromScalar(scalar));
        /// <summary>Subtracts a scalar value from each element of this tensor.</summary>
        public ITensor Subtract(float scalar) => Subtract(FromScalar(scalar));
        /// <summary>Multiplies each element of this tensor by a scalar value.</summary>
        public ITensor Multiply(float scalar) => Multiply(FromScalar(scalar));
        /// <summary>Divides each element of this tensor by a scalar value.</summary>
        public ITensor Divide(float scalar) => Divide(FromScalar(scalar));
        /// <summary>Subtracts an integer value from each element of this tensor.</summary>
        public ITensor Subtract(int other) => Subtract((float)other);
        /// <summary>Multiplies each element of this tensor by a double-precision scalar value.</summary>
        public ITensor Multiply(double scalar) => Multiply((float)scalar);
        /// <summary>Divides each element of this tensor by a double-precision scalar value.</summary>
        public ITensor Divide(double scalar) => Divide((float)scalar);
        /// <summary>
        /// Computes element-wise power with tensor-based exponent.
        /// </summary>

        public ITensor Pow(ITensor exponent) => ElementwiseBinary(exponent, CUDA.PowTensor);
        /// <summary>
        /// Broad-cast adds another tensor to this tensor.
        /// </summary>
        public ITensor BroadcastAdd(ITensor other) => Add(other);
        /// <summary>
        /// Triggers autograd backward traversal starting from this node, calculating gradients for tracking ancestors.
        /// </summary>
        /// <param name="gradient">The incoming upstream gradient. Defaults to 1.0f if null.</param>

        public void Backward(ITensor? gradient = null)
        {
            ArborNet.Core.Autograd.AutogradEngine.Backward(this, gradient);
        }
        /// <summary>
        /// Clears accumulated gradients and structural backward mapping graphs.
        /// </summary>

        public void ClearGrad()
        {
            _grad = null;
            _gradFn = null;
        }
        /// <summary>
        /// Sets the underlying values of this tensor, copying new data from the host.
        /// </summary>
        /// <param name="floats">The source host single-precision floating point array.</param>
        /// <exception cref="ArgumentException">Thrown when source array length mismatch tensor capacity.</exception>

        public void SetData(float[] floats)
        {
            if (floats.Length != _shape.TotalElements)
                throw new ArgumentException("Data size does not match tensor shape.");
            CopyHostToDevice(floats, _allocation.Ptr, floats.Length);
        }
        /// <summary>Gets a value indicating whether this tensor resides on CPU memory.</summary>

        public bool IsCpu() => false;
        /// <summary>Gets a value indicating whether this tensor resides on CUDA GPU memory.</summary>
        public bool IsCuda() => true;
        /// <summary>
        /// Returns an enumerable collection of this tensor's parameters (self-referential).
        /// </summary>

        public IEnumerable<ITensor> Parameters() { yield return this; }
        /// <summary>
        /// Performs an element-wise equality check against another tensor.
        /// </summary>
        /// <param name="other">The tensor to check.</param>
        /// <returns>A binary boolean tensor representing equality.</returns>
        /// <exception cref="InvalidOperationException">Thrown when operand tensors shape mismatch or are not on CUDA.</exception>

        public ITensor Equal(ITensor other)
        {
            if (other is not CudaBackend o || !_shape.Equals(o.Shape))
                throw new InvalidOperationException("Tensors must have same shape for equality.");

            var result = new CudaBackend(_shape, false, _device);
            NativeEqual(_allocation.Ptr, o._allocation.Ptr, result._allocation.Ptr, _shape.TotalElements);
            return result;
        }
        /// <summary>
        /// Performs a high-speed copy of data from the CPU host memory to the CUDA GPU device memory.
        /// </summary>

        private static void CopyHostToDevice(float[] source, IntPtr destination, int count)
        {
            var handle = GCHandle.Alloc(source, GCHandleType.Pinned);
            try
            {
                CudaMemcpy(destination, handle.AddrOfPinnedObject(), (ulong)(count * sizeof(float)), cudaMemcpyKind.cudaMemcpyHostToDevice);
            }
            finally { handle.Free(); }
        }
        /// <summary>
        /// Performs a high-speed copy of data from the CUDA GPU device memory to the CPU host memory.
        /// </summary>

        private static void CopyDeviceToHost(IntPtr devicePtr, float[] host, int count)
        {
            var handle = GCHandle.Alloc(host, GCHandleType.Pinned);
            try
            {
                CudaMemcpy(handle.AddrOfPinnedObject(), devicePtr, (ulong)(count * sizeof(float)), cudaMemcpyKind.cudaMemcpyDeviceToHost);
            }
            finally { handle.Free(); }
        }
        /// <summary>
        /// Disposes of the allocation by releasing the current reference and suppressing finalization.
        /// </summary>

        public void Dispose()
        {
            lock (_lock)
            {
                if (!_disposed)
                {
                    _allocation.Dispose();
                    _disposed = true;
                }
            }
            GC.SuppressFinalize(this);
        }

        ~CudaBackend() => Dispose();
    }
}