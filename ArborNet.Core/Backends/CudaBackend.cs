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

namespace ArborNet.Core.Backends
{
    public sealed class CudaBackend : ITensor, IDisposable
    {
        private IntPtr _devicePtr = IntPtr.Zero;
        private TensorShape _shape;
        private readonly Device _device;
        private bool _requiresGrad;
        private ITensor? _grad;
        private Func<ITensor, ITensor>? _gradFn;
        private bool _disposed;
        private readonly object _lock = new();

        private ITensor[] _inputs = Array.Empty<ITensor>();

        public ITensor[] Inputs
        {
            get => _inputs;
            set => _inputs = value ?? Array.Empty<ITensor>();
        }

        public TensorShape Shape => _shape;
        public Device Device => _device;
        public bool RequiresGrad { get => _requiresGrad; set => _requiresGrad = value; }
        public ITensor? Grad { get => _grad; set => _grad = value; }
        public Func<ITensor, ITensor>? GradFn { get => _gradFn; set => _gradFn = value; }
        public float[] Data => ToArray();

        public CudaBackend(TensorShape shape, bool requiresGrad = false, Device? device = null)
        {
            _shape = shape?.Clone() ?? throw new ArgumentNullException(nameof(shape));
            _device = device ?? Device.CUDA;
            _requiresGrad = requiresGrad;

            ulong bytes = (ulong)_shape.TotalElements * sizeof(float);
            _devicePtr = CudaMemoryPool.Instance.Allocate(bytes);
            CudaMemset(_devicePtr, 0, bytes);
            GC.AddMemoryPressure((long)bytes);
        }

        public CudaBackend(float[] hostData, TensorShape shape, bool requiresGrad = false, Device? device = null)
        {
            _shape = shape?.Clone() ?? throw new ArgumentNullException(nameof(shape));
            _device = device ?? Device.CUDA;
            _requiresGrad = requiresGrad;

            ulong bytes = (ulong)_shape.TotalElements * sizeof(float);
            _devicePtr = CudaMemoryPool.Instance.Allocate(bytes);
            CopyHostToDevice(hostData, _devicePtr, _shape.TotalElements);
            GC.AddMemoryPressure((long)bytes);
        }

        private CudaBackend(TensorShape shape, IntPtr devicePtr, bool requiresGrad, Device device)
        {
            _shape = shape.Clone();
            _devicePtr = devicePtr;
            _requiresGrad = requiresGrad;
            _device = device;
            GC.AddMemoryPressure((long)_shape.TotalElements * sizeof(float));
        }

        // Thread-Safe GPU Gradient Accumulation
        public void AccumulateGrad(ITensor delta)
        {
            if (delta == null) return;

            ITensor reduced = delta;
            if (!delta.Shape.Equals(_shape))
            {
                var cpuDelta = delta.To(Device.CPU);
                var cpuReduced = CpuBackend.ReduceGradientToTarget(cpuDelta, _shape);
                reduced = cpuReduced.To(_device);
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

        // On-Device Vector Math Kernels
        public void AddInPlace(ITensor other)
        {
            if (other is not CudaBackend o) throw new InvalidOperationException("Operand must reside on CUDA.");
            lock (_lock)
            {
                NativeAdd(_devicePtr, o._devicePtr, _devicePtr, _shape.TotalElements);
            }
        }

        public void AddInPlace(float scalar)
        {
            using var scalarTensor = (CudaBackend)FromScalar(scalar, _device);
            lock (_lock)
            {
                NativeAdd(_devicePtr, scalarTensor._devicePtr, _devicePtr, _shape.TotalElements);
            }
        }

        public void SubtractInPlace(ITensor other)
        {
            if (other is not CudaBackend o) throw new InvalidOperationException("Operand must reside on CUDA.");
            lock (_lock)
            {
                NativeSubtract(_devicePtr, o._devicePtr, _devicePtr, _shape.TotalElements);
            }
        }

        public void SubtractInPlace(float scalar)
        {
            using var scalarTensor = (CudaBackend)FromScalar(scalar, _device);
            lock (_lock)
            {
                NativeSubtract(_devicePtr, scalarTensor._devicePtr, _devicePtr, _shape.TotalElements);
            }
        }

        public void MultiplyInPlace(ITensor other)
        {
            if (other is not CudaBackend o) throw new InvalidOperationException("Operand must reside on CUDA.");
            lock (_lock)
            {
                NativeMultiply(_devicePtr, o._devicePtr, _devicePtr, _shape.TotalElements);
            }
        }

        public void MultiplyInPlace(float scalar)
        {
            using var scalarTensor = (CudaBackend)FromScalar(scalar, _device);
            lock (_lock)
            {
                NativeMultiply(_devicePtr, scalarTensor._devicePtr, _devicePtr, _shape.TotalElements);
            }
        }

        public ITensor Gather(int axis, ITensor indices)
        {
            var cpuSelf = this.To(Device.CPU);
            var cpuIndices = indices.To(Device.CPU);
            var cpuResult = cpuSelf.Gather(axis, cpuIndices);
            return cpuResult.To(_device);
        }

        public ITensor GreaterThan(ITensor other) => ElementwiseBinary(other, CUDA.GreaterThan);
        public ITensor LessThan(ITensor other) => ElementwiseBinary(other, CUDA.LessThan);
        public ITensor GreaterThanOrEqual(ITensor other) => LessThan(other).LogicalNot();
        public ITensor LessEqual(ITensor other) => GreaterThan(other).LogicalNot();

        public ITensor Where(ITensor condition, ITensor trueValue, ITensor falseValue)
        {
            if (condition is not CudaBackend c || trueValue is not CudaBackend tv || falseValue is not CudaBackend fv)
                throw new InvalidOperationException("All tensors must be on CUDA for Where.");

            var result = new CudaBackend(_shape, false, _device);
            CUDA.Where(c._devicePtr, tv._devicePtr, fv._devicePtr, result._devicePtr, _shape.TotalElements);
            return result;
        }

        public ITensor ArgMax(int axis)
        {
            int rank = _shape.Rank;
            int actualAxis = axis < 0 ? axis + rank : axis;
            int dim = _shape[actualAxis];

            int outer = 1; for (int i = 0; i < actualAxis; i++) outer *= _shape[i];
            int inner = 1; for (int i = actualAxis + 1; i < rank; i++) inner *= _shape[i];

            int[] outDims = _shape.Dimensions.Where((_, i) => i != actualAxis).ToArray();
            var result = new CudaBackend(new TensorShape(outDims), false, _device);

            CUDA.ArgMax(_devicePtr, result._devicePtr, outer, dim, inner);
            return result;
        }

        public ITensor ArgMin(int axis)
        {
            int rank = _shape.Rank;
            int actualAxis = axis < 0 ? axis + rank : axis;
            int dim = _shape[actualAxis];

            int outer = 1; for (int i = 0; i < actualAxis; i++) outer *= _shape[i];
            int inner = 1; for (int i = actualAxis + 1; i < rank; i++) inner *= _shape[i];

            int[] outDims = _shape.Dimensions.Where((_, i) => i != actualAxis).ToArray();
            var result = new CudaBackend(new TensorShape(outDims), false, _device);

            CUDA.ArgMin(_devicePtr, result._devicePtr, outer, dim, inner);
            return result;
        }

        public ITensor CumSum(int axis)
        {
            int rank = _shape.Rank;
            int actualAxis = axis < 0 ? axis + rank : axis;
            int dim = _shape[actualAxis];

            int outer = 1; for (int i = 0; i < actualAxis; i++) outer *= _shape[i];
            int inner = 1; for (int i = actualAxis + 1; i < rank; i++) inner *= _shape[i];

            var result = new CudaBackend(_shape, false, _device);
            CUDA.CumSum(_devicePtr, result._devicePtr, outer, dim, inner);
            return result;
        }

        public float[] ToArray()
        {
            var host = new float[_shape.TotalElements];
            CopyDeviceToHost(_devicePtr, host, _shape.TotalElements);
            return host;
        }

        public float ToScalar()
        {
            if (_shape.TotalElements != 1)
                throw new InvalidOperationException("Tensor is not a scalar.");
            var host = new float[1];
            CopyDeviceToHost(_devicePtr, host, 1);
            return host[0];
        }

        public ITensor Clone()
        {
            ulong bytes = (ulong)_shape.TotalElements * sizeof(float);
            IntPtr clonePtr = CudaMemoryPool.Instance.Allocate(bytes);
            CudaMemcpy(clonePtr, _devicePtr, bytes, cudaMemcpyKind.cudaMemcpyDeviceToDevice);
            return new CudaBackend(_shape, clonePtr, _requiresGrad, _device);
        }

        public ITensor To(Device targetDevice)
        {
            if (targetDevice.Type == DeviceType.CUDA) return Clone();
            if (targetDevice.Type == DeviceType.CPU)
                return new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, targetDevice);
            throw new NotSupportedException("Only CPU and CUDA transfers are supported.");
        }

        public static ITensor Zeros(TensorShape shape, Device device = null)
            => new CudaBackend(shape, false, device ?? Device.CUDA);

        public static ITensor Ones(TensorShape shape, Device device = null)
        {
            var t = new CudaBackend(shape, false, device ?? Device.CUDA);
            Ones(shape, t._device);
            return t;
        }

        public static ITensor FromScalar(float value, Device device = null)
        {
            var t = new CudaBackend(new TensorShape(1), false, device ?? Device.CUDA);
            SetScalar(t._devicePtr, value, 1);
            return t;
        }

        public static ITensor FromArray(float[] data, TensorShape shape, Device device = null)
            => new CudaBackend(data, shape, false, device);

        public static ITensor Rand(TensorShape shape, Device device = null)
        {
            var cpu = CpuBackend.Rand(shape);
            return FromArray(cpu.ToArray(), shape, device);
        }

        public static ITensor Randn(TensorShape shape, Device device = null)
        {
            var cpu = CpuBackend.Randn(shape);
            return FromArray(cpu.ToArray(), shape, device);
        }

        public static ITensor Eye(int size, Device device = null)
        {
            var cpu = CpuBackend.Eye(size);
            return FromArray(cpu.ToArray(), new TensorShape(size, size), device);
        }

        public ITensor Add(ITensor other) => ElementwiseBinary(other, NativeAdd);
        public ITensor Subtract(ITensor other) => ElementwiseBinary(other, NativeSubtract);
        public ITensor Multiply(ITensor other) => ElementwiseBinary(other, NativeMultiply);
        public ITensor Divide(ITensor other) => ElementwiseBinary(other, NativeDivide);

        private ITensor ElementwiseBinary(ITensor other, Action<IntPtr, IntPtr, IntPtr, int> kernel)
        {
            if (other is not CudaBackend o)
                throw new InvalidOperationException("Both tensors must be on CUDA.");

            var resultShape = _shape.BroadcastTo(o.Shape);
            var result = new CudaBackend(resultShape, false, _device);

            kernel(_devicePtr, o._devicePtr, result._devicePtr, resultShape.TotalElements);

            var resultTensor = new CudaBackend(resultShape, result._devicePtr, _requiresGrad || o.RequiresGrad, _device);
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

        public ITensor Pow(float exponent)
        {
            var result = new CudaBackend(_shape, false, _device);
            PowScalar(_devicePtr, result._devicePtr, _shape.TotalElements, exponent);
            return result;
        }

        public ITensor LogicalNot()
        {
            var result = new CudaBackend(_shape, false, _device);
            NativeLogicalNot(_devicePtr, result._devicePtr, _shape.TotalElements);
            return result;
        }

        public ITensor Clip(float min, float max)
        {
            if (min > max) (min, max) = (max, min);
            var result = new CudaBackend(_shape, _requiresGrad, _device);
            NativeClip(_devicePtr, result._devicePtr, _shape.TotalElements, min, max);

            if (_requiresGrad)
            {
                var self = this;
                result.GradFn = grad =>
                {
                    var mask = new CudaBackend(_shape, false, _device);
                    NativeClipMask(self._devicePtr, mask._devicePtr, _shape.TotalElements, min, max);
                    var finalGrad = grad.Multiply(mask);
                    self.AccumulateGrad(finalGrad);
                    return finalGrad;
                };
            }
            return result;
        }

        public ITensor Negate() => ElementwiseUnary(NativeNegate);
        public ITensor Exp() => ElementwiseUnary(NativeExp);
        public ITensor Log() => ElementwiseUnary(NativeLog);
        public ITensor Sqrt() => ElementwiseUnary(NativeSqrt);
        public ITensor Abs() => ElementwiseUnary(NativeAbs);
        public ITensor Sin() => ElementwiseUnary(NativeSin);
        public ITensor Cos() => ElementwiseUnary(NativeCos);
        public ITensor Sign() => ElementwiseUnary(NativeSign);
        public ITensor Tanh() => new Tanh().Forward(this);
        public ITensor Relu() => new ReLU().Forward(this);
        public ITensor Sigmoid() => new Sigmoid().Forward(this);
        public ITensor Softmax(int axis = -1) => new Softmax(axis).Forward(this);

        private ITensor ElementwiseUnary(Action<IntPtr, IntPtr, int> kernel)
        {
            var result = new CudaBackend(_shape, false, _device);
            kernel(_devicePtr, result._devicePtr, _shape.TotalElements);
            return result;
        }

        public ITensor MatMul(ITensor other)
        {
            if (other is not CudaBackend o || _shape.Rank != 2 || o.Shape.Rank != 2)
                throw new InvalidOperationException("MatMul requires 2D CUDA tensors.");

            int m = _shape[0], k = _shape[1], n = o.Shape[1];
            var result = new CudaBackend(new TensorShape(m, n), false, _device);
            NativeMatMul(_devicePtr, o._devicePtr, result._devicePtr, m, n, k);
            return result;
        }

        public ITensor Transpose(int[] perm)
        {
            if (perm.Length == 2 && perm[0] == 1 && perm[1] == 0)
            {
                var result = new CudaBackend(new TensorShape(_shape[1], _shape[0]), false, _device);
                NativeTranspose(_devicePtr, result._devicePtr, _shape[0], _shape[1]);
                return result;
            }

            var newShape = new TensorShape(_shape.Dimensions.Select((d, i) => _shape.Dimensions[perm[i]]).ToArray());
            var resultGen = new CudaBackend(newShape, false, _device);
            NativeGeneralTranspose(_devicePtr, resultGen._devicePtr, _shape.Dimensions, perm, perm.Length);
            return resultGen;
        }

        public ITensor Reshape(params int[] newShape)
        {
            var ns = new TensorShape(newShape);
            if (ns.TotalElements != _shape.TotalElements)
                throw new ArgumentException("Cannot reshape to a different number of elements.");

            return new CudaBackend(ns, _devicePtr, _requiresGrad, _device);
        }

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

            CUDA.Broadcast(_devicePtr, result._devicePtr, alignedInputShape.Dimensions, targetShape.Dimensions);

            var resultTensor = new CudaBackend(targetShape, result._devicePtr, _requiresGrad, _device);
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

        public ITensor Sum(int? axis = null, bool keepDims = false)
        {
            if (axis is null)
            {
                var newShape = keepDims ? new TensorShape(Enumerable.Repeat(1, _shape.Rank).ToArray()) : new TensorShape(1);
                var result = new CudaBackend(newShape, false, _device);
                NativeSumAll(_devicePtr, result._devicePtr, _shape.TotalElements);
                return result;
            }
            return ((CpuBackend)this.To(Device.CPU)).Sum(axis, keepDims).To(_device);
        }

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

        public ITensor Mean(int[] axes, bool keepDims = false)
        {
            if (axes == null || axes.Length == 0)
            {
                var result = new CudaBackend(keepDims ? new TensorShape(Enumerable.Repeat(1, _shape.Rank).ToArray()) : new TensorShape(1), false, _device);
                CUDA.MeanAll(_devicePtr, result._devicePtr, _shape.TotalElements);
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

        public ITensor Mean(int? axis = null, bool keepDims = false)
        {
            if (axis is null)
            {
                var newShape = keepDims ? new TensorShape(Enumerable.Repeat(1, _shape.Rank).ToArray()) : new TensorShape(1);
                var result = new CudaBackend(newShape, false, _device);
                CUDA.MeanAll(_devicePtr, result._devicePtr, _shape.TotalElements);
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

            CUDA.MeanAxis(_devicePtr, resultTensor._devicePtr, outer, dim, inner);
            return resultTensor;
        }

        public ITensor Max(int axis = -1, bool keepDims = false) => ((CpuBackend)this.To(Device.CPU)).Max(axis, keepDims).To(_device);
        public ITensor Min(int axis = -1, bool keepDims = false) => ((CpuBackend)this.To(Device.CPU)).Min(axis, keepDims).To(_device);

        public ITensor Slice(params (int start, int end, int step)[] slices)
            => ((CpuBackend)this.To(Device.CPU)).Slice(slices).To(_device);

        public ITensor Concat(IEnumerable<ITensor> others, int axis = 0)
        {
            var cpuOthers = others.Select(o => (CpuBackend)o.To(Device.CPU));
            return ((CpuBackend)this.To(Device.CPU)).Concat(cpuOthers, axis).To(_device);
        }

        public ITensor Add(float scalar) => Add(FromScalar(scalar));
        public ITensor Subtract(float scalar) => Subtract(FromScalar(scalar));
        public ITensor Multiply(float scalar) => Multiply(FromScalar(scalar));
        public ITensor Divide(float scalar) => Divide(FromScalar(scalar));
        public ITensor Subtract(int other) => Subtract((float)other);
        public ITensor Multiply(double scalar) => Multiply((float)scalar);
        public ITensor Divide(double scalar) => Divide((float)scalar);

        public ITensor Pow(ITensor exponent)
            => ((CpuBackend)this.To(Device.CPU)).Pow(exponent).To(_device);

        public ITensor BroadcastAdd(ITensor other) => Add(other);

        public void Backward(ITensor? gradient = null)
        {
            ArborNet.Core.Autograd.AutogradEngine.Backward(this, gradient);
        }

        public void ClearGrad()
        {
            _grad = null;
            _gradFn = null;
        }

        public void SetData(float[] floats)
        {
            if (floats.Length != _shape.TotalElements)
                throw new ArgumentException("Data size does not match tensor shape.");
            CopyHostToDevice(floats, _devicePtr, floats.Length);
        }

        public bool IsCpu() => false;
        public bool IsCuda() => true;
        public IEnumerable<ITensor> Parameters() { yield return this; }

        public ITensor Equal(ITensor other)
        {
            if (other is not CudaBackend o || !_shape.Equals(o.Shape))
                throw new InvalidOperationException("Tensors must have same shape for equality.");

            var result = new CudaBackend(_shape, false, _device);
            NativeEqual(_devicePtr, o._devicePtr, result._devicePtr, _shape.TotalElements);
            return result;
        }

        private static void CopyHostToDevice(float[] source, IntPtr destination, int count)
        {
            var handle = GCHandle.Alloc(source, GCHandleType.Pinned);
            try
            {
                CudaMemcpy(destination, handle.AddrOfPinnedObject(),
                    (ulong)(count * sizeof(float)), cudaMemcpyKind.cudaMemcpyHostToDevice);
            }
            finally { handle.Free(); }
        }

        private static void CopyDeviceToHost(IntPtr devicePtr, float[] host, int count)
        {
            var handle = GCHandle.Alloc(host, GCHandleType.Pinned);
            try
            {
                CudaMemcpy(handle.AddrOfPinnedObject(), devicePtr,
                    (ulong)(count * sizeof(float)), cudaMemcpyKind.cudaMemcpyDeviceToHost);
            }
            finally { handle.Free(); }
        }

        public void Dispose()
        {
            if (!_disposed && _devicePtr != IntPtr.Zero)
            {
                ulong bytes = (ulong)_shape.TotalElements * sizeof(float);
                CudaMemoryPool.Instance.Free(_devicePtr, bytes);
                GC.RemoveMemoryPressure((long)bytes);
                _devicePtr = IntPtr.Zero;
            }
            _disposed = true;
            GC.SuppressFinalize(this);
        }

        ~CudaBackend() => Dispose();
    }
}
