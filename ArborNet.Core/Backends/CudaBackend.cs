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
    using ArborNet.Core.Autograd;
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

        public IntPtr Ptr => _ptr;

        public CudaAllocation(ulong bytes)
        {
            _bytes = bytes;
            _ptr = CudaMemoryPool.Instance.Allocate(bytes);
            CUDA.CudaMemset(_ptr, 0, bytes);
            _refCount = 1;
            GC.AddMemoryPressure((long)bytes);
        }

        public void AddRef()
        {
            lock (_lock)
            {
                _refCount++;
            }
        }

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

        public ITensor[] Inputs { get => _inputs; set => _inputs = value ?? Array.Empty<ITensor>(); }
        public TensorShape Shape => _shape;
        public Device Device => _device;
        public bool RequiresGrad { get => _requiresGrad; set => _requiresGrad = value; }
        public ITensor? Grad { get => _grad; set => _grad = value; }
        public Func<ITensor, ITensor>? GradFn { get => _gradFn; set => _gradFn = value; }
        public float[] Data => ToArray();
        public IntPtr DevicePointer => _allocation.Ptr;


        // =================================================================================
        // VERSION TRACKING (required for correct autograd)
        // =================================================================================
        private uint _version = 0;
        public uint Version => _version;

        // Call _version++; inside every in-place method (AddInPlace, MultiplyInPlace, etc.)

        // =================================================================================
        // DATA TYPE & CAST (zero-copy identity)
        // =================================================================================
        public string DType => "float32";

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

            GCHandle handle = GCHandle.Alloc(hostData, GCHandleType.Pinned);
            try
            {
                CUDA.CudaMemcpy(_allocation.Ptr, handle.AddrOfPinnedObject(), bytes, CUDA.cudaMemcpyKind.cudaMemcpyHostToDevice);
            }
            finally { handle.Free(); }
        }

        private CudaBackend(TensorShape shape, CudaAllocation allocation, bool requiresGrad, Device device)
        {
            _shape = shape.Clone();
            _allocation = allocation;
            _allocation.AddRef();
            _requiresGrad = requiresGrad;
            _device = device;
        }

        public float[] ToArray()
        {
            CUDA.Synchronize();
            float[] host = new float[_shape.TotalElements];
            ulong bytes = (ulong)_shape.TotalElements * sizeof(float);
            GCHandle handle = GCHandle.Alloc(host, GCHandleType.Pinned);
            try
            {
                CUDA.CudaMemcpy(handle.AddrOfPinnedObject(), _allocation.Ptr, bytes, CUDA.cudaMemcpyKind.cudaMemcpyDeviceToHost);
            }
            finally { handle.Free(); }
            return host;
        }

        public float ToScalar()
        {
            if (_shape.TotalElements != 1) throw new InvalidOperationException("Tensor is not a scalar.");
            CUDA.Synchronize();
            float[] host = new float[1];
            GCHandle handle = GCHandle.Alloc(host, GCHandleType.Pinned);
            try
            {
                CUDA.CudaMemcpy(handle.AddrOfPinnedObject(), _allocation.Ptr, sizeof(float), CUDA.cudaMemcpyKind.cudaMemcpyDeviceToHost);
            }
            finally { handle.Free(); }
            return host[0];
        }

        public ITensor Clone()
        {
            ulong bytes = (ulong)_shape.TotalElements * sizeof(float);
            var cloneAlloc = new CudaAllocation(bytes);
            CUDA.CudaMemcpy(cloneAlloc.Ptr, _allocation.Ptr, bytes, CUDA.cudaMemcpyKind.cudaMemcpyDeviceToDevice);
            return new CudaBackend(_shape, cloneAlloc, _requiresGrad, _device);
        }

        public ITensor To(Device targetDevice)
        {
            if (targetDevice.Type == DeviceType.CUDA) return Clone();
            if (targetDevice.Type == DeviceType.CPU) return new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, targetDevice);
            throw new NotSupportedException("Unsupported target device.");
        }

        public void AccumulateGrad(ITensor delta)
        {
            if (delta == null) return;
            var d = Tensor.Unwrap(delta) as CudaBackend ?? throw new ArgumentException("Gradients must match GPU device.");
            lock (_lock)
            {
                if (_grad == null)
                {
                    _grad = d.Clone();
                }
                else
                {
                    _grad.AddInPlace(d);
                }
            }
        }

        public void AddInPlace(ITensor other)
        {
            var o = Tensor.Unwrap(other) as CudaBackend ?? throw new ArgumentException("Operand must reside on CUDA.");
            _version++;                                   // ← ADD
            CUDA.NativeAdd(_allocation.Ptr, o.DevicePointer, _allocation.Ptr, _shape.TotalElements);
        }

        public void AddInPlace(float scalar)
        {
            _version++;                                   // ← ADD
            CUDA.NativeAddScalarInPlace(_allocation.Ptr, scalar, _shape.TotalElements);
        }

        public void SubtractInPlace(ITensor other)
        {
            var o = Tensor.Unwrap(other) as CudaBackend ?? throw new ArgumentException("Operand must reside on CUDA.");
            _version++;                                   // ← ADD
            CUDA.NativeSubtract(_allocation.Ptr, o.DevicePointer, _allocation.Ptr, _shape.TotalElements);
        }

        public void SubtractInPlace(float scalar)
        {
            _version++;                                   // ← ADD
            CUDA.NativeSubtractScalarInPlace(_allocation.Ptr, scalar, _shape.TotalElements);
        }

        public void MultiplyInPlace(ITensor other)
        {
            var o = Tensor.Unwrap(other) as CudaBackend ?? throw new ArgumentException("Operand must reside on CUDA.");
            _version++;                                   // ← ADD
            CUDA.NativeMultiply(_allocation.Ptr, o.DevicePointer, _allocation.Ptr, _shape.TotalElements);
        }

        public void MultiplyInPlace(float scalar)
        {
            _version++;                                   // ← ADD
            CUDA.NativeMultiplyScalarInPlace(_allocation.Ptr, scalar, _shape.TotalElements);
        }


        public ITensor Slice(params (int start, int end, int step)[] slices)
        {
            if (slices.Length != _shape.Rank) throw new ArgumentException("Slicing shape mismatch.");

            int[] starts = new int[_shape.Rank];
            int[] steps = new int[_shape.Rank];
            int[] newShapeList = new int[_shape.Rank];

            for (int i = 0; i < _shape.Rank; i++)
            {
                starts[i] = slices[i].start;
                int end = slices[i].end == -1 ? _shape.Dimensions[i] : slices[i].end;
                steps[i] = slices[i].step == 0 ? 1 : slices[i].step;
                newShapeList[i] = ((end - starts[i] - 1) / steps[i]) + 1;
            }

            var outShape = new TensorShape(newShapeList);
            var result = new CudaBackend(outShape, _requiresGrad, _device) { Inputs = new[] { this } };

            CUDA.NativeSlice(_allocation.Ptr, result.DevicePointer, _shape.Dimensions, outShape.Dimensions, starts, steps, _shape.Rank);

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
                    CUDA.CudaMemset(gradInput.DevicePointer, 0, (ulong)originalShape.TotalElements * sizeof(float));

                    var gradUnwrapped = Tensor.Unwrap(grad) as CudaBackend ?? throw new InvalidOperationException("Gradient must be on CUDA.");
                    CUDA.NativeSliceGrad(gradUnwrapped.DevicePointer, gradInput.DevicePointer, originalShape.Dimensions, newShapeArr, capturedStarts, capturedSteps, originalShape.Rank);

                    capturedSelf.AccumulateGrad(gradInput);
                    return gradInput;
                };
            }

            return result;
        }

        public ITensor Transpose(int[] perm)
        {
            if (perm.Length != _shape.Rank) throw new ArgumentException("Permutation layout does not match rank.");

            var outShape = new TensorShape(perm.Select(p => _shape.Dimensions[p]).ToArray());
            var result = new CudaBackend(outShape, _requiresGrad, _device) { Inputs = new[] { this } };

            CUDA.NativeGeneralTranspose(_allocation.Ptr, result.DevicePointer, _shape.Dimensions, perm, _shape.Rank);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                var capturedPerm = (int[])perm.Clone();
                result.GradFn = grad =>
                {
                    int[] invPerm = new int[capturedPerm.Length];
                    for (int i = 0; i < capturedPerm.Length; i++) invPerm[capturedPerm[i]] = i;
                    var gradSelf = grad.Transpose(invPerm);
                    capturedSelf.AccumulateGrad(gradSelf);
                    return grad;
                };
            }

            return result;
        }

        public ITensor Concat(IEnumerable<ITensor> others, int axis = 0)
        {
            var all = new List<CudaBackend> { this };
            foreach (var o in others)
            {
                var unwrapped = Tensor.Unwrap(o) as CudaBackend ?? throw new InvalidOperationException("All tensors must be on CUDA.");
                all.Add(unwrapped);
            }

            int rank = _shape.Rank;
            int actualAxis = axis < 0 ? rank + axis : axis;

            int[] newDims = _shape.Dimensions.ToArray();
            newDims[actualAxis] = all.Sum(t => t.Shape[actualAxis]);
            var outShape = new TensorShape(newDims);

            var result = new CudaBackend(outShape, _requiresGrad || all.Any(t => t.RequiresGrad), _device) { Inputs = all.ToArray() };

            int outerSize = 1;
            for (int i = 0; i < actualAxis; i++) outerSize *= newDims[i];
            int innerSize = 1;
            for (int i = actualAxis + 1; i < rank; i++) innerSize *= newDims[i];

            IntPtr[] inputPtrs = all.Select(t => t.DevicePointer).ToArray();
            int[] concatSizes = all.Select(t => t.Shape[actualAxis]).ToArray();

            CUDA.NativeConcat(inputPtrs, result.DevicePointer, all.Count, outerSize, concatSizes, innerSize);

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
                            t.AccumulateGrad(gradOutput.Slice(slices));
                        }
                        currentOffset += tAxisSize;
                    }
                    return gradOutput;
                };
            }

            return result;
        }

        public ITensor Gather(int axis, ITensor indices)
        {
            var idx = Tensor.Unwrap(indices) as CudaBackend ?? throw new ArgumentException("Indices must be on GPU.");
            int batch = _shape[0];
            int classes = _shape[1];

            var result = new CudaBackend(new TensorShape(batch), _requiresGrad, _device) { Inputs = new[] { this } };

            CUDA.NativeGather(_allocation.Ptr, idx.DevicePointer, result.DevicePointer, batch, classes);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                var capturedIndices = idx;
                result.GradFn = gradOutput =>
                {
                    var gradIn = new CudaBackend(capturedSelf._shape, false, capturedSelf._device);
                    CUDA.CudaMemset(gradIn.DevicePointer, 0, (ulong)capturedSelf._shape.TotalElements * sizeof(float));

                    var gradOutCuda = Tensor.Unwrap(gradOutput) as CudaBackend ?? throw new InvalidOperationException("Gradients must reside on CUDA.");
                    CUDA.NativeGatherGrad(gradOutCuda.DevicePointer, capturedIndices.DevicePointer, gradIn.DevicePointer, batch, classes);

                    capturedSelf.AccumulateGrad(gradIn);
                    return gradOutput;
                };
            }

            return result;
        }

        public ITensor Reshape(params int[] newShape)
        {
            var ns = new TensorShape(newShape);
            if (ns.TotalElements != _shape.TotalElements) throw new ArgumentException("Total elements volume mismatch.");

            var result = new CudaBackend(ns, _allocation, _requiresGrad, _device) { Inputs = new[] { this } };

            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = gradOutput =>
                {
                    capturedSelf.AccumulateGrad(gradOutput.Reshape(capturedSelf._shape.Dimensions));
                    return gradOutput;
                };
            }

            return result;
        }

        public ITensor BroadcastTo(TensorShape targetShape)
        {
            if (_shape.Equals(targetShape)) return Clone();

            int[] alignedInDims = Enumerable.Repeat(1, targetShape.Rank).ToArray();
            int offset = targetShape.Rank - _shape.Rank;
            for (int i = 0; i < _shape.Rank; i++) alignedInDims[i + offset] = _shape.Dimensions[i];

            var result = new CudaBackend(targetShape, false, _device);
            CUDA.NativeBroadcast(_allocation.Ptr, result.DevicePointer, alignedInDims, targetShape.Dimensions, targetShape.Rank);

            var resultTensor = new CudaBackend(targetShape, result._allocation, _requiresGrad, _device) { Inputs = new[] { this } };

            if (resultTensor.RequiresGrad)
            {
                var @self = this;
                resultTensor.GradFn = grad =>
                {
                    @self.AccumulateGrad(grad.BroadcastTo(@self._shape));
                    return grad;
                };
            }

            return resultTensor;
        }

        public ITensor ReshapeWithBroadcast(TensorShape target, int axis)
        {
            int targetRank = target.Rank;
            int actualAxis = axis < 0 ? targetRank + axis : axis;
            var viewDims = Enumerable.Repeat(1, targetRank).ToArray();
            int origIdx = 0;
            for (int i = actualAxis; i < targetRank && origIdx < _shape.Rank; i++)
            {
                viewDims[i] = _shape.Dimensions[origIdx++];
            }
            return Reshape(viewDims).BroadcastTo(target);
        }

        public ITensor Add(ITensor other) => ElementwiseBinary(other, CUDA.NativeAdd);
        public ITensor Subtract(ITensor other) => ElementwiseBinary(other, CUDA.NativeSubtract);
        public ITensor Multiply(ITensor other) => ElementwiseBinary(other, CUDA.NativeMultiply);
        public ITensor Divide(ITensor other) => ElementwiseBinary(other, CUDA.NativeDivide);

        private ITensor ElementwiseBinary(ITensor other, Action<IntPtr, IntPtr, IntPtr, int> kernel)
        {
            var o = Tensor.Unwrap(other) as CudaBackend ?? throw new ArgumentException("Operand must reside on CUDA.");
            var outShape = _shape.BroadcastTo(o.Shape);

            CudaBackend a = this;
            CudaBackend b = o;

            bool tempA = false;
            bool tempB = false;

            if (!_shape.Equals(outShape))
            {
                a = this.BroadcastTo(outShape) as CudaBackend ?? throw new InvalidOperationException("Failed to broadcast operand A.");
                tempA = true;
            }
            if (!o.Shape.Equals(outShape))
            {
                b = o.BroadcastTo(outShape) as CudaBackend ?? throw new InvalidOperationException("Failed to broadcast operand B.");
                tempB = true;
            }

            var result = new CudaBackend(outShape, _requiresGrad || o.RequiresGrad, _device) { Inputs = new[] { this, other } };

            kernel(a.DevicePointer, b.DevicePointer, result.DevicePointer, outShape.TotalElements);

            if (tempA) a.Dispose();
            if (tempB) b.Dispose();

            if (result.RequiresGrad)
            {
                var capturedSelf = this;
                var capturedOther = o;
                result.GradFn = grad =>
                {
                    if (capturedSelf.RequiresGrad) capturedSelf.AccumulateGrad(grad.BroadcastTo(capturedSelf._shape));
                    if (capturedOther.RequiresGrad) capturedOther.AccumulateGrad(grad.BroadcastTo(capturedOther.Shape));
                    return grad;
                };
            }

            return result;
        }

        public ITensor Add(float scalar)
        {
            var result = new CudaBackend(_shape, _requiresGrad, _device) { Inputs = new[] { this } };
            CUDA.NativeAddScalar(_allocation.Ptr, result.DevicePointer, _shape.TotalElements, scalar);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = grad =>
                {
                    capturedSelf.AccumulateGrad(grad);
                    return grad;
                };
            }
            return result;
        }

        public ITensor Subtract(float scalar)
        {
            var result = new CudaBackend(_shape, _requiresGrad, _device) { Inputs = new[] { this } };
            CUDA.NativeSubtractScalar(_allocation.Ptr, result.DevicePointer, _shape.TotalElements, scalar);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = grad =>
                {
                    capturedSelf.AccumulateGrad(grad);
                    return grad;
                };
            }
            return result;
        }

        public ITensor Multiply(float scalar)
        {
            var result = new CudaBackend(_shape, _requiresGrad, _device) { Inputs = new[] { this } };
            CUDA.NativeMultiplyScalar(_allocation.Ptr, result.DevicePointer, _shape.TotalElements, scalar);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = grad =>
                {
                    capturedSelf.AccumulateGrad(grad.Multiply(scalar));
                    return grad;
                };
            }
            return result;
        }

        public ITensor Divide(float scalar)
        {
            var result = new CudaBackend(_shape, _requiresGrad, _device) { Inputs = new[] { this } };
            CUDA.NativeDivideScalar(_allocation.Ptr, result.DevicePointer, _shape.TotalElements, scalar);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = grad =>
                {
                    capturedSelf.AccumulateGrad(grad.Divide(scalar));
                    return grad;
                };
            }
            return result;
        }

        public ITensor Subtract(int other) => Subtract((float)other);
        public ITensor Multiply(double scalar) => Multiply((float)scalar);
        public ITensor Divide(double scalar) => Divide((float)scalar);
        public ITensor Pow(ITensor exponent) => ElementwiseBinary(exponent, CUDA.NativePowTensor);
        public ITensor BroadcastAdd(ITensor other) => Add(other);

        public ITensor Pow(float exponent)
        {
            var result = new CudaBackend(_shape, _requiresGrad, _device) { Inputs = new[] { this } };
            CUDA.NativePowScalar(_allocation.Ptr, result.DevicePointer, _shape.TotalElements, exponent);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = gradOutput =>
                {
                    capturedSelf.AccumulateGrad(gradOutput.Multiply(exponent).Multiply(capturedSelf.Pow(exponent - 1)));
                    return gradOutput;
                };
            }

            return result;
        }

        public ITensor Negate() => ElementwiseUnary(CUDA.NativeNegate);

        public ITensor Exp()
        {
            var result = ElementwiseUnary(CUDA.NativeExp);
            if (_requiresGrad)
            {
                var capturedSelf = this;
                var capturedResult = result;
                result.GradFn = gradOutput =>
                {
                    capturedSelf.AccumulateGrad(gradOutput.Multiply(capturedResult));
                    return gradOutput;
                };
            }
            return result;
        }

        public ITensor Log()
        {
            var result = ElementwiseUnary(CUDA.NativeLog);
            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = gradOutput =>
                {
                    capturedSelf.AccumulateGrad(gradOutput.Divide(capturedSelf));
                    return gradOutput;
                };
            }
            return result;
        }

        public ITensor Sqrt()
        {
            var result = ElementwiseUnary(CUDA.NativeSqrt);
            if (_requiresGrad)
            {
                var capturedSelf = this;
                var capturedResult = result;
                result.GradFn = gradOutput =>
                {
                    capturedSelf.AccumulateGrad(gradOutput.Divide(capturedResult.Multiply(2.0f)));
                    return gradOutput;
                };
            }
            return result;
        }

        public ITensor Abs()
        {
            var result = ElementwiseUnary(CUDA.NativeAbs);
            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = gradOutput =>
                {
                    capturedSelf.AccumulateGrad(gradOutput.Multiply(capturedSelf.Sign()));
                    return gradOutput;
                };
            }
            return result;
        }

        public ITensor Sin()
        {
            var result = ElementwiseUnary(CUDA.NativeSin);
            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = gradOutput =>
                {
                    capturedSelf.AccumulateGrad(gradOutput.Multiply(capturedSelf.Cos()));
                    return gradOutput;
                };
            }
            return result;
        }

        public ITensor Cos()
        {
            var result = ElementwiseUnary(CUDA.NativeCos);
            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = gradOutput =>
                {
                    capturedSelf.AccumulateGrad(gradOutput.Multiply(capturedSelf.Sin().Negate()));
                    return gradOutput;
                };
            }
            return result;
        }

        public ITensor Sign() => ElementwiseUnary(CUDA.NativeSign);
        public ITensor Tanh() => new Tanh().Forward(this);
        public ITensor Relu() => new ReLU().Forward(this);
        public ITensor Sigmoid() => new Sigmoid().Forward(this);
        public ITensor Softmax(int axis = -1) => new Softmax(axis).Forward(this);

        private ITensor ElementwiseUnary(Action<IntPtr, IntPtr, int> kernel)
        {
            var result = new CudaBackend(_shape, _requiresGrad, _device) { Inputs = new[] { this } };
            kernel(_allocation.Ptr, result.DevicePointer, _shape.TotalElements);
            return result;
        }

        public ITensor MatMul(ITensor other)
        {
            if (other is not CudaBackend o || _shape.Rank != 2 || o.Shape.Rank != 2)
                throw new InvalidOperationException("MatMul requires 2D CUDA tensors.");

            int m = _shape[0];
            int k = _shape[1];
            int n = o.Shape[1];

            var result = new CudaBackend(new TensorShape(m, n), _requiresGrad || o.RequiresGrad, _device) { Inputs = new[] { this, other } };
            CUDA.NativeMatMul(_allocation.Ptr, o.DevicePointer, result.DevicePointer, m, n, k);

            if (result.RequiresGrad)
            {
                var capturedSelf = this;
                var capturedOther = o;

                result.GradFn = gradOutput =>
                {
                    if (capturedSelf.RequiresGrad) capturedSelf.AccumulateGrad(gradOutput.MatMul(capturedOther.Transpose(new[] { 1, 0 })));
                    if (capturedOther.RequiresGrad) capturedOther.AccumulateGrad(capturedSelf.Transpose(new[] { 1, 0 }).MatMul(gradOutput));
                    return gradOutput;
                };
            }

            return result;
        }

        public ITensor Sum(int? axis = null, bool keepDims = false)
        {
            if (axis == null)
            {
                var outShape = keepDims ? new TensorShape(Enumerable.Repeat(1, _shape.Rank).ToArray()) : new TensorShape(1);
                var result = new CudaBackend(outShape, _requiresGrad, _device) { Inputs = new[] { this } };
                CUDA.NativeSumAll(_allocation.Ptr, result.DevicePointer, _shape.TotalElements);

                if (_requiresGrad)
                {
                    var capturedSelf = this;
                    result.GradFn = grad =>
                    {
                        capturedSelf.AccumulateGrad(Ones(capturedSelf._shape, capturedSelf._device).Multiply(grad));
                        return grad;
                    };
                }
                return result;
            }

            int normAxis = axis.Value < 0 ? axis.Value + _shape.Rank : axis.Value;
            int dimSize = _shape[normAxis];
            return Mean(normAxis, keepDims).Multiply((float)dimSize);
        }

        public ITensor Sum(int[] axes, bool keepDims = false)
        {
            ITensor current = this;
            foreach (var axis in axes.OrderByDescending(a => a)) current = current.Sum(axis, keepDims);
            return current;
        }

        public ITensor Mean(int? axis = null, bool keepDims = false)
        {
            if (axis == null)
            {
                var outShape = keepDims ? new TensorShape(Enumerable.Repeat(1, _shape.Rank).ToArray()) : new TensorShape(1);
                var result = new CudaBackend(outShape, _requiresGrad, _device) { Inputs = new[] { this } };
                CUDA.NativeMeanAll(_allocation.Ptr, result.DevicePointer, _shape.TotalElements);

                if (_requiresGrad)
                {
                    var capturedSelf = this;
                    result.GradFn = grad =>
                    {
                        capturedSelf.AccumulateGrad(Ones(capturedSelf._shape, capturedSelf._device).Multiply(grad).Divide((float)capturedSelf._shape.TotalElements));
                        return grad;
                    };
                }
                return result;
            }

            int normAxis = axis.Value < 0 ? axis.Value + _shape.Rank : axis.Value;
            int dim = _shape[normAxis];
            int outer = 1; for (int i = 0; i < normAxis; i++) outer *= _shape[i];
            int inner = 1; for (int i = normAxis + 1; i < _shape.Rank; i++) inner *= _shape[i];

            int[] outDims = keepDims ? _shape.Dimensions.Select((d, i) => i == normAxis ? 1 : d).ToArray() : _shape.Dimensions.Where((_, i) => i != normAxis).ToArray();
            var resultTensor = new CudaBackend(new TensorShape(outDims), _requiresGrad, _device) { Inputs = new[] { this } };

            CUDA.NativeMeanAxis(_allocation.Ptr, resultTensor.DevicePointer, outer, dim, inner);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                resultTensor.GradFn = grad =>
                {
                    capturedSelf.AccumulateGrad(grad.BroadcastTo(capturedSelf._shape).Divide((float)dim));
                    return grad;
                };
            }

            return resultTensor;
        }

        public ITensor Mean(int[] axes, bool keepDims = false)
        {
            ITensor current = this;
            foreach (var axis in axes.OrderByDescending(a => a)) current = current.Mean(axis, keepDims);
            return current;
        }

        public ITensor Max(int axis = -1, bool keepDims = false)
        {
            int normAxis = axis < 0 ? axis + _shape.Rank : axis;
            int dim = _shape[normAxis];
            int outer = 1; for (int i = 0; i < normAxis; i++) outer *= _shape[i];
            int inner = 1; for (int i = normAxis + 1; i < _shape.Rank; i++) inner *= _shape[i];

            int[] outDims = keepDims ? _shape.Dimensions.Select((d, i) => i == normAxis ? 1 : d).ToArray() : _shape.Dimensions.Where((_, i) => i != normAxis).ToArray();
            var result = new CudaBackend(new TensorShape(outDims), _requiresGrad, _device) { Inputs = new[] { this } };

            CUDA.NativeMaxAxis(_allocation.Ptr, result.DevicePointer, outer, dim, inner);
            return result;
        }

        public ITensor Min(int axis = -1, bool keepDims = false)
        {
            int normAxis = axis < 0 ? axis + _shape.Rank : axis;
            int dim = _shape[normAxis];
            int outer = 1; for (int i = 0; i < normAxis; i++) outer *= _shape[i];
            int inner = 1; for (int i = normAxis + 1; i < _shape.Rank; i++) inner *= _shape[i];

            int[] outDims = keepDims ? _shape.Dimensions.Select((d, i) => i == normAxis ? 1 : d).ToArray() : _shape.Dimensions.Where((_, i) => i != normAxis).ToArray();
            var result = new CudaBackend(new TensorShape(outDims), _requiresGrad, _device) { Inputs = new[] { this } };

            CUDA.NativeMinAxis(_allocation.Ptr, result.DevicePointer, outer, dim, inner);
            return result;
        }

        public ITensor ArgMin(int axis)
        {
            int normAxis = axis < 0 ? axis + _shape.Rank : axis;
            int dim = _shape[normAxis];
            int outer = 1; for (int i = 0; i < normAxis; i++) outer *= _shape[i];
            int inner = 1; for (int i = normAxis + 1; i < _shape.Rank; i++) inner *= _shape[i];

            int[] outDims = _shape.Dimensions.Where((_, i) => i != normAxis).ToArray();
            var result = new CudaBackend(new TensorShape(outDims), false, _device);

            CUDA.NativeArgMin(_allocation.Ptr, result.DevicePointer, outer, dim, inner);
            return result;
        }

        public ITensor ArgMax(int axis)
        {
            int normAxis = axis < 0 ? axis + _shape.Rank : axis;
            int dim = _shape[normAxis];
            int outer = 1; for (int i = 0; i < normAxis; i++) outer *= _shape[i];
            int inner = 1; for (int i = normAxis + 1; i < _shape.Rank; i++) inner *= _shape[i];

            int[] outDims = _shape.Dimensions.Where((_, i) => i != normAxis).ToArray();
            var result = new CudaBackend(new TensorShape(outDims), false, _device);

            CUDA.NativeArgMax(_allocation.Ptr, result.DevicePointer, outer, dim, inner);
            return result;
        }

        public ITensor CumSum(int axis)
        {
            int normAxis = axis < 0 ? axis + _shape.Rank : axis;
            int dim = _shape[normAxis];
            int outer = 1; for (int i = 0; i < normAxis; i++) outer *= _shape[i];
            int inner = 1; for (int i = normAxis + 1; i < _shape.Rank; i++) inner *= _shape[i];

            var result = new CudaBackend(_shape, _requiresGrad, _device) { Inputs = new[] { this } };
            CUDA.NativeCumSum(_allocation.Ptr, result.DevicePointer, outer, dim, inner);
            return result;
        }

        public ITensor LogicalNot()
        {
            var result = new CudaBackend(_shape, false, _device);
            CUDA.NativeLogicalNot(_allocation.Ptr, result.DevicePointer, _shape.TotalElements);
            return result;
        }

        public ITensor Clip(float min, float max)
        {
            if (min > max) (min, max) = (max, min);
            var result = new CudaBackend(_shape, _requiresGrad, _device) { Inputs = new[] { this } };
            CUDA.NativeClip(_allocation.Ptr, result.DevicePointer, _shape.TotalElements, min, max);

            if (_requiresGrad)
            {
                var self = this;
                result.GradFn = grad =>
                {
                    var mask = new CudaBackend(_shape, false, _device);
                    CUDA.NativeClipMask(self._allocation.Ptr, mask.DevicePointer, _shape.TotalElements, min, max);
                    var finalGrad = grad.Multiply(mask);
                    self.AccumulateGrad(finalGrad);
                    return finalGrad;
                };
            }
            return result;
        }

        public void Backward(ITensor? gradient = null) => AutogradEngine.Backward(this, gradient);

        public void ClearGrad()
        {
            _grad = null;
            _gradFn = null;
        }

        public static ITensor Zeros(TensorShape shape, Device? device = null) => new CudaBackend(shape, false, device ?? Device.CUDA);

        public static ITensor Ones(TensorShape shape, Device? device = null)
        {
            var t = new CudaBackend(shape, false, device ?? Device.CUDA);
            CUDA.NativeOnes(t.DevicePointer, shape.TotalElements);
            return t;
        }

        public static ITensor FromScalar(float value, Device? device = null)
        {
            var t = new CudaBackend(new TensorShape(1), false, device ?? Device.CUDA);
            CUDA.NativeSetScalar(t.DevicePointer, value, 1);
            return t;
        }

        public static ITensor FromArray(float[] data, TensorShape shape, Device? device = null)
            => new CudaBackend(data, shape, false, device ?? Device.CUDA);

        public static ITensor Rand(TensorShape shape, Device? device = null)
        {
            var t = new CudaBackend(shape, false, device ?? Device.CUDA);
            uint seed = (uint)Guid.NewGuid().GetHashCode();
            CUDA.NativeRand(t.DevicePointer, shape.TotalElements, seed);
            return t;
        }

        public static ITensor Randn(TensorShape shape, Device? device = null)
        {
            var t = new CudaBackend(shape, false, device ?? Device.CUDA);
            uint seed = (uint)Guid.NewGuid().GetHashCode();
            CUDA.NativeRandn(t.DevicePointer, shape.TotalElements, seed);
            return t;
        }

        public static ITensor Eye(int size, Device? device = null)
        {
            var t = new CudaBackend(new TensorShape(size, size), false, device ?? Device.CUDA);
            CUDA.NativeEye(t.DevicePointer, size);
            return t;
        }

        public void SetData(float[] floats)
        {
            if (floats.Length != _shape.TotalElements) throw new ArgumentException("Data volume mismatch.");
            GCHandle handle = GCHandle.Alloc(floats, GCHandleType.Pinned);
            try
            {
                CUDA.CudaMemcpy(_allocation.Ptr, handle.AddrOfPinnedObject(), (ulong)(floats.Length * sizeof(float)), CUDA.cudaMemcpyKind.cudaMemcpyHostToDevice);
            }
            finally { handle.Free(); }
        }

        public bool IsCpu() => false;
        public bool IsCuda() => true;
        public IEnumerable<ITensor> Parameters() { yield return this; }

        // =================================================================================
        // COMPARISON OPERATIONS (No CPU Copies)
        // =================================================================================

        public ITensor GreaterThan(ITensor other)
            => ElementwiseBinary(other, CUDA.NativeGreaterThan);

        public ITensor GreaterThanOrEqual(ITensor other)
            => ElementwiseBinary(other, CUDA.NativeGreaterThanOrEqual);

        public ITensor LessEqual(ITensor other)
            => ElementwiseBinary(other, CUDA.NativeLessEqual);

        public ITensor Equal(ITensor other)
            => ElementwiseBinary(other, CUDA.NativeEqual);

        // =================================================================================
        // CONDITIONAL SELECTION WITH AUTOGRAD SUPPORT (Pure GPU-side execution)
        // =================================================================================

        public ITensor Where(ITensor condition, ITensor trueValue, ITensor falseValue)
        {
            var cond = Tensor.Unwrap(condition) as CudaBackend ?? throw new ArgumentException("Condition must reside on CUDA.");
            var tv = Tensor.Unwrap(trueValue) as CudaBackend ?? throw new ArgumentException("True value must reside on CUDA.");
            var fv = Tensor.Unwrap(falseValue) as CudaBackend ?? throw new ArgumentException("False value must reside on CUDA.");

            var targetShape = _shape.BroadcastTo(cond.Shape).BroadcastTo(tv.Shape).BroadcastTo(fv.Shape);

            CudaBackend c = cond;
            CudaBackend t = tv;
            CudaBackend f = fv;

            bool tempC = false, tempT = false, tempF = false;

            if (!cond.Shape.Equals(targetShape)) { c = cond.BroadcastTo(targetShape) as CudaBackend; tempC = true; }
            if (!tv.Shape.Equals(targetShape)) { t = tv.BroadcastTo(targetShape) as CudaBackend; tempT = true; }
            if (!fv.Shape.Equals(targetShape)) { f = fv.BroadcastTo(targetShape) as CudaBackend; tempF = true; }

            var result = new CudaBackend(targetShape, tv.RequiresGrad || fv.RequiresGrad, _device)
            {
                Inputs = new[] { condition, trueValue, falseValue }
            };

            CUDA.NativeWhere(c.DevicePointer, t.DevicePointer, f.DevicePointer, result.DevicePointer, targetShape.TotalElements);

            if (tempC) c.Dispose();
            if (tempT) t.Dispose();
            if (tempF) f.Dispose();

            if (result.RequiresGrad)
            {
                var capturedCond = cond;
                var capturedTrue = tv;
                var capturedFalse = fv;

                result.GradFn = grad =>
                {
                    if (capturedTrue.RequiresGrad)
                    {
                        using var zeros = new CudaBackend(grad.Shape, false, _device);
                        var gradTrue = grad.Where(capturedCond, grad, zeros);
                        capturedTrue.AccumulateGrad(gradTrue);
                    }
                    if (capturedFalse.RequiresGrad)
                    {
                        using var zeros = new CudaBackend(grad.Shape, false, _device);
                        var gradFalse = grad.Where(capturedCond, zeros, grad);
                        capturedFalse.AccumulateGrad(gradFalse);
                    }
                    return grad;
                };
            }

            return result;
        }

        public ITensor Cast(string dtype)
        {
            if (dtype != "float32" && dtype != "float" && dtype != "f32")
                throw new NotSupportedException($"Only float32 is currently supported. Requested: {dtype}");
            return this; // pure identity – zero copy, same allocation
        }

        // =================================================================================
        // SQUEEZE (pure view – zero copy)  – FIXED
        // =================================================================================
        public ITensor Squeeze(int? axis = null)
        {
            if (axis == null)
            {
                var newDims = _shape.Dimensions.Where(d => d != 1).ToArray();
                if (newDims.Length == 0)
                    newDims = new[] { 1 };
                return Reshape(newDims);
            }

            int a = axis.Value < 0 ? _shape.Rank + axis.Value : axis.Value;
            if (a < 0 || a >= _shape.Rank)
                throw new ArgumentOutOfRangeException(nameof(axis));

            if (_shape.Dimensions[a] != 1)
                throw new InvalidOperationException($"Cannot squeeze axis {a} of size {_shape.Dimensions[a]}.");

            var dims = _shape.Dimensions.ToList();
            dims.RemoveAt(a);
            if (dims.Count == 0)
                dims.Add(1);

            return Reshape(dims.ToArray()); // shares CudaAllocation – zero copy
        }

        // =================================================================================
        // UNSQUEEZE (pure view – zero copy)
        // =================================================================================
        public ITensor Unsqueeze(int axis)
        {
            int rank = _shape.Rank;
            int actualAxis = axis < 0 ? rank + axis + 1 : axis;

            if (actualAxis < 0 || actualAxis > rank)
                throw new ArgumentOutOfRangeException(nameof(axis));

            var newDims = new int[rank + 1];
            for (int i = 0, j = 0; i < newDims.Length; i++)
            {
                newDims[i] = (i == actualAxis) ? 1 : _shape.Dimensions[j++];
            }

            return Reshape(newDims); // shares CudaAllocation – zero copy
        }

        // =================================================================================
        // TOP-K (structured for maximum efficiency)
        // Currently falls back to CPU only because a full native segmented Top-K
        // kernel is non-trivial. The structure below is ready for a native
        // CUDA implementation (CUB / bitonic / heap) with zero host traffic.
        // =================================================================================
        // =================================================================================
        // TOP-K  – NATIVE CUDA (zero host traffic)
        // =================================================================================
        public (ITensor values, ITensor indices) TopK(int k, int axis = -1)
        {
            if (k <= 0)
                throw new ArgumentOutOfRangeException(nameof(k));

            int normAxis = axis < 0 ? _shape.Rank + axis : axis;
            if (normAxis < 0 || normAxis >= _shape.Rank)
                throw new ArgumentOutOfRangeException(nameof(axis));

            int dim = _shape[normAxis];
            if (k > dim)
                throw new ArgumentOutOfRangeException(nameof(k), "k cannot be larger than the size of the axis.");

            // outer / inner calculation (same pattern used by ArgMax / Mean etc.)
            int outer = 1;
            for (int i = 0; i < normAxis; i++) outer *= _shape[i];
            int inner = 1;
            for (int i = normAxis + 1; i < _shape.Rank; i++) inner *= _shape[i];

            // Output shape: original with the reduced axis replaced by k
            int[] outDims = (int[])_shape.Dimensions.Clone();
            outDims[normAxis] = k;
            var outShape = new TensorShape(outDims);

            var valuesBackend = new CudaBackend(outShape, _requiresGrad, _device) { Inputs = new[] { this } };
            var indicesBackend = new CudaBackend(outShape, false, _device);

            // Pure device execution – zero host traffic
            CUDA.NativeTopK(
                _allocation.Ptr,
                valuesBackend.DevicePointer,
                indicesBackend.DevicePointer,
                outer, dim, inner, k);

            // Autograd: scatter gradients back to original positions (also pure device)
            if (_requiresGrad)
            {
                var capturedSelf = this;
                var capturedIndices = indicesBackend;
                var capturedOuter = outer;
                var capturedDim = dim;
                var capturedInner = inner;
                var capturedK = k;
                var originalShape = _shape.Clone();

                valuesBackend.GradFn = gradOutput =>
                {
                    var gradIn = new CudaBackend(originalShape, false, capturedSelf._device);
                    CUDA.CudaMemset(gradIn.DevicePointer, 0,
                        (ulong)originalShape.TotalElements * sizeof(float));

                    var go = Tensor.Unwrap(gradOutput) as CudaBackend
                        ?? throw new InvalidOperationException("Gradient must be on CUDA.");

                    CUDA.NativeTopKScatterGrad(
                        go.DevicePointer,
                        capturedIndices.DevicePointer,
                        gradIn.DevicePointer,
                        capturedOuter, capturedDim, capturedInner, capturedK);

                    capturedSelf.AccumulateGrad(gradIn);
                    return gradOutput;
                };
            }

            return (new Tensor(valuesBackend), new Tensor(indicesBackend));
        }


        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                // Unmanaged resource release via reference counting
                _allocation?.Release();
                _disposed = true;
            }
        }

        public void Dispose()
        {
            lock (_lock) { Dispose(true); }
            GC.SuppressFinalize(this);
        }

        ~CudaBackend() => Dispose(false);
    }
}