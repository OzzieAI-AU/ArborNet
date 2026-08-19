// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// Project: ArborNet
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Backends
{
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

    internal sealed class CudaAllocation : IDisposable
    {
        private IntPtr _ptr;
        private readonly ulong _bytes;
        private int _refCount;
        private readonly object _lock = new();
        private bool _released;

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
                if (_released) throw new ObjectDisposedException(nameof(CudaAllocation));
                _refCount++;
            }
        }

        public void Release()
        {
            lock (_lock)
            {
                if (_released) return;
                _refCount--;
                if (_refCount > 0) return;

                if (_ptr != IntPtr.Zero)
                {
                    try { CudaMemoryPool.Instance.Free(_ptr, _bytes); }
                    catch { /* context may already be torn down */ }
                    GC.RemoveMemoryPressure((long)_bytes);
                    _ptr = IntPtr.Zero;
                }
                _released = true;
            }
        }

        public void Dispose()
        {
            Release();
            GC.SuppressFinalize(this);
        }

        // Never touch CUDA from a finalizer.
        ~CudaAllocation() { }
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
        private uint _version;

        public ITensor[] Inputs { get => _inputs; set => _inputs = value ?? Array.Empty<ITensor>(); }
        public TensorShape Shape => _shape;
        public Device Device => _device;
        public bool RequiresGrad { get => _requiresGrad; set => _requiresGrad = value; }
        public ITensor? Grad { get => _grad; set => _grad = value; }
        public Func<ITensor, ITensor>? GradFn { get => _gradFn; set => _gradFn = value; }
        public float[] Data => ToArray();
        public IntPtr DevicePointer => _allocation.Ptr;
        public uint Version => _version;
        public string DType => "float32";

        public CudaBackend(TensorShape shape, bool requiresGrad = false, Device? device = null)
        {
            _shape = shape?.Clone() ?? throw new ArgumentNullException(nameof(shape));
            _device = device ?? Device.CUDA;
            _requiresGrad = requiresGrad;
            _allocation = new CudaAllocation((ulong)_shape.TotalElements * sizeof(float));
        }

        public CudaBackend(float[] hostData, TensorShape shape, bool requiresGrad = false, Device? device = null)
            : this(shape, requiresGrad, device)
        {
            if (hostData == null) throw new ArgumentNullException(nameof(hostData));
            if (hostData.Length != _shape.TotalElements)
                throw new ArgumentException("Host data length does not match shape.");
            CUDA.CopyHostToDeviceFast(hostData, _allocation.Ptr);
        }

        private CudaBackend(TensorShape shape, CudaAllocation allocation, bool requiresGrad, Device device)
        {
            _shape = shape.Clone();
            _allocation = allocation;
            _allocation.AddRef();
            _requiresGrad = requiresGrad;
            _device = device;
        }

        private static void DisposeIfTemp(ITensor? t, ITensor? keep)
        {
            if (t != null && !ReferenceEquals(t, keep) && t is IDisposable d)
                d.Dispose();
        }

        public float[] ToArray()
        {
            CUDA.Synchronize();
            float[] host = new float[_shape.TotalElements];
            CUDA.CopyDeviceToHostFast(_allocation.Ptr, host);
            return host;
        }

        public float ToScalar()
        {
            if (_shape.TotalElements != 1) throw new InvalidOperationException("Tensor is not a scalar.");
            return ToArray()[0];
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

        public ITensor SumTo(TensorShape target)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));
            if (_shape.Equals(target)) return Clone();

            if (target.Rank > _shape.Rank)
                throw new ArgumentException("Cannot SumTo a higher-rank shape.");

            int rank = _shape.Rank;
            int[] aligned = new int[rank];
            int off = rank - target.Rank;
            for (int i = 0; i < rank; i++)
                aligned[i] = i < off ? 1 : target.Dimensions[i - off];

            for (int i = 0; i < rank; i++)
            {
                int s = _shape.Dimensions[i];
                int t = aligned[i];
                if (t != s && t != 1)
                    throw new ArgumentException($"Cannot SumTo: dimension {i} is {s}, target {t}.");
            }

            var alignedShape = new TensorShape(aligned);
            var result = new CudaBackend(alignedShape, false, _device);
            CUDA.NativeSumTo(_allocation.Ptr, result.DevicePointer, _shape.Dimensions, aligned, rank);

            if (target.Rank != rank)
                return result.Reshape(target.Dimensions.Length == 0 ? new[] { 1 } : target.Dimensions);
            return result;
        }

        public void AccumulateGrad(ITensor delta)
        {
            if (delta == null) return;
            var d = Tensor.Unwrap(delta) as CudaBackend ?? throw new ArgumentException("Gradients must match GPU device.");

            CudaBackend toAdd = d;
            bool disposeToAdd = false;
            if (!d.Shape.Equals(_shape))
            {
                toAdd = (CudaBackend)d.SumTo(_shape);
                disposeToAdd = true;
            }

            try
            {
                lock (_lock)
                {
                    if (_grad == null) _grad = toAdd.Clone();
                    else _grad.AddInPlace(toAdd);
                }
            }
            finally
            {
                if (disposeToAdd) toAdd.Dispose();
            }
        }

        private void EnsureSameSize(ITensor other, string op)
        {
            var o = Tensor.Unwrap(other) as CudaBackend ?? throw new ArgumentException("Operand must reside on CUDA.");
            if (o.Shape.TotalElements != _shape.TotalElements)
                throw new ArgumentException($"{op}: in-place size mismatch ({_shape.TotalElements} vs {o.Shape.TotalElements}).");
        }

        public void AddInPlace(ITensor other)
        {
            EnsureSameSize(other, nameof(AddInPlace));
            var o = (CudaBackend)Tensor.Unwrap(other);
            _version++;
            CUDA.NativeAdd(_allocation.Ptr, o.DevicePointer, _allocation.Ptr, _shape.TotalElements);
        }

        public void AddInPlace(float scalar)
        {
            _version++;
            CUDA.NativeAddScalarInPlace(_allocation.Ptr, scalar, _shape.TotalElements);
        }

        public void SubtractInPlace(ITensor other)
        {
            EnsureSameSize(other, nameof(SubtractInPlace));
            var o = (CudaBackend)Tensor.Unwrap(other);
            _version++;
            CUDA.NativeSubtract(_allocation.Ptr, o.DevicePointer, _allocation.Ptr, _shape.TotalElements);
        }

        public void SubtractInPlace(float scalar)
        {
            _version++;
            CUDA.NativeSubtractScalarInPlace(_allocation.Ptr, scalar, _shape.TotalElements);
        }

        public void MultiplyInPlace(ITensor other)
        {
            EnsureSameSize(other, nameof(MultiplyInPlace));
            var o = (CudaBackend)Tensor.Unwrap(other);
            _version++;
            CUDA.NativeMultiply(_allocation.Ptr, o.DevicePointer, _allocation.Ptr, _shape.TotalElements);
        }

        public void MultiplyInPlace(float scalar)
        {
            _version++;
            CUDA.NativeMultiplyScalarInPlace(_allocation.Ptr, scalar, _shape.TotalElements);
        }

        public ITensor Slice(params (int start, int end, int step)[] slices)
        {
            if (slices == null || slices.Length != _shape.Rank)
                throw new ArgumentException("Slicing shape mismatch.");

            int[] starts = new int[_shape.Rank];
            int[] steps = new int[_shape.Rank];
            int[] newShapeList = new int[_shape.Rank];

            for (int i = 0; i < _shape.Rank; i++)
            {
                int dim = _shape.Dimensions[i];
                int step = slices[i].step == 0 ? 1 : slices[i].step;
                int start = slices[i].start;
                int end = slices[i].end;

                // end == -1 means "until the edge" in this API (positive: dim, negative: -1)
                if (end == -1)
                    end = step > 0 ? dim : -1;

                if (start < 0 || start >= dim)
                    throw new ArgumentOutOfRangeException(nameof(slices), $"start {start} out of range on axis {i}.");

                int count;
                if (step > 0)
                {
                    if (end < start) count = 0;
                    else count = (end - start + step - 1) / step;
                }
                else
                {
                    int absStep = -step;
                    if (end > start) count = 0;
                    else count = (start - end + absStep - 1) / absStep;
                }

                // Last visited index must lie in [0, dim)
                if (count > 0)
                {
                    int last = start + (count - 1) * step;
                    if (last < 0 || last >= dim)
                        throw new ArgumentOutOfRangeException(nameof(slices), $"slice on axis {i} walks out of bounds.");
                }

                starts[i] = start;
                steps[i] = step;
                newShapeList[i] = count;
            }

            var outShape = new TensorShape(newShapeList);
            var result = new CudaBackend(outShape, _requiresGrad, _device) { Inputs = new[] { this } };

            if (outShape.TotalElements > 0)
                CUDA.NativeSlice(_allocation.Ptr, result.DevicePointer, _shape.Dimensions, outShape.Dimensions, starts, steps, _shape.Rank);

            if (_requiresGrad)
            {
                var capturedStarts = (int[])starts.Clone();
                var capturedSteps = (int[])steps.Clone();
                var originalShape = _shape.Clone();
                var newShapeArr = (int[])newShapeList.Clone();
                var capturedSelf = this;

                result.GradFn = grad =>
                {
                    var gradInput = new CudaBackend(originalShape, false, _device);
                    var go = Tensor.Unwrap(grad) as CudaBackend
                        ?? throw new InvalidOperationException("Gradient must be on CUDA.");
                    if (go.Shape.TotalElements > 0)
                        CUDA.NativeSliceGrad(go.DevicePointer, gradInput.DevicePointer,
                            originalShape.Dimensions, newShapeArr, capturedStarts, capturedSteps, originalShape.Rank);
                    capturedSelf.AccumulateGrad(gradInput);
                    gradInput.Dispose();
                    return grad;
                };
            }
            return result;
        }

        public ITensor Transpose(int[] perm)
        {
            if (perm == null || perm.Length != _shape.Rank) throw new ArgumentException("Permutation layout does not match rank.");
            var seen = new bool[perm.Length];
            for (int i = 0; i < perm.Length; i++)
            {
                if (perm[i] < 0 || perm[i] >= perm.Length || seen[perm[i]])
                    throw new ArgumentException("perm must be a valid permutation.");
                seen[perm[i]] = true;
            }

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
                    DisposeIfTemp(gradSelf, grad);
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
            if (actualAxis < 0 || actualAxis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));

            for (int t = 1; t < all.Count; t++)
            {
                if (all[t].Shape.Rank != rank) throw new ArgumentException("Concat rank mismatch.");
                for (int i = 0; i < rank; i++)
                {
                    if (i == actualAxis) continue;
                    if (all[t].Shape[i] != _shape[i])
                        throw new ArgumentException($"Concat non-axis dimension {i} mismatch.");
                }
            }

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
                                slices[i] = i == capturedAxis
                                    ? (currentOffset, currentOffset + tAxisSize, 1)
                                    : (0, t.Shape[i], 1);
                            }
                            var part = gradOutput.Slice(slices);
                            t.AccumulateGrad(part);
                            DisposeIfTemp(part, gradOutput);
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
            var idx = Tensor.Unwrap(indices) as CudaBackend
                ?? throw new ArgumentException("Indices must be on GPU.");

            int normAxis = axis < 0 ? axis + _shape.Rank : axis;
            if (normAxis < 0 || normAxis >= _shape.Rank)
                throw new ArgumentOutOfRangeException(nameof(axis));

            int dim = _shape[normAxis];
            int outer = 1;
            for (int i = 0; i < normAxis; i++) outer *= _shape[i];
            int inner = 1;
            for (int i = normAxis + 1; i < _shape.Rank; i++) inner *= _shape[i];

            int k;
            int[] outDims;

            if (idx.Shape.Rank == _shape.Rank)
            {
                for (int i = 0; i < _shape.Rank; i++)
                {
                    if (i == normAxis) continue;
                    if (idx.Shape[i] != _shape[i])
                        throw new ArgumentException($"Gather indices dim {i} ({idx.Shape[i]}) != input ({_shape[i]}).");
                }
                k = idx.Shape[normAxis];
                if (k <= 0) throw new ArgumentException("Gather k must be > 0.");
                outDims = (int[])idx.Shape.Dimensions.Clone();
            }
            else if (idx.Shape.Rank == _shape.Rank - 1)
            {
                int[] squeezed = _shape.Dimensions.Where((_, i) => i != normAxis).ToArray();
                if (squeezed.Length == 0) squeezed = new[] { 1 };
                if (!idx.Shape.Dimensions.SequenceEqual(squeezed) && idx.Shape.TotalElements != outer * inner)
                    throw new ArgumentException("Gather squeezed indices must match input without the gathered axis.");
                k = 1;
                outDims = squeezed;
            }
            else
            {
                throw new ArgumentException("Gather indices rank must equal input rank or input rank - 1.");
            }

            if (idx.Shape.TotalElements != (long)outer * k * inner)
                throw new ArgumentException("Gather indices volume does not match outer * k * inner.");

            var result = new CudaBackend(new TensorShape(outDims), _requiresGrad, _device)
            {
                Inputs = new ITensor[] { this, indices }
            };

            CUDA.NativeGatherAxis(_allocation.Ptr, idx.DevicePointer, result.DevicePointer, outer, dim, inner, k);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                var capturedIndices = idx;
                result.GradFn = gradOutput =>
                {
                    var gradIn = new CudaBackend(capturedSelf._shape, false, capturedSelf._device);
                    var go = Tensor.Unwrap(gradOutput) as CudaBackend
                        ?? throw new InvalidOperationException("Gradient must be on CUDA.");
                    CUDA.NativeGatherAxisGrad(go.DevicePointer, capturedIndices.DevicePointer,
                        gradIn.DevicePointer, outer, dim, inner, k);
                    capturedSelf.AccumulateGrad(gradIn);
                    gradIn.Dispose();
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
                    var reshaped = gradOutput.Reshape(capturedSelf._shape.Dimensions);
                    capturedSelf.AccumulateGrad(reshaped);
                    DisposeIfTemp(reshaped, gradOutput);
                    return gradOutput;
                };
            }
            return result;
        }

        public ITensor BroadcastTo(TensorShape targetShape)
        {
            if (_shape.Equals(targetShape)) return Clone();
            if (targetShape.Rank > 12) throw new NotSupportedException("Rank > 12 is not supported.");

            int[] alignedInDims = Enumerable.Repeat(1, targetShape.Rank).ToArray();
            int offset = targetShape.Rank - _shape.Rank;
            if (offset < 0) throw new ArgumentException("Cannot broadcast a higher-rank tensor to a lower-rank shape.");
            for (int i = 0; i < _shape.Rank; i++)
            {
                int src = _shape.Dimensions[i];
                int dst = targetShape.Dimensions[i + offset];
                if (src != dst && src != 1)
                    throw new ArgumentException($"Cannot broadcast dim {src} to {dst}.");
                alignedInDims[i + offset] = src;
            }

            var result = new CudaBackend(targetShape, _requiresGrad, _device) { Inputs = new[] { this } };
            CUDA.NativeBroadcast(_allocation.Ptr, result.DevicePointer, alignedInDims, targetShape.Dimensions, targetShape.Rank);

            if (result.RequiresGrad)
            {
                var self = this;
                result.GradFn = grad =>
                {
                    self.AccumulateGrad(grad);
                    return grad;
                };
            }
            return result;
        }

        public ITensor ReshapeWithBroadcast(TensorShape target, int axis)
        {
            int targetRank = target.Rank;
            int actualAxis = axis < 0 ? targetRank + axis : axis;
            var viewDims = Enumerable.Repeat(1, targetRank).ToArray();
            int origIdx = 0;
            for (int i = actualAxis; i < targetRank && origIdx < _shape.Rank; i++)
                viewDims[i] = _shape.Dimensions[origIdx++];
            return Reshape(viewDims).BroadcastTo(target);
        }

        public ITensor Add(ITensor other) =>
            ElementwiseBinary(other, CUDA.NativeAdd, (g, a, b) => (g, g));

        public ITensor Subtract(ITensor other) =>
            ElementwiseBinary(other, CUDA.NativeSubtract, (g, a, b) => (g, g.Negate()));

        public ITensor Multiply(ITensor other) =>
            ElementwiseBinary(other, CUDA.NativeMultiply, (g, a, b) => (g.Multiply(b), g.Multiply(a)));

        public ITensor Divide(ITensor other) =>
            ElementwiseBinary(other, CUDA.NativeDivide, (g, a, b) => (g.Divide(b), g.Multiply(a.Negate()).Divide(b.Multiply(b))));

        public ITensor GreaterThan(ITensor other) => ElementwiseBinary(other, CUDA.NativeGreaterThan, null, false);
        public ITensor GreaterThanOrEqual(ITensor other) => ElementwiseBinary(other, CUDA.NativeGreaterThanOrEqual, null, false);
        public ITensor LessThan(ITensor other) => ElementwiseBinary(other, CUDA.NativeLessThan, null, false);
        public ITensor LessEqual(ITensor other) => ElementwiseBinary(other, CUDA.NativeLessEqual, null, false);
        public ITensor Equal(ITensor other) => ElementwiseBinary(other, CUDA.NativeEqual, null, false);

        private ITensor ElementwiseBinary(
            ITensor other,
            Action<IntPtr, IntPtr, IntPtr, int> kernel,
            Func<ITensor, ITensor, ITensor, (ITensor ga, ITensor gb)>? gradRule,
            bool allowGrad = true)
        {
            var o = Tensor.Unwrap(other) as CudaBackend ?? throw new ArgumentException("Operand must reside on CUDA.");
            var outShape = _shape.BroadcastTo(o.Shape);

            CudaBackend a = this;
            CudaBackend b = o;
            bool tempA = false, tempB = false;

            if (!_shape.Equals(outShape))
            {
                a = (CudaBackend)BroadcastTo(outShape);
                tempA = true;
            }
            if (!o.Shape.Equals(outShape))
            {
                b = (CudaBackend)o.BroadcastTo(outShape);
                tempB = true;
            }

            bool req = allowGrad && (_requiresGrad || o.RequiresGrad);
            var result = new CudaBackend(outShape, req, _device) { Inputs = new[] { this, other } };
            try
            {
                kernel(a.DevicePointer, b.DevicePointer, result.DevicePointer, outShape.TotalElements);
            }
            finally
            {
                if (tempA) a.Dispose();
                if (tempB) b.Dispose();
            }

            if (req && gradRule != null)
            {
                var capturedSelf = this;
                var capturedOther = o;
                result.GradFn = grad =>
                {
                    var (ga, gb) = gradRule(grad, capturedSelf, capturedOther);
                    try
                    {
                        if (capturedSelf.RequiresGrad) capturedSelf.AccumulateGrad(ga);
                        if (capturedOther.RequiresGrad) capturedOther.AccumulateGrad(gb);
                    }
                    finally
                    {
                        DisposeIfTemp(ga, grad);
                        DisposeIfTemp(gb, grad);
                    }
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
                result.GradFn = grad => { capturedSelf.AccumulateGrad(grad); return grad; };
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
                result.GradFn = grad => { capturedSelf.AccumulateGrad(grad); return grad; };
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
                    var scaled = grad.Multiply(scalar);
                    capturedSelf.AccumulateGrad(scaled);
                    DisposeIfTemp(scaled, grad);
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
                    var scaled = grad.Divide(scalar);
                    capturedSelf.AccumulateGrad(scaled);
                    DisposeIfTemp(scaled, grad);
                    return grad;
                };
            }
            return result;
        }

        public ITensor Subtract(int other) => Subtract((float)other);
        public ITensor Multiply(double scalar) => Multiply((float)scalar);
        public ITensor Divide(double scalar) => Divide((float)scalar);
        public ITensor BroadcastAdd(ITensor other) => Add(other);

        public ITensor Pow(float exponent)
        {
            var result = new CudaBackend(_shape, _requiresGrad, _device) { Inputs = new[] { this } };
            CUDA.NativePowScalar(_allocation.Ptr, result.DevicePointer, _shape.TotalElements, exponent);
            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = grad =>
                {
                    var basePow = capturedSelf.Pow(exponent - 1f);
                    var local = grad.Multiply(basePow).Multiply(exponent);
                    capturedSelf.AccumulateGrad(local);
                    DisposeIfTemp(basePow, grad);
                    DisposeIfTemp(local, grad);
                    return grad;
                };
            }
            return result;
        }

        public ITensor Pow(ITensor exponent)
        {
            var o = Tensor.Unwrap(exponent) as CudaBackend ?? throw new ArgumentException("Operand must reside on CUDA.");
            return ElementwiseBinary(o, CUDA.NativePow, (g, a, b) =>
            {
                var ga = g.Multiply(b).Multiply(a.Pow(b.Subtract(1f)));
                var gb = g.Multiply(a.Pow(b)).Multiply(a.Log());
                return (ga, gb);
            });
        }

        public ITensor Negate()
        {
            var result = ElementwiseUnary(CUDA.NativeNegate);
            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = grad =>
                {
                    var n = grad.Negate();
                    capturedSelf.AccumulateGrad(n);
                    DisposeIfTemp(n, grad);
                    return grad;
                };
            }
            return result;
        }

        public ITensor Exp()
        {
            var result = ElementwiseUnary(CUDA.NativeExp);
            if (_requiresGrad)
            {
                var capturedSelf = this;
                var capturedOut = result;
                result.GradFn = grad =>
                {
                    var local = grad.Multiply(capturedOut);
                    capturedSelf.AccumulateGrad(local);
                    DisposeIfTemp(local, grad);
                    return grad;
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
                result.GradFn = grad =>
                {
                    var local = grad.Divide(capturedSelf);
                    capturedSelf.AccumulateGrad(local);
                    DisposeIfTemp(local, grad);
                    return grad;
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
                var capturedOut = result;
                result.GradFn = grad =>
                {
                    var local = grad.Divide(capturedOut.Multiply(2f));
                    capturedSelf.AccumulateGrad(local);
                    DisposeIfTemp(local, grad);
                    return grad;
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
                result.GradFn = grad =>
                {
                    var local = grad.Multiply(capturedSelf.Sign());
                    capturedSelf.AccumulateGrad(local);
                    DisposeIfTemp(local, grad);
                    return grad;
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
                    var local = gradOutput.Multiply(capturedSelf.Cos());
                    capturedSelf.AccumulateGrad(local);
                    DisposeIfTemp(local, gradOutput);
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
                    var local = gradOutput.Multiply(capturedSelf.Sin().Negate());
                    capturedSelf.AccumulateGrad(local);
                    DisposeIfTemp(local, gradOutput);
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

            int m = _shape[0], k = _shape[1], n = o.Shape[1];
            if (o.Shape[0] != k) throw new ArgumentException("MatMul inner dimensions must match.");

            var result = new CudaBackend(new TensorShape(m, n), _requiresGrad || o.RequiresGrad, _device) { Inputs = new[] { this, other } };
            CUDA.NativeMatMul(_allocation.Ptr, o.DevicePointer, result.DevicePointer, m, n, k);

            if (result.RequiresGrad)
            {
                var capturedSelf = this;
                var capturedOther = o;
                result.GradFn = gradOutput =>
                {
                    if (capturedSelf.RequiresGrad)
                    {
                        var gt = capturedOther.Transpose(new[] { 1, 0 });
                        var ga = gradOutput.MatMul(gt);
                        capturedSelf.AccumulateGrad(ga);
                        DisposeIfTemp(gt, gradOutput);
                        DisposeIfTemp(ga, gradOutput);
                    }
                    if (capturedOther.RequiresGrad)
                    {
                        var st = capturedSelf.Transpose(new[] { 1, 0 });
                        var gb = st.MatMul(gradOutput);
                        capturedOther.AccumulateGrad(gb);
                        DisposeIfTemp(st, gradOutput);
                        DisposeIfTemp(gb, gradOutput);
                    }
                    return gradOutput;
                };
            }
            return result;
        }

        private static void SplitAxis(TensorShape shape, int normAxis, out int outer, out int dim, out int inner)
        {
            dim = shape[normAxis];
            outer = 1;
            for (int i = 0; i < normAxis; i++) outer *= shape[i];
            inner = 1;
            for (int i = normAxis + 1; i < shape.Rank; i++) inner *= shape[i];
        }

        private static int[] ReducedDims(TensorShape shape, int normAxis, bool keepDims)
        {
            return keepDims
                ? shape.Dimensions.Select((d, i) => i == normAxis ? 1 : d).ToArray()
                : shape.Dimensions.Where((_, i) => i != normAxis).DefaultIfEmpty(1).ToArray();
        }

        public ITensor Sum(int? axis = null, bool keepDims = false)
        {
            if (axis == null)
            {
                var outShape = keepDims
                    ? new TensorShape(Enumerable.Repeat(1, _shape.Rank).ToArray())
                    : new TensorShape(1);
                var result = new CudaBackend(outShape, _requiresGrad, _device) { Inputs = new[] { this } };
                CUDA.NativeSumAll(_allocation.Ptr, result.DevicePointer, _shape.TotalElements);

                if (_requiresGrad)
                {
                    var capturedSelf = this;
                    result.GradFn = grad =>
                    {
                        // ones(parent) * grad  broadcasts a scalar / all-1s grad
                        var ones = Ones(capturedSelf._shape, capturedSelf._device);
                        var local = ones.Multiply(grad);
                        capturedSelf.AccumulateGrad(local);
                        (ones as IDisposable)?.Dispose();
                        DisposeIfTemp(local, grad);
                        return grad;
                    };
                }
                return result;
            }

            int normAxis = axis.Value < 0 ? axis.Value + _shape.Rank : axis.Value;
            if (normAxis < 0 || normAxis >= _shape.Rank)
                throw new ArgumentOutOfRangeException(nameof(axis));

            SplitAxis(_shape, normAxis, out int outer, out int dim, out int inner);
            var resultTensor = new CudaBackend(
                new TensorShape(ReducedDims(_shape, normAxis, keepDims)),
                _requiresGrad, _device)
            { Inputs = new[] { this } };

            CUDA.NativeSumAxis(_allocation.Ptr, resultTensor.DevicePointer, outer, dim, inner);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                bool capturedKeep = keepDims;
                int capturedAxis = normAxis;
                resultTensor.GradFn = grad =>
                {
                    ITensor expanded = ExpandForReductionBackward(grad, capturedSelf._shape, capturedAxis, capturedKeep);
                    try
                    {
                        var broadcasted = expanded.BroadcastTo(capturedSelf._shape);
                        try { capturedSelf.AccumulateGrad(broadcasted); }
                        finally { DisposeIfTemp(broadcasted, expanded); }
                    }
                    finally { DisposeIfTemp(expanded, grad); }
                    return grad;
                };
            }
            return resultTensor;
        }

        public ITensor Mean(int? axis = null, bool keepDims = false)
        {
            if (axis == null)
            {
                var outShape = keepDims
                    ? new TensorShape(Enumerable.Repeat(1, _shape.Rank).ToArray())
                    : new TensorShape(1);
                var result = new CudaBackend(outShape, _requiresGrad, _device) { Inputs = new[] { this } };
                CUDA.NativeMeanAll(_allocation.Ptr, result.DevicePointer, _shape.TotalElements);

                if (_requiresGrad)
                {
                    var capturedSelf = this;
                    result.GradFn = grad =>
                    {
                        var ones = Ones(capturedSelf._shape, capturedSelf._device);
                        var local = ones.Multiply(grad).Divide((float)capturedSelf._shape.TotalElements);
                        capturedSelf.AccumulateGrad(local);
                        (ones as IDisposable)?.Dispose();
                        DisposeIfTemp(local, grad);
                        return grad;
                    };
                }
                return result;
            }

            int normAxis = axis.Value < 0 ? axis.Value + _shape.Rank : axis.Value;
            if (normAxis < 0 || normAxis >= _shape.Rank)
                throw new ArgumentOutOfRangeException(nameof(axis));

            SplitAxis(_shape, normAxis, out int outer, out int dim, out int inner);
            var resultTensor = new CudaBackend(
                new TensorShape(ReducedDims(_shape, normAxis, keepDims)),
                _requiresGrad, _device)
            { Inputs = new[] { this } };

            CUDA.NativeMeanAxis(_allocation.Ptr, resultTensor.DevicePointer, outer, dim, inner);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                bool capturedKeep = keepDims;
                int capturedAxis = normAxis;
                float scale = dim;
                resultTensor.GradFn = grad =>
                {
                    var scaled = grad.Divide(scale);
                    try
                    {
                        ITensor expanded = ExpandForReductionBackward(scaled, capturedSelf._shape, capturedAxis, capturedKeep);
                        try
                        {
                            var broadcasted = expanded.BroadcastTo(capturedSelf._shape);
                            try { capturedSelf.AccumulateGrad(broadcasted); }
                            finally { DisposeIfTemp(broadcasted, expanded); }
                        }
                        finally { DisposeIfTemp(expanded, scaled); }
                    }
                    finally { DisposeIfTemp(scaled, grad); }
                    return grad;
                };
            }
            return resultTensor;
        }

        /// <summary>
        /// Insert a size-1 axis at the reduced dimension when keepDims was false,
        /// so BroadcastTo can legally expand back to the parent shape.
        /// </summary>
        private static ITensor ExpandForReductionBackward(ITensor grad, TensorShape parent, int normAxis, bool keepDims)
        {
            if (keepDims)
                return grad;

            int[] dims = new int[parent.Rank];
            for (int i = 0, g = 0; i < parent.Rank; i++)
                dims[i] = (i == normAxis) ? 1 : grad.Shape.Dimensions[g++];

            return grad.Reshape(dims);
        }


        public ITensor Sum(int[] axes, bool keepDims = false)
        {
            ITensor current = this;
            foreach (var ax in axes.OrderByDescending(a => a))
            {
                var next = current.Sum(ax, keepDims);
                if (!ReferenceEquals(current, this)) (current as IDisposable)?.Dispose();
                current = next;
            }
            return current;
        }


        public ITensor Mean(int[] axes, bool keepDims = false)
        {
            ITensor current = this;
            foreach (var ax in axes.OrderByDescending(a => a))
            {
                var next = current.Mean(ax, keepDims);
                if (!ReferenceEquals(current, this)) (current as IDisposable)?.Dispose();
                current = next;
            }
            return current;
        }

        public ITensor Max(int axis = -1, bool keepDims = false)
        {
            int normAxis = axis < 0 ? axis + _shape.Rank : axis;
            SplitAxis(_shape, normAxis, out int outer, out int dim, out int inner);
            var result = new CudaBackend(new TensorShape(ReducedDims(_shape, normAxis, keepDims)), _requiresGrad, _device) { Inputs = new[] { this } };
            CUDA.NativeMaxAxis(_allocation.Ptr, result.DevicePointer, outer, dim, inner);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = grad =>
                {
                    var idx = new CudaBackend(result.Shape, false, _device);
                    CUDA.NativeArgMax(capturedSelf._allocation.Ptr, idx.DevicePointer, outer, dim, inner);
                    var go = Tensor.Unwrap(grad) as CudaBackend ?? throw new InvalidOperationException("Gradient must be on CUDA.");
                    var gradIn = new CudaBackend(capturedSelf._shape, false, capturedSelf._device);
                    CUDA.NativeGatherAxisGrad(go.DevicePointer, idx.DevicePointer, gradIn.DevicePointer, outer, dim, inner, 1);
                    capturedSelf.AccumulateGrad(gradIn);
                    idx.Dispose();
                    gradIn.Dispose();
                    return grad;
                };
            }
            return result;
        }

        public ITensor Min(int axis = -1, bool keepDims = false)
        {
            int normAxis = axis < 0 ? axis + _shape.Rank : axis;
            SplitAxis(_shape, normAxis, out int outer, out int dim, out int inner);
            var result = new CudaBackend(new TensorShape(ReducedDims(_shape, normAxis, keepDims)), _requiresGrad, _device) { Inputs = new[] { this } };
            CUDA.NativeMinAxis(_allocation.Ptr, result.DevicePointer, outer, dim, inner);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = grad =>
                {
                    var idx = new CudaBackend(result.Shape, false, _device);
                    CUDA.NativeArgMin(capturedSelf._allocation.Ptr, idx.DevicePointer, outer, dim, inner);
                    var go = Tensor.Unwrap(grad) as CudaBackend ?? throw new InvalidOperationException("Gradient must be on CUDA.");
                    var gradIn = new CudaBackend(capturedSelf._shape, false, capturedSelf._device);
                    CUDA.NativeGatherAxisGrad(go.DevicePointer, idx.DevicePointer, gradIn.DevicePointer, outer, dim, inner, 1);
                    capturedSelf.AccumulateGrad(gradIn);
                    idx.Dispose();
                    gradIn.Dispose();
                    return grad;
                };
            }
            return result;
        }

        public ITensor ArgMin(int axis)
        {
            int normAxis = axis < 0 ? axis + _shape.Rank : axis;
            SplitAxis(_shape, normAxis, out int outer, out int dim, out int inner);
            var result = new CudaBackend(new TensorShape(ReducedDims(_shape, normAxis, false)), false, _device);
            CUDA.NativeArgMin(_allocation.Ptr, result.DevicePointer, outer, dim, inner);
            return result;
        }

        public ITensor ArgMax(int axis)
        {
            int normAxis = axis < 0 ? axis + _shape.Rank : axis;
            SplitAxis(_shape, normAxis, out int outer, out int dim, out int inner);
            var result = new CudaBackend(new TensorShape(ReducedDims(_shape, normAxis, false)), false, _device);
            CUDA.NativeArgMax(_allocation.Ptr, result.DevicePointer, outer, dim, inner);
            return result;
        }

        public ITensor CumSum(int axis)
        {
            int normAxis = axis < 0 ? axis + _shape.Rank : axis;
            SplitAxis(_shape, normAxis, out int outer, out int dim, out int inner);
            var result = new CudaBackend(_shape, _requiresGrad, _device) { Inputs = new[] { this } };
            CUDA.NativeCumSum(_allocation.Ptr, result.DevicePointer, outer, dim, inner);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                result.GradFn = grad =>
                {
                    var go = Tensor.Unwrap(grad) as CudaBackend ?? throw new InvalidOperationException("Gradient must be on CUDA.");
                    var gradIn = new CudaBackend(capturedSelf._shape, false, capturedSelf._device);
                    CUDA.NativeReverseCumSum(go.DevicePointer, gradIn.DevicePointer, outer, dim, inner);
                    capturedSelf.AccumulateGrad(gradIn);
                    gradIn.Dispose();
                    return grad;
                };
            }
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
                    mask.Dispose();
                    DisposeIfTemp(finalGrad, grad);
                    return grad;
                };
            }
            return result;
        }

        public void Backward(ITensor? gradient = null) => AutogradEngine.Backward(this, gradient);

        public void ClearGrad()
        {
            if (_grad is IDisposable d) d.Dispose();
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
            CUDA.NativeRand(t.DevicePointer, shape.TotalElements, (uint)Guid.NewGuid().GetHashCode());
            return t;
        }

        public static ITensor Randn(TensorShape shape, Device? device = null)
        {
            var t = new CudaBackend(shape, false, device ?? Device.CUDA);
            CUDA.NativeRandn(t.DevicePointer, shape.TotalElements, (uint)Guid.NewGuid().GetHashCode());
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
            CUDA.CopyHostToDeviceFast(floats, _allocation.Ptr);
        }

        public bool IsCpu() => false;
        public bool IsCuda() => true;
        public IEnumerable<ITensor> Parameters() { yield return this; }

        public ITensor Where(ITensor condition, ITensor trueValue, ITensor falseValue)
        {
            var cond = Tensor.Unwrap(condition) as CudaBackend ?? throw new ArgumentException("Condition must reside on CUDA.");
            var tv = Tensor.Unwrap(trueValue) as CudaBackend ?? throw new ArgumentException("True value must reside on CUDA.");
            var fv = Tensor.Unwrap(falseValue) as CudaBackend ?? throw new ArgumentException("False value must reside on CUDA.");

            var targetShape = _shape.BroadcastTo(cond.Shape).BroadcastTo(tv.Shape).BroadcastTo(fv.Shape);

            CudaBackend c = cond, t = tv, f = fv;
            bool tempC = false, tempT = false, tempF = false;
            if (!cond.Shape.Equals(targetShape)) { c = (CudaBackend)cond.BroadcastTo(targetShape); tempC = true; }
            if (!tv.Shape.Equals(targetShape)) { t = (CudaBackend)tv.BroadcastTo(targetShape); tempT = true; }
            if (!fv.Shape.Equals(targetShape)) { f = (CudaBackend)fv.BroadcastTo(targetShape); tempF = true; }

            var result = new CudaBackend(targetShape, tv.RequiresGrad || fv.RequiresGrad, _device)
            {
                Inputs = new[] { condition, trueValue, falseValue }
            };

            try
            {
                CUDA.NativeWhere(c.DevicePointer, t.DevicePointer, f.DevicePointer, result.DevicePointer, targetShape.TotalElements);
            }
            finally
            {
                if (tempC) c.Dispose();
                if (tempT) t.Dispose();
                if (tempF) f.Dispose();
            }

            if (result.RequiresGrad)
            {
                var capturedCond = cond;
                var capturedTrue = tv;
                var capturedFalse = fv;
                result.GradFn = grad =>
                {
                    var go = Tensor.Unwrap(grad) as CudaBackend ?? throw new InvalidOperationException("Gradient must be on CUDA.");
                    var condB = capturedCond.Shape.Equals(go.Shape) ? capturedCond : (CudaBackend)capturedCond.BroadcastTo(go.Shape);
                    try
                    {
                        if (capturedTrue.RequiresGrad)
                        {
                            using var zeros = new CudaBackend(go.Shape, false, _device);
                            using var masked = new CudaBackend(go.Shape, false, _device);
                            CUDA.NativeWhere(condB.DevicePointer, go.DevicePointer, zeros.DevicePointer, masked.DevicePointer, go.Shape.TotalElements);
                            capturedTrue.AccumulateGrad(masked);
                        }
                        if (capturedFalse.RequiresGrad)
                        {
                            using var zeros = new CudaBackend(go.Shape, false, _device);
                            using var masked = new CudaBackend(go.Shape, false, _device);
                            CUDA.NativeWhere(condB.DevicePointer, zeros.DevicePointer, go.DevicePointer, masked.DevicePointer, go.Shape.TotalElements);
                            capturedFalse.AccumulateGrad(masked);
                        }
                    }
                    finally
                    {
                        if (!ReferenceEquals(condB, capturedCond)) condB.Dispose();
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
            return this;
        }

        public ITensor Squeeze(int? axis = null)
        {
            if (axis == null)
            {
                var newDims = _shape.Dimensions.Where(d => d != 1).ToArray();
                if (newDims.Length == 0) newDims = new[] { 1 };
                return Reshape(newDims);
            }

            int a = axis.Value < 0 ? _shape.Rank + axis.Value : axis.Value;
            if (a < 0 || a >= _shape.Rank) throw new ArgumentOutOfRangeException(nameof(axis));
            if (_shape.Dimensions[a] != 1)
                throw new InvalidOperationException($"Cannot squeeze axis {a} of size {_shape.Dimensions[a]}.");

            var dims = _shape.Dimensions.ToList();
            dims.RemoveAt(a);
            if (dims.Count == 0) dims.Add(1);
            return Reshape(dims.ToArray());
        }

        public ITensor Unsqueeze(int axis)
        {
            int rank = _shape.Rank;
            int actualAxis = axis < 0 ? rank + axis + 1 : axis;
            if (actualAxis < 0 || actualAxis > rank) throw new ArgumentOutOfRangeException(nameof(axis));

            var newDims = new int[rank + 1];
            for (int i = 0, j = 0; i < newDims.Length; i++)
                newDims[i] = (i == actualAxis) ? 1 : _shape.Dimensions[j++];
            return Reshape(newDims);
        }

        public (ITensor values, ITensor indices) TopK(int k, int axis = -1)
        {
            if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
            int normAxis = axis < 0 ? _shape.Rank + axis : axis;
            if (normAxis < 0 || normAxis >= _shape.Rank) throw new ArgumentOutOfRangeException(nameof(axis));

            SplitAxis(_shape, normAxis, out int outer, out int dim, out int inner);
            if (k > dim) throw new ArgumentOutOfRangeException(nameof(k), "k cannot be larger than the size of the axis.");

            int[] outDims = (int[])_shape.Dimensions.Clone();
            outDims[normAxis] = k;
            var outShape = new TensorShape(outDims);

            var valuesBackend = new CudaBackend(outShape, _requiresGrad, _device) { Inputs = new[] { this } };
            var indicesBackend = new CudaBackend(outShape, false, _device);
            CUDA.NativeTopK(_allocation.Ptr, valuesBackend.DevicePointer, indicesBackend.DevicePointer, outer, dim, inner, k);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                var capturedIndices = indicesBackend;
                var originalShape = _shape.Clone();
                valuesBackend.GradFn = gradOutput =>
                {
                    var gradIn = new CudaBackend(originalShape, false, capturedSelf._device);
                    var go = Tensor.Unwrap(gradOutput) as CudaBackend ?? throw new InvalidOperationException("Gradient must be on CUDA.");
                    CUDA.NativeTopKScatterGrad(go.DevicePointer, capturedIndices.DevicePointer, gradIn.DevicePointer, outer, dim, inner, k);
                    capturedSelf.AccumulateGrad(gradIn);
                    gradIn.Dispose();
                    return gradOutput;
                };
            }

            return (new Tensor(valuesBackend), new Tensor(indicesBackend));
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (!_disposed)
                {
                    _allocation?.Release();
                    _disposed = true;
                }
            }
            GC.SuppressFinalize(this);
        }

        ~CudaBackend() { }
    }
}
