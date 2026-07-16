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

namespace ArborNet.Core.Backends
{
    public sealed class CpuBackend : ITensor, IDisposable
    {
        private float[] _data;
        private readonly int _length;
        private bool _isRented;
        private TensorShape _shape;
        private Device _device;
        private bool _requiresGrad;
        private ITensor? _grad;
        private Func<ITensor, ITensor>? _gradFn;
        private ITensor[] _inputs = Array.Empty<ITensor>();
        private readonly object _lock = new();

        public ITensor[] Inputs { get => _inputs; set => _inputs = value; }
        public TensorShape Shape => _shape;
        public Device Device => _device;
        public bool RequiresGrad { get => _requiresGrad; set => _requiresGrad = value; }
        public ITensor? Grad { get => _grad; set => _grad = value; }
        public Func<ITensor, ITensor>? GradFn { get => _gradFn; set => _gradFn = value; }
        public float[] Data => ToArray();

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

        public CpuBackend(float[] data, TensorShape shape, bool requiresGrad = false, Device? device = null)
        {
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));
            _length = shape.TotalElements;
            _data = data ?? throw new ArgumentNullException(nameof(data));
            _isRented = false;
            _requiresGrad = requiresGrad;
            _device = device ?? Device.CPU;
        }

        // Thread-Safe Atomic Gradient Accumulation
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

        // High-Speed, Zero-Allocation In-Place Vector Math
        public void AddInPlace(ITensor other)
        {
            var otherRaw = Unwrap(other);
            lock (_lock)
            {
                int len = Math.Min(_length, otherRaw._length);
                ArborNet.Core.Native.SIMD.Accelerate.Add(_data, _data, otherRaw._data, len);
            }
        }

        public void AddInPlace(float scalar)
        {
            lock (_lock)
            {
                for (int i = 0; i < _length; i++) _data[i] += scalar;
            }
        }

        public void SubtractInPlace(ITensor other)
        {
            var otherRaw = Unwrap(other);
            lock (_lock)
            {
                int len = Math.Min(_length, otherRaw._length);
                ArborNet.Core.Native.SIMD.Accelerate.Subtract(_data, _data, otherRaw._data, len);
            }
        }

        public void SubtractInPlace(float scalar)
        {
            lock (_lock)
            {
                for (int i = 0; i < _length; i++) _data[i] -= scalar;
            }
        }

        public void MultiplyInPlace(ITensor other)
        {
            var otherRaw = Unwrap(other);
            lock (_lock)
            {
                int len = Math.Min(_length, otherRaw._length);
                ArborNet.Core.Native.SIMD.Accelerate.Multiply(_data, _data, otherRaw._data, len);
            }
        }

        public void MultiplyInPlace(float scalar)
        {
            lock (_lock)
            {
                for (int i = 0; i < _length; i++) _data[i] *= scalar;
            }
        }

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

        public float[] ToArray()
        {
            var copy = new float[_length];
            lock (_lock)
            {
                Array.Copy(_data, 0, copy, 0, _length);
            }
            return copy;
        }

        public float ToScalar()
        {
            if (_length != 1) throw new InvalidOperationException("Tensor is not a scalar.");
            return _data[0];
        }

        public ITensor Clone() => new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);

        public ITensor To(Device device)
        {
            if (device.Type == DeviceType.CPU) return Clone();
            if (device.Type == DeviceType.CUDA) return new CudaBackend(ToArray(), _shape.Clone(), _requiresGrad, device);
            throw new NotSupportedException($"Transfer to {device.Type} is not supported.");
        }

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

        public bool IsCpu() => true;
        public bool IsCuda() => false;
        public IEnumerable<ITensor> Parameters() { yield return this; }

        private static (TensorShape shape, long[] strideA, long[] strideB) GetBroadcastShapeAndStrides(ITensor a, ITensor b)
        {
            var ca = Unwrap(a);
            var cb = Unwrap(b);
            var resultShape = ca._shape.BroadcastTo(cb._shape);
            var strideA = ComputeBroadcastStrides(ca._shape.Dimensions, resultShape.Dimensions);
            var strideB = ComputeBroadcastStrides(cb._shape.Dimensions, resultShape.Dimensions);
            return (resultShape, strideA, strideB);
        }

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

        public ITensor Divide(ITensor other)
        {
            return Elementwise(other, (a, b) => b != 0 ? a / b : 0f,
                (g, a, b) => (g.Divide(b), g.Multiply(a.Negate()).Divide(b.Multiply(b))));
        }

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

        public ITensor Add(float scalar) => ElementwiseScalar(x => x + scalar, g => g);
        public ITensor Subtract(float scalar) => ElementwiseScalar(x => x - scalar, g => g);
        public ITensor Multiply(float scalar) => ElementwiseScalar(x => x * scalar, g => g.Multiply(scalar));
        public ITensor Divide(float scalar) => Multiply(1f / scalar);
        public ITensor Subtract(int other) => Subtract((float)other);
        public ITensor Multiply(double scalar) => Multiply((float)scalar);
        public ITensor Divide(double scalar) => Multiply(1.0 / scalar);

        public ITensor Negate() => ElementwiseScalar(x => -x, g => g.Negate());
        public ITensor Exp() => ElementwiseScalar(MathF.Exp, g => g.Multiply(this.Exp()));
        public ITensor Log() => ElementwiseScalar(MathF.Log, g => g.Divide(this));
        public ITensor Sqrt() => ElementwiseScalar(MathF.Sqrt, g => g.Divide(this.Sqrt().Multiply(2)));
        public ITensor Abs() => ElementwiseScalar(MathF.Abs, g => g.Multiply(this.GreaterThan(Tensor.Zeros(_shape)).Multiply(2).Subtract(1)));
        public ITensor Sin() => ElementwiseScalar(MathF.Sin, g => g.Multiply(this.Cos()));
        public ITensor Cos() => ElementwiseScalar(MathF.Cos, g => g.Multiply(this.Sin().Negate()));
        public ITensor Sign() => ElementwiseScalar(x => MathF.Sign(x), g => Tensor.Zeros(g.Shape, g.Device));

        public ITensor Pow(float exponent)
        {
            return ElementwiseScalar(x => MathF.Pow(x, exponent),
                g => g.Multiply(Tensor.FromScalar(exponent)).Multiply(this.Pow(exponent - 1)));
        }

        public ITensor Pow(ITensor exponent) => Elementwise(exponent, (a, b) => MathF.Pow(a, b),
            (g, a, b) => (
                g.Multiply(b.Multiply(a.Pow(b.Subtract(Tensor.FromScalar(1f))))),
                g.Multiply(a.Pow(b).Multiply(a.Log()))
            ));

        public ITensor BroadcastAdd(ITensor other) => Add(other);

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

        private static int[] InvertPerm(int[] perm)
        {
            int[] inv = new int[perm.Length];
            for (int i = 0; i < perm.Length; i++) inv[perm[i]] = i;
            return inv;
        }

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

        public ITensor Max(int axis = -1, bool keepDims = false) => ReduceAlongAxis(axis < 0 ? _shape.Rank - 1 : axis, false, keepDims);
        public ITensor Min(int axis = -1, bool keepDims = false) => ReduceAlongAxis(axis < 0 ? _shape.Rank - 1 : axis, false, keepDims);

        public ITensor LogicalNot()
        {
            var resultData = _data.Select(x => x == 0f ? 1f : 0f).ToArray();
            return new Tensor(new CpuBackend(resultData, _shape.Clone(), false, _device));
        }

        public ITensor Clip(float v1, float v2)
        {
            if (v1 > v2) (v1, v2) = (v2, v1);
            var resultData = _data.Select(x => x < v1 ? v1 : (x > v2 ? v2 : x)).ToArray();
            var rawResult = new CpuBackend(resultData, _shape.Clone(), _requiresGrad, _device);

            if (_requiresGrad)
            {
                var capturedSelf = this;
                rawResult.GradFn = gradOutput =>
                {
                    var mask = _data.Select(x => (x >= v1 && x <= v2) ? 1f : 0f).ToArray();
                    var finalGrad = gradOutput.Multiply(new Tensor(new CpuBackend(mask, _shape.Clone(), false, _device)));
                    capturedSelf.AccumulateGrad(finalGrad);
                    return finalGrad;
                };
            }
            return new Tensor(rawResult);
        }

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

        public ITensor Equal(ITensor other) => Comparison(other, (a, b) => Math.Abs(a - b) < 1e-6f ? 1f : 0f);
        public ITensor GreaterThan(ITensor other) => Comparison(other, (a, b) => a > b ? 1f : 0f);
        public ITensor GreaterThanOrEqual(ITensor other) => Comparison(other, (a, b) => a >= b ? 1f : 0f);
        public ITensor LessEqual(ITensor other) => Comparison(other, (a, b) => a <= b ? 1f : 0f);

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

        public ITensor Tanh() => new Tanh().Forward(this);
        public ITensor Relu() => new ReLU().Forward(this);
        public ITensor Sigmoid() => new Sigmoid().Forward(this);
        public ITensor Softmax(int axis = -1) => new Softmax(axis).Forward(this);

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

        private int GetStride(int dim)
        {
            int stride = 1;
            for (int i = dim + 1; i < _shape.Rank; i++) stride *= _shape.Dimensions[i];
            return stride;
        }

        public ITensor Concat(IEnumerable<ITensor> others, int axis = 0)
        {
            var all = new List<CpuBackend> { this };
            all.AddRange(others.Cast<CpuBackend>());
            if (axis < 0) axis = _shape.Rank + axis;

            var newShape = _shape.Dimensions.ToArray();
            newShape[axis] = all.Sum(t => t._shape.Dimensions[axis]);

            var resultData = new float[all.Sum(t => t._data.Length)];
            int offset = 0;
            foreach (var t in all)
            {
                Array.Copy(t._data, 0, resultData, offset, t._data.Length);
                offset += t._data.Length;
            }

            var result = new CpuBackend(resultData, new TensorShape(newShape), _requiresGrad || all.Any(t => t.RequiresGrad), _device);
            return new Tensor(result);
        }

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

        public void Backward(ITensor? gradient = null)
        {
            ArborNet.Core.Autograd.AutogradEngine.Backward(this, gradient);
        }

        public void ClearGrad()
        {
            _grad = null;
            _gradFn = null;
        }

        public static ITensor Zeros(TensorShape shape, Device? device = null) => new CpuBackend(shape, false, device);
        public static ITensor Ones(TensorShape shape, Device? device = null)
        {
            var data = new float[shape.TotalElements];
            Array.Fill(data, 1f);
            return new CpuBackend(data, shape, false, device);
        }
        public static ITensor FromScalar(float value, Device? device = null) => new CpuBackend(new[] { value }, new TensorShape(1), false, device);
        public static ITensor FromArray(float[] data, TensorShape shape, Device? device = null) => new CpuBackend(data, shape, false, device);
        public static ITensor Rand(TensorShape shape, Device? device = null)
        {
            var r = new Random();
            return new CpuBackend(Enumerable.Range(0, shape.TotalElements).Select(_ => (float)r.NextDouble()).ToArray(), shape, false, device);
        }
        public static ITensor Randn(TensorShape shape, Device? device = null)
        {
            var r = new Random();
            return new CpuBackend(Enumerable.Range(0, shape.TotalElements).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray(), shape, false, device);
        }
        public static ITensor Eye(int size, Device? device = null)
        {
            var data = new float[size * size];
            for (int i = 0; i < size; i++) data[i * size + i] = 1f;
            return new CpuBackend(data, new TensorShape(size, size), false, device);
        }

        public ITensor ArgMin(int axis) => ReduceIndices(axis, false);
        public ITensor ArgMax(int axis) => ReduceIndices(axis, true);

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

        public void Dispose()
        {
            if (_isRented && _data != null)
            {
                ArrayPool<float>.Shared.Return(_data);
                _data = null!;
                _isRented = false;
            }
        }

        ~CpuBackend() => Dispose();
    }
}