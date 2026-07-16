using ArborNet.Activations;
using ArborNet.Core.Backends;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using System;
using System.Collections.Generic;

namespace ArborNet.Core.Tensors
{
    /// <summary>
    /// Thread-safe variable wrapper that prevents race conditions and memory corruption
    /// during concurrent backpropagation using atomic synchronization.
    /// </summary>
    public class Variable : ITensor
    {
        internal readonly ITensor _inner;
        private readonly object _gradLock = new();

        public ITensor[] Inputs { get => _inner.Inputs; set => _inner.Inputs = value; }
        public TensorShape Shape => _inner.Shape;
        public Device Device => _inner.Device;
        public bool RequiresGrad { get => _inner.RequiresGrad; set => _inner.RequiresGrad = value; }

        private ITensor? _grad;
        public ITensor? Grad
        {
            get { lock (_gradLock) { return _grad; } }
            set { lock (_gradLock) { _grad = value; } }
        }

        public Func<ITensor, ITensor>? GradFn { get => _inner.GradFn; set => _inner.GradFn = value; }
        public float[] Data => _inner.ToArray();

        public Variable(ITensor inner, bool requiresGrad = false)
        {
            _inner = inner ?? throw new ArgumentNullException(nameof(inner));
            RequiresGrad = requiresGrad || inner.RequiresGrad;
        }

        public void SetData(float[] floats) => _inner.SetData(floats);
        public float[] ToArray() => _inner.ToArray();
        public float ToScalar() => _inner.ToScalar();
        public ITensor Clone() => new Variable(_inner.Clone(), RequiresGrad);
        public ITensor To(Device device) => new Variable(_inner.To(device), RequiresGrad);
        public bool IsCpu() => _inner.IsCpu();
        public bool IsCuda() => _inner.IsCuda();
        public IEnumerable<ITensor> Parameters() => _inner.Parameters();

        // ====================================================================
        // ITENSOR INTERFACE DELEGATION METHODS
        // ====================================================================

        public ITensor Gather(int axis, ITensor indices)
            => new Variable(_inner.Gather(axis, indices), RequiresGrad || indices.RequiresGrad);

        public void AccumulateGrad(ITensor delta)
        {
            if (delta == null) return;
            lock (_gradLock)
            {
                _inner.AccumulateGrad(delta);
                _grad = _inner.Grad;
            }
        }

        public ITensor Add(ITensor other) => new Variable(_inner.Add(other), RequiresGrad || other.RequiresGrad);
        public ITensor Subtract(ITensor other) => new Variable(_inner.Subtract(other), RequiresGrad || other.RequiresGrad);
        public ITensor Multiply(ITensor other) => new Variable(_inner.Multiply(other), RequiresGrad || other.RequiresGrad);
        public ITensor Divide(ITensor other) => new Variable(_inner.Divide(other), RequiresGrad || other.RequiresGrad);
        public ITensor Add(float scalar) => new Variable(_inner.Add(scalar), RequiresGrad);
        public ITensor Subtract(float scalar) => new Variable(_inner.Subtract(scalar), RequiresGrad);
        public ITensor Multiply(float scalar) => new Variable(_inner.Multiply(scalar), RequiresGrad);
        public ITensor Divide(float scalar) => new Variable(_inner.Divide(scalar), RequiresGrad);
        public ITensor Subtract(int other) => Subtract((float)other);
        public ITensor Multiply(double scalar) => Multiply((float)scalar);
        public ITensor Divide(double scalar) => Multiply(1.0 / scalar);
        public ITensor Negate() => new Variable(_inner.Negate(), RequiresGrad);
        public ITensor Exp() => new Variable(_inner.Exp(), RequiresGrad);
        public ITensor Log() => new Variable(_inner.Log(), RequiresGrad);
        public ITensor Sqrt() => new Variable(_inner.Sqrt(), RequiresGrad);
        public ITensor Abs() => new Variable(_inner.Abs(), RequiresGrad);
        public ITensor Sin() => new Variable(_inner.Sin(), RequiresGrad);
        public ITensor Cos() => new Variable(_inner.Cos(), RequiresGrad);
        public ITensor Sign() => new Variable(_inner.Sign(), false);
        public ITensor Pow(ITensor exponent) => new Variable(_inner.Pow(exponent), RequiresGrad || exponent.RequiresGrad);
        public ITensor Pow(float exponent) => new Variable(_inner.Pow(exponent), RequiresGrad);
        public ITensor MatMul(ITensor other) => new Variable(_inner.MatMul(other), RequiresGrad || other.RequiresGrad);
        public ITensor Transpose(int[] perm) => new Variable(_inner.Transpose(perm), RequiresGrad);
        public ITensor Reshape(params int[] newShape) => new Variable(_inner.Reshape(newShape), RequiresGrad);
        public ITensor Slice(params (int start, int end, int step)[] slices) => new Variable(_inner.Slice(slices), RequiresGrad);
        public ITensor Concat(IEnumerable<ITensor> others, int axis = 0) => new Variable(_inner.Concat(others, axis), RequiresGrad);
        public ITensor BroadcastTo(TensorShape targetShape) => new Variable(_inner.BroadcastTo(targetShape), RequiresGrad);
        public ITensor ReshapeWithBroadcast(TensorShape target, int axis) => new Variable(_inner.ReshapeWithBroadcast(target, axis), RequiresGrad);
        public ITensor Sum(int? axis = null, bool keepDims = false) => new Variable(_inner.Sum(axis, keepDims), RequiresGrad);
        public ITensor Sum(int[] axes, bool keepDims = false) => new Variable(_inner.Sum(axes, keepDims), RequiresGrad);
        public ITensor Mean(int? axis = null, bool keepDims = false) => new Variable(_inner.Mean(axis, keepDims), RequiresGrad);
        public ITensor Mean(int[] axes, bool keepDims = false) => new Variable(_inner.Mean(axes, keepDims), RequiresGrad);
        public ITensor Max(int axis = -1, bool keepDims = false) => new Variable(_inner.Max(axis, keepDims), RequiresGrad);
        public ITensor Min(int axis = -1, bool keepDims = false) => new Variable(_inner.Min(axis, keepDims), RequiresGrad);
        public ITensor CumSum(int axis) => new Variable(_inner.CumSum(axis), RequiresGrad);
        public ITensor GreaterThan(ITensor other) => new Variable(_inner.GreaterThan(other), false);
        public ITensor GreaterThanOrEqual(ITensor other) => new Variable(_inner.GreaterThanOrEqual(other), false);
        public ITensor LessEqual(ITensor other) => new Variable(_inner.LessEqual(other), false);
        public ITensor Equal(ITensor other) => new Variable(_inner.Equal(other), false);
        public ITensor Where(ITensor condition, ITensor trueValue, ITensor falseValue)
            => new Variable(_inner.Where(condition, trueValue, falseValue), false);
        public ITensor Tanh() => new Variable(new Tanh().Forward(_inner), RequiresGrad);
        public ITensor Relu() => new Variable(new ReLU().Forward(_inner), RequiresGrad);
        public ITensor Sigmoid() => new Variable(new Sigmoid().Forward(_inner), RequiresGrad);
        public ITensor Softmax(int axis = -1) => new Variable(new Softmax(axis).Forward(_inner), RequiresGrad);
        public void Backward(ITensor? gradient = null) => _inner.Backward(gradient);
        public void ClearGrad() { lock (_gradLock) { _grad = null; } _inner.ClearGrad(); }
        public ITensor ArgMin(int axis) => new Variable(_inner.ArgMin(axis), false);
        public ITensor ArgMax(int axis) => new Variable(_inner.ArgMax(axis), false);
        public ITensor BroadcastAdd(ITensor other) => Add(other);
        public ITensor LogicalNot() => new Variable(_inner.LogicalNot(), RequiresGrad);
        public ITensor Clip(float v1, float v2) => new Variable(_inner.Clip(v1, v2), RequiresGrad);
    }
}