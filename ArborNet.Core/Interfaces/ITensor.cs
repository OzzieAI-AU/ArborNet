using System;
using System.Collections.Generic;
using ArborNet.Core.Devices;
using ArborNet.Core.Tensors;

namespace ArborNet.Core.Interfaces
{
    /// <summary>
    /// Core tensor interface for ArborNet - the foundation of all operations, autograd, and device abstraction.
    /// All backends (CPU/CUDA) and wrappers must implement this exactly.
    /// </summary>
    public interface ITensor
    {
        TensorShape Shape { get; }
        Device Device { get; }
        ITensor[] Inputs { get; set; }
        bool RequiresGrad { get; set; }
        ITensor? Grad { get; set; }
        Func<ITensor, ITensor>? GradFn { get; set; }
        float[] Data { get; }

        void AccumulateGrad(ITensor delta);
        ITensor Gather(int axis, ITensor indices);
        float[] ToArray();
        float ToScalar();
        ITensor Clone();
        ITensor To(Device device);
        bool IsCpu();
        bool IsCuda();
        IEnumerable<ITensor> Parameters();

        // High-Performance In-Place Operators for Optimizers
        void AddInPlace(ITensor other);
        void AddInPlace(float scalar);
        void SubtractInPlace(ITensor other);
        void SubtractInPlace(float scalar);
        void MultiplyInPlace(ITensor other);
        void MultiplyInPlace(float scalar);

        ITensor Add(ITensor other);
        ITensor Subtract(ITensor other);
        ITensor Multiply(ITensor other);
        ITensor Divide(ITensor other);
        ITensor Add(float scalar);
        ITensor Subtract(float scalar);
        ITensor Multiply(float scalar);
        ITensor Divide(float scalar);
        ITensor Subtract(int other);
        ITensor Multiply(double scalar);
        ITensor Divide(double scalar);

        ITensor Negate();
        ITensor Exp();
        ITensor Log();
        ITensor Sqrt();
        ITensor Abs();
        ITensor Sin();
        ITensor Cos();
        ITensor Pow(float exponent);
        ITensor Pow(ITensor exponent);
        ITensor MatMul(ITensor other);
        ITensor Transpose(int[] perm);
        ITensor Reshape(params int[] newShape);
        ITensor Slice(params (int start, int end, int step)[] slices);
        ITensor Concat(IEnumerable<ITensor> others, int axis = 0);
        ITensor BroadcastTo(TensorShape targetShape);
        ITensor BroadcastAdd(ITensor other);
        ITensor ReshapeWithBroadcast(TensorShape target, int axis);

        ITensor Sum(int? axis = null, bool keepDims = false);
        ITensor Sum(int[] axes, bool keepDims = false);
        ITensor Mean(int? axis = null, bool keepDims = false);
        ITensor Mean(int[] axes, bool keepDims = false);
        ITensor Max(int axis = -1, bool keepDims = false);
        ITensor Min(int axis = -1, bool keepDims = false);
        ITensor ArgMin(int axis);
        ITensor ArgMax(int axis);
        ITensor CumSum(int axis);

        ITensor GreaterThan(ITensor other);
        ITensor GreaterThanOrEqual(ITensor other);
        ITensor LessEqual(ITensor other);
        ITensor Equal(ITensor other);
        ITensor Where(ITensor condition, ITensor trueValue, ITensor falseValue);
        ITensor Sign();

        ITensor Tanh();
        ITensor Relu();
        ITensor Sigmoid();
        ITensor Softmax(int axis = -1);

        void Backward(ITensor? gradient = null);
        void ClearGrad();
        void SetData(float[] floats);
        ITensor LogicalNot();
        ITensor Clip(float v1, float v2);
    }
}
