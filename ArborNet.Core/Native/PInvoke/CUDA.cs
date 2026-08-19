// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// Project: ArborNet
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Native.PInvoke
{
    using System;
    using System.Runtime.InteropServices;

    public static partial class CUDA
    {
        private const string CudaRuntime = "cudart64_12.dll";
        private const string CustomKernel = "cuda_backend.dll";

        static CUDA()
        {
            NativeResolver.Register();
        }

        public static bool IsAvailable()
        {
            try
            {
                CudaError err = cudaGetDeviceCount(out int count);
                return err == CudaError.Success && count > 0;
            }
            catch
            {
                return false;
            }
        }

        public enum CudaError
        {
            Success = 0,
            InvalidValue = 1,
            OutOfMemory = 2,
            NotInitialized = 3,
            InvalidDevice = 10,
            InvalidPointer = 17,
            InvalidMemcpyDirection = 21,
            InvalidResourceHandle = 33,
            Unknown = 999
        }

        public enum cudaMemcpyKind : int
        {
            cudaMemcpyHostToHost = 0,
            cudaMemcpyHostToDevice = 1,
            cudaMemcpyDeviceToHost = 2,
            cudaMemcpyDeviceToDevice = 3,
            cudaMemcpyDefault = 4
        }

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CudaError cudaMalloc(out IntPtr devicePtr, ulong size);

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CudaError cudaFree(IntPtr devicePtr);

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CudaError cudaMemset(IntPtr devicePtr, int value, ulong count);

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CudaError cudaMemcpy(IntPtr dst, IntPtr src, ulong count, cudaMemcpyKind kind);

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CudaError cudaDeviceSynchronize();

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CudaError cudaGetDeviceCount(out int count);

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CudaError cudaHostRegister(IntPtr ptr, ulong size, uint flags);

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CudaError cudaHostUnregister(IntPtr ptr);

        [DllImport(CustomKernel, EntryPoint = "NativeGetLastError", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NativeGetLastErrorImpl();

        public static void Check(CudaError err, string method)
        {
            if (err != CudaError.Success)
                throw new InvalidOperationException($"CUDA runtime error in {method}: {err}");
        }

        public static void ThrowIfKernelFailed(string method)
        {
            int err = NativeGetLastErrorImpl();
            if (err != 0)
                throw new InvalidOperationException($"CUDA kernel launch failed in {method}: {(CudaError)err} ({err})");
        }

        public static void CudaMalloc(out IntPtr devicePtr, ulong size)
            => Check(cudaMalloc(out devicePtr, size), nameof(CudaMalloc));

        public static void CudaFree(IntPtr devicePtr)
        {
            if (devicePtr == IntPtr.Zero) return;
            Check(cudaFree(devicePtr), nameof(CudaFree));
        }

        public static void CudaHostRegister(IntPtr ptr, ulong size, uint flags)
            => Check(cudaHostRegister(ptr, size, flags), nameof(CudaHostRegister));

        public static void CudaHostUnregister(IntPtr ptr)
            => Check(cudaHostUnregister(ptr), nameof(CudaHostUnregister));

        public static void CudaMemcpy(IntPtr dst, IntPtr src, ulong count, cudaMemcpyKind kind)
            => Check(cudaMemcpy(dst, src, count, kind), nameof(CudaMemcpy));

        public static void CudaMemset(IntPtr devicePtr, int value, ulong count)
            => Check(cudaMemset(devicePtr, value, count), nameof(CudaMemset));

        public static void Synchronize()
            => Check(cudaDeviceSynchronize(), nameof(Synchronize));

        public static void Initialize(out IntPtr context)
        {
            Check(cudaGetDeviceCount(out int count), nameof(cudaGetDeviceCount));
            if (count <= 0) throw new InvalidOperationException("No CUDA devices.");
            context = new IntPtr(1);
        }

        public static void Cleanup(IntPtr context) { }

        public static void MatMul(IntPtr A, IntPtr B, IntPtr C, int m, int n, int k) => NativeMatMul(A, B, C, m, n, k);
        public static void Add(IntPtr a, IntPtr b, IntPtr c, int n) => NativeAdd(a, b, c, n);

        public static void Reshape(IntPtr context, IntPtr input, IntPtr output, int[] inputShape, int[] outputShape)
        {
            long size = 1;
            foreach (int dim in inputShape) size *= dim;
            CudaMemcpy(output, input, (ulong)(size * sizeof(float)), cudaMemcpyKind.cudaMemcpyDeviceToDevice);
        }

        private static void Checked(Action call, string name)
        {
            call();
            ThrowIfKernelFailed(name);
        }

        [DllImport(CustomKernel, EntryPoint = "NativeAdd", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeAddImpl(IntPtr a, IntPtr b, IntPtr c, int n);
        public static void NativeAdd(IntPtr a, IntPtr b, IntPtr c, int n) => Checked(() => NativeAddImpl(a, b, c, n), nameof(NativeAdd));

        [DllImport(CustomKernel, EntryPoint = "NativeSubtract", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeSubtractImpl(IntPtr a, IntPtr b, IntPtr c, int n);
        public static void NativeSubtract(IntPtr a, IntPtr b, IntPtr c, int n) => Checked(() => NativeSubtractImpl(a, b, c, n), nameof(NativeSubtract));

        [DllImport(CustomKernel, EntryPoint = "NativeMultiply", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeMultiplyImpl(IntPtr a, IntPtr b, IntPtr c, int n);
        public static void NativeMultiply(IntPtr a, IntPtr b, IntPtr c, int n) => Checked(() => NativeMultiplyImpl(a, b, c, n), nameof(NativeMultiply));

        [DllImport(CustomKernel, EntryPoint = "NativeDivide", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeDivideImpl(IntPtr a, IntPtr b, IntPtr c, int n);
        public static void NativeDivide(IntPtr a, IntPtr b, IntPtr c, int n) => Checked(() => NativeDivideImpl(a, b, c, n), nameof(NativeDivide));

        [DllImport(CustomKernel, EntryPoint = "NativePow", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativePowImpl(IntPtr a, IntPtr b, IntPtr c, int n);
        public static void NativePow(IntPtr a, IntPtr b, IntPtr c, int n) => Checked(() => NativePowImpl(a, b, c, n), nameof(NativePow));

        [DllImport(CustomKernel, EntryPoint = "NativePowTensor", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativePowTensorImpl(IntPtr a, IntPtr b, IntPtr c, int n);
        public static void NativePowTensor(IntPtr a, IntPtr b, IntPtr c, int n) => Checked(() => NativePowTensorImpl(a, b, c, n), nameof(NativePowTensor));

        [DllImport(CustomKernel, EntryPoint = "NativeEqual", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeEqualImpl(IntPtr a, IntPtr b, IntPtr c, int n);
        public static void NativeEqual(IntPtr a, IntPtr b, IntPtr c, int n) => Checked(() => NativeEqualImpl(a, b, c, n), nameof(NativeEqual));

        [DllImport(CustomKernel, EntryPoint = "NativeGreaterThan", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeGreaterThanImpl(IntPtr a, IntPtr b, IntPtr c, int n);
        public static void NativeGreaterThan(IntPtr a, IntPtr b, IntPtr c, int n) => Checked(() => NativeGreaterThanImpl(a, b, c, n), nameof(NativeGreaterThan));

        [DllImport(CustomKernel, EntryPoint = "NativeGreaterThanOrEqual", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeGreaterThanOrEqualImpl(IntPtr a, IntPtr b, IntPtr c, int n);
        public static void NativeGreaterThanOrEqual(IntPtr a, IntPtr b, IntPtr c, int n) => Checked(() => NativeGreaterThanOrEqualImpl(a, b, c, n), nameof(NativeGreaterThanOrEqual));

        [DllImport(CustomKernel, EntryPoint = "NativeLessThan", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeLessThanImpl(IntPtr a, IntPtr b, IntPtr c, int n);
        public static void NativeLessThan(IntPtr a, IntPtr b, IntPtr c, int n) => Checked(() => NativeLessThanImpl(a, b, c, n), nameof(NativeLessThan));

        [DllImport(CustomKernel, EntryPoint = "NativeLessEqual", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeLessEqualImpl(IntPtr a, IntPtr b, IntPtr c, int n);
        public static void NativeLessEqual(IntPtr a, IntPtr b, IntPtr c, int n) => Checked(() => NativeLessEqualImpl(a, b, c, n), nameof(NativeLessEqual));

        [DllImport(CustomKernel, EntryPoint = "NativeWhere", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeWhereImpl(IntPtr cond, IntPtr a, IntPtr b, IntPtr c, int n);
        public static void NativeWhere(IntPtr cond, IntPtr a, IntPtr b, IntPtr c, int n) => Checked(() => NativeWhereImpl(cond, a, b, c, n), nameof(NativeWhere));

        [DllImport(CustomKernel, EntryPoint = "NativeReLU", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeReLUImpl(IntPtr input, IntPtr output, int n);
        public static void NativeReLU(IntPtr input, IntPtr output, int n) => Checked(() => NativeReLUImpl(input, output, n), nameof(NativeReLU));

        [DllImport(CustomKernel, EntryPoint = "NativeReLUGrad", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeReLUGradImpl(IntPtr gradOut, IntPtr originIn, IntPtr gradIn, int n);
        public static void NativeReLUGrad(IntPtr gradOut, IntPtr originIn, IntPtr gradIn, int n) => Checked(() => NativeReLUGradImpl(gradOut, originIn, gradIn, n), nameof(NativeReLUGrad));

        [DllImport(CustomKernel, EntryPoint = "NativeSigmoid", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeSigmoidImpl(IntPtr input, IntPtr output, int n);
        public static void NativeSigmoid(IntPtr input, IntPtr output, int n) => Checked(() => NativeSigmoidImpl(input, output, n), nameof(NativeSigmoid));

        [DllImport(CustomKernel, EntryPoint = "NativeSigmoidGrad", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeSigmoidGradImpl(IntPtr gradOut, IntPtr originOut, IntPtr gradIn, int n);
        public static void NativeSigmoidGrad(IntPtr gradOut, IntPtr originOut, IntPtr gradIn, int n) => Checked(() => NativeSigmoidGradImpl(gradOut, originOut, gradIn, n), nameof(NativeSigmoidGrad));

        [DllImport(CustomKernel, EntryPoint = "NativeTanh", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeTanhImpl(IntPtr input, IntPtr output, int n);
        public static void NativeTanh(IntPtr input, IntPtr output, int n) => Checked(() => NativeTanhImpl(input, output, n), nameof(NativeTanh));

        [DllImport(CustomKernel, EntryPoint = "NativeNegate", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeNegateImpl(IntPtr input, IntPtr output, int n);
        public static void NativeNegate(IntPtr input, IntPtr output, int n) => Checked(() => NativeNegateImpl(input, output, n), nameof(NativeNegate));

        [DllImport(CustomKernel, EntryPoint = "NativeExp", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeExpImpl(IntPtr input, IntPtr output, int n);
        public static void NativeExp(IntPtr input, IntPtr output, int n) => Checked(() => NativeExpImpl(input, output, n), nameof(NativeExp));

        [DllImport(CustomKernel, EntryPoint = "NativeLog", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeLogImpl(IntPtr input, IntPtr output, int n);
        public static void NativeLog(IntPtr input, IntPtr output, int n) => Checked(() => NativeLogImpl(input, output, n), nameof(NativeLog));

        [DllImport(CustomKernel, EntryPoint = "NativeSqrt", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeSqrtImpl(IntPtr input, IntPtr output, int n);
        public static void NativeSqrt(IntPtr input, IntPtr output, int n) => Checked(() => NativeSqrtImpl(input, output, n), nameof(NativeSqrt));

        [DllImport(CustomKernel, EntryPoint = "NativeAbs", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeAbsImpl(IntPtr input, IntPtr output, int n);
        public static void NativeAbs(IntPtr input, IntPtr output, int n) => Checked(() => NativeAbsImpl(input, output, n), nameof(NativeAbs));

        [DllImport(CustomKernel, EntryPoint = "NativeSin", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeSinImpl(IntPtr input, IntPtr output, int n);
        public static void NativeSin(IntPtr input, IntPtr output, int n) => Checked(() => NativeSinImpl(input, output, n), nameof(NativeSin));

        [DllImport(CustomKernel, EntryPoint = "NativeCos", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeCosImpl(IntPtr input, IntPtr output, int n);
        public static void NativeCos(IntPtr input, IntPtr output, int n) => Checked(() => NativeCosImpl(input, output, n), nameof(NativeCos));

        [DllImport(CustomKernel, EntryPoint = "NativeSign", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeSignImpl(IntPtr input, IntPtr output, int n);
        public static void NativeSign(IntPtr input, IntPtr output, int n) => Checked(() => NativeSignImpl(input, output, n), nameof(NativeSign));

        [DllImport(CustomKernel, EntryPoint = "NativeOnes", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeOnesImpl(IntPtr output, int n);
        public static void NativeOnes(IntPtr output, int n) => Checked(() => NativeOnesImpl(output, n), nameof(NativeOnes));

        [DllImport(CustomKernel, EntryPoint = "NativeSetScalar", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeSetScalarImpl(IntPtr output, float value, int n);
        public static void NativeSetScalar(IntPtr output, float value, int n) => Checked(() => NativeSetScalarImpl(output, value, n), nameof(NativeSetScalar));

        [DllImport(CustomKernel, EntryPoint = "NativePowScalar", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativePowScalarImpl(IntPtr input, IntPtr output, int n, float exponent);
        public static void NativePowScalar(IntPtr input, IntPtr output, int n, float exponent) => Checked(() => NativePowScalarImpl(input, output, n, exponent), nameof(NativePowScalar));

        [DllImport(CustomKernel, EntryPoint = "NativeAddScalar", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeAddScalarImpl(IntPtr input, IntPtr output, int n, float value);
        public static void NativeAddScalar(IntPtr input, IntPtr output, int n, float value) => Checked(() => NativeAddScalarImpl(input, output, n, value), nameof(NativeAddScalar));

        [DllImport(CustomKernel, EntryPoint = "NativeSubtractScalar", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeSubtractScalarImpl(IntPtr input, IntPtr output, int n, float value);
        public static void NativeSubtractScalar(IntPtr input, IntPtr output, int n, float value) => Checked(() => NativeSubtractScalarImpl(input, output, n, value), nameof(NativeSubtractScalar));

        [DllImport(CustomKernel, EntryPoint = "NativeMultiplyScalar", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeMultiplyScalarImpl(IntPtr input, IntPtr output, int n, float value);
        public static void NativeMultiplyScalar(IntPtr input, IntPtr output, int n, float value) => Checked(() => NativeMultiplyScalarImpl(input, output, n, value), nameof(NativeMultiplyScalar));

        [DllImport(CustomKernel, EntryPoint = "NativeDivideScalar", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeDivideScalarImpl(IntPtr input, IntPtr output, int n, float value);
        public static void NativeDivideScalar(IntPtr input, IntPtr output, int n, float value) => Checked(() => NativeDivideScalarImpl(input, output, n, value), nameof(NativeDivideScalar));

        [DllImport(CustomKernel, EntryPoint = "NativeAddScalarInPlace", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeAddScalarInPlaceImpl(IntPtr data, float value, int n);
        public static void NativeAddScalarInPlace(IntPtr data, float value, int n) => Checked(() => NativeAddScalarInPlaceImpl(data, value, n), nameof(NativeAddScalarInPlace));

        [DllImport(CustomKernel, EntryPoint = "NativeSubtractScalarInPlace", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeSubtractScalarInPlaceImpl(IntPtr data, float value, int n);
        public static void NativeSubtractScalarInPlace(IntPtr data, float value, int n) => Checked(() => NativeSubtractScalarInPlaceImpl(data, value, n), nameof(NativeSubtractScalarInPlace));

        [DllImport(CustomKernel, EntryPoint = "NativeMultiplyScalarInPlace", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeMultiplyScalarInPlaceImpl(IntPtr data, float value, int n);
        public static void NativeMultiplyScalarInPlace(IntPtr data, float value, int n) => Checked(() => NativeMultiplyScalarInPlaceImpl(data, value, n), nameof(NativeMultiplyScalarInPlace));

        [DllImport(CustomKernel, EntryPoint = "NativeEye", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeEyeImpl(IntPtr data, int size);
        public static void NativeEye(IntPtr data, int size) => Checked(() => NativeEyeImpl(data, size), nameof(NativeEye));

        [DllImport(CustomKernel, EntryPoint = "NativeRand", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeRandImpl(IntPtr data, int n, uint seed);
        public static void NativeRand(IntPtr data, int n, uint seed) => Checked(() => NativeRandImpl(data, n, seed), nameof(NativeRand));

        [DllImport(CustomKernel, EntryPoint = "NativeRandn", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeRandnImpl(IntPtr data, int n, uint seed);
        public static void NativeRandn(IntPtr data, int n, uint seed) => Checked(() => NativeRandnImpl(data, n, seed), nameof(NativeRandn));

        [DllImport(CustomKernel, EntryPoint = "NativeTranspose", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeTransposeImpl(IntPtr input, IntPtr output, int rows, int cols);
        public static void NativeTranspose(IntPtr input, IntPtr output, int rows, int cols) => Checked(() => NativeTransposeImpl(input, output, rows, cols), nameof(NativeTranspose));

        [DllImport(CustomKernel, EntryPoint = "NativeSumAll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeSumAllImpl(IntPtr input, IntPtr output, int n);
        public static void NativeSumAll(IntPtr input, IntPtr output, int n) => Checked(() => NativeSumAllImpl(input, output, n), nameof(NativeSumAll));

        [DllImport(CustomKernel, EntryPoint = "NativeMeanAll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeMeanAllImpl(IntPtr input, IntPtr output, int n);
        public static void NativeMeanAll(IntPtr input, IntPtr output, int n) => Checked(() => NativeMeanAllImpl(input, output, n), nameof(NativeMeanAll));

        [DllImport(CustomKernel, EntryPoint = "NativeArgMax", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeArgMaxImpl(IntPtr input, IntPtr output, int outer, int dim, int inner);
        public static void NativeArgMax(IntPtr input, IntPtr output, int outer, int dim, int inner) => Checked(() => NativeArgMaxImpl(input, output, outer, dim, inner), nameof(NativeArgMax));

        [DllImport(CustomKernel, EntryPoint = "NativeArgMin", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeArgMinImpl(IntPtr input, IntPtr output, int outer, int dim, int inner);
        public static void NativeArgMin(IntPtr input, IntPtr output, int outer, int dim, int inner) => Checked(() => NativeArgMinImpl(input, output, outer, dim, inner), nameof(NativeArgMin));

        [DllImport(CustomKernel, EntryPoint = "NativeCumSum", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeCumSumImpl(IntPtr input, IntPtr output, int outer, int dim, int inner);
        public static void NativeCumSum(IntPtr input, IntPtr output, int outer, int dim, int inner) => Checked(() => NativeCumSumImpl(input, output, outer, dim, inner), nameof(NativeCumSum));

        [DllImport(CustomKernel, EntryPoint = "NativeReverseCumSum", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeReverseCumSumImpl(IntPtr input, IntPtr output, int outer, int dim, int inner);
        public static void NativeReverseCumSum(IntPtr input, IntPtr output, int outer, int dim, int inner) => Checked(() => NativeReverseCumSumImpl(input, output, outer, dim, inner), nameof(NativeReverseCumSum));

        [DllImport(CustomKernel, EntryPoint = "NativeGeneralTranspose", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeGeneralTransposeImpl(IntPtr input, IntPtr output, int[] shape, int[] perm, int rank);
        public static void NativeGeneralTranspose(IntPtr input, IntPtr output, int[] shape, int[] perm, int rank)
            => Checked(() => NativeGeneralTransposeImpl(input, output, shape, perm, rank), nameof(NativeGeneralTranspose));

        [DllImport(CustomKernel, EntryPoint = "NativeBroadcast", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeBroadcastImpl(IntPtr input, IntPtr output, int[] inputShape, int[] outputShape, int rank);
        public static void NativeBroadcast(IntPtr input, IntPtr output, int[] inputShape, int[] outputShape, int rank)
            => Checked(() => NativeBroadcastImpl(input, output, inputShape, outputShape, rank), nameof(NativeBroadcast));

        [DllImport(CustomKernel, EntryPoint = "NativeSumTo", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeSumToImpl(IntPtr input, IntPtr output, int[] inShape, int[] outShape, int rank);
        public static void NativeSumTo(IntPtr input, IntPtr output, int[] inShape, int[] outShape, int rank)
            => Checked(() => NativeSumToImpl(input, output, inShape, outShape, rank), nameof(NativeSumTo));

        [DllImport(CustomKernel, EntryPoint = "NativeMatMul", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeMatMulImpl(IntPtr A, IntPtr B, IntPtr C, int m, int n, int k);
        public static void NativeMatMul(IntPtr A, IntPtr B, IntPtr C, int m, int n, int k) => Checked(() => NativeMatMulImpl(A, B, C, m, n, k), nameof(NativeMatMul));

        [DllImport(CustomKernel, EntryPoint = "NativeSoftmax", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeSoftmaxImpl(IntPtr input, IntPtr output, int rows, int cols);
        public static void NativeSoftmax(IntPtr input, IntPtr output, int rows, int cols) => Checked(() => NativeSoftmaxImpl(input, output, rows, cols), nameof(NativeSoftmax));

        [DllImport(CustomKernel, EntryPoint = "NativeLogicalNot", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeLogicalNotImpl(IntPtr input, IntPtr output, int count);
        public static void NativeLogicalNot(IntPtr input, IntPtr output, int count) => Checked(() => NativeLogicalNotImpl(input, output, count), nameof(NativeLogicalNot));

        [DllImport(CustomKernel, EntryPoint = "NativeClip", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeClipImpl(IntPtr input, IntPtr output, int count, float minVal, float maxVal);
        public static void NativeClip(IntPtr input, IntPtr output, int count, float minVal, float maxVal)
            => Checked(() => NativeClipImpl(input, output, count, minVal, maxVal), nameof(NativeClip));

        [DllImport(CustomKernel, EntryPoint = "NativeClipMask", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeClipMaskImpl(IntPtr input, IntPtr output, int count, float minVal, float maxVal);
        public static void NativeClipMask(IntPtr input, IntPtr output, int count, float minVal, float maxVal)
            => Checked(() => NativeClipMaskImpl(input, output, count, minVal, maxVal), nameof(NativeClipMask));

        [DllImport(CustomKernel, EntryPoint = "NativeMeanAxis", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeMeanAxisImpl(IntPtr input, IntPtr output, int outer, int dim, int inner);
        public static void NativeMeanAxis(IntPtr input, IntPtr output, int outer, int dim, int inner)
            => Checked(() => NativeMeanAxisImpl(input, output, outer, dim, inner), nameof(NativeMeanAxis));

        [DllImport(CustomKernel, EntryPoint = "NativeSumAxis", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeSumAxisImpl(IntPtr input, IntPtr output, int outer, int dim, int inner);
        public static void NativeSumAxis(IntPtr input, IntPtr output, int outer, int dim, int inner)
            => Checked(() => NativeSumAxisImpl(input, output, outer, dim, inner), nameof(NativeSumAxis));

        [DllImport(CustomKernel, EntryPoint = "NativeMaxAxis", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeMaxAxisImpl(IntPtr input, IntPtr output, int outer, int dim, int inner);
        public static void NativeMaxAxis(IntPtr input, IntPtr output, int outer, int dim, int inner)
            => Checked(() => NativeMaxAxisImpl(input, output, outer, dim, inner), nameof(NativeMaxAxis));

        [DllImport(CustomKernel, EntryPoint = "NativeMinAxis", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeMinAxisImpl(IntPtr input, IntPtr output, int outer, int dim, int inner);
        public static void NativeMinAxis(IntPtr input, IntPtr output, int outer, int dim, int inner)
            => Checked(() => NativeMinAxisImpl(input, output, outer, dim, inner), nameof(NativeMinAxis));

        [DllImport(CustomKernel, EntryPoint = "NativeGather", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeGatherImpl(IntPtr input, IntPtr indices, IntPtr output, int batch, int classes);
        public static void NativeGather(IntPtr input, IntPtr indices, IntPtr output, int batch, int classes)
            => Checked(() => NativeGatherImpl(input, indices, output, batch, classes), nameof(NativeGather));

        [DllImport(CustomKernel, EntryPoint = "NativeGatherGrad", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeGatherGradImpl(IntPtr gradOut, IntPtr indices, IntPtr gradIn, int batch, int classes);
        public static void NativeGatherGrad(IntPtr gradOut, IntPtr indices, IntPtr gradIn, int batch, int classes)
            => Checked(() => NativeGatherGradImpl(gradOut, indices, gradIn, batch, classes), nameof(NativeGatherGrad));

        [DllImport(CustomKernel, EntryPoint = "NativeGatherAxis", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeGatherAxisImpl(IntPtr input, IntPtr indices, IntPtr output, int outer, int dim, int inner, int k);
        public static void NativeGatherAxis(IntPtr input, IntPtr indices, IntPtr output, int outer, int dim, int inner, int k)
            => Checked(() => NativeGatherAxisImpl(input, indices, output, outer, dim, inner, k), nameof(NativeGatherAxis));

        [DllImport(CustomKernel, EntryPoint = "NativeGatherAxisGrad", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeGatherAxisGradImpl(IntPtr gradOut, IntPtr indices, IntPtr gradIn, int outer, int dim, int inner, int k);
        public static void NativeGatherAxisGrad(IntPtr gradOut, IntPtr indices, IntPtr gradIn, int outer, int dim, int inner, int k)
            => Checked(() => NativeGatherAxisGradImpl(gradOut, indices, gradIn, outer, dim, inner, k), nameof(NativeGatherAxisGrad));

        [DllImport(CustomKernel, EntryPoint = "NativeEmbedding", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeEmbeddingImpl(IntPtr weights, IntPtr indices, IntPtr output, int numWords, int embedDim, int totalIndices);
        public static void NativeEmbedding(IntPtr weights, IntPtr indices, IntPtr output, int numWords, int embedDim, int totalIndices)
            => Checked(() => NativeEmbeddingImpl(weights, indices, output, numWords, embedDim, totalIndices), nameof(NativeEmbedding));

        [DllImport(CustomKernel, EntryPoint = "NativeEmbeddingGrad", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeEmbeddingGradImpl(IntPtr gradOut, IntPtr indices, IntPtr gradWeights, int numWords, int embedDim, int totalIndices);
        public static void NativeEmbeddingGrad(IntPtr gradOut, IntPtr indices, IntPtr gradWeights, int numWords, int embedDim, int totalIndices)
            => Checked(() => NativeEmbeddingGradImpl(gradOut, indices, gradWeights, numWords, embedDim, totalIndices), nameof(NativeEmbeddingGrad));

        [DllImport(CustomKernel, EntryPoint = "NativeSlice", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeSliceImpl(IntPtr input, IntPtr output, int[] inShape, int[] outShape, int[] starts, int[] steps, int rank);
        public static void NativeSlice(IntPtr input, IntPtr output, int[] inShape, int[] outShape, int[] starts, int[] steps, int rank)
            => Checked(() => NativeSliceImpl(input, output, inShape, outShape, starts, steps, rank), nameof(NativeSlice));

        [DllImport(CustomKernel, EntryPoint = "NativeSliceGrad", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeSliceGradImpl(IntPtr gradOut, IntPtr gradIn, int[] originalShape, int[] newShape, int[] starts, int[] steps, int rank);
        public static void NativeSliceGrad(IntPtr gradOut, IntPtr gradIn, int[] originalShape, int[] newShape, int[] starts, int[] steps, int rank)
            => Checked(() => NativeSliceGradImpl(gradOut, gradIn, originalShape, newShape, starts, steps, rank), nameof(NativeSliceGrad));

        [DllImport(CustomKernel, EntryPoint = "NativeConcat", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeConcatImpl(IntPtr[] inputs, IntPtr output, int numInputs, int outerSize, int[] concatSizes, int innerSize);
        public static void NativeConcat(IntPtr[] inputs, IntPtr output, int numInputs, int outerSize, int[] concatSizes, int innerSize)
            => Checked(() => NativeConcatImpl(inputs, output, numInputs, outerSize, concatSizes, innerSize), nameof(NativeConcat));

        [DllImport(CustomKernel, EntryPoint = "NativeConv2DForward", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeConv2DForwardImpl(IntPtr input, IntPtr weight, IntPtr output, int batch, int inChannels, int inH, int inW, int outChannels, int outH, int outW, int kH, int kW, int stride, int padding);
        public static void NativeConv2DForward(IntPtr input, IntPtr weight, IntPtr output, int batch, int inChannels, int inH, int inW, int outChannels, int outH, int outW, int kH, int kW, int stride, int padding)
            => Checked(() => NativeConv2DForwardImpl(input, weight, output, batch, inChannels, inH, inW, outChannels, outH, outW, kH, kW, stride, padding), nameof(NativeConv2DForward));

        [DllImport(CustomKernel, EntryPoint = "NativeConv2DGradWeight", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeConv2DGradWeightImpl(IntPtr input, IntPtr gradOut, IntPtr gradWeight, int batch, int inChannels, int inH, int inW, int outChannels, int outH, int outW, int kH, int kW, int stride, int padding);
        public static void NativeConv2DGradWeight(IntPtr input, IntPtr gradOut, IntPtr gradWeight, int batch, int inChannels, int inH, int inW, int outChannels, int outH, int outW, int kH, int kW, int stride, int padding)
            => Checked(() => NativeConv2DGradWeightImpl(input, gradOut, gradWeight, batch, inChannels, inH, inW, outChannels, outH, outW, kH, kW, stride, padding), nameof(NativeConv2DGradWeight));

        [DllImport(CustomKernel, EntryPoint = "NativeConv2DGradInput", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeConv2DGradInputImpl(IntPtr gradOut, IntPtr weight, IntPtr gradInput, int batch, int inChannels, int inH, int inW, int outChannels, int outH, int outW, int kH, int kW, int stride, int padding);
        public static void NativeConv2DGradInput(IntPtr gradOut, IntPtr weight, IntPtr gradInput, int batch, int inChannels, int inH, int inW, int outChannels, int outH, int outW, int kH, int kW, int stride, int padding)
            => Checked(() => NativeConv2DGradInputImpl(gradOut, weight, gradInput, batch, inChannels, inH, inW, outChannels, outH, outW, kH, kW, stride, padding), nameof(NativeConv2DGradInput));

        [DllImport(CustomKernel, EntryPoint = "NativeConv3DForward", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeConv3DForwardImpl(IntPtr input, IntPtr weight, IntPtr output, int batch, int inChannels, int inD, int inH, int inW, int outChannels, int outD, int outH, int outW, int kD, int kH, int kW, int stride, int padding);
        public static void NativeConv3DForward(IntPtr input, IntPtr weight, IntPtr output, int batch, int inChannels, int inD, int inH, int inW, int outChannels, int outD, int outH, int outW, int kD, int kH, int kW, int stride, int padding)
            => Checked(() => NativeConv3DForwardImpl(input, weight, output, batch, inChannels, inD, inH, inW, outChannels, outD, outH, outW, kD, kH, kW, stride, padding), nameof(NativeConv3DForward));

        [DllImport(CustomKernel, EntryPoint = "NativeConv3DGradWeight", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeConv3DGradWeightImpl(IntPtr input, IntPtr gradOut, IntPtr gradWeight, int batch, int inChannels, int inD, int inH, int inW, int outChannels, int outD, int outH, int outW, int kD, int kH, int kW, int stride, int padding);
        public static void NativeConv3DGradWeight(IntPtr input, IntPtr gradOut, IntPtr gradWeight, int batch, int inChannels, int inD, int inH, int inW, int outChannels, int outD, int outH, int outW, int kD, int kH, int kW, int stride, int padding)
            => Checked(() => NativeConv3DGradWeightImpl(input, gradOut, gradWeight, batch, inChannels, inD, inH, inW, outChannels, outD, outH, outW, kD, kH, kW, stride, padding), nameof(NativeConv3DGradWeight));

        [DllImport(CustomKernel, EntryPoint = "NativeConv3DGradInput", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeConv3DGradInputImpl(IntPtr gradOut, IntPtr weight, IntPtr gradInput, int batch, int inChannels, int inD, int inH, int inW, int outChannels, int outD, int outH, int outW, int kD, int kH, int kW, int stride, int padding);
        public static void NativeConv3DGradInput(IntPtr gradOut, IntPtr weight, IntPtr gradInput, int batch, int inChannels, int inD, int inH, int inW, int outChannels, int outD, int outH, int outW, int kD, int kH, int kW, int stride, int padding)
            => Checked(() => NativeConv3DGradInputImpl(gradOut, weight, gradInput, batch, inChannels, inD, inH, inW, outChannels, outD, outH, outW, kD, kH, kW, stride, padding), nameof(NativeConv3DGradInput));

        [DllImport(CustomKernel, EntryPoint = "InvokeHolonomicKernel", CallingConvention = CallingConvention.Cdecl)]
        private static extern int InvokeHolonomicKernelImpl(IntPtr inputs, IntPtr weights, IntPtr intWeights, IntPtr outputs, int inputSize, int neuronCount, int fractalDepth);
        public static int InvokeHolonomicKernel(IntPtr inputs, IntPtr weights, IntPtr intWeights, IntPtr outputs, int inputSize, int neuronCount, int fractalDepth)
        {
            int code = InvokeHolonomicKernelImpl(inputs, weights, intWeights, outputs, inputSize, neuronCount, fractalDepth);
            if (code != 0)
                throw new InvalidOperationException($"Holonomic kernel failed: {(CudaError)code}");
            ThrowIfKernelFailed(nameof(InvokeHolonomicKernel));
            return code;
        }

        [DllImport(CustomKernel, EntryPoint = "NativeTopK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeTopKImpl(IntPtr input, IntPtr outValues, IntPtr outIndices, int outer, int dim, int inner, int k);
        public static void NativeTopK(IntPtr input, IntPtr outValues, IntPtr outIndices, int outer, int dim, int inner, int k)
            => Checked(() => NativeTopKImpl(input, outValues, outIndices, outer, dim, inner, k), nameof(NativeTopK));

        [DllImport(CustomKernel, EntryPoint = "NativeTopKScatterGrad", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NativeTopKScatterGradImpl(IntPtr gradOut, IntPtr indices, IntPtr gradIn, int outer, int dim, int inner, int k);
        public static void NativeTopKScatterGrad(IntPtr gradOut, IntPtr indices, IntPtr gradIn, int outer, int dim, int inner, int k)
            => Checked(() => NativeTopKScatterGradImpl(gradOut, indices, gradIn, outer, dim, inner, k), nameof(NativeTopKScatterGrad));
    }
}
