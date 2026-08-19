// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Native.PInvoke
{

    #region Using Statements:

    using System;
    using System.Runtime.InteropServices;
    /// <summary>
    /// Provides managed wrappers for CUDA runtime API, cuBLAS, and custom GPU kernels.
    /// Fully compatible with existing code while maintaining perfect naming and robustness.
    /// </summary>
    /// <remarks>
    /// This static class serves as the primary bridge between managed .NET code and native CUDA operations.
    /// It encapsulates raw P/Invoke signatures targeting the CUDA Runtime library (<c>cudart64_12.dll</c>) 
    /// and custom compiled CUDA kernels (<c>cuda_backend.dll</c>). Developers should ensure that proper 
    /// synchronization and lifetime management of GPU memory resources are maintained when calling these APIs.
    /// </remarks>

    #endregion

    /// <summary>
    /// Provides managed wrappers for CUDA runtime API, cuBLAS, and custom GPU kernels.
    /// Fully compatible with existing code while maintaining perfect naming and robustness.
    /// </summary>
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
                int count = 0;
                CudaError err = cudaGetDeviceCount(out count);
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

        // =================================================================================
        // CUDA RUNTIME APIs (P/Invokes)
        // =================================================================================
        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        private static extern CudaError cudaMalloc(out IntPtr devicePtr, ulong size);

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        private static extern CudaError cudaFree(IntPtr devicePtr);

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        private static extern CudaError cudaMemset(IntPtr devicePtr, int value, ulong count);

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        private static extern CudaError cudaMemcpy(IntPtr dst, IntPtr src, ulong count, cudaMemcpyKind kind);

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        private static extern CudaError cudaDeviceSynchronize();

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        private static extern CudaError cudaGetDeviceCount(out int count);

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        private static extern CudaError cudaHostRegister(IntPtr ptr, ulong size, uint flags);

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        private static extern CudaError cudaHostUnregister(IntPtr ptr);

        // =================================================================================
        // CUSTOM ARBORNET NATIVE GPU KERNELS (P/Invokes)
        // =================================================================================
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeAdd(IntPtr a, IntPtr b, IntPtr c, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSubtract(IntPtr a, IntPtr b, IntPtr c, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeMultiply(IntPtr a, IntPtr b, IntPtr c, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeDivide(IntPtr a, IntPtr b, IntPtr c, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativePow(IntPtr a, IntPtr b, IntPtr c, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativePowTensor(IntPtr a, IntPtr b, IntPtr c, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeEqual(IntPtr a, IntPtr b, IntPtr c, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeGreaterThan(IntPtr a, IntPtr b, IntPtr c, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeGreaterThanOrEqual(IntPtr a, IntPtr b, IntPtr c, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeLessThan(IntPtr a, IntPtr b, IntPtr c, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeLessEqual(IntPtr a, IntPtr b, IntPtr c, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeWhere(IntPtr cond, IntPtr a, IntPtr b, IntPtr c, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeReLU(IntPtr input, IntPtr output, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeReLUGrad(IntPtr gradOut, IntPtr originIn, IntPtr gradIn, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSigmoid(IntPtr input, IntPtr output, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSigmoidGrad(IntPtr gradOut, IntPtr originOut, IntPtr gradIn, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeTanh(IntPtr input, IntPtr output, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeNegate(IntPtr input, IntPtr output, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeExp(IntPtr input, IntPtr output, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeLog(IntPtr input, IntPtr output, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSqrt(IntPtr input, IntPtr output, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeAbs(IntPtr input, IntPtr output, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSin(IntPtr input, IntPtr output, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeCos(IntPtr input, IntPtr output, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSign(IntPtr input, IntPtr output, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeOnes(IntPtr output, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSetScalar(IntPtr output, float value, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativePowScalar(IntPtr input, IntPtr output, int n, float exponent);

        // Out-of-place Scalar math
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeAddScalar(IntPtr input, IntPtr output, int n, float value);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSubtractScalar(IntPtr input, IntPtr output, int n, float value);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeMultiplyScalar(IntPtr input, IntPtr output, int n, float value);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeDivideScalar(IntPtr input, IntPtr output, int n, float value);

        // In-place Scalar math
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeAddScalarInPlace(IntPtr data, float value, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSubtractScalarInPlace(IntPtr data, float value, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeMultiplyScalarInPlace(IntPtr data, float value, int n);

        // Native GPU generation (No CPU roundtrips)
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeEye(IntPtr data, int size);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeRand(IntPtr data, int n, uint seed);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeRandn(IntPtr data, int n, uint seed);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeTranspose(IntPtr input, IntPtr output, int rows, int cols);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSumAll(IntPtr input, IntPtr output, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeMeanAll(IntPtr input, IntPtr output, int n);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeArgMax(IntPtr input, IntPtr output, int outer, int dim, int inner);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeArgMin(IntPtr input, IntPtr output, int outer, int dim, int inner);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeCumSum(IntPtr input, IntPtr output, int outer, int dim, int inner);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeGeneralTranspose(IntPtr input, IntPtr output, int[] shape, int[] perm, int rank);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeBroadcast(IntPtr input, IntPtr output, int[] inputShape, int[] outputShape, int rank);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeMatMul(IntPtr A, IntPtr B, IntPtr C, int m, int n, int k);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSoftmax(IntPtr input, IntPtr output, int rows, int cols);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeLogicalNot(IntPtr input, IntPtr output, int count);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeClip(IntPtr input, IntPtr output, int count, float minVal, float maxVal);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeClipMask(IntPtr input, IntPtr output, int count, float minVal, float maxVal);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeMeanAxis(IntPtr input, IntPtr output, int outer, int dim, int inner);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeMaxAxis(IntPtr input, IntPtr output, int outer, int dim, int inner);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeMinAxis(IntPtr input, IntPtr output, int outer, int dim, int inner);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeGather(IntPtr input, IntPtr indices, IntPtr output, int batch, int classes);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeGatherGrad(IntPtr gradOut, IntPtr indices, IntPtr gradIn, int batch, int classes);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeEmbedding(IntPtr weights, IntPtr indices, IntPtr output, int numWords, int embedDim, int totalIndices);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeEmbeddingGrad(IntPtr gradOut, IntPtr indices, IntPtr gradWeights, int numWords, int embedDim, int totalIndices);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSlice(IntPtr input, IntPtr output, int[] inShape, int[] outShape, int[] starts, int[] steps, int rank);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSliceGrad(IntPtr gradOut, IntPtr gradIn, int[] originalShape, int[] newShape, int[] starts, int[] steps, int rank);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeConcat(IntPtr[] inputs, IntPtr output, int numInputs, int outerSize, int[] concatSizes, int innerSize);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeConv2DForward(IntPtr input, IntPtr weight, IntPtr output, int batch, int inChannels, int inH, int inW, int outChannels, int outH, int outW, int kH, int kW, int stride, int padding);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeConv2DGradWeight(IntPtr input, IntPtr gradOut, IntPtr gradWeight, int batch, int inChannels, int inH, int inW, int outChannels, int outH, int outW, int kH, int kW, int stride, int padding);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeConv2DGradInput(IntPtr gradOut, IntPtr weight, IntPtr gradInput, int batch, int inChannels, int inH, int inW, int outChannels, int outH, int outW, int kH, int kW, int stride, int padding);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeConv3DForward(IntPtr input, IntPtr weight, IntPtr output, int batch, int inChannels, int inD, int inH, int inW, int outChannels, int outD, int outH, int outW, int kD, int kH, int kW, int stride, int padding);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeConv3DGradWeight(IntPtr input, IntPtr gradOut, IntPtr gradWeight, int batch, int inChannels, int inD, int inH, int inW, int outChannels, int outD, int outH, int outW, int kD, int kH, int kW, int stride, int padding);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeConv3DGradInput(IntPtr gradOut, IntPtr weight, IntPtr gradInput, int batch, int inChannels, int inD, int inH, int inW, int outChannels, int outD, int outH, int outW, int kD, int kH, int kW, int stride, int padding);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern int InvokeHolonomicKernel(IntPtr inputs, IntPtr weights, IntPtr intWeights, IntPtr outputs, int inputSize, int neuronCount, int fractalDepth);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeTopK(
    IntPtr input,
    IntPtr outValues,
    IntPtr outIndices,
    int outer, int dim, int inner, int k);

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeTopKScatterGrad(
            IntPtr gradOut,
            IntPtr indices,
            IntPtr gradIn,
            int outer, int dim, int inner, int k);

        // =// =================================================================================
        // CUDA WORKER ERROR CONTROLLER
        // =================================================================================
        public static void Check(CudaError err, string method)
        {
            if (err != CudaError.Success)
                throw new InvalidOperationException($"CUDA Driver Exception in {method}: {err}");
        }

        // =================================================================================
        // HIGH-LEVEL DISPATCHER WRAPPERS & PASCALCASE TRANSLATORS
        // =================================================================================
        public static void CudaMalloc(out IntPtr devicePtr, ulong size)
        {
            Check(cudaMalloc(out devicePtr, size), nameof(CudaMalloc));
        }

        public static void CudaFree(IntPtr devicePtr)
        {
            Check(cudaFree(devicePtr), nameof(CudaFree));
        }

        public static void CudaHostRegister(IntPtr ptr, ulong size, uint flags)
        {
            Check(cudaHostRegister(ptr, size, flags), nameof(CudaHostRegister));
        }

        public static void CudaHostUnregister(IntPtr ptr)
        {
            Check(cudaHostUnregister(ptr), nameof(CudaHostUnregister));
        }

        public static void CudaMemcpy(IntPtr dst, IntPtr src, ulong count, cudaMemcpyKind kind)
        {
            Check(cudaMemcpy(dst, src, count, kind), nameof(CudaMemcpy));
        }

        public static void CudaMemset(IntPtr devicePtr, int value, ulong count)
        {
            Check(cudaMemset(devicePtr, value, count), nameof(CudaMemset));
        }

        public static void Synchronize()
        {
            Check(cudaDeviceSynchronize(), nameof(Synchronize));
        }

        public static void Initialize(out IntPtr context)
        {
            int count = 0;
            Check(cudaGetDeviceCount(out count), nameof(cudaGetDeviceCount));
            // Set dummy non-zero active state handle representing initialization success
            context = new IntPtr(1);
        }

        public static void Cleanup(IntPtr context)
        {
            // No-op for runtime context virtualization layers
        }

        public static void MatMul(IntPtr A, IntPtr B, IntPtr C, int m, int n, int k)
        {
            NativeMatMul(A, B, C, m, n, k);
        }

        public static void Add(IntPtr a, IntPtr b, IntPtr c, int n)
        {
            NativeAdd(a, b, c, n);
        }

        public static void Reshape(IntPtr context, IntPtr input, IntPtr output, int[] inputShape, int[] outputShape)
        {
            long size = 1;
            foreach (int dim in inputShape) size *= dim;
            CudaMemcpy(output, input, (ulong)(size * sizeof(float)), cudaMemcpyKind.cudaMemcpyDeviceToDevice);
        }
    }
}