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

    public static partial class CUDA
    {
        private const string CudaRuntime = "cudart64_12.dll";
        private const string CustomKernel = "cuda_backend.dll";

        static CUDA()
        {
            NativeResolver.Register();
        }
        /// <summary>
        /// Determines whether CUDA-capable devices are available on the current system.
        /// </summary>
        /// <returns><c>true</c> if at least one CUDA device is available and accessible; otherwise, <c>false</c>.</returns>

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

        public enum CudaError { Success = 0, InvalidValue = 1, OutOfMemory = 2, NotInitialized = 3, InvalidDevice = 10, InvalidPointer = 17, InvalidMemcpyDirection = 21, InvalidResourceHandle = 33, Unknown = 999 }
        public enum cudaMemcpyKind : int { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4 }
        /// <summary>
        /// Allocates physical memory on the device.
        /// </summary>
        /// <param name="devicePtr">Outputs the pointer to the allocated device memory.</param>
        /// <param name="size">The size in bytes of the requested allocation.</param>
        /// <returns>A <see cref="CudaError"/> representing the operation outcome.</returns>

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        public static extern CudaError cudaMalloc(out IntPtr devicePtr, ulong size);
        /// <summary>
        /// Frees physical memory allocated on the device.
        /// </summary>
        /// <param name="devicePtr">The pointer to the device memory to be freed.</param>
        /// <returns>A <see cref="CudaError"/> representing the operation outcome.</returns>
        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        public static extern CudaError cudaFree(IntPtr devicePtr);
        /// <summary>
        /// Fills a block of device memory with a specific value.
        /// </summary>
        /// <param name="devicePtr">The pointer to the device memory to fill.</param>
        /// <param name="value">The byte value with which to fill the memory block.</param>
        /// <param name="count">The number of bytes to set.</param>
        /// <returns>A <see cref="CudaError"/> representing the operation outcome.</returns>
        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        public static extern CudaError cudaMemset(IntPtr devicePtr, int value, ulong count);
        /// <summary>
        /// Copies data between host and device memory buffers.
        /// </summary>
        /// <param name="dst">The destination memory pointer.</param>
        /// <param name="src">The source memory pointer.</param>
        /// <param name="count">The size in bytes of the data to copy.</param>
        /// <param name="kind">The direction of the copy operation.</param>
        /// <returns>A <see cref="CudaError"/> representing the operation outcome.</returns>
        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        public static extern CudaError cudaMemcpy(IntPtr dst, IntPtr src, ulong count, cudaMemcpyKind kind);
        /// <summary>
        /// Blocks the host execution thread until the device has completed all preceding requested tasks.
        /// </summary>
        /// <returns>A <see cref="CudaError"/> representing the operation outcome.</returns>
        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        public static extern CudaError cudaDeviceSynchronize();
        /// <summary>
        /// Retrieves the total number of CUDA-capable devices on the host system.
        /// </summary>
        /// <param name="count">Outputs the number of CUDA-capable devices.</param>
        /// <returns>A <see cref="CudaError"/> representing the operation outcome.</returns>
        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        public static extern CudaError cudaGetDeviceCount(out int count);
        /// <summary>
        /// Registers a range of host memory for use by CUDA, locking it in place and optimizing access speed.
        /// </summary>
        /// <param name="ptr">The pointer to the host memory buffer to register.</param>
        /// <param name="size">The size in bytes of the host memory buffer.</param>
        /// <param name="flags">Registration options and flags.</param>
        /// <returns>A <see cref="CudaError"/> representing the operation outcome.</returns>
        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        public static extern CudaError cudaHostRegister(IntPtr ptr, ulong size, uint flags);
        /// <summary>
        /// Unregisters a previously page-locked host memory range.
        /// </summary>
        /// <param name="ptr">The pointer to the registered host memory buffer.</param>
        /// <returns>A <see cref="CudaError"/> representing the operation outcome.</returns>
        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        public static extern CudaError cudaHostUnregister(IntPtr ptr);
        /// <summary>
        /// Creates an asynchronous execution stream.
        /// </summary>
        /// <param name="stream">Outputs the handle to the newly created stream.</param>
        /// <returns>A <see cref="CudaError"/> representing the operation outcome.</returns>

        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        public static extern CudaError cudaStreamCreate(out IntPtr stream);
        /// <summary>
        /// Destroys an asynchronous execution stream.
        /// </summary>
        /// <param name="stream">The handle to the stream to destroy.</param>
        /// <returns>A <see cref="CudaError"/> representing the operation outcome.</returns>
        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        public static extern CudaError cudaStreamDestroy(IntPtr stream);
        /// <summary>
        /// Asynchronously copies data between host and device memory buffers using a specified execution stream.
        /// </summary>
        /// <param name="dst">The destination memory pointer.</param>
        /// <param name="src">The source memory pointer.</param>
        /// <param name="count">The size in bytes of the data to copy.</param>
        /// <param name="kind">The direction of the copy operation.</param>
        /// <param name="stream">The stream handle associated with the asynchronous copy operation.</param>
        /// <returns>A <see cref="CudaError"/> representing the operation outcome.</returns>
        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        public static extern CudaError cudaMemcpyAsync(IntPtr dst, IntPtr src, ulong count, cudaMemcpyKind kind, IntPtr stream);
        /// <summary>
        /// Blocks host execution until all operations in the specified stream have finished executing.
        /// </summary>
        /// <param name="stream">The stream handle to synchronize.</param>
        /// <returns>A <see cref="CudaError"/> representing the operation outcome.</returns>
        [DllImport(CudaRuntime, CallingConvention = CallingConvention.Cdecl)]
        public static extern CudaError cudaStreamSynchronize(IntPtr stream);
        /// <summary>
        /// Gathers values along an axis specified by indices from a source GPU memory buffer.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="indices">The pointer to the indices tensor in device memory.</param>
        /// <param name="output">The pointer to the gathered output tensor in device memory.</param>
        /// <param name="batch">The batch dimension size.</param>
        /// <param name="classes">The class dimension size.</param>

        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeGather(IntPtr input, IntPtr indices, IntPtr output, int batch, int classes);
        /// <summary>
        /// Computes the gradient for the gather operation, mapping back gradients to their source locations.
        /// </summary>
        /// <param name="gradOut">The pointer to the outgoing gradient tensor in device memory.</param>
        /// <param name="indices">The pointer to the indices tensor used in the forward gather pass.</param>
        /// <param name="gradIn">The pointer to the incoming gradient tensor to be accumulated on device memory.</param>
        /// <param name="batch">The batch dimension size.</param>
        /// <param name="classes">The class dimension size.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl, EntryPoint = "NativeGatherGrad")]
        public static extern void NativeGatherGrad(IntPtr gradOut, IntPtr indices, IntPtr gradIn, int batch, int classes);
        /// <summary>
        /// Performs element-wise addition on two device vectors: <c>c = a + b</c>.
        /// </summary>
        /// <param name="a">The pointer to the first input device tensor.</param>
        /// <param name="b">The pointer to the second input device tensor.</param>
        /// <param name="c">The pointer to the output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeAdd(IntPtr a, IntPtr b, IntPtr c, int n);
        /// <summary>
        /// Performs element-wise subtraction on two device vectors: <c>c = a - b</c>.
        /// </summary>
        /// <param name="a">The pointer to the minuend device tensor.</param>
        /// <param name="b">The pointer to the subtrahend device tensor.</param>
        /// <param name="c">The pointer to the output difference device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSubtract(IntPtr a, IntPtr b, IntPtr c, int n);
        /// <summary>
        /// Performs element-wise multiplication on two device vectors: <c>c = a * b</c>.
        /// </summary>
        /// <param name="a">The pointer to the first input device tensor.</param>
        /// <param name="b">The pointer to the second input device tensor.</param>
        /// <param name="c">The pointer to the output product device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeMultiply(IntPtr a, IntPtr b, IntPtr c, int n);
        /// <summary>
        /// Performs element-wise division on two device vectors: <c>c = a / b</c>.
        /// </summary>
        /// <param name="a">The pointer to the dividend device tensor.</param>
        /// <param name="b">The pointer to the divisor device tensor.</param>
        /// <param name="c">The pointer to the output quotient device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeDivide(IntPtr a, IntPtr b, IntPtr c, int n);
        /// <summary>
        /// Raises elements of base device vector <paramref name="a"/> to the power of exponent device vector <paramref name="b"/>: <c>c = a^b</c>.
        /// </summary>
        /// <param name="a">The pointer to the base device tensor.</param>
        /// <param name="b">The pointer to the exponent device tensor.</param>
        /// <param name="c">The pointer to the output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativePow(IntPtr a, IntPtr b, IntPtr c, int n);
        /// <summary>
        /// Compares element-wise equality between two device vectors: <c>c = (a == b)</c>.
        /// </summary>
        /// <param name="a">The pointer to the first input device tensor.</param>
        /// <param name="b">The pointer to the second input device tensor.</param>
        /// <param name="c">The pointer to the boolean/numeric result mask device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeEqual(IntPtr a, IntPtr b, IntPtr c, int n);
        /// <summary>
        /// Compares whether elements of vector <paramref name="a"/> are greater than vector <paramref name="b"/>: <c>c = (a > b)</c>.
        /// </summary>
        /// <param name="a">The pointer to the left-hand input device tensor.</param>
        /// <param name="b">The pointer to the right-hand input device tensor.</param>
        /// <param name="c">The pointer to the output mask device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeGreaterThan(IntPtr a, IntPtr b, IntPtr c, int n);
        /// <summary>
        /// Compares whether elements of vector <paramref name="a"/> are less than vector <paramref name="b"/>: <c>c = (a &lt; b)</c>.
        /// </summary>
        /// <param name="a">The pointer to the left-hand input device tensor.</param>
        /// <param name="b">The pointer to the right-hand input device tensor.</param>
        /// <param name="c">The pointer to the output mask device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeLessThan(IntPtr a, IntPtr b, IntPtr c, int n);
        /// <summary>
        /// Selects elements from either <paramref name="a"/> or <paramref name="b"/> based on the truth value of <paramref name="cond"/>.
        /// </summary>
        /// <param name="cond">The pointer to the condition mask device tensor.</param>
        /// <param name="a">The pointer to the source device tensor chosen where condition is true.</param>
        /// <param name="b">The pointer to the source device tensor chosen where condition is false.</param>
        /// <param name="c">The pointer to the resulting output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeWhere(IntPtr cond, IntPtr a, IntPtr b, IntPtr c, int n);
        /// <summary>
        /// Applies the Rectified Linear Unit activation function element-wise on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the computed activation output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeReLU(IntPtr input, IntPtr output, int n);
        /// <summary>
        /// Computes the gradient of the Rectified Linear Unit function during backpropagation.
        /// </summary>
        /// <param name="gradOut">The pointer to the outgoing gradient device tensor.</param>
        /// <param name="originIn">The pointer to the original forward pass input device tensor.</param>
        /// <param name="gradIn">The pointer to the incoming computed gradient device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeReLUGrad(IntPtr gradOut, IntPtr originIn, IntPtr gradIn, int n);
        /// <summary>
        /// Applies the Sigmoid activation function element-wise on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the computed activation output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSigmoid(IntPtr input, IntPtr output, int n);
        /// <summary>
        /// Computes the gradient of the Sigmoid function during backpropagation.
        /// </summary>
        /// <param name="gradOut">The pointer to the outgoing gradient device tensor.</param>
        /// <param name="originOut">The pointer to the original forward pass computed output device tensor.</param>
        /// <param name="gradIn">The pointer to the incoming computed gradient device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSigmoidGrad(IntPtr gradOut, IntPtr originOut, IntPtr gradIn, int n);
        /// <summary>
        /// Applies the Hyperbolic Tangent (tanh) activation function element-wise on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the computed activation output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeTanh(IntPtr input, IntPtr output, int n);
        /// <summary>
        /// Performs element-wise arithmetic negation on device memory: <c>output = -input</c>.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the negated output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeNegate(IntPtr input, IntPtr output, int n);
        /// <summary>
        /// Computes the natural exponential function element-wise on device memory: <c>output = e^input</c>.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the exponentiated output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeExp(IntPtr input, IntPtr output, int n);
        /// <summary>
        /// Computes the natural logarithm element-wise on device memory: <c>output = ln(input)</c>.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the logarithm output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeLog(IntPtr input, IntPtr output, int n);
        /// <summary>
        /// Computes the square root element-wise on device memory: <c>output = sqrt(input)</c>.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the square root output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSqrt(IntPtr input, IntPtr output, int n);
        /// <summary>
        /// Computes the absolute value element-wise on device memory: <c>output = |input|</c>.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the absolute value output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeAbs(IntPtr input, IntPtr output, int n);
        /// <summary>
        /// Computes the sine function element-wise on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor in radians.</param>
        /// <param name="output">The pointer to the sine output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSin(IntPtr input, IntPtr output, int n);
        /// <summary>
        /// Computes the cosine function element-wise on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor in radians.</param>
        /// <param name="output">The pointer to the cosine output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeCos(IntPtr input, IntPtr output, int n);
        /// <summary>
        /// Computes the signum function element-wise on device memory (-1 for negative, 0 for zero, 1 for positive).
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the sign output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSign(IntPtr input, IntPtr output, int n);
        /// <summary>
        /// Fills a device memory buffer with the scalar value 1.0f.
        /// </summary>
        /// <param name="output">The pointer to the output device tensor.</param>
        /// <param name="n">The total number of elements to initialize.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeOnes(IntPtr output, int n);
        /// <summary>
        /// Fills a device memory buffer with a specified float scalar value.
        /// </summary>
        /// <param name="output">The pointer to the output device tensor.</param>
        /// <param name="value">The float value to populate across the memory block.</param>
        /// <param name="n">The total number of elements to initialize.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSetScalar(IntPtr output, float value, int n);
        /// <summary>
        /// Raises elements of a base device vector to a constant scalar power.
        /// </summary>
        /// <param name="input">The pointer to the input base device tensor.</param>
        /// <param name="output">The pointer to the calculated output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        /// <param name="exponent">The float power exponent value.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativePowScalar(IntPtr input, IntPtr output, int n, float exponent);
        /// <summary>
        /// Computes the sum of all elements within a device tensor and stores the result in a 1-element tensor.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the single-value output sum device tensor.</param>
        /// <param name="n">The total number of elements to sum.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSumAll(IntPtr input, IntPtr output, int n);
        /// <summary>
        /// Computes the arithmetic mean of all elements in a device tensor and stores the result in a 1-element tensor.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the single-value output mean device tensor.</param>
        /// <param name="n">The total number of elements to compute.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeMeanAll(IntPtr input, IntPtr output, int n);
        /// <summary>
        /// Finds the indices of maximum values along a given reduction dimension.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the calculated index output device tensor.</param>
        /// <param name="outer">The size of dimensions outer to the reduction dimension.</param>
        /// <param name="dim">The size of the reduction dimension.</param>
        /// <param name="inner">The size of dimensions inner to the reduction dimension.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeArgMax(IntPtr input, IntPtr output, int outer, int dim, int inner);
        /// <summary>
        /// Finds the indices of minimum values along a given reduction dimension.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the calculated index output device tensor.</param>
        /// <param name="outer">The size of dimensions outer to the reduction dimension.</param>
        /// <param name="dim">The size of the reduction dimension.</param>
        /// <param name="inner">The size of dimensions inner to the reduction dimension.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeArgMin(IntPtr input, IntPtr output, int outer, int dim, int inner);
        /// <summary>
        /// Computes the cumulative sum of elements along a specified dimension.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the cumulative sum output device tensor.</param>
        /// <param name="outer">The size of dimensions outer to the accumulation dimension.</param>
        /// <param name="dim">The size of the accumulation dimension.</param>
        /// <param name="inner">The size of dimensions inner to the accumulation dimension.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeCumSum(IntPtr input, IntPtr output, int outer, int dim, int inner);
        /// <summary>
        /// Transposes a 2D matrix on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input matrix in device memory.</param>
        /// <param name="output">The pointer to the transposed output matrix in device memory.</param>
        /// <param name="rows">The number of rows in the input matrix.</param>
        /// <param name="cols">The number of columns in the input matrix.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeTranspose(IntPtr input, IntPtr output, int rows, int cols);
        /// <summary>
        /// Performs a multi-dimensional permutation transpose of a tensor on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input multi-dimensional device tensor.</param>
        /// <param name="output">The pointer to the transposed output device tensor.</param>
        /// <param name="shape">The shape array of the input tensor.</param>
        /// <param name="perm">The target dimension permutation index order mapping array.</param>
        /// <param name="rank">The total number of dimensions (rank) of the input tensor.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeGeneralTranspose(IntPtr input, IntPtr output, int[] shape, int[] perm, int rank);
        /// <summary>
        /// Broadcasts elements from a lower dimensional source tensor to match a larger target shape.
        /// </summary>
        /// <param name="input">The pointer to the source input device tensor.</param>
        /// <param name="output">The pointer to the broadcasted output device tensor.</param>
        /// <param name="inShape">The shape array of the source input tensor.</param>
        /// <param name="outShape">The target destination shape array.</param>
        /// <param name="rank">The dimensional rank of the target output tensor.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeBroadcast(IntPtr input, IntPtr output, int[] inShape, int[] outShape, int rank);
        /// <summary>
        /// Computes standard matrix multiplication of general matrices: <c>c = a * b</c>.
        /// </summary>
        /// <param name="a">The pointer to left-hand side matrix A on device memory.</param>
        /// <param name="b">The pointer to right-hand side matrix B on device memory.</param>
        /// <param name="c">The pointer to resulting matrix C on device memory.</param>
        /// <param name="m">The number of rows of matrix A and C.</param>
        /// <param name="n">The number of columns of matrix B and C.</param>
        /// <param name="k">The number of columns of matrix A and rows of matrix B.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeMatMul(IntPtr a, IntPtr b, IntPtr c, int m, int n, int k);
        /// <summary>
        /// Computes Softmax normalized probabilities across rows of a 2D matrix.
        /// </summary>
        /// <param name="input">The pointer to the unnormalized logits device tensor.</param>
        /// <param name="output">The pointer to the calculated probability distribution output device tensor.</param>
        /// <param name="rows">The total number of rows in the matrix.</param>
        /// <param name="cols">The total number of columns in the matrix.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSoftmax(IntPtr input, IntPtr output, int rows, int cols);
        /// <summary>
        /// Performs element-wise logical negation on a boolean/numeric mask on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input condition device tensor.</param>
        /// <param name="output">The pointer to the logically inverted output device tensor.</param>
        /// <param name="count">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeLogicalNot(IntPtr input, IntPtr output, int count);
        /// <summary>
        /// Clamps all values in a device tensor to stay within the boundaries of a closed interval.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the clamped output device tensor.</param>
        /// <param name="count">The total number of elements to process.</param>
        /// <param name="minVal">The lower bound scalar clamp threshold.</param>
        /// <param name="maxVal">The upper bound scalar clamp threshold.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeClip(IntPtr input, IntPtr output, int count, float minVal, float maxVal);
        /// <summary>
        /// Generates a binary mask indicating which elements were clamped during a clip operation.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the output binary mask device tensor.</param>
        /// <param name="count">The total number of elements to process.</param>
        /// <param name="minVal">The lower bound clamp threshold.</param>
        /// <param name="maxVal">The upper bound clamp threshold.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeClipMask(IntPtr input, IntPtr output, int count, float minVal, float maxVal);
        /// <summary>
        /// Computes the mean along a specified reduction axis.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the calculated mean output device tensor.</param>
        /// <param name="outer">The size of dimensions outer to the reduction dimension.</param>
        /// <param name="dim">The size of the reduction dimension.</param>
        /// <param name="inner">The size of dimensions inner to the reduction dimension.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl, EntryPoint = "NativeMeanAxis")]
        public static extern void NativeMeanAxis(IntPtr input, IntPtr output, int outer, int dim, int inner);
        /// <summary>
        /// Finds the maximum values along a specified reduction axis.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the computed maximum output device tensor.</param>
        /// <param name="outer">The size of dimensions outer to the reduction dimension.</param>
        /// <param name="dim">The size of the reduction dimension.</param>
        /// <param name="inner">The size of dimensions inner to the reduction dimension.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeMaxAxis(IntPtr input, IntPtr output, int outer, int dim, int inner);
        /// <summary>
        /// Finds the minimum values along a specified reduction axis.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the computed minimum output device tensor.</param>
        /// <param name="outer">The size of dimensions outer to the reduction dimension.</param>
        /// <param name="dim">The size of the reduction dimension.</param>
        /// <param name="inner">The size of dimensions inner to the reduction dimension.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeMinAxis(IntPtr input, IntPtr output, int outer, int dim, int inner);
        /// <summary>
        /// Performs element-wise power computation with tensor-valued base and exponent: <c>c = a^b</c>.
        /// </summary>
        /// <param name="a">The pointer to the base device tensor.</param>
        /// <param name="b">The pointer to the exponent device tensor.</param>
        /// <param name="c">The pointer to the calculated output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativePowTensor(IntPtr a, IntPtr b, IntPtr c, int n);
        /// <summary>
        /// Extracts a sub-tensor slice from an input multi-dimensional tensor.
        /// </summary>
        /// <param name="input">The pointer to the input source device tensor.</param>
        /// <param name="output">The pointer to the extracted output slice device tensor.</param>
        /// <param name="inShape">The shape array of the input tensor.</param>
        /// <param name="outShape">The shape array of the output sliced tensor.</param>
        /// <param name="starts">The start indices of the slice along each dimension.</param>
        /// <param name="steps">The step stride intervals along each dimension.</param>
        /// <param name="rank">The total dimensional rank of the tensor.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSlice(IntPtr input, IntPtr output, int[] inShape, int[] outShape, int[] starts, int[] steps, int rank);
        /// <summary>
        /// Computes the backpropagation gradient for a slice extraction operation.
        /// </summary>
        /// <param name="gradOut">The pointer to the outgoing gradient device tensor.</param>
        /// <param name="gradIn">The pointer to the incoming computed gradient device tensor.</param>
        /// <param name="originalShape">The shape array of the original tensor before slicing.</param>
        /// <param name="newShape">The shape array of the sliced sub-tensor.</param>
        /// <param name="starts">The start indices of the slice along each dimension.</param>
        /// <param name="steps">The step stride intervals along each dimension.</param>
        /// <param name="rank">The total dimensional rank of the tensor.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeSliceGrad(IntPtr gradOut, IntPtr gradIn, int[] originalShape, int[] newShape, int[] starts, int[] steps, int rank);
        /// <summary>
        /// Concatenates multiple tensors along a specified target dimension on device memory.
        /// </summary>
        /// <param name="inputs">The array of pointers to input device tensors.</param>
        /// <param name="output">The pointer to the concatenated destination output device tensor.</param>
        /// <param name="numInputs">The total number of input tensors to concatenate.</param>
        /// <param name="outerSize">The size of dimensions outer to the concatenation dimension.</param>
        /// <param name="concatSizes">The array containing the sizes of the concatenation dimension for each input tensor.</param>
        /// <param name="innerSize">The size of dimensions inner to the concatenation dimension.</param>
        [DllImport(CustomKernel, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NativeConcat(IntPtr[] inputs, IntPtr output, int numInputs, int outerSize, int[] concatSizes, int innerSize);
        /// <summary>
        /// Copies data asynchronously in the specified stream. Throws on error.
        /// </summary>
        /// <param name="dst">The destination pointer on device or host.</param>
        /// <param name="src">The source pointer on device or host.</param>
        /// <param name="count">The number of bytes to copy.</param>
        /// <param name="kind">The copy direction.</param>
        /// <param name="stream">The CUDA stream execution context.</param>
        /// <exception cref="InvalidOperationException">Thrown if the CUDA runtime returns an error.</exception>

        // PUBLIC WRAPPERS TO FIX "DOES NOT EXIST IN CURRENT CONTEXT"
        public static void CudaMemcpyAsync(IntPtr dst, IntPtr src, ulong count, cudaMemcpyKind kind, IntPtr stream) => Check(cudaMemcpyAsync(dst, src, count, kind, stream), nameof(cudaMemcpyAsync));
        /// <summary>
        /// Creates an asynchronous execution stream. Throws on error.
        /// </summary>
        /// <param name="stream">Outputs the handle to the newly created stream.</param>
        /// <exception cref="InvalidOperationException">Thrown if the CUDA runtime returns an error.</exception>
        public static void CudaStreamCreate(out IntPtr stream) => Check(cudaStreamCreate(out stream), nameof(cudaStreamCreate));
        /// <summary>
        /// Destroys an asynchronous execution stream. Throws on error.
        /// </summary>
        /// <param name="stream">The handle of the stream to destroy.</param>
        /// <exception cref="InvalidOperationException">Thrown if the CUDA runtime returns an error.</exception>
        public static void CudaStreamDestroy(IntPtr stream) => Check(cudaStreamDestroy(stream), nameof(cudaStreamDestroy));
        /// <summary>
        /// Synchronizes the specified CUDA stream. Throws on error.
        /// </summary>
        /// <param name="stream">The handle of the stream to synchronize.</param>
        /// <exception cref="InvalidOperationException">Thrown if the CUDA runtime returns an error.</exception>
        public static void CudaStreamSynchronize(IntPtr stream) => Check(cudaStreamSynchronize(stream), nameof(cudaStreamSynchronize));
        /// <summary>
        /// Allocates a block of device memory. Throws on error.
        /// </summary>
        /// <param name="devicePtr">Outputs the pointer to the allocated device memory.</param>
        /// <param name="byteCount">The size in bytes of the allocation.</param>
        /// <exception cref="InvalidOperationException">Thrown if the CUDA runtime returns an error.</exception>
        public static void CudaMalloc(out IntPtr devicePtr, ulong byteCount) => Check(cudaMalloc(out devicePtr, byteCount), nameof(cudaMalloc));
        /// <summary>
        /// Frees a block of device memory. Safe to call with <see cref="IntPtr.Zero"/>. Throws on error.
        /// </summary>
        /// <param name="devicePtr">The pointer to the device memory block to free.</param>
        /// <exception cref="InvalidOperationException">Thrown if the CUDA runtime returns an error.</exception>
        public static void CudaFree(IntPtr devicePtr) { if (devicePtr != IntPtr.Zero) Check(cudaFree(devicePtr), nameof(cudaFree)); }
        /// <summary>
        /// Sets a block of device memory to a specific byte value. Throws on error.
        /// </summary>
        /// <param name="devicePtr">The pointer to the target device memory block.</param>
        /// <param name="value">The byte value to set.</param>
        /// <param name="count">The size in bytes to set.</param>
        /// <exception cref="InvalidOperationException">Thrown if the CUDA runtime returns an error.</exception>
        public static void CudaMemset(IntPtr devicePtr, int value, ulong count) => Check(cudaMemset(devicePtr, value, count), nameof(cudaMemset));
        /// <summary>
        /// Copies a block of memory between the host and device. Throws on error.
        /// </summary>
        /// <param name="dst">The destination pointer.</param>
        /// <param name="src">The source pointer.</param>
        /// <param name="count">The size in bytes to copy.</param>
        /// <param name="kind">The copy direction.</param>
        /// <exception cref="InvalidOperationException">Thrown if the CUDA runtime returns an error.</exception>
        public static void CudaMemcpy(IntPtr dst, IntPtr src, ulong count, cudaMemcpyKind kind) => Check(cudaMemcpy(dst, src, count, kind), nameof(cudaMemcpy));
        /// <summary>
        /// Blocks the host execution thread until all preceding requested tasks are complete. Throws on error.
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown if the CUDA runtime returns an error.</exception>
        public static void Synchronize() => Check(cudaDeviceSynchronize(), nameof(cudaDeviceSynchronize));
        /// <summary>
        /// Evaluates a CUDA API return code, throwing an exception if the operation failed.
        /// </summary>
        /// <param name="err">The status code returned by the native CUDA API call.</param>
        /// <param name="method">The name of the method that was invoked.</param>
        /// <exception cref="InvalidOperationException">Thrown if <paramref name="err"/> is not <see cref="CudaError.Success"/>.</exception>

        public static void Check(CudaError err, string method)
        {
            if (err != CudaError.Success)
                throw new InvalidOperationException($"CUDA Error in {method}: {err}");
        }
        /// <summary>
        /// Performs element-wise addition on two device vectors: <c>c = a + b</c>.
        /// </summary>
        /// <param name="a">The pointer to the first input device tensor.</param>
        /// <param name="b">The pointer to the second input device tensor.</param>
        /// <param name="c">The pointer to the output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>

        // Keep existing wrappers...
        public static void Add(IntPtr a, IntPtr b, IntPtr c, int n) => NativeAdd(a, b, c, n);
        /// <summary>
        /// Performs element-wise subtraction on two device vectors: <c>c = a - b</c>.
        /// </summary>
        /// <param name="a">The pointer to the minuend device tensor.</param>
        /// <param name="b">The pointer to the subtrahend device tensor.</param>
        /// <param name="c">The pointer to the output difference device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Subtract(IntPtr a, IntPtr b, IntPtr c, int n) => NativeSubtract(a, b, c, n);
        /// <summary>
        /// Performs element-wise multiplication on two device vectors: <c>c = a * b</c>.
        /// </summary>
        /// <param name="a">The pointer to the first input device tensor.</param>
        /// <param name="b">The pointer to the second input device tensor.</param>
        /// <param name="c">The pointer to the output product device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Multiply(IntPtr a, IntPtr b, IntPtr c, int n) => NativeMultiply(a, b, c, n);
        /// <summary>
        /// Performs element-wise division on two device vectors: <c>c = a / b</c>.
        /// </summary>
        /// <param name="a">The pointer to the dividend device tensor.</param>
        /// <param name="b">The pointer to the divisor device tensor.</param>
        /// <param name="c">The pointer to the output quotient device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Divide(IntPtr a, IntPtr b, IntPtr c, int n) => NativeDivide(a, b, c, n);
        /// <summary>
        /// Raises elements of base device vector <paramref name="a"/> to the power of exponent device vector <paramref name="b"/>: <c>c = a^b</c>.
        /// </summary>
        /// <param name="a">The pointer to the base device tensor.</param>
        /// <param name="b">The pointer to the exponent device tensor.</param>
        /// <param name="c">The pointer to the output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Pow(IntPtr a, IntPtr b, IntPtr c, int n) => NativePow(a, b, c, n);
        /// <summary>
        /// Compares element-wise equality between two device vectors: <c>c = (a == b)</c>.
        /// </summary>
        /// <param name="a">The pointer to the first input device tensor.</param>
        /// <param name="b">The pointer to the second input device tensor.</param>
        /// <param name="c">The pointer to the boolean/numeric result mask device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Equal(IntPtr a, IntPtr b, IntPtr c, int n) => NativeEqual(a, b, c, n);
        /// <summary>
        /// Compares whether elements of vector <paramref name="a"/> are greater than vector <paramref name="b"/>: <c>c = (a > b)</c>.
        /// </summary>
        /// <param name="a">The pointer to the left-hand input device tensor.</param>
        /// <param name="b">The pointer to the right-hand input device tensor.</param>
        /// <param name="c">The pointer to the output mask device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void GreaterThan(IntPtr a, IntPtr b, IntPtr c, int n) => NativeGreaterThan(a, b, c, n);
        /// <summary>
        /// Compares whether elements of vector <paramref name="a"/> are less than vector <paramref name="b"/>: <c>c = (a &lt; b)</c>.
        /// </summary>
        /// <param name="a">The pointer to the left-hand input device tensor.</param>
        /// <param name="b">The pointer to the right-hand input device tensor.</param>
        /// <param name="c">The pointer to the output mask device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void LessThan(IntPtr a, IntPtr b, IntPtr c, int n) => NativeLessThan(a, b, c, n);
        /// <summary>
        /// Selects elements from either <paramref name="a"/> or <paramref name="b"/> based on the truth value of <paramref name="cond"/>.
        /// </summary>
        /// <param name="cond">The pointer to the condition mask device tensor.</param>
        /// <param name="a">The pointer to the source device tensor chosen where condition is true.</param>
        /// <param name="b">The pointer to the source device tensor chosen where condition is false.</param>
        /// <param name="c">The pointer to the resulting output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Where(IntPtr cond, IntPtr a, IntPtr b, IntPtr c, int n) => NativeWhere(cond, a, b, c, n);
        /// <summary>
        /// Applies the Rectified Linear Unit activation function element-wise on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the computed activation output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void ReLU(IntPtr input, IntPtr output, int n) => NativeReLU(input, output, n);
        /// <summary>
        /// Computes the gradient of the Rectified Linear Unit function during backpropagation.
        /// </summary>
        /// <param name="gradOut">The pointer to the outgoing gradient device tensor.</param>
        /// <param name="originIn">The pointer to the original forward pass input device tensor.</param>
        /// <param name="gradIn">The pointer to the incoming computed gradient device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void ReLUGrad(IntPtr gradOut, IntPtr originIn, IntPtr gradIn, int n) => NativeReLUGrad(gradOut, originIn, gradIn, n);
        /// <summary>
        /// Applies the Sigmoid activation function element-wise on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the computed activation output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Sigmoid(IntPtr input, IntPtr output, int n) => NativeSigmoid(input, output, n);
        /// <summary>
        /// Computes the gradient of the Sigmoid function during backpropagation.
        /// </summary>
        /// <param name="gradOut">The pointer to the outgoing gradient device tensor.</param>
        /// <param name="originOut">The pointer to the original forward pass computed output device tensor.</param>
        /// <param name="gradIn">The pointer to the incoming computed gradient device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void SigmoidGrad(IntPtr gradOut, IntPtr originOut, IntPtr gradIn, int n) => NativeSigmoidGrad(gradOut, originOut, gradIn, n);
        /// <summary>
        /// Applies the Hyperbolic Tangent (tanh) activation function element-wise on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the computed activation output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Tanh(IntPtr input, IntPtr output, int n) => NativeTanh(input, output, n);
        /// <summary>
        /// Performs element-wise arithmetic negation on device memory: <c>output = -input</c>.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the negated output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Negate(IntPtr input, IntPtr output, int n) => NativeNegate(input, output, n);
        /// <summary>
        /// Computes the natural exponential function element-wise on device memory: <c>output = e^input</c>.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the exponentiated output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Exp(IntPtr input, IntPtr output, int n) => NativeExp(input, output, n);
        /// <summary>
        /// Computes the natural logarithm element-wise on device memory: <c>output = ln(input)</c>.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the logarithm output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Log(IntPtr input, IntPtr output, int n) => NativeLog(input, output, n);
        /// <summary>
        /// Computes the square root element-wise on device memory: <c>output = sqrt(input)</c>.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the square root output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Sqrt(IntPtr input, IntPtr output, int n) => NativeSqrt(input, output, n);
        /// <summary>
        /// Computes the absolute value element-wise on device memory: <c>output = |input|</c>.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the absolute value output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Abs(IntPtr input, IntPtr output, int n) => NativeAbs(input, output, n);
        /// <summary>
        /// Computes the sine function element-wise on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor in radians.</param>
        /// <param name="output">The pointer to the sine output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Sin(IntPtr input, IntPtr output, int n) => NativeSin(input, output, n);
        /// <summary>
        /// Computes the cosine function element-wise on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor in radians.</param>
        /// <param name="output">The pointer to the cosine output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Cos(IntPtr input, IntPtr output, int n) => NativeCos(input, output, n);
        /// <summary>
        /// Computes the signum function element-wise on device memory (-1 for negative, 0 for zero, 1 for positive).
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the sign output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void Sign(IntPtr input, IntPtr output, int n) => NativeSign(input, output, n);
        /// <summary>
        /// Fills a device memory buffer with the scalar value 1.0f.
        /// </summary>
        /// <param name="output">The pointer to the output device tensor.</param>
        /// <param name="n">The total number of elements to initialize.</param>
        public static void Ones(IntPtr output, int n) => NativeOnes(output, n);
        /// <summary>
        /// Fills a device memory buffer with a specified float scalar value.
        /// </summary>
        /// <param name="output">The pointer to the output device tensor.</param>
        /// <param name="value">The float value to populate across the memory block.</param>
        /// <param name="n">The total number of elements to initialize.</param>
        public static void SetScalar(IntPtr output, float value, int n) => NativeSetScalar(output, value, n);
        /// <summary>
        /// Raises elements of a base device vector to a constant scalar power.
        /// </summary>
        /// <param name="input">The pointer to the input base device tensor.</param>
        /// <param name="output">The pointer to the calculated output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        /// <param name="exponent">The float power exponent value.</param>
        public static void PowScalar(IntPtr input, IntPtr output, int n, float exponent) => NativePowScalar(input, output, n, exponent);
        /// <summary>
        /// Computes the sum of all elements within a device tensor and stores the result in a 1-element tensor.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the single-value output sum device tensor.</param>
        /// <param name="n">The total number of elements to sum.</param>
        public static void SumAll(IntPtr input, IntPtr output, int n) => NativeSumAll(input, output, n);
        /// <summary>
        /// Computes the arithmetic mean of all elements in a device tensor and stores the result in a 1-element tensor.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the single-value output mean device tensor.</param>
        /// <param name="n">The total number of elements to compute.</param>
        public static void MeanAll(IntPtr input, IntPtr output, int n) => NativeMeanAll(input, output, n);
        /// <summary>
        /// Computes the mean along a specified reduction axis.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the calculated mean output device tensor.</param>
        /// <param name="outer">The size of dimensions outer to the reduction dimension.</param>
        /// <param name="dim">The size of the reduction dimension.</param>
        /// <param name="inner">The size of dimensions inner to the reduction dimension.</param>
        public static void MeanAxis(IntPtr input, IntPtr output, int outer, int dim, int inner) => NativeMeanAxis(input, output, outer, dim, inner);
        /// <summary>
        /// Finds the indices of maximum values along a given reduction dimension.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the calculated index output device tensor.</param>
        /// <param name="outer">The size of dimensions outer to the reduction dimension.</param>
        /// <param name="dim">The size of the reduction dimension.</param>
        /// <param name="inner">The size of dimensions inner to the reduction dimension.</param>
        public static void ArgMax(IntPtr input, IntPtr output, int outer, int dim, int inner) => NativeArgMax(input, output, outer, dim, inner);
        /// <summary>
        /// Finds the indices of minimum values along a given reduction dimension.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the calculated index output device tensor.</param>
        /// <param name="outer">The size of dimensions outer to the reduction dimension.</param>
        /// <param name="dim">The size of the reduction dimension.</param>
        /// <param name="inner">The size of dimensions inner to the reduction dimension.</param>
        public static void ArgMin(IntPtr input, IntPtr output, int outer, int dim, int inner) => NativeArgMin(input, output, outer, dim, inner);
        /// <summary>
        /// Computes the cumulative sum of elements along a specified dimension.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the cumulative sum output device tensor.</param>
        /// <param name="outer">The size of dimensions outer to the accumulation dimension.</param>
        /// <param name="dim">The size of the accumulation dimension.</param>
        /// <param name="inner">The size of dimensions inner to the accumulation dimension.</param>
        public static void CumSum(IntPtr input, IntPtr output, int outer, int dim, int inner) => NativeCumSum(input, output, outer, dim, inner);
        /// <summary>
        /// Transposes a 2D matrix on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input matrix in device memory.</param>
        /// <param name="output">The pointer to the transposed output matrix in device memory.</param>
        /// <param name="rows">The number of rows in the input matrix.</param>
        /// <param name="cols">The number of columns in the input matrix.</param>
        public static void Transpose(IntPtr input, IntPtr output, int rows, int cols) => NativeTranspose(input, output, rows, cols);
        /// <summary>
        /// Performs a multi-dimensional permutation transpose of a tensor on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input multi-dimensional device tensor.</param>
        /// <param name="output">The pointer to the transposed output device tensor.</param>
        /// <param name="shape">The shape array of the input tensor.</param>
        /// <param name="perm">The target dimension permutation index order mapping array.</param>
        public static void GeneralTranspose(IntPtr input, IntPtr output, int[] shape, int[] perm) { NativeGeneralTranspose(input, output, shape, perm, shape.Length); }
        /// <summary>
        /// Broadcasts elements from a lower dimensional source tensor to match a larger target shape.
        /// </summary>
        /// <param name="input">The pointer to the source input device tensor.</param>
        /// <param name="output">The pointer to the broadcasted output device tensor.</param>
        /// <param name="inputShape">The shape array of the source input tensor.</param>
        /// <param name="outputShape">The target destination shape array.</param>
        public static void Broadcast(IntPtr input, IntPtr output, int[] inputShape, int[] outputShape) { NativeBroadcast(input, output, inputShape, outputShape, inputShape.Length); }
        /// <summary>
        /// Computes standard matrix multiplication of general matrices: <c>c = a * b</c>.
        /// </summary>
        /// <param name="a">The pointer to left-hand side matrix A on device memory.</param>
        /// <param name="b">The pointer to right-hand side matrix B on device memory.</param>
        /// <param name="c">The pointer to resulting matrix C on device memory.</param>
        /// <param name="m">The number of rows of matrix A and C.</param>
        /// <param name="n">The number of columns of matrix B and C.</param>
        /// <param name="k">The number of columns of matrix A and rows of matrix B.</param>
        public static void MatMul(IntPtr a, IntPtr b, IntPtr c, int m, int n, int k) => NativeMatMul(a, b, c, m, n, k);
        /// <summary>
        /// Computes Softmax normalized probabilities across rows of a 2D matrix.
        /// </summary>
        /// <param name="input">The pointer to the unnormalized logits device tensor.</param>
        /// <param name="output">The pointer to the calculated probability distribution output device tensor.</param>
        /// <param name="rows">The total number of rows in the matrix.</param>
        /// <param name="cols">The total number of columns in the matrix.</param>
        public static void Softmax(IntPtr input, IntPtr output, int rows, int cols) => NativeSoftmax(input, output, rows, cols);
        /// <summary>
        /// Performs element-wise logical negation on a boolean/numeric mask on device memory.
        /// </summary>
        /// <param name="input">The pointer to the input condition device tensor.</param>
        /// <param name="output">The pointer to the logically inverted output device tensor.</param>
        /// <param name="count">The total number of elements to process.</param>
        public static void LogicalNot(IntPtr input, IntPtr output, int count) => NativeLogicalNot(input, output, count);
        /// <summary>
        /// Clamps all values in a device tensor to stay within the boundaries of a closed interval.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the clamped output device tensor.</param>
        /// <param name="count">The total number of elements to process.</param>
        /// <param name="minVal">The lower bound scalar clamp threshold.</param>
        /// <param name="maxVal">The upper bound scalar clamp threshold.</param>
        public static void Clip(IntPtr input, IntPtr output, int count, float minVal, float maxVal) => NativeClip(input, output, count, minVal, maxVal);
        /// <summary>
        /// Generates a binary mask indicating which elements were clamped during a clip operation.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the output binary mask device tensor.</param>
        /// <param name="count">The total number of elements to process.</param>
        /// <param name="minVal">The lower bound clamp threshold.</param>
        /// <param name="maxVal">The upper bound clamp threshold.</param>
        public static void ClipMask(IntPtr input, IntPtr output, int count, float minVal, float maxVal) => NativeClipMask(input, output, count, minVal, maxVal);
        /// <summary>
        /// Finds the maximum values along a specified reduction axis.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the computed maximum output device tensor.</param>
        /// <param name="outer">The size of dimensions outer to the reduction dimension.</param>
        /// <param name="dim">The size of the reduction dimension.</param>
        /// <param name="inner">The size of dimensions inner to the reduction dimension.</param>
        public static void MaxAxis(IntPtr input, IntPtr output, int outer, int dim, int inner) => NativeMaxAxis(input, output, outer, dim, inner);
        /// <summary>
        /// Finds the minimum values along a specified reduction axis.
        /// </summary>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the computed minimum output device tensor.</param>
        /// <param name="outer">The size of dimensions outer to the reduction dimension.</param>
        /// <param name="dim">The size of the reduction dimension.</param>
        /// <param name="inner">The size of dimensions inner to the reduction dimension.</param>
        public static void MinAxis(IntPtr input, IntPtr output, int outer, int dim, int inner) => NativeMinAxis(input, output, outer, dim, inner);
        /// <summary>
        /// Performs element-wise power computation with tensor-valued base and exponent: <c>c = a^b</c>.
        /// </summary>
        /// <param name="a">The pointer to the base device tensor.</param>
        /// <param name="b">The pointer to the exponent device tensor.</param>
        /// <param name="c">The pointer to the calculated output device tensor.</param>
        /// <param name="n">The total number of elements to process.</param>
        public static void PowTensor(IntPtr a, IntPtr b, IntPtr c, int n) => NativePowTensor(a, b, c, n);
        /// <summary>
        /// Extracts a sub-tensor slice from an input multi-dimensional tensor.
        /// </summary>
        /// <param name="input">The pointer to the input source device tensor.</param>
        /// <param name="output">The pointer to the extracted output slice device tensor.</param>
        /// <param name="inShape">The shape array of the input tensor.</param>
        /// <param name="outShape">The shape array of the output sliced tensor.</param>
        /// <param name="starts">The start indices of the slice along each dimension.</param>
        /// <param name="steps">The step stride intervals along each dimension.</param>
        /// <param name="rank">The total dimensional rank of the tensor.</param>
        public static void Slice(IntPtr input, IntPtr output, int[] inShape, int[] outShape, int[] starts, int[] steps, int rank) => NativeSlice(input, output, inShape, outShape, starts, steps, rank);
        /// <summary>
        /// Computes the backpropagation gradient for a slice extraction operation.
        /// </summary>
        /// <param name="gradOut">The pointer to the outgoing gradient device tensor.</param>
        /// <param name="gradIn">The pointer to the incoming computed gradient device tensor.</param>
        /// <param name="originalShape">The shape array of the original tensor before slicing.</param>
        /// <param name="newShape">The shape array of the sliced sub-tensor.</param>
        /// <param name="starts">The start indices of the slice along each dimension.</param>
        /// <param name="steps">The step stride intervals along each dimension.</param>
        /// <param name="rank">The total dimensional rank of the tensor.</param>
        public static void SliceGrad(IntPtr gradOut, IntPtr gradIn, int[] originalShape, int[] newShape, int[] starts, int[] steps, int rank) => NativeSliceGrad(gradOut, gradIn, originalShape, newShape, starts, steps, rank);
        /// <summary>
        /// Concatenates multiple tensors along a specified target dimension on device memory.
        /// </summary>
        /// <param name="inputs">The array of pointers to input device tensors.</param>
        /// <param name="output">The pointer to the concatenated destination output device tensor.</param>
        /// <param name="numInputs">The total number of input tensors to concatenate.</param>
        /// <param name="outerSize">The size of dimensions outer to the concatenation dimension.</param>
        /// <param name="concatSizes">The array containing the sizes of the concatenation dimension for each input tensor.</param>
        /// <param name="innerSize">The size of dimensions inner to the concatenation dimension.</param>
        public static void Concat(IntPtr[] inputs, IntPtr output, int numInputs, int outerSize, int[] concatSizes, int innerSize) => NativeConcat(inputs, output, numInputs, outerSize, concatSizes, innerSize);
        /// <summary>
        /// Computes the gradient for the gather operation, mapping back gradients to their source locations.
        /// </summary>
        /// <param name="gradOut">The pointer to the outgoing gradient tensor in device memory.</param>
        /// <param name="indices">The pointer to the indices tensor used in the forward gather pass.</param>
        /// <param name="gradIn">The pointer to the incoming gradient tensor to be accumulated on device memory.</param>
        /// <param name="batch">The batch dimension size.</param>
        /// <param name="classes">The class dimension size.</param>
        public static void GatherGrad(IntPtr gradOut, IntPtr indices, IntPtr gradIn, int batch, int classes) => NativeGatherGrad(gradOut, indices, gradIn, batch, classes);
        /// <summary>
        /// Initializes the CUDA context. This is a stub placeholder for compatibility.
        /// </summary>
        /// <param name="context">Outputs the initialized context handle (currently returns <see cref="IntPtr.Zero"/>).</param>
        public static void Initialize(out IntPtr context) { context = IntPtr.Zero; }
        /// <summary>
        /// Cleans up and releases the specified CUDA context. This is a stub placeholder for compatibility.
        /// </summary>
        /// <param name="context">The context handle to release.</param>
        public static void Cleanup(IntPtr context) { }
        /// <summary>
        /// Registers page-locked host memory. This is a stub placeholder for compatibility.
        /// </summary>
        /// <param name="ptr">The pointer to the host memory.</param>
        /// <param name="size">The size in bytes.</param>
        /// <param name="flags">Registration flags.</param>
        public static void CudaHostRegister(IntPtr ptr, ulong size, uint flags = 0) { }
        /// <summary>
        /// Unregisters page-locked host memory. This is a stub placeholder for compatibility.
        /// </summary>
        /// <param name="ptr">The pointer to the host memory.</param>
        public static void CudaHostUnregister(IntPtr ptr) { }
        /// <summary>
        /// Reshapes a tensor layout on device memory. This is a stub placeholder for compatibility.
        /// </summary>
        /// <param name="context">The execution context pointer.</param>
        /// <param name="input">The pointer to the input device tensor.</param>
        /// <param name="output">The pointer to the reshaped output device tensor.</param>
        /// <param name="inputShape">The shape array of the input tensor.</param>
        /// <param name="outputShape">The target destination shape array.</param>
        public static void Reshape(IntPtr context, IntPtr input, IntPtr output, int[] inputShape, int[] outputShape) { }
    }
}