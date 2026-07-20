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
    /// P/Invoke wrapper for AMD ROCm (HIP) and hipBLAS API functions.
    /// Uses an encapsulation pattern: private extern methods for raw native calls 
    /// and public accessor methods for managed application logic and resource management.
    /// </summary>
    /// <remarks>
    /// This class acts as a bridge between managed .NET applications and the AMD ROCm platform.
    /// It exposes low-level memory operations, execution control, and high-level tensor operations
    /// by wrapping native driver calls from 'amdhip64.dll' and 'hipblas.dll'.
    /// </remarks>

    #endregion

    public static class ROCm
    {
        /// <summary>
        /// The native DLL name for the AMD HIP runtime.
        /// </summary>
        private const string HipRuntimeDll = "amdhip64.dll";

        /// <summary>
        /// The native DLL name for the hipBLAS library.
        /// </summary>
        private const string HipBlasDll = "hipblas.dll";

        #region Enums and Constants

        /// <summary>
        /// Error codes returned by HIP runtime functions.
        /// </summary>
        /// <remarks>
        /// These values correspond directly to the native <c>hipError_t</c> enumeration in the HIP runtime API.
        /// </remarks>
        public enum hipError_t
        {
            /// <summary>
            /// The API call returned with no errors.
            /// </summary>
            hipSuccess = 0,

            /// <summary>
            /// This indicates that the HIP context passed to the API call was invalid.
            /// </summary>
            hipErrorInvalidContext = 1,

            /// <summary>
            /// This indicates that one or more of the parameters passed to the API call is invalid.
            /// </summary>
            hipErrorInvalidValue = 2,
            // Additional codes can be added as needed
        }

        /// <summary>
        /// Status codes returned by hipBLAS library functions.
        /// </summary>
        /// <remarks>
        /// These values correspond directly to the native <c>hipblasStatus_t</c> enumeration in the hipBLAS API.
        /// </remarks>
        public enum hipblasStatus_t
        {
            /// <summary>
            /// The hipBLAS operation completed successfully.
            /// </summary>
            HIPBLAS_STATUS_SUCCESS = 0,

            /// <summary>
            /// The hipBLAS library was not initialized.
            /// </summary>
            HIPBLAS_STATUS_NOT_INITIALIZED = 1,

            /// <summary>
            /// Resource allocation failed inside the hipBLAS library.
            /// </summary>
            HIPBLAS_STATUS_ALLOC_FAILED = 2,

            /// <summary>
            /// An invalid value or parameter was passed to the hipBLAS function.
            /// </summary>
            HIPBLAS_STATUS_INVALID_VALUE = 3
        }

        /// <summary>
        /// Specifies the direction of a HIP memory copy operation.
        /// </summary>
        /// <remarks>
        /// Used by the HIP memory copy functions to determine data transfer topology.
        /// </remarks>
        public enum hipMemcpyKind
        {
            /// <summary>
            /// Memory copy from host (CPU) to host (CPU).
            /// </summary>
            hipMemcpyHostToHost = 0,

            /// <summary>
            /// Memory copy from host (CPU) to device (GPU).
            /// </summary>
            hipMemcpyHostToDevice = 1,

            /// <summary>
            /// Memory copy from device (GPU) to host (CPU).
            /// </summary>
            hipMemcpyDeviceToHost = 2,

            /// <summary>
            /// Memory copy from device (GPU) to device (GPU).
            /// </summary>
            hipMemcpyDeviceToDevice = 3,

            /// <summary>
            /// Direction is automatically inferred based on the pointers (requires unified memory).
            /// </summary>
            hipMemcpyDefault = 4
        }
        /// <summary>
        /// Native P/Invoke signature for the HIP runtime initialization function.
        /// </summary>
        /// <param name="flags">Initialization flags (must be 0).</param>
        /// <returns>The <see cref="hipError_t"/> result of the operation.</returns>

        #endregion

        #region Private Native Signatures (P/Invoke)

        [DllImport(HipRuntimeDll, EntryPoint = "hipInit", CallingConvention = CallingConvention.Cdecl)]
        private static extern hipError_t hip_init(uint flags);
        /// <summary>
        /// Native P/Invoke signature for retrieving the number of HIP devices.
        /// </summary>
        /// <param name="count">Receives the number of available devices.</param>
        /// <returns>The <see cref="hipError_t"/> result of the operation.</returns>

        [DllImport(HipRuntimeDll, EntryPoint = "hipGetDeviceCount", CallingConvention = CallingConvention.Cdecl)]
        private static extern hipError_t hip_get_device_count(ref int count);
        /// <summary>
        /// Native P/Invoke signature for HIP memory copy operations.
        /// </summary>
        /// <param name="dst">Destination memory pointer.</param>
        /// <param name="src">Source memory pointer.</param>
        /// <param name="size">Number of bytes to copy.</param>
        /// <param name="kind">The kind of memory transfer.</param>
        /// <returns>The <see cref="hipError_t"/> result of the operation.</returns>

        [DllImport(HipRuntimeDll, EntryPoint = "hipMemcpy", CallingConvention = CallingConvention.Cdecl)]
        private static extern hipError_t hip_memcpy(IntPtr dst, IntPtr src, IntPtr size, hipMemcpyKind kind);
        /// <summary>
        /// Native P/Invoke signature for creating a hipBLAS handle.
        /// </summary>
        /// <param name="handle">Receives the newly created hipBLAS handle.</param>
        /// <returns>The <see cref="hipblasStatus_t"/> result of the operation.</returns>

        [DllImport(HipBlasDll, EntryPoint = "hipblasCreate", CallingConvention = CallingConvention.Cdecl)]
        private static extern hipblasStatus_t hip_blas_create(ref IntPtr handle);
        /// <summary>
        /// Native P/Invoke signature for destroying a hipBLAS handle.
        /// </summary>
        /// <param name="handle">The hipBLAS handle to destroy.</param>
        /// <returns>The <see cref="hipblasStatus_t"/> result of the operation.</returns>

        [DllImport(HipBlasDll, EntryPoint = "hipblasDestroy", CallingConvention = CallingConvention.Cdecl)]
        private static extern hipblasStatus_t hip_blas_destroy(IntPtr handle);
        /// <summary>
        /// Native P/Invoke signature for single-precision general matrix multiplication (SGEMM).
        /// </summary>
        /// <param name="handle">The hipBLAS handle.</param>
        /// <param name="transa">Transpose operation for matrix A.</param>
        /// <param name="transb">Transpose operation for matrix B.</param>
        /// <param name="m">Number of rows in the output matrix.</param>
        /// <param name="n">Number of columns in the output matrix.</param>
        /// <param name="k">Inner dimension shared by the input matrices.</param>
        /// <param name="alpha">Scalar multiplier for the product of A and B.</param>
        /// <param name="A">Pointer to matrix A in device memory.</param>
        /// <param name="lda">Leading dimension of matrix A.</param>
        /// <param name="B">Pointer to matrix B in device memory.</param>
        /// <param name="ldb">Leading dimension of matrix B.</param>
        /// <param name="beta">Scalar multiplier for matrix C.</param>
        /// <param name="C">Pointer to matrix C (output) in device memory.</param>
        /// <param name="ldc">Leading dimension of matrix C.</param>
        /// <returns>The <see cref="hipblasStatus_t"/> result of the operation.</returns>

        [DllImport(HipBlasDll, EntryPoint = "hipblasSgemm", CallingConvention = CallingConvention.Cdecl)]
        private static extern hipblasStatus_t hip_blas_sgemm(IntPtr handle, int transa, int transb, int m, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc);
        /// <summary>
        /// Native P/Invoke signature for single-precision AXPY operation (y = alpha * x + y).
        /// </summary>
        /// <param name="handle">The hipBLAS handle.</param>
        /// <param name="n">Number of elements in the vectors.</param>
        /// <param name="alpha">Scalar multiplier applied to vector x.</param>
        /// <param name="x">Pointer to vector x in device memory.</param>
        /// <param name="incx">Stride between consecutive elements of x.</param>
        /// <param name="y">Pointer to vector y in device memory (input and output).</param>
        /// <param name="incy">Stride between consecutive elements of y.</param>
        /// <returns>The <see cref="hipblasStatus_t"/> result of the operation.</returns>

        [DllImport(HipBlasDll, EntryPoint = "hipblasSaxpy", CallingConvention = CallingConvention.Cdecl)]
        private static extern hipblasStatus_t hip_blas_saxpy(IntPtr handle, int n, ref float alpha, IntPtr x, int incx, IntPtr y, int incy);
        /// <summary>
        /// Initializes the HIP primary context.
        /// </summary>
        /// <param name="flags">Initialization flags (typically 0).</param>
        /// <returns>The <see cref="hipError_t"/> result of the initialization.</returns>

        #endregion

        #region Public Accessor Methods

        public static hipError_t HipInit(uint flags)
        {
            return hip_init(flags);
        }
        /// <summary>
        /// Returns the number of compute-capable devices.
        /// </summary>
        /// <param name="count">Receives the number of compute-capable devices.</param>
        /// <returns>The <see cref="hipError_t"/> result of the query.</returns>

        public static hipError_t HipGetDeviceCount(ref int count)
        {
            return hip_get_device_count(ref count);
        }
        /// <summary>
        /// Copies data between host and device memory ranges using HIP.
        /// </summary>
        /// <param name="dst">Destination memory pointer.</param>
        /// <param name="src">Source memory pointer.</param>
        /// <param name="size">Number of bytes to copy.</param>
        /// <param name="kind">Direction of the memory copy.</param>
        /// <returns>The <see cref="hipError_t"/> result of the copy operation.</returns>

        public static hipError_t HipMemcpy(IntPtr dst, IntPtr src, IntPtr size, hipMemcpyKind kind)
        {
            return hip_memcpy(dst, src, size, kind);
        }
        /// <summary>
        /// Initializes the ROCm environment and creates a hipBLAS context handle.
        /// </summary>
        /// <param name="context">Outputs the generated hipBLAS handle.</param>
        /// <exception cref="InvalidOperationException">
        /// Thrown when HIP fails to initialize or if the hipBLAS handle cannot be created.
        /// </exception>
        /// <remarks>
        /// This method represents a simplified setup procedure and will execute internal HIP calls.
        /// </remarks>

        #endregion

        #region High-Level Dispatcher Logic

        public static void Initialize(out IntPtr context)
        {
            if (hip_init(0) != hipError_t.hipSuccess)
                throw new InvalidOperationException("ROCm HIP initialization failed.");

            IntPtr handle = IntPtr.Zero;
            if (hip_blas_create(ref handle) != hipblasStatus_t.HIPBLAS_STATUS_SUCCESS)
                throw new InvalidOperationException("Failed to create hipBLAS context.");

            context = handle;
        }
        /// <summary>
        /// Checks if a compatible AMD ROCm device is present on the system.
        /// </summary>
        /// <returns><c>true</c> if a compatible ROCm device is available and the query succeeded; otherwise, <c>false</c>.</returns>

        public static bool IsAvailable()
        {
            try
            {
                int count = 0;
                return hip_get_device_count(ref count) == hipError_t.hipSuccess && count > 0;
            }
            catch { return false; }
        }
        /// <summary>
        /// Dispatches a matrix multiplication operation (C = A * B) to hipBLAS.
        /// </summary>
        /// <param name="context">The hipBLAS context handle.</param>
        /// <param name="a">Pointer to the first input matrix in device memory.</param>
        /// <param name="b">Pointer to the second input matrix in device memory.</param>
        /// <param name="c">Pointer to the output matrix in device memory.</param>
        /// <param name="m">Number of rows in matrix A.</param>
        /// <param name="n">Number of columns in matrix B.</param>
        /// <param name="k">Shared inner dimension of the matrices.</param>
        /// <remarks>
        /// This method performs an internal row-major to column-major adjustment to accommodate the
        /// column-major requirement of the underlying hipBLAS SGEMM implementation.
        /// </remarks>

        public static void MatMul(IntPtr context, IntPtr a, IntPtr b, IntPtr c, int m, int n, int k)
        {
            float alpha = 1.0f, beta = 0.0f;
            // standard row-major to column-major adjustment for BLAS
            hip_blas_sgemm(context, 0, 0, n, m, k, ref alpha, b, n, a, k, ref beta, c, n);
        }
        /// <summary>
        /// Dispatches an element-wise addition (C = A + B) using hipMemcpy and hipBLAS SAXPY.
        /// </summary>
        /// <param name="context">The hipBLAS context handle.</param>
        /// <param name="a">Pointer to the first input tensor in device memory.</param>
        /// <param name="b">Pointer to the second input tensor in device memory.</param>
        /// <param name="c">Pointer to the output tensor in device memory.</param>
        /// <param name="size">Number of elements in each tensor.</param>

        public static void Add(IntPtr context, IntPtr a, IntPtr b, IntPtr c, int size)
        {
            IntPtr byteSize = new IntPtr(size * sizeof(float));
            hip_memcpy(c, a, byteSize, hipMemcpyKind.hipMemcpyDeviceToDevice);

            float alpha = 1.0f;
            hip_blas_saxpy(context, size, ref alpha, b, 1, c, 1);
        }
        /// <summary>
        /// Performs a tensor reshape by copying data to the output pointer.
        /// </summary>
        /// <param name="context">The hipBLAS context handle (currently unused).</param>
        /// <param name="input">Pointer to the input tensor in device memory.</param>
        /// <param name="output">Pointer to the output tensor in device memory.</param>
        /// <param name="inputShape">Dimensions of the input tensor.</param>
        /// <param name="outputShape">Dimensions of the output tensor.</param>
        /// <exception cref="NullReferenceException">Thrown if <paramref name="inputShape"/> is null.</exception>

        public static void Reshape(IntPtr context, IntPtr input, IntPtr output, int[] inputShape, int[] outputShape)
        {
            long size = 1;
            foreach (int dim in inputShape) size *= dim;

            hip_memcpy(output, input, (IntPtr)(size * sizeof(float)), hipMemcpyKind.hipMemcpyDeviceToDevice);
        }
        /// <summary>
        /// Releases the hipBLAS context handle.
        /// </summary>
        /// <param name="context">The hipBLAS handle to destroy.</param>
        /// <remarks>
        /// Safe to call with <see cref="IntPtr.Zero"/>. Only the native hipblasDestroy call is performed.
        /// </remarks>

        public static void Cleanup(IntPtr context)
        {
            if (context != IntPtr.Zero)
            {
                hip_blas_destroy(context);
            }
        }

        #endregion
    }
}