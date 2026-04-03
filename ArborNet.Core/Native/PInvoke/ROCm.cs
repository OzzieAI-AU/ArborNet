using System;
using System.Runtime.InteropServices;

namespace ArborNet.Core.Native.PInvoke
{
    /// <summary>
    /// P/Invoke wrapper for AMD ROCm (HIP) and hipBLAS API functions.
    /// Uses an encapsulation pattern: private extern methods for raw native calls 
    /// and public accessor methods for managed application logic and resource management.
    /// </summary>
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
        public enum hipError_t
        {
            hipSuccess = 0,
            hipErrorInvalidContext = 1,
            hipErrorInvalidValue = 2,
            // Additional codes can be added as needed
        }

        /// <summary>
        /// Status codes returned by hipBLAS library functions.
        /// </summary>
        public enum hipblasStatus_t
        {
            HIPBLAS_STATUS_SUCCESS = 0,
            HIPBLAS_STATUS_NOT_INITIALIZED = 1,
            HIPBLAS_STATUS_ALLOC_FAILED = 2,
            HIPBLAS_STATUS_INVALID_VALUE = 3
        }

        /// <summary>
        /// Specifies the direction of a HIP memory copy operation.
        /// </summary>
        public enum hipMemcpyKind
        {
            hipMemcpyHostToHost = 0,
            hipMemcpyHostToDevice = 1,
            hipMemcpyDeviceToHost = 2,
            hipMemcpyDeviceToDevice = 3,
            hipMemcpyDefault = 4
        }

        #endregion

        #region Private Native Signatures (P/Invoke)

        /// <summary>
        /// Native P/Invoke signature for the HIP runtime initialization function.
        /// </summary>
        /// <param name="flags">Initialization flags (must be 0).</param>
        /// <returns>The <see cref="hipError_t"/> result of the operation.</returns>
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

        #endregion

        #region Public Accessor Methods

        /// <summary>
        /// Initializes the HIP primary context.
        /// </summary>
        /// <param name="flags">Initialization flags (typically 0).</param>
        /// <returns>The <see cref="hipError_t"/> result of the initialization.</returns>
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

        #endregion

        #region High-Level Dispatcher Logic

        /// <summary>
        /// Initializes the ROCm environment and creates a hipBLAS context handle.
        /// </summary>
        /// <param name="context">Outputs the generated hipBLAS handle.</param>
        /// <remarks>
        /// Throws <see cref="InvalidOperationException"/> if HIP initialization or hipBLAS handle creation fails.
        /// </remarks>
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