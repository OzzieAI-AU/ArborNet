// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Native
{

    #region Using Statements:

    using System;
    using System.Reflection;
    using System.Runtime.InteropServices;
    using ArborNet.Core.Native;
    using ArborNet.Core.Native.PInvoke;
    /// <summary>
    /// Provides a unified dispatcher for executing high-performance native GPU operations across supported hardware architectures,
    /// specifically NVIDIA CUDA and AMD ROCm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This class dynamically detects the underlying GPU hardware capability at runtime during instantiation. 
    /// It abstractly wraps the platform-specific native API calls, allowing the consuming application to remain agnostic 
    /// of the underlying GPU vendor (NVIDIA vs. AMD).
    /// </para>
    /// <para>
    /// <b>Resource Management:</b> This class implements the <see cref="IDisposable"/> interface. It allocates and manages 
    /// unmanaged native contexts (<c>hipCtx_t</c> or <c>CUcontext</c> equivalents represented via <see cref="IntPtr"/>). 
    /// To prevent resource and memory leaks in the native driver layer, callers must explicitly call <see cref="Dispose()"/> 
    /// or utilize a <c>using</c> block when the dispatcher instance is no longer required. A finalizer is provided as a fallback safety net.
    /// </para>
    /// <para>
    /// <b>Thread Safety:</b> Instance members of this class are not guaranteed to be safe for concurrent thread access. Concurrent dispatch 
    /// of operations on the same dispatcher instance may lead to race conditions or undefined behavior within the native contexts.
    /// </para>
    /// </remarks>

    #endregion

    public class NativeDispatcher : IDisposable
    {
        /// <summary>
        /// The native CUDA context handle allocated during initialization if the CUDA backend is selected.
        /// </summary>
        /// <value>
        /// An <see cref="IntPtr"/> pointing to the native CUDA context, or <see cref="IntPtr.Zero"/> if not initialized or using a different architecture.
        /// </value>
        private readonly IntPtr _cudaContext;

        /// <summary>
        /// The native ROCm/HIP context handle allocated during initialization if the ROCm backend is selected.
        /// </summary>
        /// <value>
        /// An <see cref="IntPtr"/> pointing to the native ROCm context, or <see cref="IntPtr.Zero"/> if not initialized or using a different architecture.
        /// </value>
        private readonly IntPtr _rocmContext;

        /// <summary>
        /// The active GPU hardware architecture detected and utilized by this dispatcher instance.
        /// </summary>
        private readonly GpuArchitecture _architecture;

        /// <summary>
        /// A flag indicating whether the resources managed by this instance have been disposed.
        /// </summary>
        /// <remarks>
        /// Used to ensure that unmanaged resources are not released multiple times, guaranteeing idempotency of the <see cref="Dispose()"/> method.
        /// </remarks>
        private bool _disposed;

        /// <summary>
        /// Specifies the supported graphics processing unit (GPU) hardware architectures.
        /// </summary>
        public enum GpuArchitecture
        {
            /// <summary>
            /// NVIDIA Compute Unified Device Architecture (CUDA).
            /// </summary>
            CUDA,

            /// <summary>
            /// AMD Radeon Open Compute platform (ROCm).
            /// </summary>
            ROCm,

            /// <summary>
            /// No supported GPU architecture or driver was detected on the host system.
            /// </summary>
            Unknown
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NativeDispatcher"/> class.
        /// </summary>
        /// <remarks>
        /// During initialization, this constructor queries the host system to detect compatible GPU hardware.
        /// If an NVIDIA GPU with functional CUDA drivers is detected, a CUDA context is initialized.
        /// Otherwise, if an AMD GPU with functional ROCm drivers is detected, a ROCm context is initialized.
        /// If neither is available, the dispatcher falls back to an <see cref="GpuArchitecture.Unknown"/> state,
        /// and subsequent dispatch calls will fail with a <see cref="NotSupportedException"/>.
        /// </remarks>
        public NativeDispatcher()
        {
            _architecture = DetectGpuArchitecture();
            switch (_architecture)
            {
                case GpuArchitecture.CUDA:
                    CUDA.Initialize(out _cudaContext);
                    break;
                case GpuArchitecture.ROCm:
                    ROCm.Initialize(out _rocmContext);
                    break;
                default:
                    break;
            }
        }
        /// <summary>
        /// Probes the host system to detect the presence of supported GPU runtimes and architectures.
        /// </summary>
        /// <returns>
        /// The detected <see cref="GpuArchitecture"/> corresponding to the first available and functional GPU platform.
        /// Returns <see cref="GpuArchitecture.Unknown"/> if no supported GPU hardware or driver runtime is found.
        /// </returns>
        /// <remarks>
        /// <para>
        /// The detection logic prioritizes CUDA over ROCm. 
        /// </para>
        /// <para>
        /// Any exceptions encountered during the loading of native libraries or invocation of initialization APIs 
        /// (such as <see cref="DllNotFoundException"/> or <see cref="BadImageFormatException"/>) are caught internally 
        /// to allow graceful fallback to alternative architectures or CPU execution paths.
        /// </para>
        /// </remarks>

        private static GpuArchitecture DetectGpuArchitecture()
        {
            try
            {
                if (CUDA.IsAvailable())
                {
                    return GpuArchitecture.CUDA;
                }
            }
            catch { }

            try
            {
                if (ROCm.IsAvailable())
                {
                    return GpuArchitecture.ROCm;
                }
            }
            catch { }

            return GpuArchitecture.Unknown;
        }
        /// <summary>
        /// Dispatches a General Matrix Multiplication (GEMM) operation to the active native GPU backend.
        /// </summary>
        /// <param name="a">A device memory pointer (<see cref="IntPtr"/>) to the input matrix A.</param>
        /// <param name="b">A device memory pointer (<see cref="IntPtr"/>) to the input matrix B.</param>
        /// <param name="c">A device memory pointer (<see cref="IntPtr"/>) where the resulting matrix C (A * B) will be stored.</param>
        /// <param name="m">The number of rows in matrix A and matrix C. Must be greater than 0.</param>
        /// <param name="n">The number of columns in matrix B and matrix C. Must be greater than 0.</param>
        /// <param name="k">The number of columns in matrix A and rows in matrix B. Must be greater than 0.</param>
        /// <exception cref="NotSupportedException">
        /// Thrown when the dispatcher's detected architecture is <see cref="GpuArchitecture.Unknown"/>, 
        /// indicating that matrix multiplication cannot be performed on the current platform.
        /// </exception>
        /// <exception cref="ObjectDisposedException">
        /// Thrown if this operation is called after the dispatcher has been disposed.
        /// </exception>
        /// <remarks>
        /// The caller must ensure that the device pointers <paramref name="a"/>, <paramref name="b"/>, and <paramref name="c"/> 
        /// point to valid allocated memory segments on the active GPU device, and that their sizes correspond 
        /// to the dimensions specified by <paramref name="m"/>, <paramref name="n"/>, and <paramref name="k"/>.
        /// </remarks>

        public void DispatchMatMul(IntPtr a, IntPtr b, IntPtr c, int m, int n, int k)
        {
            switch (_architecture)
            {
                case GpuArchitecture.CUDA:
                    CUDA.MatMul(a, b, c, m, n, k);
                    break;
                case GpuArchitecture.ROCm:
                    ROCm.MatMul(_rocmContext, a, b, c, m, n, k);
                    break;
                default:
                    throw new NotSupportedException("Matrix multiplication not supported on this architecture.");
            }
        }
        /// <summary>
        /// Dispatches an element-wise addition operation (<c>C = A + B</c>) to the active native GPU backend.
        /// </summary>
        /// <param name="a">A device memory pointer (<see cref="IntPtr"/>) to the first input array/tensor.</param>
        /// <param name="b">A device memory pointer (<see cref="IntPtr"/>) to the second input array/tensor.</param>
        /// <param name="c">A device memory pointer (<see cref="IntPtr"/>) where the computed element-wise sum will be stored.</param>
        /// <param name="size">The total number of elements to process in the arrays. Must be greater than 0.</param>
        /// <exception cref="NotSupportedException">
        /// Thrown when the dispatcher's detected architecture is <see cref="GpuArchitecture.Unknown"/>,
        /// indicating that element-wise addition is not supported on the host.
        /// </exception>
        /// <exception cref="ObjectDisposedException">
        /// Thrown if this operation is called after the dispatcher has been disposed.
        /// </exception>
        /// <remarks>
        /// All input and output pointers must reside on the same GPU device and correspond to memory buffers 
        /// allocated with at least <paramref name="size"/> elements of the target data type.
        /// </remarks>

        public void DispatchAdd(IntPtr a, IntPtr b, IntPtr c, int size)
        {
            switch (_architecture)
            {
                case GpuArchitecture.CUDA:
                    CUDA.Add(a, b, c, size);
                    break;
                case GpuArchitecture.ROCm:
                    ROCm.Add(_rocmContext, a, b, c, size);
                    break;
                default:
                    throw new NotSupportedException("Addition not supported on this architecture.");
            }
        }
        /// <summary>
        /// Dispatches a tensor reshape operation to the active native GPU backend, restructuring the input layout into the output layout.
        /// </summary>
        /// <param name="input">A device memory pointer (<see cref="IntPtr"/>) to the source tensor data.</param>
        /// <param name="output">A device memory pointer (<see cref="IntPtr"/>) to the destination tensor data.</param>
        /// <param name="inputShape">An array of integers representing the dimensions (shape) of the source tensor. Must not be null.</param>
        /// <param name="outputShape">An array of integers representing the dimensions (shape) of the target tensor. Must not be null.</param>
        /// <exception cref="NotSupportedException">
        /// Thrown when the dispatcher's detected architecture is <see cref="GpuArchitecture.Unknown"/>,
        /// indicating that reshape operations are not supported on the host.
        /// </exception>
        /// <exception cref="ObjectDisposedException">
        /// Thrown if this operation is called after the dispatcher has been disposed.
        /// </exception>
        /// <remarks>
        /// The caller is responsible for verifying that the total number of elements represented by <paramref name="inputShape"/> 
        /// matches the total number of elements represented by <paramref name="outputShape"/> to preserve data integrity.
        /// </remarks>

        public void DispatchReshape(IntPtr input, IntPtr output, int[] inputShape, int[] outputShape)
        {
            switch (_architecture)
            {
                case GpuArchitecture.CUDA:
                    CUDA.Reshape(_cudaContext, input, output, inputShape, outputShape);
                    break;
                case GpuArchitecture.ROCm:
                    ROCm.Reshape(_rocmContext, input, output, inputShape, outputShape);
                    break;
                default:
                    throw new NotSupportedException("Reshape not supported on this architecture.");
            }
        }
        /// <summary>
        /// Releases all unmanaged resources and contexts associated with the active native GPU backend.
        /// </summary>
        /// <remarks>
        /// This method performs deterministic cleanup of native GPU handles. 
        /// Once disposed, calling any dispatch operations on this instance will throw an <see cref="ObjectDisposedException"/>.
        /// This method is idempotent; subsequent calls will return immediately without throwing an exception.
        /// </remarks>

        public void Dispose()
        {
            if (!_disposed)
            {
                switch (_architecture)
                {
                    case GpuArchitecture.CUDA:
                        CUDA.Cleanup(_cudaContext);
                        break;
                    case GpuArchitecture.ROCm:
                        ROCm.Cleanup(_rocmContext);
                        break;
                }
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Finalizes an instance of the <see cref="NativeDispatcher"/> class.
        /// </summary>
        /// <remarks>
        /// The finalizer acts as a safety mechanism to release unmanaged GPU context resources in the event that 
        /// <see cref="Dispose()"/> was not explicitly called by the consumer.
        /// </remarks>
        ~NativeDispatcher()
        {
            Dispose();
        }
    }
}