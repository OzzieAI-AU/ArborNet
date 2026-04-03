using System;
using System.Reflection;
using System.Runtime.InteropServices;
using ArborNet.Core.Native;
using ArborNet.Core.Native.PInvoke;

namespace ArborNet.Core.Native
{
    /// <summary>
    /// Provides a unified dispatcher for native GPU operations across supported architectures (CUDA and ROCm).
    /// Automatically detects the available GPU backend at construction time and routes tensor operations
    /// to the appropriate native implementation while managing associated native contexts and resources.
    /// </summary>
    /// <remarks>
    /// This class implements the <see cref="IDisposable"/> pattern to ensure proper cleanup of native GPU contexts.
    /// CUDA is preferred over ROCm when both are available. If no supported architecture is detected,
    /// operations will throw <see cref="NotSupportedException"/>.
    /// </remarks>
    public class NativeDispatcher : IDisposable
    {
        /// <summary>
        /// CUDA context handle obtained during initialization.
        /// </summary>
        private readonly IntPtr _cudaContext;
        /// <summary>
        /// ROCm context handle obtained during initialization.
        /// </summary>
        private readonly IntPtr _rocmContext;
        /// <summary>
        /// The GPU architecture detected on the current system.
        /// </summary>
        private readonly GpuArchitecture _architecture;
        /// <summary>
        /// Indicates whether this instance has already been disposed.
        /// </summary>
        private bool _disposed;

        /// <summary>
        /// Defines the supported GPU architectures for native operations.
        /// </summary>
        public enum GpuArchitecture
        {
            /// <summary>
            /// NVIDIA CUDA architecture.
            /// </summary>
            CUDA,
            /// <summary>
            /// AMD ROCm architecture.
            /// </summary>
            ROCm,
            /// <summary>
            /// No supported GPU architecture was detected.
            /// </summary>
            Unknown
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NativeDispatcher"/> class.
        /// </summary>
        /// <remarks>
        /// Performs GPU architecture detection and initializes the corresponding native backend context.
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
        /// Detects the available GPU architecture by attempting to query CUDA then ROCm.
        /// </summary>
        /// <returns>The first available <see cref="GpuArchitecture"/>, or <see cref="GpuArchitecture.Unknown"/> if none is available.</returns>
        /// <remarks>
        /// CUDA is checked first. Any exceptions during availability checks are silently caught to enable graceful fallback.
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
        /// Dispatches a general matrix multiplication (GEMM) operation to the appropriate native GPU backend.
        /// </summary>
        /// <param name="a">Pointer to the first input matrix in device memory.</param>
        /// <param name="b">Pointer to the second input matrix in device memory.</param>
        /// <param name="c">Pointer to the output matrix in device memory.</param>
        /// <param name="m">Number of rows in matrices A and C.</param>
        /// <param name="n">Number of columns in matrices B and C.</param>
        /// <param name="k">Number of columns in matrix A and rows in matrix B.</param>
        /// <exception cref="NotSupportedException">Thrown when the current architecture does not support matrix multiplication.</exception>
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
        /// Dispatches an element-wise addition operation to the appropriate native GPU backend.
        /// </summary>
        /// <param name="a">Pointer to the first input array in device memory.</param>
        /// <param name="b">Pointer to the second input array in device memory.</param>
        /// <param name="c">Pointer to the output array in device memory.</param>
        /// <param name="size">Number of elements to process.</param>
        /// <exception cref="NotSupportedException">Thrown when the current architecture does not support addition.</exception>
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
        /// Dispatches a tensor reshape operation to the appropriate native GPU backend.
        /// </summary>
        /// <param name="input">Pointer to the input tensor data in device memory.</param>
        /// <param name="output">Pointer to the output tensor data in device memory.</param>
        /// <param name="inputShape">Array describing the dimensions of the input tensor.</param>
        /// <param name="outputShape">Array describing the dimensions of the output tensor.</param>
        /// <exception cref="NotSupportedException">Thrown when the current architecture does not support reshape.</exception>
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
        /// Releases all unmanaged resources used by the <see cref="NativeDispatcher"/>.
        /// </summary>
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
        /// Finalizer that ensures native resources are released if <see cref="Dispose"/> was not called.
        /// </summary>
        ~NativeDispatcher()
        {
            Dispose();
        }
    }
}