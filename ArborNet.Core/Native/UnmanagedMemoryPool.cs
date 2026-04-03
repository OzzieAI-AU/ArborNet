using ArborNet.Core.Native;
using ArborNet.Core.Native.PInvoke;
using System;
using System.Runtime.InteropServices;

namespace ArborNet.Core.Native
{
    /// <summary>
    /// Provides functionality for allocating unmanaged host memory that is registered
    /// with CUDA as pinned memory, along with high-performance host-device memory transfers.
    /// </summary>
    /// <remarks>
    /// This class uses <see cref="Marshal.AllocHGlobal"/> combined with CUDA host memory registration
    /// to enable efficient zero-copy or pinned transfers between host and device.
    /// Memory allocated via this class must be explicitly freed using <see cref="Free"/>.
    /// </remarks>
    public class UnmanagedMemoryPool : IDisposable
    {
        /// <summary>
        /// Indicates whether the current instance has been disposed.
        /// </summary>
        private bool _disposed = false;

        /// <summary>
        /// Allocates unmanaged host memory of the specified size and registers it with CUDA
        /// as pinned memory for optimal transfer performance.
        /// </summary>
        /// <param name="size">The number of bytes to allocate.</param>
        /// <returns>An <see cref="IntPtr"/> to the allocated unmanaged memory block.</returns>
        /// <exception cref="ArgumentException">Thrown when <paramref name="size"/> is zero.</exception>
        /// <exception cref="InvalidOperationException">Thrown when the memory allocation or CUDA registration fails.</exception>
        public IntPtr Alloc(ulong size)
        {
            if (size == 0)
            {
                throw new ArgumentException("Size must be greater than zero.", nameof(size));
            }

            IntPtr ptr = Marshal.AllocHGlobal((IntPtr)size);
            if (ptr == IntPtr.Zero)
            {
                throw new InvalidOperationException("Failed to allocate unmanaged memory.");
            }

            CUDA.CudaHostRegister(ptr, size, 0);
            return ptr;
        }

        /// <summary>
        /// Unregisters the memory from CUDA and frees the associated unmanaged memory.
        /// </summary>
        /// <param name="ptr">Pointer to the memory block previously allocated by <see cref="Alloc"/>.</param>
        /// <exception cref="ArgumentException">Thrown when <paramref name="ptr"/> is <see cref="IntPtr.Zero"/>.</exception>
        public void Free(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
            {
                throw new ArgumentException("Pointer must not be IntPtr.Zero.", nameof(ptr));
            }

            CUDA.CudaHostUnregister(ptr);
            Marshal.FreeHGlobal(ptr);
        }

        /// <summary>
        /// Copies data from host memory to device memory using CUDA's cudaMemcpy.
        /// </summary>
        /// <param name="hostPtr">Pointer to the source data in host (unmanaged) memory.</param>
        /// <param name="devicePtr">Pointer to the destination buffer in device memory.</param>
        /// <param name="size">The number of bytes to copy.</param>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="hostPtr"/>, <paramref name="devicePtr"/>, or <paramref name="size"/> is invalid.
        /// </exception>
        public void TransferToDevice(IntPtr hostPtr, IntPtr devicePtr, ulong size)
        {
            if (hostPtr == IntPtr.Zero) throw new ArgumentException("Host pointer must not be IntPtr.Zero.", nameof(hostPtr));
            if (devicePtr == IntPtr.Zero) throw new ArgumentException("Device pointer must not be IntPtr.Zero.", nameof(devicePtr));
            if (size == 0) throw new ArgumentException("Size must be greater than zero.", nameof(size));

            CUDA.CudaMemcpy(devicePtr, hostPtr, size, CUDA.cudaMemcpyKind.cudaMemcpyHostToDevice);
        }

        /// <summary>
        /// Copies data from device memory to host memory using CUDA's cudaMemcpy.
        /// </summary>
        /// <param name="devicePtr">Pointer to the source data in device memory.</param>
        /// <param name="hostPtr">Pointer to the destination buffer in host (unmanaged) memory.</param>
        /// <param name="size">The number of bytes to copy.</param>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="devicePtr"/>, <paramref name="hostPtr"/>, or <paramref name="size"/> is invalid.
        /// </exception>
        public void TransferFromDevice(IntPtr devicePtr, IntPtr hostPtr, ulong size)
        {
            if (devicePtr == IntPtr.Zero) throw new ArgumentException("Device pointer must not be IntPtr.Zero.", nameof(devicePtr));
            if (hostPtr == IntPtr.Zero) throw new ArgumentException("Host pointer must not be IntPtr.Zero.", nameof(hostPtr));
            if (size == 0) throw new ArgumentException("Size must be greater than zero.", nameof(size));

            CUDA.CudaMemcpy(hostPtr, devicePtr, size, CUDA.cudaMemcpyKind.cudaMemcpyDeviceToHost);
        }

        /// <summary>
        /// Releases all resources used by the current instance of the <see cref="UnmanagedMemoryPool"/> class.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Finalizer that ensures resources are released if Dispose was not called.
        /// </summary>
        ~UnmanagedMemoryPool()
        {
            Dispose();
        }
    }
}