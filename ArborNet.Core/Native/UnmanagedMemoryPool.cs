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

    using ArborNet.Core.Native;
    using ArborNet.Core.Native.PInvoke;
    using System;
    using System.Runtime.InteropServices;
    /// <summary>
    /// Provides functionality for allocating unmanaged host memory that is registered
    /// with CUDA as pinned memory, along with high-performance host-device memory transfers.
    /// </summary>
    /// <remarks>
    /// This class uses <see cref="Marshal.AllocHGlobal"/> combined with CUDA host memory registration
    /// to enable efficient zero-copy or pinned transfers between host and device.
    /// Memory allocated via this class must be explicitly freed using <see cref="Free"/>.
    /// </remarks>
    /// <seealso cref="IDisposable"/>

    #endregion

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
        /// <param name="size">The number of bytes to allocate. Must be greater than zero.</param>
        /// <returns>An <see cref="IntPtr"/> to the allocated unmanaged memory block.</returns>
        /// <exception cref="ArgumentException">Thrown when <paramref name="size"/> is zero.</exception>
        /// <exception cref="InvalidOperationException">Thrown when the memory allocation or CUDA registration fails.</exception>
        /// <remarks>
        /// The allocated memory is registered using CUDA's host register API to ensure page-locked (pinned) 
        /// CPU memory, which enables high-bandwidth transfers to and from GPU device memory.
        /// </remarks>

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
        /// <remarks>
        /// This method must be called to release the OS-level unmanaged allocation and to unpin the memory from the CUDA driver.
        /// </remarks>

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
        /// <param name="size">The number of bytes to copy. Must be greater than zero.</param>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="hostPtr"/> is <see cref="IntPtr.Zero"/>, 
        /// <paramref name="devicePtr"/> is <see cref="IntPtr.Zero"/>, or 
        /// <paramref name="size"/> is zero.
        /// </exception>
        /// <remarks>
        /// Utilizes <see cref="CUDA.cudaMemcpyKind.cudaMemcpyHostToDevice"/> to execute the synchronous transfer.
        /// </remarks>

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
        /// <param name="size">The number of bytes to copy. Must be greater than zero.</param>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="devicePtr"/> is <see cref="IntPtr.Zero"/>, 
        /// <paramref name="hostPtr"/> is <see cref="IntPtr.Zero"/>, or 
        /// <paramref name="size"/> is zero.
        /// </exception>
        /// <remarks>
        /// Utilizes <see cref="CUDA.cudaMemcpyKind.cudaMemcpyDeviceToHost"/> to execute the synchronous transfer.
        /// </remarks>

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
        /// <remarks>
        /// Disposes internal tracking flags. Pointers allocated via <see cref="Alloc"/> must still be 
        /// explicitly freed using <see cref="Free"/> to prevent unmanaged memory leaks.
        /// </remarks>

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Finalizer that ensures resources are released if <see cref="Dispose"/> was not called.
        /// </summary>
        /// <remarks>
        /// Invokes the <see cref="Dispose"/> method to signal resource cleanup.
        /// </remarks>
        ~UnmanagedMemoryPool()
        {
            Dispose();
        }
    }
}