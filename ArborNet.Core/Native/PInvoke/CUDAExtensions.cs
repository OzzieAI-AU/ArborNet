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
    using System.Runtime.CompilerServices;
    using System.Runtime.InteropServices;
    /// <summary>
    /// Provides low-level, high-performance bindings and utility methods for interacting with the NVIDIA CUDA API.
    /// </summary>
    /// <remarks>
    /// This partial class wraps native CUDA driver and runtime API calls. It incorporates aggressive optimization 
    /// strategies, such as memory pinning and direct pointer manipulation, to achieve zero-allocation and zero-GC-overhead operations.
    /// All methods in this class are designed to minimize latency and overhead in high-throughput applications.
    /// </remarks>

    #endregion

    public static partial class CUDA
    {
        /// <summary>
        /// Copies memory synchronously from the host (CPU) to the device (GPU) memory space using a high-performance,
        /// zero-allocation, and zero-GC-overhead pathway.
        /// </summary>
        /// <param name="hostSource">The read-only source memory span residing on the host CPU containing the data to copy. This memory is pinned temporarily during the transfer.</param>
        /// <param name="deviceDestination">An unmanaged <see cref="IntPtr"/> pointing to the pre-allocated GPU device memory destination.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="deviceDestination"/> is <see cref="IntPtr.Zero"/>.</exception>
        /// <exception cref="InvalidOperationException">Thrown when the underlying CUDA memory copy operation fails, returning a non-success <see cref="CudaError"/>.</exception>
        /// <remarks>
        /// This method temporarily pins the <paramref name="hostSource"/> span using a <see langword="fixed"/> statement 
        /// to resolve a direct pointer, bypassing the garbage collector and avoiding any managed allocation. 
        /// <para/>
        /// Because this operates synchronously on the calling thread, it blocks execution until the transfer to the GPU is complete.
        /// Ensure that the buffer pointed to by <paramref name="deviceDestination"/> has a capacity of at least 
        /// <c>hostSource.Length * sizeof(float)</c> bytes to prevent memory corruption.
        /// </remarks>
        /// <example>
        /// This example demonstrates how to copy an array of floats to a pre-allocated CUDA device memory buffer.
        /// <code>
        /// float[] hostData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        /// IntPtr deviceBuffer = CUDA.AllocateDeviceMemory(hostData.Length * sizeof(float)); // Example allocation call
        /// 
        /// try
        /// {
        ///     CUDA.CopyHostToDeviceFast(hostData, deviceBuffer);
        ///     // Perform GPU calculations...
        /// }
        /// finally
        /// {
        ///     CUDA.FreeDeviceMemory(deviceBuffer); // Example free call
        /// }
        /// </code>
        /// </example>
        /// <seealso cref="CopyDeviceToHostFast(IntPtr, Span{float})"/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void CopyHostToDeviceFast(ReadOnlySpan<float> hostSource, IntPtr deviceDestination)
        {
            if (deviceDestination == IntPtr.Zero)
                throw new ArgumentNullException(nameof(deviceDestination), "Device pointer cannot be null.");

            fixed (float* hostPtr = hostSource)
            {
                ulong bytes = (ulong)hostSource.Length * sizeof(float);
                // Dest: deviceDestination, Src: hostPtr
                CudaError err = cudaMemcpy(deviceDestination, (IntPtr)hostPtr, bytes, cudaMemcpyKind.cudaMemcpyHostToDevice);
                if (err != CudaError.Success)
                    throw new InvalidOperationException($"Fast Host-to-Device Memcpy failed: {err}");
            }
        }
        /// <summary>
        /// Copies memory synchronously from the device (GPU) to the host (CPU) memory space using a high-performance,
        /// zero-allocation, and zero-GC-overhead pathway.
        /// </summary>
        /// <param name="deviceSource">An unmanaged <see cref="IntPtr"/> pointing to the initialized GPU device memory source to copy from.</param>
        /// <param name="hostDestination">The destination memory span on the host CPU where the copied data will be written. This memory is pinned temporarily during the transfer.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="deviceSource"/> is <see cref="IntPtr.Zero"/>.</exception>
        /// <exception cref="InvalidOperationException">Thrown when the underlying CUDA memory copy operation fails, returning a non-success <see cref="CudaError"/>.</exception>
        /// <remarks>
        /// This method pins the <paramref name="hostDestination"/> span using a <see langword="fixed"/> statement to obtain 
        /// a direct pointer to the target host memory, avoiding intermediate buffering or GC allocation overhead.
        /// <para/>
        /// This operation is synchronous and blocks the caller. The caller must guarantee that the allocated size 
        /// of the native memory at <paramref name="deviceSource"/> is at least equal to <c>hostDestination.Length * sizeof(float)</c> bytes.
        /// </remarks>
        /// <example>
        /// This example demonstrates how to copy data from a CUDA device memory buffer back to a host float array.
        /// <code>
        /// float[] hostBuffer = new float[4];
        /// IntPtr deviceBuffer = GetDeviceDataPointer(); // Example retrieval call
        /// 
        /// CUDA.CopyDeviceToHostFast(deviceBuffer, hostBuffer);
        /// 
        /// foreach (var val in hostBuffer)
        /// {
        ///     Console.WriteLine(val);
        /// }
        /// </code>
        /// </example>
        /// <seealso cref="CopyHostToDeviceFast(ReadOnlySpan{float}, IntPtr)"/>

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void CopyDeviceToHostFast(IntPtr deviceSource, Span<float> hostDestination)
        {
            if (deviceSource == IntPtr.Zero)
                throw new ArgumentNullException(nameof(deviceSource), "Device source pointer cannot be null.");

            fixed (float* hostPtr = hostDestination)
            {
                ulong bytes = (ulong)hostDestination.Length * sizeof(float);
                // Dest: hostPtr, Src: deviceSource
                CudaError err = cudaMemcpy((IntPtr)hostPtr, deviceSource, bytes, cudaMemcpyKind.cudaMemcpyDeviceToHost);
                if (err != CudaError.Success)
                    throw new InvalidOperationException($"Fast Device-to-Host Memcpy failed: {err}");
            }
        }
    }
}