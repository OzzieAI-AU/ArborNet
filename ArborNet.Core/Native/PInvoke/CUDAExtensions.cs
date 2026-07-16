using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace ArborNet.Core.Native.PInvoke
{
    public static partial class CUDA
    {
        /// <summary>
        /// Copies host memory to device memory with zero allocations and zero GC overhead.
        /// </summary>
        /// <param name="hostSource">The source memory span on the host CPU.</param>
        /// <param name="deviceDestination">The destination pointer in GPU device memory.</param>
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
        /// Copies device memory to host memory with zero allocations and zero GC overhead.
        /// </summary>
        /// <param name="deviceSource">The source pointer in GPU device memory.</param>
        /// <param name="hostDestination">The destination memory span on the host CPU.</param>
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