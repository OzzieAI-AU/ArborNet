// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// Project: ArborNet
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Native.PInvoke
{
    using System;
    using System.Runtime.CompilerServices;

    public static partial class CUDA
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void CopyHostToDeviceFast(ReadOnlySpan<float> hostSource, IntPtr deviceDestination)
        {
            if (deviceDestination == IntPtr.Zero)
                throw new ArgumentNullException(nameof(deviceDestination));
            if (hostSource.IsEmpty) return;

            fixed (float* hostPtr = hostSource)
            {
                ulong bytes = (ulong)hostSource.Length * sizeof(float);
                Check(cudaMemcpy(deviceDestination, (IntPtr)hostPtr, bytes, cudaMemcpyKind.cudaMemcpyHostToDevice),
                    nameof(CopyHostToDeviceFast));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void CopyDeviceToHostFast(IntPtr deviceSource, Span<float> hostDestination)
        {
            if (deviceSource == IntPtr.Zero)
                throw new ArgumentNullException(nameof(deviceSource));
            if (hostDestination.IsEmpty) return;

            fixed (float* hostPtr = hostDestination)
            {
                ulong bytes = (ulong)hostDestination.Length * sizeof(float);
                Check(cudaMemcpy((IntPtr)hostPtr, deviceSource, bytes, cudaMemcpyKind.cudaMemcpyDeviceToHost),
                    nameof(CopyDeviceToHostFast));
            }
        }

        public static void CopyRaw(IntPtr dst, IntPtr src, ulong bytes, cudaMemcpyKind kind)
        {
            if (bytes == 0) return;
            if (dst == IntPtr.Zero) throw new ArgumentNullException(nameof(dst));
            if (src == IntPtr.Zero) throw new ArgumentNullException(nameof(src));
            Check(cudaMemcpy(dst, src, bytes, kind), nameof(CopyRaw));
        }
    }
}
