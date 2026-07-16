using System;
using System.Runtime.InteropServices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Core.Native.Distributed
{
    /// <summary>
    /// Thread-safe managed wrapper for the NVIDIA Collective Communications Library (NCCL).
    /// Provides collective synchronization primitives for high-performance multi-GPU distributed training.
    /// </summary>
    public sealed class CollectiveComm : IDisposable
    {
        private const string NcclDll = "nccl.dll"; // For Linux, runtime maps this automatically to libnccl.so

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct NcclUniqueId
        {
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]
            public byte[] InternalId;
        }

        public enum NcclRedOp : int
        {
            Sum = 0,
            Prod = 1,
            Max = 2,
            Min = 3
        }

        public enum NcclDataType : int
        {
            Float32 = 7,
            Float16 = 6
        }

        #region Native NCCL P/Invoke Signatures

        [DllImport(NcclDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int ncclGetUniqueId(out NcclUniqueId uniqueId);

        [DllImport(NcclDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int ncclCommInitRank(out IntPtr comm, int nranks, NcclUniqueId uniqueId, int rank);

        [DllImport(NcclDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int ncclCommDestroy(IntPtr comm);

        [DllImport(NcclDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int ncclAllReduce(
            IntPtr sendbuff,
            IntPtr recvbuff,
            ulong count,
            NcclDataType datatype,
            NcclRedOp op,
            IntPtr comm,
            IntPtr stream);

        [DllImport(NcclDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int ncclGroupStart();

        [DllImport(NcclDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int ncclGroupEnd();

        #endregion

        private readonly IntPtr _commHandle;
        private readonly int _rank;
        private readonly int _worldSize;
        private bool _isDisposed;

        public int Rank => _rank;
        public int WorldSize => _worldSize;

        /// <summary>
        /// Generates a unique communication ID used to initialize multiple distributed ranks.
        /// </summary>
        public static NcclUniqueId GenerateUniqueId()
        {
            try
            {
                int err = ncclGetUniqueId(out NcclUniqueId id);
                if (err != 0) throw new InvalidOperationException($"NCCL failed to generate Unique ID. Code: {err}");
                return id;
            }
            catch (DllNotFoundException)
            {
                // Managed single-process fallback placeholder ID
                return new NcclUniqueId { InternalId = new byte[128] };
            }
        }

        /// <summary>
        /// Initializes a distributed collective communication group rank.
        /// </summary>
        public CollectiveComm(int rank, int worldSize, NcclUniqueId uniqueId)
        {
            _rank = rank;
            _worldSize = worldSize;

            try
            {
                int err = ncclCommInitRank(out _commHandle, worldSize, uniqueId, rank);
                if (err != 0) throw new InvalidOperationException($"NCCL initialization failed with code: {err}");
            }
            catch (DllNotFoundException)
            {
                Console.WriteLine("[CollectiveComm] Warning: NCCL DLL not found. Operating in CPU-Fallback Single Mode.");
                _commHandle = IntPtr.Zero;
            }
        }

        /// <summary>
        /// Redundantly synchronizes and accumulates gradients or outputs across all active GPUs.
        /// </summary>
        public void AllReduce(ITensor tensor, NcclRedOp reductionOp = NcclRedOp.Sum)
        {
            if (tensor == null) throw new ArgumentNullException(nameof(tensor));

            if (_commHandle == IntPtr.Zero || tensor.IsCpu())
            {
                // Safe single-process bypass: No operation needed on single host nodes
                return;
            }

            // Extract the native GPU memory pointer from the CUDA Backend
            IntPtr devicePtr = GetNativeCudaPointer(tensor);
            ulong elementsCount = (ulong)tensor.Shape.TotalElements;

            ncclGroupStart();
            int err = ncclAllReduce(
                devicePtr,
                devicePtr,
                elementsCount,
                NcclDataType.Float32,
                reductionOp,
                _commHandle,
                IntPtr.Zero);
            ncclGroupEnd();

            if (err != 0)
            {
                throw new InvalidOperationException($"NCCL AllReduce failed with driver error: {err}");
            }
        }

        private static IntPtr GetNativeCudaPointer(ITensor tensor)
        {
            ITensor unwrapped = Tensor.Unwrap(tensor);
            var field = unwrapped.GetType().GetField("_devicePtr",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            if (field == null)
                throw new NotSupportedException("Unwrapped tensor backend does not expose a native GPU device pointer.");

            return (IntPtr)field.GetValue(unwrapped)!;
        }

        public void Dispose()
        {
            if (_isDisposed) return;

            if (_commHandle != IntPtr.Zero)
            {
                ncclCommDestroy(_commHandle);
            }
            _isDisposed = true;
            GC.SuppressFinalize(this);
        }

        ~CollectiveComm() => Dispose();
    }
}