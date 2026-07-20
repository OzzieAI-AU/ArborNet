// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Native.Distributed
{

    #region Using Statements:

    using System;
    using System.Runtime.InteropServices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Thread-safe managed wrapper for the NVIDIA Collective Communications Library (NCCL).
    /// Provides collective synchronization primitives for high-performance multi-GPU distributed training.
    /// </summary>
    /// <remarks>
    /// This class encapsulates the native pointers and operations required to coordinate collective communications
    /// (such as AllReduce) across multiple CUDA devices. It supports graceful fallback to single-process CPU mode
    /// if the NCCL library (nccl.dll / libnccl.so) is unavailable in the execution environment.
    /// Thread safety is guaranteed as long as operations on the underlying CUDA stream are synchronized.
    /// </remarks>

    #endregion

    public sealed class CollectiveComm : IDisposable
    {
        /// <summary>
        /// The filename of the NCCL shared library used for P/Invoke operations.
        /// </summary>
        /// <remarks>
        /// Under Linux/Unix environments, the Mono or CoreCLR runtime automatically maps this name to 'libnccl.so'.
        /// </remarks>
        private const string NcclDll = "nccl.dll";

        /// <summary>
        /// Represents a unique 128-byte identifier used by NCCL to coordinate connections
        /// between distributed processes (ranks) during communicator initialization.
        /// </summary>
        /// <remarks>
        /// This structure must be serialized and distributed from the rank 0 process to all other participating ranks
        /// before initializing the communicator group via <see cref="ncclCommInitRank"/>.
        /// </remarks>
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct NcclUniqueId
        {
            /// <summary>
            /// The raw 128-byte array holding the unique NCCL connection token.
            /// </summary>
            /// <remarks>
            /// Marshaled as a fixed-size byte array of 128 elements to match the native NCCL unique ID structure.
            /// </remarks>
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]
            public byte[] InternalId;
        }

        /// <summary>
        /// Specifies the reduction operations supported by NCCL collective communications.
        /// </summary>
        /// <remarks>
        /// These operations are executed on the GPU devices using optimized collective algorithms.
        /// </remarks>
        public enum NcclRedOp : int
        {
            /// <summary>
            /// Computes the sum across all ranks.
            /// </summary>
            Sum = 0,

            /// <summary>
            /// Computes the product across all ranks.
            /// </summary>
            Prod = 1,

            /// <summary>
            /// Computes the maximum value across all ranks.
            /// </summary>
            Max = 2,

            /// <summary>
            /// Computes the minimum value across all ranks.
            /// </summary>
            Min = 3
        }

        /// <summary>
        /// Specifies the tensor element data types supported by NCCL.
        /// </summary>
        /// <remarks>
        /// Matches the native NCCL enumerations for supported floating-point formats.
        /// </remarks>
        public enum NcclDataType : int
        {
            /// <summary>
            /// Represents a 32-bit single-precision floating-point number.
            /// </summary>
            Float32 = 7,

            /// <summary>
            /// Represents a 16-bit half-precision floating-point number.
            /// </summary>
            Float16 = 6
        }
        /// <summary>
        /// Native P/Invoke entry point for generating a globally unique NCCL execution context ID.
        /// This ID must be distributed to all ranks in the communicator group.
        /// </summary>
        /// <param name="uniqueId">Output parameter that receives the generated <see cref="NcclUniqueId"/>.</param>
        /// <returns>A native NCCL status code (<c>ncclResult_t</c>) where 0 indicates success, and any non-zero value indicates failure.</returns>

        #region Native NCCL P/Invoke Signatures

        [DllImport(NcclDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int ncclGetUniqueId(out NcclUniqueId uniqueId);
        /// <summary>
        /// Native P/Invoke entry point for initializing a communicator rank context.
        /// </summary>
        /// <param name="comm">Output parameter that receives the initialized native communicator pointer (<c>ncclComm_t</c>).</param>
        /// <param name="nranks">The total number of participating ranks in the communicator group.</param>
        /// <param name="uniqueId">The shared unique identifier generated by <see cref="ncclGetUniqueId"/> and distributed to all ranks.</param>
        /// <param name="rank">The unique index of the current rank within the communicator group (0 to <paramref name="nranks"/> - 1).</param>
        /// <returns>A native NCCL status code (<c>ncclResult_t</c>) where 0 indicates success, and any non-zero value indicates failure.</returns>

        [DllImport(NcclDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int ncclCommInitRank(out IntPtr comm, int nranks, NcclUniqueId uniqueId, int rank);
        /// <summary>
        /// Native P/Invoke entry point for destroying and releasing a communicator context.
        /// </summary>
        /// <param name="comm">The native pointer representing the communicator context (<c>ncclComm_t</c>) to destroy.</param>
        /// <returns>A native NCCL status code (<c>ncclResult_t</c>) where 0 indicates success, and any non-zero value indicates failure.</returns>

        [DllImport(NcclDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int ncclCommDestroy(IntPtr comm);
        /// <summary>
        /// Native P/Invoke entry point to execute an in-place or out-of-place AllReduce operation.
        /// AllReduce averages, sums, multiplies, or finds min/max values across all GPUs in the group.
        /// </summary>
        /// <param name="sendbuff">The native device pointer to the input tensor buffer on the GPU.</param>
        /// <param name="recvbuff">The native device pointer to the output tensor buffer on the GPU. For in-place reduction, this should equal <paramref name="sendbuff"/>.</param>
        /// <param name="count">The total number of elements in the tensor to reduce.</param>
        /// <param name="datatype">The native data type of the tensor elements.</param>
        /// <param name="op">The reduction operation to apply.</param>
        /// <param name="comm">The native pointer representing the active communicator context (<c>ncclComm_t</c>).</param>
        /// <param name="stream">The native CUDA stream handle (<c>cudaStream_t</c>) associated with this operation.</param>
        /// <returns>A native NCCL status code (<c>ncclResult_t</c>) where 0 indicates success, and any non-zero value indicates failure.</returns>

        [DllImport(NcclDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int ncclAllReduce(
    IntPtr sendbuff,
    IntPtr recvbuff,
    ulong count,
    NcclDataType datatype,
    NcclRedOp op,
    IntPtr comm,
    IntPtr stream);
        /// <summary>
        /// Native P/Invoke entry point to begin a NCCL group execution. 
        /// Used to queue and batch collective operations across multiple GPUs to reduce synchronization overhead.
        /// </summary>
        /// <returns>A native NCCL status code (<c>ncclResult_t</c>) where 0 indicates success, and any non-zero value indicates failure.</returns>

        [DllImport(NcclDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int ncclGroupStart();
        /// <summary>
        /// Native P/Invoke entry point to end a NCCL group execution, triggering all queued batched collective operations.
        /// </summary>
        /// <returns>A native NCCL status code (<c>ncclResult_t</c>) where 0 indicates success, and any non-zero value indicates failure.</returns>

        [DllImport(NcclDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int ncclGroupEnd();

        #endregion

        /// <summary>
        /// The native pointer identifying the active NCCL communicator instance.
        /// </summary>
        /// <remarks>
        /// This is initialized by <see cref="ncclCommInitRank"/> and released by <see cref="ncclCommDestroy"/>.
        /// </remarks>
        private readonly IntPtr _commHandle;

        /// <summary>
        /// The unique ID (index) of this specific process/GPU within the collective group.
        /// </summary>
        private readonly int _rank;

        /// <summary>
        /// The total number of processes/GPUs participating in the collective group.
        /// </summary>
        private readonly int _worldSize;

        /// <summary>
        /// Tracks whether the unmanaged and managed resources have been disposed to prevent double-free errors.
        /// </summary>
        private bool _isDisposed;
        /// <summary>
        /// Gets the unique rank identifier of this node within the distributed cluster.
        /// </summary>
        /// <value>
        /// An <see cref="int"/> representing the 0-indexed identifier of this process or GPU.
        /// </value>

        public int Rank => _rank;
        /// <summary>
        /// Gets the total number of active ranks (GPUs) in the distributed network context.
        /// </summary>
        /// <value>
        /// An <see cref="int"/> representing the total count of participating cluster nodes.
        /// </value>

        public int WorldSize => _worldSize;
        /// <summary>
        /// Generates a unique communication ID used to initialize multiple distributed ranks.
        /// </summary>
        /// <returns>A globally unique <see cref="NcclUniqueId"/> connection token.</returns>
        /// <exception cref="InvalidOperationException">Thrown if NCCL is present but fails to generate a unique ID.</exception>
        /// <remarks>
        /// If the NCCL shared library cannot be located, a dummy zeroed-out ID is returned for fallback purposes.
        /// This is useful during local testing, continuous integration, or non-distributed CPU execution.
        /// </remarks>

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
        /// Initializes a new instance of the <see cref="CollectiveComm"/> class for a specific distributed rank.
        /// </summary>
        /// <param name="rank">The unique rank of the current GPU/process (0-indexed).</param>
        /// <param name="worldSize">The total number of GPUs/processes in the distributed training run.</param>
        /// <param name="uniqueId">The unique synchronization ID retrieved from <see cref="GenerateUniqueId"/>.</param>
        /// <exception cref="InvalidOperationException">Thrown if NCCL is found but fails to initialize the collective context.</exception>
        /// <remarks>
        /// If the native NCCL library is missing, this constructor catches the <see cref="DllNotFoundException"/>,
        /// logs a warning, and falls back to a dummy CPU-based operational mode where collective calls are bypassed.
        /// </remarks>
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
        /// Redundantly synchronizes and accumulates gradients or outputs across all active GPUs using the specified reduction operation.
        /// Performs this in-place operation directly on the provided device tensor.
        /// </summary>
        /// <param name="tensor">The tensor containing values to be reduced across the active ranks.</param>
        /// <param name="reductionOp">The type of math reduction operation to apply (e.g., Sum, Prod, Max, Min). Defaults to <see cref="NcclRedOp.Sum"/>.</param>
        /// <exception cref="ArgumentNullException">Thrown if the <paramref name="tensor"/> is null.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the native NCCL execution encounters driver failures.</exception>
        /// <remarks>
        /// If the communicator handle is uninitialized (<see cref="IntPtr.Zero"/>) or the tensor resides on the CPU,
        /// this method will silently return, serving as a safe single-process bypass.
        /// The operation runs in-place, meaning the output of the reduction will overwrite the input <paramref name="tensor"/>'s values.
        /// Currently, this operation is optimized for 32-bit floating-point data types (<see cref="NcclDataType.Float32"/>).
        /// </remarks>

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
        /// <summary>
        /// Extracts the internal native device pointer from a managed <see cref="ITensor"/> implementation using reflection.
        /// </summary>
        /// <param name="tensor">The managed tensor object targeting the CUDA backend.</param>
        /// <returns>An <see cref="IntPtr"/> representing the raw GPU memory address.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="tensor"/> is null.</exception>
        /// <exception cref="NotSupportedException">Thrown if the tensor backend cannot be unwrapped or lacks a native device pointer.</exception>
        /// <remarks>
        /// This method relies on reflection to look up the private field <c>_devicePtr</c> on the unwrapped tensor type.
        /// It is sensitive to changes in the underlying tensor implementation.
        /// </remarks>

        private static IntPtr GetNativeCudaPointer(ITensor tensor)
        {
            if (tensor == null) throw new ArgumentNullException(nameof(tensor));

            ITensor unwrapped = Tensor.Unwrap(tensor);
            var field = unwrapped.GetType().GetField("_devicePtr",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            if (field == null)
                throw new NotSupportedException("Unwrapped tensor backend does not expose a native GPU device pointer.");

            return (IntPtr)field.GetValue(unwrapped)!;
        }
        /// <summary>
        /// Releases all unmanaged resources used by the <see cref="CollectiveComm"/> instance, 
        /// destroying the active NCCL communicator.
        /// </summary>
        /// <remarks>
        /// Closes the native communicator handle using <see cref="ncclCommDestroy(IntPtr)"/> and suppresses finalization.
        /// This method is safe to call multiple times (idempotent).
        /// </remarks>

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

        /// <summary>
        /// Finalizes an instance of the <see cref="CollectiveComm"/> class to release native communicator contexts
        /// if they were not explicitly disposed.
        /// </summary>
        ~CollectiveComm() => Dispose();
    }
}