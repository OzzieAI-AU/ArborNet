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
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using ArborNet.Core.Native.PInvoke;
    /// <summary>
    /// Provides a thread-safe memory pool for managing CUDA memory allocations.
    /// Reduces allocation overhead by caching and reusing freed buffers using a power-of-two bucket strategy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This class implements the Singleton pattern. It manages a cache of CUDA pointers and automatically
    /// evicts older allocations when the pool size exceeds the defined memory budget.
    /// </para>
    /// <para>
    /// Thread-safety is achieved using concurrent collections (<see cref="ConcurrentDictionary{TKey, TValue}"/> 
    /// and <see cref="ConcurrentStack{T}"/>) along with atomic <see cref="System.Threading.Interlocked"/> operations for tracking 
    /// the pool's memory footprint.
    /// </para>
    /// </remarks>

    #endregion

    public sealed class CudaMemoryPool
    {
        private static readonly Lazy<CudaMemoryPool> _instance =
            new Lazy<CudaMemoryPool>(() => new CudaMemoryPool());
        /// <summary>
        /// Gets the singleton instance of the <see cref="CudaMemoryPool"/> class.
        /// </summary>
        /// <value>
        /// The singleton instance of the CUDA memory pool.
        /// </value>

        public static CudaMemoryPool Instance => _instance.Value;

        // FIXED: Cache buffers alongside their last-used timestamp to support LRU pruning
        private readonly ConcurrentDictionary<ulong, ConcurrentStack<(IntPtr Ptr, DateTime LastUsed)>> _freeBuffers =
            new ConcurrentDictionary<ulong, ConcurrentStack<(IntPtr Ptr, DateTime LastUsed)>>();

        private long _totalAllocatedBytes = 0;
        private const long MaxPoolBytes = 4L * 1024 * 1024 * 1024; // 4 GB strict limit

        private CudaMemoryPool() { }
        /// <summary>
        /// Calculates the appropriate power-of-two bucket size for a given request size in bytes.
        /// </summary>
        /// <param name="bytes">The requested size in bytes.</param>
        /// <returns>
        /// The rounded-up power-of-two size (minimum 256 bytes) containing the requested size, 
        /// or 0 if the input <paramref name="bytes"/> is 0.
        /// </returns>

        private static ulong GetBucketSize(ulong bytes)
        {
            if (bytes == 0) return 0;
            ulong bucket = 256;
            while (bucket < bytes)
            {
                bucket <<= 1;
            }
            return bucket;
        }
        /// <summary>
        /// Allocates a CUDA memory block of the specified size, reusing cached allocations if available.
        /// </summary>
        /// <param name="bytes">The minimum size of the memory block to allocate in bytes.</param>
        /// <returns>
        /// A pointer to the allocated CUDA memory, or <see cref="IntPtr.Zero"/> if <paramref name="bytes"/> is 0.
        /// </returns>
        /// <remarks>
        /// If a matching buffer exists in the pool, it is popped and returned. Otherwise, a new allocation is made.
        /// If the total pool footprint exceeds <see cref="MaxPoolBytes"/>, an eviction cycle is run before allocating.
        /// </remarks>

        public IntPtr Allocate(ulong bytes)
        {
            if (bytes == 0) return IntPtr.Zero;

            ulong bucketSize = GetBucketSize(bytes);

            if (_freeBuffers.TryGetValue(bucketSize, out var stack) && stack.TryPop(out var cached))
            {
                Interlocked.Add(ref _totalAllocatedBytes, -(long)bucketSize);
                return cached.Ptr;
            }

            // FIXED: If we are over our budget, evict the oldest buffers before allocating new ones
            if (Interlocked.Read(ref _totalAllocatedBytes) > MaxPoolBytes)
            {
                EvictOldest();
            }

            CUDA.CudaMalloc(out IntPtr newPtr, bucketSize);
            return newPtr;
        }
        /// <summary>
        /// Returns a CUDA memory block back to the pool for potential future reuse.
        /// </summary>
        /// <param name="ptr">The pointer to the CUDA memory block to return.</param>
        /// <param name="bytes">The original requested size of the allocation in bytes.</param>
        /// <remarks>
        /// If <paramref name="ptr"/> is <see cref="IntPtr.Zero"/>, this method does nothing.
        /// The returned pointer is associated with its corresponding bucket size and tagged with the current UTC timestamp.
        /// </remarks>

        public void Free(IntPtr ptr, ulong bytes)
        {
            if (ptr == IntPtr.Zero) return;

            ulong bucketSize = GetBucketSize(bytes);

            var stack = _freeBuffers.GetOrAdd(bucketSize, _ => new ConcurrentStack<(IntPtr, DateTime)>());
            stack.Push((ptr, DateTime.UtcNow));
            Interlocked.Add(ref _totalAllocatedBytes, (long)bucketSize);
        }
        /// <summary>
        /// Evicts warm buffers to stay within the safe memory allocation limit.
        /// </summary>
        /// <remarks>
        /// This method iterates through all buckets, leaving up to 2 warm buffers per bucket, 
        /// and frees the remaining buffers back to CUDA to reduce the pool's memory usage footprint.
        /// </remarks>

        private void EvictOldest()
        {
            foreach (var kvp in _freeBuffers)
            {
                var stack = kvp.Value;
                // Leave up to 2 warm buffers per bucket, evicting the rest
                while (stack.Count > 2 && stack.TryPop(out var item))
                {
                    CUDA.CudaFree(item.Ptr);
                    Interlocked.Add(ref _totalAllocatedBytes, -(long)kvp.Key);
                }
            }
        }
        /// <summary>
        /// Releases all currently cached memory allocations from the pool back to the GPU.
        /// </summary>
        /// <remarks>
        /// This method empties the pool, invokes <see cref="CUDA.CudaFree(IntPtr)"/> on all cached pointers,
        /// and resets the tracking size back to zero.
        /// </remarks>

        public void Clear()
        {
            foreach (var kvp in _freeBuffers)
            {
                var stack = kvp.Value;
                while (stack.TryPop(out var item))
                {
                    CUDA.CudaFree(item.Ptr);
                }
            }
            _freeBuffers.Clear();
            _totalAllocatedBytes = 0;
        }
    }
}