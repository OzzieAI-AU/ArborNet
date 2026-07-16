using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using ArborNet.Core.Native.PInvoke;

namespace ArborNet.Core.Native
{
    public sealed class CudaMemoryPool
    {
        private static readonly Lazy<CudaMemoryPool> _instance =
            new Lazy<CudaMemoryPool>(() => new CudaMemoryPool());

        public static CudaMemoryPool Instance => _instance.Value;

        // Buckets of freed memory pointers keyed by byte size
        private readonly ConcurrentDictionary<ulong, ConcurrentStack<IntPtr>> _freeBuffers =
            new ConcurrentDictionary<ulong, ConcurrentStack<IntPtr>>();

        private CudaMemoryPool() { }

        /// <summary>
        /// Rounds allocations to power-of-2 buckets to improve cache hit rates.
        /// </summary>
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

        public IntPtr Allocate(ulong bytes)
        {
            if (bytes == 0) return IntPtr.Zero;

            ulong bucketSize = GetBucketSize(bytes);

            if (_freeBuffers.TryGetValue(bucketSize, out var stack) && stack.TryPop(out IntPtr cachedPtr))
            {
                return cachedPtr;
            }

            // Fallback to CUDA allocation
            CUDA.CudaMalloc(out IntPtr newPtr, bucketSize);
            return newPtr;
        }

        public void Free(IntPtr ptr, ulong bytes)
        {
            if (ptr == IntPtr.Zero) return;

            ulong bucketSize = GetBucketSize(bytes);

            var stack = _freeBuffers.GetOrAdd(bucketSize, _ => new ConcurrentStack<IntPtr>());
            stack.Push(ptr);
        }

        public void Clear()
        {
            foreach (var kvp in _freeBuffers)
            {
                var stack = kvp.Value;
                while (stack.TryPop(out IntPtr ptr))
                {
                    CUDA.CudaFree(ptr);
                }
            }
            _freeBuffers.Clear();
        }
    }
}