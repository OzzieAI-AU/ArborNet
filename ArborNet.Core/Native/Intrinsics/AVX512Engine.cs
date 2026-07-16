using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace ArborNet.Core.Native.Intrinsics
{
    /// <summary>
    /// Highly optimized hardware SIMD calculation engine.
    /// Direct integration with Intel AVX-512 FMA instruction registers with 
    /// safe fallback tiers to AVX2, AVX, or SSE when run on standard consumer processors.
    /// </summary>
    public static class AVX512Engine
    {
        /// <summary>
        /// Performs vectorized element-wise addition (result = left + right) across floats.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static unsafe void Add(float[] result, float[] left, float[] right, int length)
        {
            int i = 0;

            // AVX-512 Path (512-bit width: Processes 16 floats simultaneously)
            if (Avx512F.IsSupported)
            {
                fixed (float* pL = left, pR = right, pRes = result)
                {
                    for (; i <= length - 16; i += 16)
                    {
                        var vecL = Vector512.Load(pL + i);
                        var vecR = Vector512.Load(pR + i);
                        var sum = vecL + vecR;
                        sum.Store(pRes + i);
                    }
                }
            }
            // AVX2 / AVX Path Fallback (256-bit width: Processes 8 floats simultaneously)
            else if (Avx.IsSupported)
            {
                fixed (float* pL = left, pR = right, pRes = result)
                {
                    for (; i <= length - 8; i += 8)
                    {
                        var vecL = Vector256.Load(pL + i);
                        var vecR = Vector256.Load(pR + i);
                        var sum = vecL + vecR;
                        sum.Store(pRes + i);
                    }
                }
            }
            // SSE Fallback (128-bit width: Processes 4 floats simultaneously)
            else if (Sse.IsSupported)
            {
                fixed (float* pL = left, pR = right, pRes = result)
                {
                    for (; i <= length - 4; i += 4)
                    {
                        var vecL = Vector128.Load(pL + i);
                        var vecR = Vector128.Load(pR + i);
                        var sum = vecL + vecR;
                        sum.Store(pRes + i);
                    }
                }
            }

            // Scalar loop handles residual tails
            for (; i < length; i++)
            {
                result[i] = left[i] + right[i];
            }
        }

        /// <summary>
        /// Performs vectorized element-wise subtraction (result = left - right).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static unsafe void Subtract(float[] result, float[] left, float[] right, int length)
        {
            int i = 0;

            if (Avx512F.IsSupported)
            {
                fixed (float* pL = left, pR = right, pRes = result)
                {
                    for (; i <= length - 16; i += 16)
                    {
                        var vecL = Vector512.Load(pL + i);
                        var vecR = Vector512.Load(pR + i);
                        var diff = vecL - vecR;
                        diff.Store(pRes + i);
                    }
                }
            }
            else if (Avx.IsSupported)
            {
                fixed (float* pL = left, pR = right, pRes = result)
                {
                    for (; i <= length - 8; i += 8)
                    {
                        var vecL = Vector256.Load(pL + i);
                        var vecR = Vector256.Load(pR + i);
                        var diff = vecL - vecR;
                        diff.Store(pRes + i);
                    }
                }
            }
            else if (Sse.IsSupported)
            {
                fixed (float* pL = left, pR = right, pRes = result)
                {
                    for (; i <= length - 4; i += 4)
                    {
                        var vecL = Vector128.Load(pL + i);
                        var vecR = Vector128.Load(pR + i);
                        var diff = vecL - vecR;
                        diff.Store(pRes + i);
                    }
                }
            }

            for (; i < length; i++)
            {
                result[i] = left[i] - right[i];
            }
        }

        /// <summary>
        /// Performs vectorized element-wise multiplication (result = left * right).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static unsafe void Multiply(float[] result, float[] left, float[] right, int length)
        {
            int i = 0;

            if (Avx512F.IsSupported)
            {
                fixed (float* pL = left, pR = right, pRes = result)
                {
                    for (; i <= length - 16; i += 16)
                    {
                        var vecL = Vector512.Load(pL + i);
                        var vecR = Vector512.Load(pR + i);
                        var prod = vecL * vecR;
                        prod.Store(pRes + i);
                    }
                }
            }
            else if (Avx.IsSupported)
            {
                fixed (float* pL = left, pR = right, pRes = result)
                {
                    for (; i <= length - 8; i += 8)
                    {
                        var vecL = Vector256.Load(pL + i);
                        var vecR = Vector256.Load(pR + i);
                        var prod = vecL * vecR;
                        prod.Store(pRes + i);
                    }
                }
            }
            else if (Sse.IsSupported)
            {
                fixed (float* pL = left, pR = right, pRes = result)
                {
                    for (; i <= length - 4; i += 4)
                    {
                        var vecL = Vector128.Load(pL + i);
                        var vecR = Vector128.Load(pR + i);
                        var prod = vecL * vecR;
                        prod.Store(pRes + i);
                    }
                }
            }

            for (; i < length; i++)
            {
                result[i] = left[i] * right[i];
            }
        }

        /// <summary>
        /// Computes vectorized Fused Multiply-Add operation (result = (left * multiplier) + addend).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static unsafe void FusedMultiplyAdd(float[] result, float[] left, float[] multiplier, float[] addend, int length)
        {
            int i = 0;

            // AVX-512 FMA Integration
            if (Avx512F.IsSupported)
            {
                fixed (float* pL = left, pM = multiplier, pA = addend, pRes = result)
                {
                    for (; i <= length - 16; i += 16)
                    {
                        var vecL = Vector512.Load(pL + i);
                        var vecM = Vector512.Load(pM + i);
                        var vecA = Vector512.Load(pA + i);

                        // Fused evaluation: result = (L * M) + A
                        var fma = Vector512.FusedMultiplyAdd(vecL, vecM, vecA);
                        fma.Store(pRes + i);
                    }
                }
            }
            // AVX2 FMA Fallback
            else if (Fma.IsSupported)
            {
                fixed (float* pL = left, pM = multiplier, pA = addend, pRes = result)
                {
                    for (; i <= length - 8; i += 8)
                    {
                        var vecL = Vector256.Load(pL + i);
                        var vecM = Vector256.Load(pM + i);
                        var vecA = Vector256.Load(pA + i);

                        var fma = Vector256.FusedMultiplyAdd(vecL, vecM, vecA);
                        fma.Store(pRes + i);
                    }
                }
            }

            // Scalar tail fallback
            for (; i < length; i++)
            {
                result[i] = (left[i] * multiplier[i]) + addend[i];
            }
        }
    }
}