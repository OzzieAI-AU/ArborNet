// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Native.Intrinsics
{

    #region Using Statements:

    using System;
    using System.Runtime.CompilerServices;
    using System.Runtime.Intrinsics;
    using System.Runtime.Intrinsics.X86;

    #endregion

    /// <summary>
    /// Provides highly optimized, hardware-accelerated SIMD (Single Instruction, Multiple Data) calculation engines.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This class features direct integration with Intel AVX-512 instruction registers, enabling processing of 
    /// up to 512 bits of data in a single clock cycle. It implements a multi-tiered fallback architecture 
    /// that gracefully degrades to AVX2, AVX, or SSE instruction sets depending on the capabilities of the host processor.
    /// </para>
    /// <para>
    /// If no hardware acceleration is supported by the target CPU, or if there are residual elements remaining at the 
    /// end of the vectorization loop (the "tail"), operations safely fall back to a standard sequential scalar loop.
    /// This design guarantees both maximum execution velocity on modern server architectures and safe fallback on legacy hardware.
    /// </para>
    /// </remarks>
    public static class AVX512Engine
    {
        /// <summary>
        /// Performs a vectorized, element-wise addition operation (<c>result[i] = left[i] + right[i]</c>) across float arrays.
        /// </summary>
        /// <param name="result">The destination array where the computed element-wise sums are written.</param>
        /// <param name="left">The first source array containing the left-hand operands.</param>
        /// <param name="right">The second source array containing the right-hand operands.</param>
        /// <param name="length">The total number of elements to process from the arrays.</param>
        /// <remarks>
        /// <para>
        /// This method leverages hardware acceleration through CPU SIMD registers. Depending on host architecture support:
        /// <list type="bullet">
        /// <item>
        /// <term>AVX-512 (512-bit)</term>
        /// <description>Processes 16 single-precision floats (64 bytes) simultaneously per iteration.</description>
        /// </item>
        /// <item>
        /// <term>AVX/AVX2 (256-bit)</term>
        /// <description>Processes 8 single-precision floats (32 bytes) simultaneously per iteration.</description>
        /// </item>
        /// <item>
        /// <term>SSE (128-bit)</term>
        /// <description>Processes 4 single-precision floats (16 bytes) simultaneously per iteration.</description>
        /// </item>
        /// </list>
        /// </para>
        /// <para>
        /// Any remaining elements that do not fit evenly into the largest supported hardware register size 
        /// are processed sequentially by a scalar fallback loop.
        /// </para>
        /// </remarks>
        /// <exception cref="NullReferenceException">
        /// Thrown when <paramref name="result"/>, <paramref name="left"/>, or <paramref name="right"/> is <see langword="null"/>.
        /// </exception>
        /// <exception cref="IndexOutOfRangeException">
        /// Thrown when <paramref name="length"/> is greater than the length of <paramref name="result"/>, 
        /// <paramref name="left"/>, or <paramref name="right"/>.
        /// </exception>
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
        /// ********************************************************************************
        /// <summary>
        /// Performs a vectorized, element-wise subtraction operation (<c>result[i] = left[i] - right[i]</c>) across float arrays.
        /// </summary>
        /// <param name="result">The destination array where the computed element-wise differences are written.</param>
        /// <param name="left">The source array containing the minuend operands.</param>
        /// <param name="right">The source array containing the subtrahend operands.</param>
        /// <param name="length">The total number of elements to process from the arrays.</param>
        /// <remarks>
        /// <para>
        /// This method leverages hardware acceleration through CPU SIMD registers. Depending on host architecture support:
        /// <list type="bullet">
        /// <item>
        /// <term>AVX-512 (512-bit)</term>
        /// <description>Processes 16 single-precision floats (64 bytes) simultaneously per iteration.</description>
        /// </item>
        /// <item>
        /// <term>AVX/AVX2 (256-bit)</term>
        /// <description>Processes 8 single-precision floats (32 bytes) simultaneously per iteration.</description>
        /// </item>
        /// <item>
        /// <term>SSE (128-bit)</term>
        /// <description>Processes 4 single-precision floats (16 bytes) simultaneously per iteration.</description>
        /// </item>
        /// </list>
        /// </para>
        /// <para>
        /// Any remaining elements that do not fit evenly into the largest supported hardware register size 
        /// are processed sequentially by a scalar fallback loop.
        /// </para>
        /// </remarks>
        /// <exception cref="NullReferenceException">
        /// Thrown when <paramref name="result"/>, <paramref name="left"/>, or <paramref name="right"/> is <see langword="null"/>.
        /// </exception>
        /// <exception cref="IndexOutOfRangeException">
        /// Thrown when <paramref name="length"/> is greater than the length of <paramref name="result"/>, 
        /// <paramref name="left"/>, or <paramref name="right"/>.
        /// </exception>

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
        /// ********************************************************************************
        /// <summary>
        /// Performs a vectorized, element-wise multiplication operation (<c>result[i] = left[i] * right[i]</c>) across float arrays.
        /// </summary>
        /// <param name="result">The destination array where the computed element-wise products are written.</param>
        /// <param name="left">The first source array containing the multiplicand operands.</param>
        /// <param name="right">The second source array containing the multiplier operands.</param>
        /// <param name="length">The total number of elements to process from the arrays.</param>
        /// <remarks>
        /// <para>
        /// This method leverages hardware acceleration through CPU SIMD registers. Depending on host architecture support:
        /// <list type="bullet">
        /// <item>
        /// <term>AVX-512 (512-bit)</term>
        /// <description>Processes 16 single-precision floats (64 bytes) simultaneously per iteration.</description>
        /// </item>
        /// <item>
        /// <term>AVX/AVX2 (256-bit)</term>
        /// <description>Processes 8 single-precision floats (32 bytes) simultaneously per iteration.</description>
        /// </item>
        /// <item>
        /// <term>SSE (128-bit)</term>
        /// <description>Processes 4 single-precision floats (16 bytes) simultaneously per iteration.</description>
        /// </item>
        /// </list>
        /// </para>
        /// <para>
        /// Any remaining elements that do not fit evenly into the largest supported hardware register size 
        /// are processed sequentially by a scalar fallback loop.
        /// </para>
        /// </remarks>
        /// <exception cref="NullReferenceException">
        /// Thrown when <paramref name="result"/>, <paramref name="left"/>, or <paramref name="right"/> is <see langword="null"/>.
        /// </exception>
        /// <exception cref="IndexOutOfRangeException">
        /// Thrown when <paramref name="length"/> is greater than the length of <paramref name="result"/>, 
        /// <paramref name="left"/>, or <paramref name="right"/>.
        /// </exception>

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
        /// ********************************************************************************
        /// <summary>
        /// Computes a vectorized, hardware-accelerated Fused Multiply-Add (FMA) operation (<c>result[i] = (left[i] * multiplier[i]) + addend[i]</c>).
        /// </summary>
        /// <param name="result">The destination array where calculated values will be stored.</param>
        /// <param name="left">The source array containing factors to multiply.</param>
        /// <param name="multiplier">The source array containing multipliers to apply to the values in <paramref name="left"/>.</param>
        /// <param name="addend">The source array containing values to add to the computed product.</param>
        /// <param name="length">The total number of elements to process from the arrays.</param>
        /// <remarks>
        /// <para>
        /// This method leverages hardware Fused Multiply-Add (FMA) instructions. FMA executes the multiplication 
        /// and addition operations with a single rounding step. This significantly enhances calculation speed 
        /// and increases mathematical precision by preventing intermediate rounding errors.
        /// </para>
        /// <para>
        /// Depending on host architecture support:
        /// <list type="bullet">
        /// <item>
        /// <term>AVX-512 FMA</term>
        /// <description>Processes 16 single-precision floats simultaneously using 512-bit registers.</description>
        /// </item>
        /// <item>
        /// <term>AVX2 FMA (FMA3)</term>
        /// <description>Processes 8 single-precision floats simultaneously using 256-bit registers.</description>
        /// </item>
        /// </list>
        /// </para>
        /// <para>
        /// Any remaining elements that do not fit evenly into the largest supported hardware register size 
        /// are processed sequentially by a scalar fallback loop.
        /// </para>
        /// </remarks>
        /// <exception cref="NullReferenceException">
        /// Thrown when <paramref name="result"/>, <paramref name="left"/>, <paramref name="multiplier"/>, or <paramref name="addend"/> is <see langword="null"/>.
        /// </exception>
        /// <exception cref="IndexOutOfRangeException">
        /// Thrown when <paramref name="length"/> is greater than the length of any of the provided arrays.
        /// </exception>

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