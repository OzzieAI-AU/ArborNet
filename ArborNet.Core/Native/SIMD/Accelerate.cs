using System;
using System.Runtime.Intrinsics;

namespace ArborNet.Core.Native.SIMD
{
    /// <summary>
    /// Provides high-performance SIMD-accelerated binary operations for arrays of single-precision floating-point numbers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This class uses <see cref="Vector128{T}"/> (128-bit SIMD vectors) to process four <see cref="float"/> values 
    /// in parallel where possible, falling back to scalar operations for any remaining elements.
    /// </para>
    /// <para>
    /// All methods assume that <paramref name="result"/>, <paramref name="left"/>, and <paramref name="right"/> 
    /// are valid arrays with at least <paramref name="length"/> elements, and that the arrays do not overlap in memory.
    /// </para>
    /// </remarks>
    public static class Accelerate
    {
        // Binary operations for float arrays using SIMD acceleration
        /// <summary>
        /// Performs element-wise addition of two float arrays using SIMD acceleration.
        /// </summary>
        /// <param name="result">The array that receives the element-wise sums.</param>
        /// <param name="left">The first input array.</param>
        /// <param name="right">The second input array.</param>
        /// <param name="length">The number of elements to process.</param>
        /// <remarks>
        /// Uses <see cref="Vector128.Add(Vector128{float}, Vector128{float})"/> to process four elements per iteration
        /// when possible. Remaining elements are processed using standard scalar addition.
        /// </remarks>
        public static void Add(float[] result, float[] left, float[] right, int length)
        {
            int i = 0;
            for (; i <= length - Vector128<float>.Count; i += Vector128<float>.Count)
            {
                Vector128<float> l = Vector128.LoadUnsafe(ref left[i]);
                Vector128<float> r = Vector128.LoadUnsafe(ref right[i]);
                Vector128<float> res = Vector128.Add(l, r);
                Vector128.StoreUnsafe(res, ref result[i]);
            }
            // Handle remaining elements scalarly
            for (; i < length; i++)
            {
                result[i] = left[i] + right[i];
            }
        }

        /// <summary>
        /// Performs element-wise subtraction of two float arrays using SIMD acceleration.
        /// </summary>
        /// <param name="result">The array that receives the element-wise differences.</param>
        /// <param name="left">The first input array (minuend).</param>
        /// <param name="right">The second input array (subtrahend).</param>
        /// <param name="length">The number of elements to process.</param>
        /// <remarks>
        /// Uses <see cref="Vector128.Subtract(Vector128{float}, Vector128{float})"/> to process four elements per iteration
        /// when possible. Remaining elements are processed using standard scalar subtraction.
        /// </remarks>
        public static void Subtract(float[] result, float[] left, float[] right, int length)
        {
            int i = 0;
            for (; i <= length - Vector128<float>.Count; i += Vector128<float>.Count)
            {
                Vector128<float> l = Vector128.LoadUnsafe(ref left[i]);
                Vector128<float> r = Vector128.LoadUnsafe(ref right[i]);
                Vector128<float> res = Vector128.Subtract(l, r);
                Vector128.StoreUnsafe(res, ref result[i]);
            }
            // Handle remaining elements scalarly
            for (; i < length; i++)
            {
                result[i] = left[i] - right[i];
            }
        }

        /// <summary>
        /// Performs element-wise multiplication of two float arrays using SIMD acceleration.
        /// </summary>
        /// <param name="result">The array that receives the element-wise products.</param>
        /// <param name="left">The first input array.</param>
        /// <param name="right">The second input array.</param>
        /// <param name="length">The number of elements to process.</param>
        /// <remarks>
        /// Uses <see cref="Vector128.Multiply(Vector128{float}, Vector128{float})"/> to process four elements per iteration
        /// when possible. Remaining elements are processed using standard scalar multiplication.
        /// </remarks>
        public static void Multiply(float[] result, float[] left, float[] right, int length)
        {
            int i = 0;
            for (; i <= length - Vector128<float>.Count; i += Vector128<float>.Count)
            {
                Vector128<float> l = Vector128.LoadUnsafe(ref left[i]);
                Vector128<float> r = Vector128.LoadUnsafe(ref right[i]);
                Vector128<float> res = Vector128.Multiply(l, r);
                Vector128.StoreUnsafe(res, ref result[i]);
            }
            // Handle remaining elements scalarly
            for (; i < length; i++)
            {
                result[i] = left[i] * right[i];
            }
        }

        /// <summary>
        /// Performs element-wise division of two float arrays using SIMD acceleration.
        /// </summary>
        /// <param name="result">The array that receives the element-wise quotients.</param>
        /// <param name="left">The first input array (dividend).</param>
        /// <param name="right">The second input array (divisor).</param>
        /// <param name="length">The number of elements to process.</param>
        /// <remarks>
        /// Uses <see cref="Vector128.Divide(Vector128{float}, Vector128{float})"/> to process four elements per iteration
        /// when possible. Remaining elements are processed using standard scalar division.
        /// </remarks>
        public static void Divide(float[] result, float[] left, float[] right, int length)
        {
            int i = 0;
            for (; i <= length - Vector128<float>.Count; i += Vector128<float>.Count)
            {
                Vector128<float> l = Vector128.LoadUnsafe(ref left[i]);
                Vector128<float> r = Vector128.LoadUnsafe(ref right[i]);
                Vector128<float> res = Vector128.Divide(l, r);
                Vector128.StoreUnsafe(res, ref result[i]);
            }
            // Handle remaining elements scalarly
            for (; i < length; i++)
            {
                result[i] = left[i] / right[i];
            }
        }

        /// <summary>
        /// Performs element-wise maximum of two float arrays using SIMD acceleration.
        /// </summary>
        /// <param name="result">The array that receives the element-wise maximum values.</param>
        /// <param name="left">The first input array.</param>
        /// <param name="right">The second input array.</param>
        /// <param name="length">The number of elements to process.</param>
        /// <remarks>
        /// Uses <see cref="Vector128.Max(Vector128{float}, Vector128{float})"/> to process four elements per iteration
        /// when possible. Remaining elements are processed using <see cref="Math.Max(float, float)"/>.
        /// </remarks>
        public static void Max(float[] result, float[] left, float[] right, int length)
        {
            int i = 0;
            for (; i <= length - Vector128<float>.Count; i += Vector128<float>.Count)
            {
                Vector128<float> l = Vector128.LoadUnsafe(ref left[i]);
                Vector128<float> r = Vector128.LoadUnsafe(ref right[i]);
                Vector128<float> res = Vector128.Max(l, r);
                Vector128.StoreUnsafe(res, ref result[i]);
            }
            // Handle remaining elements scalarly
            for (; i < length; i++)
            {
                result[i] = Math.Max(left[i], right[i]);
            }
        }

        /// <summary>
        /// Performs element-wise minimum of two float arrays using SIMD acceleration.
        /// </summary>
        /// <param name="result">The array that receives the element-wise minimum values.</param>
        /// <param name="left">The first input array.</param>
        /// <param name="right">The second input array.</param>
        /// <param name="length">The number of elements to process.</param>
        /// <remarks>
        /// Uses <see cref="Vector128.Min(Vector128{float}, Vector128{float})"/> to process four elements per iteration
        /// when possible. Remaining elements are processed using <see cref="Math.Min(float, float)"/>.
        /// </remarks>
        public static void Min(float[] result, float[] left, float[] right, int length)
        {
            int i = 0;
            for (; i <= length - Vector128<float>.Count; i += Vector128<float>.Count)
            {
                Vector128<float> l = Vector128.LoadUnsafe(ref left[i]);
                Vector128<float> r = Vector128.LoadUnsafe(ref right[i]);
                Vector128<float> res = Vector128.Min(l, r);
                Vector128.StoreUnsafe(res, ref result[i]);
            }
            // Handle remaining elements scalarly
            for (; i < length; i++)
            {
                result[i] = Math.Min(left[i], right[i]);
            }
        }

        // Additional SIMD operations can be added here as needed
    }
}