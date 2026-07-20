// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Initializers
{

    #region Using Statements:

    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    /// <summary>
    /// Provides static factory methods for generating structured pseudo-random noise matrices
    /// based on various mathematical and chaotic sequences (fractals).
    /// </summary>
    /// <remarks>
    /// These initializers generate deterministic, structured patterns that can be used as weight
    /// initializers in neural networks, offering an alternative to standard Gaussian or Uniform distributions.
    /// The generated values are automatically scaled using a Xavier/Glorot-like scaling factor based on the column dimensions.
    /// </remarks>

    #endregion


    public static class FractalInitializers
    {
        /// <summary>
        /// Generates a two-dimensional <see cref="ITensor"/> filled with structured pseudo-random values
        /// derived from a specified mathematical fractal sequence, scaled for neural network initialization.
        /// </summary>
        /// <remarks>
        /// The generated values from the fractal pool are scaled by a factor of <c>sqrt(2 / cols)</c> to help sustain 
        /// stable variance of activation levels across layers, analogous to He/Xavier initialization schemes.
        /// </remarks>
        /// <param name="rows">The number of rows in the output tensor.</param>
        /// <param name="cols">The number of columns in the output tensor.</param>
        /// <param name="type">The type of fractal or mathematical sequence to use for generation.</param>
        /// <param name="device">The hardware device on which the tensor should be allocated. Defaults to <see langword="null"/>.</param>
        /// <returns>An initialized <see cref="ITensor"/> containing the generated fractal-based weights.</returns>
        public static ITensor Generate(int rows, int cols, FractalType type, Device device = null)
        {
            int size = rows * cols;
            float scale = (float)Math.Sqrt(2.0 / cols);
            List<float> pool = GetFractalPool(size, type);

            float[] flatData = new float[size];
            for (int i = 0; i < size; i++)
            {
                flatData[i] = pool[i] * scale;
            }

            return Tensor.FromArray(flatData, new TensorShape(new int[] { rows, cols }));
        }
        /// <summary>
        /// Retrieves a raw list of float values representing the unscaled fractal pool
        /// of the specified type and size.
        /// </summary>
        /// <param name="size">The total number of elements required in the pool.</param>
        /// <param name="type">The type of fractal algorithm to execute.</param>
        /// <returns>A list of float values containing the raw sequence values.</returns>

        private static List<float> GetFractalPool(int size, FractalType type)
        {
            switch (type)
            {
                case FractalType.PrimeGapSignature: return GetPrimeGapPool(size);
                case FractalType.MandelbrotBifurcation: return GetMandelbrotPool(size);
                case FractalType.CantorDustSieve: return GetCantorDustPool(size);
                case FractalType.CollatzSequence: return GetCollatzPool(size);
                case FractalType.GoldenRatioPhase:
                default:
                    return GetGoldenRatioPool(size);
            }
        }
        /// <summary>
        /// Generates a pool of values based on the mathematical gaps between consecutive prime numbers.
        /// </summary>
        /// <remarks>
        /// This method finds consecutive prime numbers starting from 2, computes the gap between them,
        /// and applies a trigonometric transformation (<c>Math.Sin(gap) * Math.Cos(gap * Math.PI / 4.0)</c>)
        /// to project the gaps into a bounded pseudo-random space.
        /// </remarks>
        /// <param name="size">The total number of elements to generate.</param>
        /// <returns>A list of float values representing the prime gap signature sequence.</returns>

        private static List<float> GetPrimeGapPool(int size)
        {
            List<int> primes = new List<int>();
            List<float> pool = new List<float>(size);
            int num = 2;
            while (pool.Count < size)
            {
                bool isPrime = true;
                for (int i = 2; i * i <= num; i++)
                    if (num % i == 0) { isPrime = false; break; }

                if (isPrime)
                {
                    primes.Add(num);
                    if (primes.Count > 1)
                    {
                        int gap = primes[primes.Count - 1] - primes[primes.Count - 2];
                        pool.Add((float)(Math.Sin(gap) * Math.Cos(gap * Math.PI / 4.0)));
                    }
                }
                num++;
            }
            return pool;
        }
        /// <summary>
        /// Generates a pool of values based on the escape velocity iterations of the Mandelbrot set bifurcation.
        /// </summary>
        /// <remarks>
        /// Maps a 2D grid onto the complex plane coordinates, computes the Mandelbrot iteration count (up to 100 iterations)
        /// for each point, and applies a hyperbolic tangent scaling to normalize the values between -1.0 and 1.0.
        /// </remarks>
        /// <param name="size">The total number of elements to generate.</param>
        /// <returns>A list of float values representing the normalized Mandelbrot bifurcation sequence.</returns>

        private static List<float> GetMandelbrotPool(int size)
        {
            List<float> pool = new List<float>(size);
            int side = (int)Math.Ceiling(Math.Sqrt(size));
            for (int x = 0; x < side; x++)
            {
                for (int y = 0; y < side; y++)
                {
                    double cr = -2.0 + (x * 3.0 / side);
                    double ci = -1.5 + (y * 3.0 / side);
                    double zr = 0, zi = 0;
                    int iter = 0;
                    while (zr * zr + zi * zi <= 4.0 && iter < 100)
                    {
                        double temp = zr * zr - zi * zi + cr;
                        zi = 2.0 * zr * zi + ci;
                        zr = temp;
                        iter++;
                    }
                    pool.Add((float)Math.Tanh((iter / 100.0) * 2.0 - 1.0));
                    if (pool.Count == size) return pool;
                }
            }
            return pool;
        }
        /// <summary>
        /// Generates a pool of values using a ternary-based Cantor Dust sieve simulation.
        /// </summary>
        /// <remarks>
        /// Iteratively removes the middle third of the interval (classic Cantor set construction) up to 6 levels of recursion.
        /// Points remaining in the Cantor set are assigned a high value (0.85f), while removed points are assigned a low value (-0.85f).
        /// </remarks>
        /// <param name="size">The total number of elements to generate.</param>
        /// <returns>A list of float values representing the binary Cantor Dust density sequence.</returns>

        private static List<float> GetCantorDustPool(int size)
        {
            List<float> pool = new List<float>(size);
            for (int i = 0; i < size; i++)
            {
                double val = (double)i / size;
                bool inDust = true;
                for (int level = 0; level < 6; level++)
                {
                    double ternaryDigit = Math.Floor(val * 3.0);
                    if (ternaryDigit == 1) { inDust = false; break; }
                    val = (val * 3.0) - ternaryDigit;
                }
                pool.Add(inDust ? 0.85f : -0.85f);
            }
            return pool;
        }
        /// <summary>
        /// Generates a pool of values based on the fractional part of successive multiples of the Golden Ratio.
        /// </summary>
        /// <remarks>
        /// Uses the low-discrepancy Weyl sequence based on the Golden Ratio (<c>phi = 1.618033988749895</c>)
        /// to distribute values evenly across a bounded interval, mapped to the range [-1.0, 1.0].
        /// </remarks>
        /// <param name="size">The total number of elements to generate.</param>
        /// <returns>A list of float values representing the Golden Ratio phase distribution.</returns>

        private static List<float> GetGoldenRatioPool(int size)
        {
            List<float> pool = new List<float>(size);
            double phi = 1.618033988749895;
            for (int i = 0; i < size; i++)
            {
                double val = (i * phi) - Math.Floor(i * phi);
                pool.Add((float)((val * 2.0) - 1.0));
            }
            return pool;
        }
        /// <summary>
        /// Generates a pool of values based on the number of steps required to reach 1 in the Collatz Conjecture sequence.
        /// </summary>
        /// <remarks>
        /// For each integer from 1 to <paramref name="size"/>, the Collatz (3n + 1) sequence is evaluated to count the
        /// number of halving/tripling steps needed to reach the value 1. The sine of the total step count is stored to 
        /// ensure a balanced trigonometric distribution.
        /// </remarks>
        /// <param name="size">The total number of elements to generate.</param>
        /// <returns>A list of float values representing the Collatz sequence progression.</returns>

        private static List<float> GetCollatzPool(int size)
        {
            List<float> pool = new List<float>(size);
            for (int i = 1; i <= size; i++)
            {
                long n = i;
                int steps = 0;
                while (n > 1)
                {
                    if (n % 2 == 0) n /= 2;
                    else n = (3 * n) + 1;
                    steps++;
                }
                pool.Add((float)Math.Sin(steps));
            }
            return pool;
        }
    }
}