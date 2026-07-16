namespace ArborNet.Core.Initializers
{

    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;


    /// <summary>
    /// Generates structured pseudo-random noise matrices based on mathematical sequences.
    /// Outputs native ArborNet ITensors.
    /// </summary>
    public static class FractalInitializers
    {
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

            return Tensor.FromArray(flatData, new TensorShape(new int[] { rows, cols } ));
        }

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