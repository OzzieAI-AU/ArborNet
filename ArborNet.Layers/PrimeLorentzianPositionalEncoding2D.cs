namespace ArborNet.Layers
{
    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Tensors;

    /// <summary>
    /// Pure 2D Tensor Lorentzian Prime-Gap Positional Encoding.
    /// Injects hyperbolic geometry into sequence features [T, D] using fixed non-Euclidean phase shifts.
    /// </summary>
    public sealed class PrimeLorentzianPositionalEncoding2D : BaseLayer
    {
        private readonly int _dModel;
        private readonly int _maxLen;
        private readonly ITensor _peTensor;
        private readonly ITensor _scale;

        public ITensor Table => _peTensor;

        public PrimeLorentzianPositionalEncoding2D(
            int dModel,
            int maxLen = 512,
            double hausdorffDim = 1.2618595,
            Device? device = null)
        {
            _dModel = dModel;
            _maxLen = maxLen;
            device ??= Device.CPU;
            this.device = device;

            _peTensor = PrecomputeHyperbolicGrid(maxLen, dModel, hausdorffDim, device);

            // Learnable scale initialized to 0.1f to prevent hyperbolic features from drowning embeddings
            _scale = Tensor.FromScalar(0.1f, device);
            _scale.RequiresGrad = true;
        }

        private static ITensor PrecomputeHyperbolicGrid(int maxLen, int dModel, double hausdorff, Device device)
        {
            float[] pe = new float[maxLen * dModel];
            var primes = SievePrimes(maxLen + 2);

            for (int pos = 0; pos < maxLen; pos++)
            {
                long pCurrent = primes[pos];
                long pNext = primes[pos + 1];
                long gap = pNext - pCurrent;

                double normGap = Math.Log(Math.Max(1.0, (double)gap));
                double normP = Math.Log(Math.Max(2.0, (double)pCurrent));
                double primeRatio = normGap / normP;

                float rowEnergy = 0f;
                int offset = pos * dModel;

                for (int d = 0; d < dModel; d++)
                {
                    double freq = Math.Pow(d + 1, hausdorff / 2.0);
                    double phase = ((pos + 1) * freq) % (2.0 * Math.PI);
                    float val = (float)(primeRatio * Math.Cos(phase));

                    pe[offset + d] = val;
                    rowEnergy += val * val;
                }

                // Strictly standardize row magnitude to unit variance
                float rms = MathF.Sqrt(rowEnergy / dModel + 1e-6f);
                for (int d = 0; d < dModel; d++)
                {
                    pe[offset + d] /= rms;
                }
            }

            var tensor = Tensor.FromArray(pe, new TensorShape(maxLen, dModel), device);
            tensor.RequiresGrad = false;
            return tensor;
        }

        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input);
            int seqLen = input.Shape[0];

            if (seqLen > _maxLen)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(input),
                    $"Sequence length {seqLen} exceeds max precomputed Prime PE limit of {_maxLen}.");
            }

            var peSlice = (seqLen == _maxLen)
                ? _peTensor
                : _peTensor.Slice((0, seqLen, 1), (0, _dModel, 1));

            return input.Add(peSlice.Multiply(_scale));
        }

        public override IEnumerable<ITensor> Parameters()
        {
            yield return _scale;
        }

        private static List<long> SievePrimes(int n)
        {
            var primes = new List<long>(n) { 2 };
            long candidate = 3;

            while (primes.Count < n)
            {
                bool isPrime = true;
                long limit = (long)Math.Sqrt(candidate);

                for (int i = 0; i < primes.Count; i++)
                {
                    long p = primes[i];
                    if (p > limit) break;
                    if (candidate % p == 0)
                    {
                        isPrime = false;
                        break;
                    }
                }

                if (isPrime) primes.Add(candidate);
                candidate += 2;
            }

            return primes;
        }
    }
}