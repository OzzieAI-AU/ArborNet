namespace ArborNet.Data
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;
    using System.Text.RegularExpressions;
    using ArborNet.Core.Interfaces;

    public readonly struct PrimeGapCoordinate
    {
        public readonly int TokenId;
        public readonly long Prime;
        public readonly long Gap;
        public readonly float NormalizedRatio;

        public PrimeGapCoordinate(int tokenId, long prime, long gap)
        {
            TokenId = tokenId;
            Prime = prime;
            Gap = gap;
            NormalizedRatio = (prime > 1 && gap > 0)
                ? (MathF.Log(gap) / MathF.Log(prime))
                : 0f;
        }
    }

    public sealed class PrimeGapTokenizer : ITokenizer
    {
        private static readonly Regex TokenSplitRegex = new(@"\w+|[^\w\s]|\s+", RegexOptions.Compiled);

        private readonly Dictionary<string, int> _tokenToId;
        private readonly Dictionary<int, string> _idToToken;
        private readonly PrimeGapCoordinate[] _primeCoordinates;

        // ITokenizer requires mutable get/set accessors
        public int PadId { get; set; }
        public int UnkId { get; set; }

        public int VocabSize => _tokenToId.Count;
        public IReadOnlyList<PrimeGapCoordinate> PrimeCoordinates => _primeCoordinates;
        public Dictionary<string, int> VocabDictionary => _tokenToId;

        public PrimeGapTokenizer(Dictionary<string, int> vocab, int padId = 0, int unkId = 1)
        {
            _tokenToId = vocab;
            _idToToken = vocab.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
            PadId = padId;
            UnkId = unkId;

            _primeCoordinates = ComputePrimeGapCoordinates(_tokenToId.Count);
        }

        public List<int> Encode(string text)
        {
            if (string.IsNullOrEmpty(text)) return new List<int>();

            var matches = TokenSplitRegex.Matches(text);
            var ids = new List<int>(matches.Count);

            foreach (Match m in matches)
            {
                ids.Add(_tokenToId.TryGetValue(m.Value, out int id) ? id : UnkId);
            }
            return ids;
        }

        public string Decode(List<int> tokens)
        {
            var sb = new StringBuilder();
            foreach (var id in tokens)
            {
                if (_idToToken.TryGetValue(id, out var word))
                {
                    sb.Append(word);
                }
                else
                {
                    sb.Append("<unk>");
                }
            }
            return sb.ToString();
        }

        public PrimeGapCoordinate GetCoordinate(int tokenId)
        {
            if ((uint)tokenId >= (uint)_primeCoordinates.Length)
                return _primeCoordinates[UnkId];
            return _primeCoordinates[tokenId];
        }

        private static PrimeGapCoordinate[] ComputePrimeGapCoordinates(int count)
        {
            var coords = new PrimeGapCoordinate[count];
            var primes = GeneratePrimes(count + 1);

            for (int i = 0; i < count; i++)
            {
                long currentPrime = primes[i];
                long nextPrime = primes[i + 1];
                long gap = nextPrime - currentPrime;
                coords[i] = new PrimeGapCoordinate(i, currentPrime, gap);
            }

            return coords;
        }

        private static List<long> GeneratePrimes(int n)
        {
            var primes = new List<long>(n);
            if (n <= 0) return primes;

            primes.Add(2);
            long candidate = 3;

            while (primes.Count < n)
            {
                bool isPrime = true;
                long sqrt = (long)Math.Sqrt(candidate);

                for (int i = 0; i < primes.Count; i++)
                {
                    long p = primes[i];
                    if (p > sqrt) break;
                    if (candidate % p == 0)
                    {
                        isPrime = false;
                        break;
                    }
                }

                if (isPrime)
                {
                    primes.Add(candidate);
                }
                candidate += 2;
            }

            return primes;
        }

        public static PrimeGapTokenizer BuildAndSave(IEnumerable<string> lines, string vocabPath, int targetVocabSize)
        {
            string? dir = Path.GetDirectoryName(vocabPath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }

            var vocab = new Dictionary<string, int>
            {
                ["<pad>"] = 0,
                ["<unk>"] = 1,
                ["<s>"] = 2,
                ["</s>"] = 3
            };

            var freq = new Dictionary<string, int>();
            foreach (var line in lines)
            {
                foreach (Match m in TokenSplitRegex.Matches(line))
                {
                    freq[m.Value] = freq.GetValueOrDefault(m.Value, 0) + 1;
                }
            }

            foreach (var token in freq.OrderByDescending(kv => kv.Value).Select(kv => kv.Key))
            {
                if (!vocab.ContainsKey(token))
                {
                    vocab[token] = vocab.Count;
                    if (vocab.Count >= targetVocabSize)
                        break;
                }
            }

            using (var vw = new StreamWriter(vocabPath, false, Encoding.UTF8))
            {
                foreach (var kv in vocab.OrderBy(k => k.Value))
                {
                    vw.WriteLine($"{kv.Key}\t{kv.Value}");
                }
            }

            return new PrimeGapTokenizer(vocab);
        }
    }
}