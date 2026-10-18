namespace ArborNet.Data
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
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
        private readonly Dictionary<string, int> _tokenToId;
        private readonly Dictionary<int, string> _idToToken;
        private readonly PrimeGapCoordinate[] _primeCoordinates;

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

            var tokens = TokenizeSpan(text.AsSpan());
            var ids = new List<int>(tokens.Count);

            foreach (var token in tokens)
            {
                ids.Add(_tokenToId.TryGetValue(token, out int id) ? id : UnkId);
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

        /// <summary>
        /// Ultra-fast, zero-allocation Span-based scanner replacing Regex entirely.
        /// Matches \w+ | [^\w\s] | \s+
        /// </summary>
        private static List<string> TokenizeSpan(ReadOnlySpan<char> text)
        {
            var tokens = new List<string>();
            int i = 0;
            while (i < text.Length)
            {
                char c = text[i];
                if (char.IsWhiteSpace(c))
                {
                    int start = i;
                    while (i < text.Length && char.IsWhiteSpace(text[i])) i++;
                    tokens.Add(text.Slice(start, i - start).ToString());
                }
                else if (char.IsLetterOrDigit(c) || c == '_')
                {
                    int start = i;
                    while (i < text.Length && (char.IsLetterOrDigit(text[i]) || text[i] == '_')) i++;
                    tokens.Add(text.Slice(start, i - start).ToString());
                }
                else
                {
                    tokens.Add(text.Slice(i, 1).ToString());
                    i++;
                }
            }
            return tokens;
        }

        private static PrimeGapCoordinate[] ComputePrimeGapCoordinates(int count)
        {
            var coords = new PrimeGapCoordinate[count];
            var primes = GeneratePrimesFast(count + 1);

            for (int i = 0; i < count; i++)
            {
                long currentPrime = primes[i];
                long nextPrime = primes[i + 1];
                long gap = nextPrime - currentPrime;
                coords[i] = new PrimeGapCoordinate(i, currentPrime, gap);
            }

            return coords;
        }

        private static List<long> GeneratePrimesFast(int n)
        {
            var primes = new List<long>(n);
            if (n <= 0) return primes;
            if (n == 1) { primes.Add(2); return primes; }

            int limit;
            if (n < 6)
            {
                limit = 15;
            }
            else
            {
                double nd = n;
                limit = (int)(nd * (Math.Log(nd) + Math.Log(Math.Log(nd)))) + 128;
            }

            bool[] isPrime = new bool[limit + 1];
            Array.Fill(isPrime, true, 2, limit - 1);

            for (long p = 2; p * p <= limit; p++)
            {
                if (isPrime[p])
                {
                    for (long i = p * p; i <= limit; i += p)
                        isPrime[i] = false;
                }
            }

            primes.Add(2);
            for (int i = 3; i <= limit && primes.Count < n; i += 2)
            {
                if (isPrime[i]) primes.Add(i);
            }

            long candidate = limit % 2 == 0 ? limit + 1 : limit + 2;
            while (primes.Count < n)
            {
                bool prime = true;
                long sqrt = (long)Math.Sqrt(candidate);
                for (int i = 0; i < primes.Count; i++)
                {
                    long p = primes[i];
                    if (p > sqrt) break;
                    if (candidate % p == 0) { prime = false; break; }
                }
                if (prime) primes.Add(candidate);
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

            // Thread-local dictionaries to completely eliminate concurrent dictionary lock contention
            var mergedFreq = new Dictionary<string, int>(StringComparer.Ordinal);

            Parallel.ForEach(
                lines,
                () => new Dictionary<string, int>(StringComparer.Ordinal),
                (line, _, localDict) =>
                {
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        var tokens = TokenizeSpan(line.AsSpan());
                        foreach (var t in tokens)
                        {
                            localDict[t] = localDict.GetValueOrDefault(t, 0) + 1;
                        }
                    }
                    return localDict;
                },
                localDict =>
                {
                    lock (mergedFreq)
                    {
                        foreach (var kvp in localDict)
                        {
                            mergedFreq[kvp.Key] = mergedFreq.GetValueOrDefault(kvp.Key, 0) + kvp.Value;
                        }
                    }
                }
            );

            var sortedTokens = mergedFreq
                .OrderByDescending(kv => kv.Value)
                .Select(kv => kv.Key);

            foreach (var token in sortedTokens)
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