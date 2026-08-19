// -----------------------------------------------------------------------------------------
// Project:      ArborNet
// Description:  LLM Token Sampling (Temperature, Top-K, Top-P)
// -----------------------------------------------------------------------------------------

namespace ArborNet.Generation
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;

    public static class Sampler
    {
        private static readonly Random _rng = new Random();

        /// <summary>
        /// Samples the next token from a 1D logits tensor using Temperature, Top-K, and Top-P.
        /// </summary>
        public static int SampleToken(ITensor logits, float temperature = 0.7f, int topK = 50, float topP = 0.9f)
        {
            if (logits.Shape.Rank != 1)
                throw new ArgumentException("Logits must be a 1D tensor representing vocabulary scores.");

            float[] logitsData = logits.ToArray();
            int vocabSize = logitsData.Length;

            // 1. Apply Temperature
            if (temperature <= 0f)
            {
                // Greedy fallback if temperature is 0
                return Array.IndexOf(logitsData, logitsData.Max());
            }

            for (int i = 0; i < vocabSize; i++)
            {
                logitsData[i] /= temperature;
            }

            // 2. Compute Softmax to get probabilities
            float maxLogit = logitsData.Max();
            double sumExp = 0.0;
            double[] probs = new double[vocabSize];

            for (int i = 0; i < vocabSize; i++)
            {
                probs[i] = Math.Exp(logitsData[i] - maxLogit);
                sumExp += probs[i];
            }
            for (int i = 0; i < vocabSize; i++)
            {
                probs[i] /= sumExp;
            }

            // 3. Sort probabilities for Top-K / Top-P
            var sortedProbs = probs
                .Select((p, idx) => new { Prob = p, Index = idx })
                .OrderByDescending(x => x.Prob)
                .ToList();

            // 4. Apply Top-K
            if (topK > 0 && topK < vocabSize)
            {
                sortedProbs = sortedProbs.Take(topK).ToList();
            }

            // 5. Apply Top-P (Nucleus Sampling)
            if (topP > 0f && topP < 1.0f)
            {
                double cumulativeProb = 0.0;
                int cutoffIndex = sortedProbs.Count;
                for (int i = 0; i < sortedProbs.Count; i++)
                {
                    cumulativeProb += sortedProbs[i].Prob;
                    if (cumulativeProb > topP)
                    {
                        cutoffIndex = i + 1;
                        break;
                    }
                }
                sortedProbs = sortedProbs.Take(cutoffIndex).ToList();
            }

            // 6. Re-normalize probabilities after slicing
            double newSum = sortedProbs.Sum(x => x.Prob);
            double randomThreshold = _rng.NextDouble() * newSum;

            // 7. Sample from the distribution
            double cumulative = 0.0;
            foreach (var item in sortedProbs)
            {
                cumulative += item.Prob;
                if (randomThreshold <= cumulative)
                {
                    return item.Index;
                }
            }

            return sortedProbs.Last().Index; // Fallback
        }
    }
}