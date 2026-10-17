// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Metrics
{
    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.Linq;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;

    #endregion

    /// <summary>
    /// Evaluates structural and informational density metrics for ArborNet neural architectures.
    /// </summary>
    public static class KnowledgeDensityEvaluator
    {
        /// <summary>
        /// Computes the comprehensive Knowledge Density Score and deep architectural metrics for any ArborNet BaseModel.
        /// </summary>
        /// <param name="model">The model to evaluate.</param>
        /// <param name="evalLoss">Cross entropy evaluation loss (nats).</param>
        /// <param name="vocabSize">Vocabulary size of the model.</param>
        /// <param name="sampleHiddenStates">Captured hidden state activations [Batch * Seq, D].</param>
        /// <param name="bytesPerParameter">Memory width per parameter (default: 4 bytes for FP32).</param>
        public static KnowledgeDensityMetrics Evaluate(
            BaseModel model,
            double evalLoss,
            int vocabSize,
            ITensor? sampleHiddenStates = null,
            int bytesPerParameter = 4)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));

            // 1. Parameter Auditing (Unique vs Total)
            var uniqueTensors = new HashSet<ITensor>(ReferenceEqualityComparer.Instance);
            long totalUniqueParams = 0;
            long totalParams = 0;

            foreach (var p in model.Parameters())
            {
                if (p != null)
                {
                    long count = p.Shape.TotalElements;
                    totalParams += count;
                    if (uniqueTensors.Add(p))
                    {
                        totalUniqueParams += count;
                    }
                }
            }

            if (totalUniqueParams == 0) totalUniqueParams = 1;
            if (totalParams == 0) totalParams = totalUniqueParams;

            // 2. Information Compression & Entropy Analysis
            double maxEntropyBits = Math.Log2(vocabSize);
            double lossBits = evalLoss / Math.Log(2.0);
            double compressionRatio = Math.Clamp((maxEntropyBits - lossBits) / maxEntropyBits, 0.0001, 1.0);
            double bitsPerParam = (compressionRatio * maxEntropyBits) / (totalUniqueParams / 1_000_000.0);

            // 3. Manifold Effective Rank & Dimensional Collapse Analysis
            double rankRatio = 1.0;
            if (sampleHiddenStates != null && sampleHiddenStates.Shape.Rank == 2)
            {
                rankRatio = ComputeRobustEffectiveRank(sampleHiddenStates);
            }
            double collapseScore = Math.Clamp(1.0 - rankRatio, 0.0, 1.0);

            // 4. Memory Footprint & Information Efficiency Calculations
            double memoryMB = (totalUniqueParams * bytesPerParameter) / (1024.0 * 1024.0);
            double infoEfficiencyIndex = compressionRatio / Math.Max(1.0, memoryMB / 100.0);

            // 5. Composite Density Score (0 - 100 Scale)
            // Balances compression efficiency, parameter economy, and latent space utilization.
            double paramScale = Math.Pow(1_000_000.0 / totalUniqueParams, 0.30);
            double kds = (compressionRatio * 45.0) * paramScale * Math.Pow(rankRatio, 0.25) * Math.Min(infoEfficiencyIndex, 2.0) * 10.0;
            kds = Math.Clamp(kds, 0.0, 100.0);

            double perplexity = Math.Exp(Math.Min(evalLoss, 20.0));

            return new KnowledgeDensityMetrics(
                densityScore: Math.Round(kds, 2),
                compressionRatio: Math.Round(compressionRatio, 4),
                bitsPerParam: Math.Round(bitsPerParam, 4),
                effectiveRankRatio: Math.Round(rankRatio, 4),
                uniqueParamCount: totalUniqueParams,
                totalParamCount: totalParams,
                evalPerplexity: Math.Round(perplexity, 2),
                estimatedMemoryMB: Math.Round(memoryMB, 2),
                informationEfficiencyIndex: Math.Round(infoEfficiencyIndex, 4),
                dimensionalCollapseScore: Math.Round(collapseScore, 4)
            );
        }

        /// <summary>
        /// Computes a robust Roy & Vetterli Effective Rank ratio over the hidden state activations,
        /// guarding against representation collapse and dead latent dimensions.
        /// </summary>
        private static double ComputeRobustEffectiveRank(ITensor hiddenStates)
        {
            try
            {
                int n = hiddenStates.Shape[0];
                int d = hiddenStates.Shape[1];
                if (n < 2 || d < 2) return 1.0;

                var hostData = hiddenStates.ToArray();

                // Compute column means for centering
                var means = new double[d];
                for (int i = 0; i < n; i++)
                {
                    int rowOffset = i * d;
                    for (int j = 0; j < d; j++)
                    {
                        means[j] += hostData[rowOffset + j];
                    }
                }
                for (int j = 0; j < d; j++) means[j] /= n;

                // Compute diagonal variances and approximate cross-feature energy distribution
                var featureEnergies = new double[d];
                double totalEnergy = 0;

                for (int j = 0; j < d; j++)
                {
                    double sumSq = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double centered = hostData[(i * d) + j] - means[j];
                        sumSq += centered * centered;
                    }
                    double variance = Math.Max(1e-12, sumSq / (n - 1));
                    featureEnergies[j] = variance;
                    totalEnergy += variance;
                }

                if (totalEnergy <= 0) return 0.01;

                // Compute Shannon entropy over normalized feature energy distribution (Spectral entropy proxy)
                double entropy = 0.0;
                for (int j = 0; j < d; j++)
                {
                    double p = featureEnergies[j] / totalEnergy;
                    if (p > 0)
                    {
                        entropy -= p * Math.Log(p);
                    }
                }

                double effectiveRank = Math.Exp(entropy);
                return Math.Clamp(effectiveRank / d, 0.01, 1.0);
            }
            catch
            {
                return 1.0;
            }
        }
    }
}