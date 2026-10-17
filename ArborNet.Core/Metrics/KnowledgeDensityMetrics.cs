// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Metrics
{
    /// <summary>
    /// Represents comprehensive quantitative metrics evaluating a neural network model's parameter efficiency,
    /// information compression capability, and latent representation manifold quality within the ArborNet framework.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The <see cref="KnowledgeDensityMetrics"/> suite moves beyond simple loss evaluation by auditing structural bloat, 
    /// memory overhead, latent space dimensional collapse, and information entropy capture.
    /// </para>
    /// <para>
    /// Key theoretical foundations include:
    /// <list type="bullet">
    /// <item><description>Roy &amp; Vetterli Effective Rank for latent manifold representation health.</description></item>
    /// <item><description>Shannon entropy-based information compression ratios relative to vocabulary limits.</description></item>
    /// <item><description>Unique parameter accounting for weight-shared and recursive architectures.</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public sealed class KnowledgeDensityMetrics
    {
        /// <summary>
        /// Gets the composite Knowledge Density Score (KDS) normalized on a scale from $0.0$ to $100.0$.
        /// </summary>
        /// <value>
        /// A weighted score penalizing parameter bloat while rewarding high validation entropy capture 
        /// and healthy latent manifold rank utilization.
        /// </value>
        public double DensityScore { get; }

        /// <summary>
        /// Gets the information compression ratio derived from cross-entropy evaluation loss versus maximum vocabulary entropy.
        /// </summary>
        /// <value>
        /// A clamped ratio between $0.0001$ and $1.0$, calculated as:
        /// $\text{CompressionRatio} = \frac{H_{\max} - H_{\text{loss}}}{H_{\max}}$
        /// where $H_{\max} = \log_2(\text{vocabSize})$ and $H_{\text{loss}} = \frac{\text{evalLoss}}{\ln(2)}$.
        /// </value>
        public double CompressionRatio { get; }

        /// <summary>
        /// Gets the information density metric representing bits compressed per $1,000,000$ unique parameters.
        /// </summary>
        /// <value>
        /// Calculated as: $\frac{\text{CompressionRatio} \times \log_2(\text{vocabSize})}{\text{UniqueParamCount} / 10^6}$.
        /// </value>
        public double BitsPerParam { get; }

        /// <summary>
        /// Gets the normalized Roy &amp; Vetterli Effective Rank ratio of the model's latent hidden state representations.
        /// </summary>
        /// <value>
        /// A value between $0.01$ and $1.0$ representing the active utilization capacity of the latent feature space $D$. 
        /// Values close to $0.0$ indicate severe dimensional collapse, while values near $1.0$ indicate maximal representation diversity.
        /// </value>
        public double EffectiveRankRatio { get; }

        /// <summary>
        /// Gets the count of strictly unique parameters in the model (deduplicated by reference evaluation).
        /// </summary>
        /// <value>
        /// Essential for accurate efficiency profiling in parameter-shared, recurrent, or looped network topologies.
        /// </value>
        public long UniqueParamCount { get; }

        /// <summary>
        /// Gets the total instantiated parameter count across all model layers without deduplication.
        /// </summary>
        public long TotalParamCount { get; }

        /// <summary>
        /// Gets the evaluation perplexity derived directly from the evaluation loss.
        /// </summary>
        /// <value>
        /// Calculated as $\exp(\min(\text{evalLoss}, 20.0))$ to prevent numerical overflow under high-loss conditions.
        /// </value>
        public double EvalPerplexity { get; }

        /// <summary>
        /// Gets the estimated memory footprint of the model parameters in megabytes (MB).
        /// </summary>
        /// <value>
        /// Computed based on unique parameter counts and the active data precision width (defaulting to 4 bytes per FP32 parameter).
        /// </value>
        public double EstimatedMemoryMB { get; }

        /// <summary>
        /// Gets the Information Efficiency Index measuring compression gain relative to memory utilization overhead.
        /// </summary>
        /// <value>
        /// A balanced index comparing the model's compression ratio against its structural memory footprint per $100\text{MB}$.
        /// </value>
        public double InformationEfficiencyIndex { get; }

        /// <summary>
        /// Gets the representation dimensional collapse score.
        /// </summary>
        /// <value>
        /// Calculated as $1.0 - \text{EffectiveRankRatio}$. A high score indicates that the model's latent space 
        /// is suffering from subspace collapse and losing expressive dimensionality.
        /// </value>
        public double DimensionalCollapseScore { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="KnowledgeDensityMetrics"/> class with pre-computed evaluation parameters.
        /// </summary>
        /// <param name="densityScore">The composite knowledge density score ($0 - 100$).</param>
        /// <param name="compressionRatio">The normalized information compression ratio.</param>
        /// <param name="bitsPerParam">The compressed bits per million unique parameters.</param>
        /// <param name="effectiveRankRatio">The normalized Roy &amp; Vetterli effective rank ratio.</param>
        /// <param name="uniqueParamCount">The deduplicated parameter count.</param>
        /// <param name="totalParamCount">The total raw parameter count.</param>
        /// <param name="evalPerplexity">The evaluation perplexity.</param>
        /// <param name="estimatedMemoryMB">The estimated memory consumption in megabytes.</param>
        /// <param name="informationEfficiencyIndex">The computed information efficiency index.</param>
        /// <param name="dimensionalCollapseScore">The latent space dimensional collapse penalty score.</param>
        public KnowledgeDensityMetrics(
            double densityScore,
            double compressionRatio,
            double bitsPerParam,
            double effectiveRankRatio,
            long uniqueParamCount,
            long totalParamCount,
            double evalPerplexity,
            double estimatedMemoryMB,
            double informationEfficiencyIndex,
            double dimensionalCollapseScore)
        {
            DensityScore = densityScore;
            CompressionRatio = compressionRatio;
            BitsPerParam = bitsPerParam;
            EffectiveRankRatio = effectiveRankRatio;
            UniqueParamCount = uniqueParamCount;
            TotalParamCount = totalParamCount;
            EvalPerplexity = evalPerplexity;
            EstimatedMemoryMB = estimatedMemoryMB;
            InformationEfficiencyIndex = informationEfficiencyIndex;
            DimensionalCollapseScore = dimensionalCollapseScore;
        }
    }
}