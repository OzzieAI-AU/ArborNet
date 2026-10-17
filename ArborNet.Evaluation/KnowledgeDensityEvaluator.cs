namespace ArborNet.Evaluation
{
    using System;
    using ArborNet.Models;

    public record KnowledgeDensityMetrics(
        float BestValLoss,
        float FinalValLoss,
        float BaselineValLoss,
        float ValidationAdvantagePct,
        float OverfittingGap,
        float CompositeScore,
        bool GatePassed
    );

    public static class KnowledgeDensityEvaluator
    {
        public static KnowledgeDensityMetrics Evaluate(
            RecurrentDepthTransformer model,
            float bestValLoss,
            float finalValLoss,
            float baselineValLoss = 0.4897f)
        {
            float valAdvantage = ((baselineValLoss - bestValLoss) / baselineValLoss) * 100.0f;
            float overfitGap = finalValLoss - bestValLoss;
            
            // Gate criteria: Advantage >= 9.0%, Overfit gap <= 0.030
            float scoreAdvantage = Math.Clamp(valAdvantage / 9.0f * 50.0f, 0.0f, 50.0f);
            float scoreConvergence = Math.Clamp((1.0f - (overfitGap / 0.030f)) * 50.0f, 0.0f, 50.0f);
            float compositeScore = scoreAdvantage + scoreConvergence;

            bool passed = valAdvantage >= 9.0f && overfitGap <= 0.030f && model.ComputeTotalUniqueParameters() <= 1260;

            return new KnowledgeDensityMetrics(
                BestValLoss: bestValLoss,
                FinalValLoss: finalValLoss,
                BaselineValLoss: baselineValLoss,
                ValidationAdvantagePct: valAdvantage,
                OverfittingGap: overfitGap,
                CompositeScore: compositeScore,
                GatePassed: passed
            );
        }
    }
}
