// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Losses
{
    using System;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;

    /// <summary>
    /// Cross-Entropy loss tailored for recurrent depth models. Clamps extreme logit excursions
    /// from deep recurrence and provides optional label smoothing to mitigate overconfidence.
    /// </summary>
    public class LoopStableCrossEntropyLoss : ILoss
    {
        private readonly float _labelSmoothing;
        private readonly float _logitCap;

        public LoopStableCrossEntropyLoss(float labelSmoothing = 0.0f, float logitCap = 30.0f)
        {
            if (labelSmoothing < 0.0f || labelSmoothing >= 1.0f)
            {
                throw new ArgumentOutOfRangeException(nameof(labelSmoothing), "Label smoothing must be in [0, 1).");
            }
            if (logitCap <= 0.0f)
            {
                throw new ArgumentOutOfRangeException(nameof(logitCap), "Logit cap must be strictly positive.");
            }

            _labelSmoothing = labelSmoothing;
            _logitCap = logitCap;
        }

        public ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean")
        {
            ArgumentNullException.ThrowIfNull(predictions);
            ArgumentNullException.ThrowIfNull(targets);

            // Stabilize unrolled recurrence by clamping logit bounds symmetrically
            ITensor clamped = predictions.Clip(-_logitCap, _logitCap);

            // Numerically stable LogSoftmax along the vocabulary dimension (-1):
            // log_softmax(x) = (x - max(x)) - log(sum(exp(x - max(x))))
            ITensor maxVal = clamped.Max(axis: -1, keepDims: true);
            ITensor shifted = clamped.Subtract(maxVal);
            ITensor logSumExp = shifted.Exp().Sum(axis: -1, keepDims: true).Log();
            ITensor logProbs = shifted.Subtract(logSumExp);

            // Align targets for Gather across both CPU and CUDA backends
            ITensor targetIndices = (targets.Shape.Rank == logProbs.Shape.Rank && targets.Shape[^1] == 1)
                ? targets.Squeeze(-1)
                : targets;

            ITensor targetLogProbs = logProbs.Gather(axis: -1, indices: targetIndices);
            if (targetLogProbs.Shape.Rank == logProbs.Shape.Rank && targetLogProbs.Shape[^1] == 1)
            {
                targetLogProbs = targetLogProbs.Squeeze(-1);
            }

            ITensor loss;
            if (_labelSmoothing > 0.0f)
            {
                ITensor smoothedLoss = logProbs.Mean(axis: -1).Negate();
                ITensor nllLoss = targetLogProbs.Negate();

                loss = nllLoss.Multiply(1.0f - _labelSmoothing)
                              .Add(smoothedLoss.Multiply(_labelSmoothing));
            }
            else
            {
                loss = targetLogProbs.Negate();
            }

            return reduction?.ToLowerInvariant() switch
            {
                "none" => loss,
                "sum" => loss.Sum(),
                _ => loss.Mean()
            };
        }
    }
}