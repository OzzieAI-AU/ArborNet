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
    using System.Collections.Generic;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;

    /// <summary>
    /// Numerically stable cross-entropy loss designed for deep recurrent and looped architectures.
    /// Computes log-softmax with max subtraction to prevent numerical overflow across multi-step iterations.
    /// </summary>
    public class LoopStableCrossEntropy : ILoss
    {
        private const float Epsilon = 1e-9f;

        /// <summary>
        /// Executes the forward pass of the cross entropy loss.
        /// </summary>
        /// <param name="predictions">Logits tensor of shape [..., VocabSize].</param>
        /// <param name="targets">Target indices or one-hot targets of shape matching predictions.</param>
        /// <param name="reduction">Reduction type: "mean", "sum", or "none". Defaults to "mean".</param>
        /// <returns>Computed scalar loss or reduced loss tensor.</returns>
        public ITensor Forward(ITensor predictions, ITensor targets, string reduction = "mean")
        {
            ArgumentNullException.ThrowIfNull(predictions);
            ArgumentNullException.ThrowIfNull(targets);

            int lastDim = predictions.Shape[^1];

            // 1. Numerically stable LogSoftmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
            ITensor maxLogits = predictions.Max(-1, keepDims: true);
            ITensor shifted = predictions.Subtract(maxLogits);
            ITensor sumExp = shifted.Exp().Sum(-1, keepDims: true).Add(Epsilon);
            ITensor logProbs = shifted.Subtract(sumExp.Log());

            // 2. Cross entropy = -sum(targets * logProbs)
            ITensor lossMatrix;
            if (targets.Shape.Rank == predictions.Shape.Rank && targets.Shape[^1] == lastDim)
            {
                // Targets are already one-hot or soft probabilities
                lossMatrix = targets.Multiply(logProbs).Negate().Sum(-1);
            }
            else
            {
                // If targets are integer token indices, convert to one-hot representation
                int numTokens = targets.Shape.TotalElements;
                float[] targetArray = targets.ToArray();
                float[] oneHotArray = new float[numTokens * lastDim];

                for (int i = 0; i < numTokens; i++)
                {
                    int classIdx = (int)targetArray[i];
                    if (classIdx >= 0 && classIdx < lastDim)
                    {
                        oneHotArray[i * lastDim + classIdx] = 1.0f;
                    }
                }

                var oneHot = Tensor.FromArray(oneHotArray, predictions.Shape, predictions.Device);
                lossMatrix = oneHot.Multiply(logProbs).Negate().Sum(-1);
            }

            // 3. Apply reduction
            return reduction.ToLowerInvariant() switch
            {
                "mean" => lossMatrix.Mean(),
                "sum" => lossMatrix.Sum(),
                "none" => lossMatrix,
                _ => lossMatrix.Mean()
            };
        }

        /// <summary>
        /// Default forward pass with "mean" reduction.
        /// </summary>
        public ITensor Forward(ITensor predictions, ITensor targets)
        {
            return Forward(predictions, targets, "mean");
        }

        public IEnumerable<ITensor> Parameters()
        {
            return Array.Empty<ITensor>();
        }
    }
}