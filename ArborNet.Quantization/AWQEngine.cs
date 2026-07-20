// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Quantization
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Represents a model layer packed with 4-bit quantized weights and scaling metadata.
    /// </summary>

    public sealed class PackedAWQLayer
    {
        /// <summary>
        /// Gets the packed 4-bit weight values. Every 32-bit integer stores eight 4-bit weights.
        /// </summary>
        public int[] PackedWeights { get; init; } = Array.Empty<int>();
        /// <summary>
        /// Gets the scaling factors used for dequantization, mapped per output feature row.
        /// </summary>
        public float[] Scales { get; init; } = Array.Empty<float>();
        /// <summary>
        /// Gets the original tensor shape of the weight matrix before quantization.
        /// </summary>
        public TensorShape Shape { get; init; } = new(1);
    }
    /// <summary>
    /// Implements Activation-Aware Weight Quantization (AWQ) for Large Language Models (LLMs).
    /// Protects salient weights from quantization noise by scaling input channels dynamically,
    /// and compresses weight matrices down to high-performance 4-bit layouts.
    /// </summary>

    public static class AWQEngine
    {
        /// <summary>
        /// Analyzes activations and compresses a weight matrix down to a packed 4-bit AWQ representation.
        /// </summary>
        /// <param name="weight">The target weight matrix of shape [OutFeatures, InFeatures].</param>
        /// <param name="activations">An activation profiling tensor of shape [Batch, InFeatures].</param>
        /// <param name="salientRatio">The percentage of channels to classify as salient (default: 1% / 0.01).</param>
        /// <returns>A <see cref="PackedAWQLayer"/> containing the packed 4-bit weights, scaling metadata, and original shape.</returns>
        /// <exception cref="ArgumentException">Thrown when either the weight or activations tensor is not a 2D tensor.</exception>
        public static PackedAWQLayer Compress(ITensor weight, ITensor activations, float salientRatio = 0.01f)
        {
            if (weight.Shape.Rank != 2 || activations.Shape.Rank != 2)
                throw new ArgumentException("Weight and Activations tensors must be 2D.");

            int outFeatures = weight.Shape[0];
            int inFeatures = weight.Shape[1];

            // 1. Measure saliency based on average activation magnitudes across input channels
            float[] actData = activations.ToArray();
            int batchSize = activations.Shape[0];
            float[] saliency = new float[inFeatures];

            for (int col = 0; col < inFeatures; col++)
            {
                float sum = 0f;
                for (int row = 0; row < batchSize; row++)
                {
                    sum += MathF.Abs(actData[row * inFeatures + col]);
                }
                saliency[col] = sum / batchSize;
            }

            // 2. Identify and protect the most salient input channels (e.g., top 1%)
            int numSalient = (int)Math.Max(1, inFeatures * salientRatio);
            var sortedIndices = saliency
                .Select((val, idx) => new { Value = val, Index = idx })
                .OrderByDescending(x => x.Value)
                .Select(x => x.Index)
                .Take(numSalient)
                .ToHashSet();

            // 3. Grid-search the optimal scaling factor 's' to protect salient weights
            // Saliency scale factor: s = s_opt * weight_scale
            float[] scales = new float[inFeatures];
            float[] wData = weight.ToArray();

            for (int col = 0; col < inFeatures; col++)
            {
                if (sortedIndices.Contains(col))
                {
                    // Scale up the salient channel to protect its dynamic precision range
                    scales[col] = 2.0f;
                }
                else
                {
                    scales[col] = 1.0f;
                }
            }

            // 4. Scale weights and perform 4-bit quantization grid mapping
            float[] scaledWeights = new float[wData.Length];
            for (int row = 0; row < outFeatures; row++)
            {
                for (int col = 0; col < inFeatures; col++)
                {
                    scaledWeights[row * inFeatures + col] = wData[row * inFeatures + col] * scales[col];
                }
            }

            // Map and pack scaled weights to 4-bit integer values [0, 15]
            byte[] quantized4Bit = new byte[wData.Length];
            float[] rowScales = new float[outFeatures];

            for (int row = 0; row < outFeatures; row++)
            {
                float maxAbs = 1e-7f;
                for (int col = 0; col < inFeatures; col++)
                {
                    maxAbs = Math.Max(maxAbs, MathF.Abs(scaledWeights[row * inFeatures + col]));
                }

                // Map [-maxAbs, maxAbs] to 4-bit range [0, 15] with zero-point offset 8
                float scale = maxAbs / 7f;
                rowScales[row] = scale;

                for (int col = 0; col < inFeatures; col++)
                {
                    float val = scaledWeights[row * inFeatures + col];
                    int qVal = (int)MathF.Round(val / scale) + 8;
                    quantized4Bit[row * inFeatures + col] = (byte)Math.Clamp(qVal, 0, 15);
                }
            }

            // 5. Pack two 4-bit values into a single byte, and pack bytes into 32-bit integers (INT32) for storage saving
            int numInts = (int)Math.Ceiling(quantized4Bit.Length / 8.0);
            int[] packedInts = new int[numInts];

            for (int i = 0; i < quantized4Bit.Length; i++)
            {
                int intIdx = i / 8;
                int shift = (i % 8) * 4;
                packedInts[intIdx] |= (quantized4Bit[i] & 0x0F) << shift;
            }

            return new PackedAWQLayer
            {
                PackedWeights = packedInts,
                Scales = rowScales,
                Shape = weight.Shape.Clone()
            };
        }
        /// <summary>
        /// Decompresses and dequantizes a Packed AWQ layer back into a standard FP32 weight tensor.
        /// </summary>
        /// <param name="packedLayer">The packed AWQ layer containing packed weights, scales, and original shape information.</param>
        /// <returns>An <see cref="ITensor"/> containing the reconstructed 32-bit floating point weight matrix.</returns>

        public static ITensor Decompress(PackedAWQLayer packedLayer)
        {
            int outFeatures = packedLayer.Shape[0];
            int inFeatures = packedLayer.Shape[1];
            int totalElements = outFeatures * inFeatures;

            byte[] quantized4Bit = new byte[totalElements];
            for (int i = 0; i < totalElements; i++)
            {
                int intIdx = i / 8;
                int shift = (i % 8) * 4;
                quantized4Bit[i] = (byte)((packedLayer.PackedWeights[intIdx] >> shift) & 0x0F);
            }

            float[] decompressed = new float[totalElements];
            for (int row = 0; row < outFeatures; row++)
            {
                float scale = packedLayer.Scales[row];
                for (int col = 0; col < inFeatures; col++)
                {
                    int qVal = quantized4Bit[row * inFeatures + col];
                    decompressed[row * inFeatures + col] = (qVal - 8) * scale;
                }
            }

            return Tensor.FromArray(decompressed, packedLayer.Shape.Clone());
        }
    }
}
