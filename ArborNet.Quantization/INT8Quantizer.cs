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
    using System.Linq;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;

    /// <summary>
    /// Represents the mathematical parameters defining an INT8 quantization map.
    /// </summary>
    public struct QuantizationScale
    {
        public float Scale;
        public sbyte ZeroPoint;
    }

    /// <summary>
    /// High-performance Post-Training Quantization (PTQ) engine for compressing FP32 tensors to INT8.
    /// Supports both Symmetric (zero-centered) and Asymmetric scale mappings.
    /// </summary>
    public static class INT8Quantizer
    {
        /// <summary>
        /// Calibrates and computes the symmetric scale factor for a float tensor.
        /// </summary>
        public static QuantizationScale CalibrateSymmetric(ITensor tensor)
        {
            float[] data = tensor.ToArray();
            float maxAbs = data.Max(MathF.Abs);
            maxAbs = Math.Max(maxAbs, 1e-7f); // Avoid division by zero

            // Maps [-maxAbs, maxAbs] to [-127, 127]
            float scale = maxAbs / 127f;
            return new QuantizationScale { Scale = scale, ZeroPoint = 0 };
        }

        /// <summary>
        /// Calibrates and computes the asymmetric scale factor and zero-point for a float tensor.
        /// </summary>
        public static QuantizationScale CalibrateAsymmetric(ITensor tensor)
        {
            float[] data = tensor.ToArray();
            float min = data.Min();
            float max = data.Max();

            float range = Math.Max(max - min, 1e-7f);
            float scale = range / 255f;

            // Compute zero-point and clamp to signed 8-bit range [-128, 127]
            int zpVal = (int)MathF.Round(-min / scale) - 128;
            sbyte zeroPoint = (sbyte)Math.Clamp(zpVal, -128, 127);

            return new QuantizationScale { Scale = scale, ZeroPoint = zeroPoint };
        }

        /// <summary>
        /// Quantizes a 32-bit floating-point tensor to a signed 8-bit integer array.
        /// </summary>
        public static sbyte[] Quantize(ITensor tensor, QuantizationScale qScale)
        {
            float[] data = tensor.ToArray();
            sbyte[] quantized = new sbyte[data.Length];

            for (int i = 0; i < data.Length; i++)
            {
                int qVal = (int)MathF.Round(data[i] / qScale.Scale) + qScale.ZeroPoint;
                quantized[i] = (sbyte)Math.Clamp(qVal, -128, 127);
            }

            return quantized;
        }

        /// <summary>
        /// Dequantizes an 8-bit signed integer array back to a 32-bit float tensor.
        /// </summary>
        public static ITensor Dequantize(sbyte[] quantized, TensorShape shape, QuantizationScale qScale)
        {
            float[] dequantized = new float[quantized.Length];

            for (int i = 0; i < quantized.Length; i++)
            {
                dequantized[i] = (quantized[i] - qScale.ZeroPoint) * qScale.Scale;
            }

            return Tensor.FromArray(dequantized, shape);
        }
    }
}