// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Native.Dispatcher
{

    #region Using Statements:

    using System;
    using System.Threading.Tasks;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Core.Devices;
    /// <summary>
    /// Provides high-performance fused operations that merge multiple compute steps
    /// into a single pass, avoiding costly intermediate allocations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Fused kernels optimize memory bandwidth and cache locality by performing
    /// sequential element-wise and reduction operations within a single loop execution, 
    /// significantly reducing garbage collection overhead and memory transfer bottlenecks.
    /// </para>
    /// <para>
    /// Thread safety is inherited from the underlying device memory models. When executing on CPU, 
    /// parallel loops partition the workload across thread pool workers.
    /// </para>
    /// </remarks>

    #endregion

    public static class FusedKernels
    {
        /// <summary>
        /// Fuses Dense Linear Projection (Matrix Multiplication) with a Bias addition and Rectified Linear Unit (ReLU) activation.
        /// </summary>
        /// <param name="input">The input tensor of shape [M, K]. Must not be null.</param>
        /// <param name="weight">The weight tensor of shape [K, N]. Must not be null.</param>
        /// <param name="bias">The optional bias tensor of shape [N] to be added to the projected results. If null, the bias step is omitted.</param>
        /// <returns>A new <see cref="ITensor"/> of shape [M, N] containing the activated projection results.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> or <paramref name="weight"/> is <see langword="null"/>.</exception>
        /// <exception cref="IndexOutOfRangeException">Thrown during CPU parallel execution if tensor shapes are mismatched or out of bounds.</exception>
        /// <remarks>
        /// <para>
        /// On CPU devices, this operation is executed as a parallelized, single-pass fused loop to bypass 
        /// intermediate buffer allocations. On non-CPU devices (e.g., CUDA), it falls back to chained hardware operations.
        /// </para>
        /// <para>
        /// Mathematical Formula: <c>Output = Max(0, Input * Weight + Bias)</c>
        /// </para>
        /// </remarks>
        public static ITensor FusedLinearReLU(ITensor input, ITensor weight, ITensor? bias)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (weight == null) throw new ArgumentNullException(nameof(weight));

            if (input.Device.Type == DeviceType.CPU)
            {
                int m = input.Shape[0];
                int k = input.Shape[1];
                int n = weight.Shape[1];

                float[] inData = input.ToArray();
                float[] wData = weight.ToArray();
                float[] bData = bias != null ? bias.ToArray() : Array.Empty<float>();
                float[] outData = new float[m * n];

                Parallel.For(0, m, i =>
                {
                    for (int j = 0; j < n; j++)
                    {
                        float sum = 0f;
                        for (int l = 0; l < k; l++)
                        {
                            sum += inData[i * k + l] * wData[l * n + j];
                        }
                        if (bias != null)
                        {
                            sum += bData[j];
                        }
                        outData[i * n + j] = sum > 0f ? sum : 0f; // ReLU step
                    }
                });

                return Tensor.FromArray(outData, new TensorShape(m, n), input.Device);
            }
            else
            {
                // CUDA / Accelerator Fallback
                var proj = input.MatMul(weight);
                if (bias != null) proj = proj.Add(bias);
                return proj.Relu();
            }
        }
        /// <summary>
        /// Fuses element-wise scaling, bias addition, and Sigmoid activation into a single pass.
        /// </summary>
        /// <param name="input">The input tensor containing the source values. Must not be null.</param>
        /// <param name="scale">The scaling factor multiplier to apply to each element.</param>
        /// <param name="bias">The bias offset value to add to each scaled element.</param>
        /// <returns>A new <see cref="ITensor"/> of the same shape as <paramref name="input"/> containing the processed activations.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <see langword="null"/>.</exception>
        /// <remarks>
        /// <para>
        /// Minimizes memory access roundtrips on CPU by wrapping the scale, shift, and activation logic into 
        /// a single parallel loop, bypassing intermediate allocations for scale and bias results.
        /// </para>
        /// <para>
        /// Mathematical Formula: <c>Output = 1 / (1 + exp(-(Input * Scale + Bias)))</c>
        /// </para>
        /// </remarks>

        public static ITensor FusedScaleBiasSigmoid(ITensor input, float scale, float bias)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            if (input.Device.Type == DeviceType.CPU)
            {
                float[] inData = input.ToArray();
                float[] outData = new float[inData.Length];

                Parallel.For(0, inData.Length, i =>
                {
                    float val = inData[i] * scale + bias;
                    outData[i] = 1f / (1f + MathF.Exp(-val)); // Sigmoid step
                });

                return Tensor.FromArray(outData, input.Shape, input.Device);
            }
            else
            {
                return input.Multiply(scale).Add(bias).Sigmoid();
            }
        }
    }
}