using System;
using System.Threading.Tasks;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Devices;

namespace ArborNet.Core.Native.Dispatcher
{
    /// <summary>
    /// Provides high-performance fused operations that merge multiple compute steps
    /// into a single pass, avoiding costly intermediate allocations.
    /// </summary>
    public static class FusedKernels
    {
        /// <summary>
        /// Fuses Dense Linear Projection with ReLU: Output = ReLU(Input * Weight + Bias)
        /// </summary>
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
        /// Fuses Scale, Bias, and Sigmoid activations: Output = Sigmoid(Input * Scale + Bias)
        /// </summary>
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