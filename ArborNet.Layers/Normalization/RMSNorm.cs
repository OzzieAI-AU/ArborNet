using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace ArborNet.Layers.Normalization
{

    /// <summary>
    /// RMSNorm: Root Mean Square Normalization (used in Llama/Mistral).
    /// Normalizes by RMS instead of mean+variance. Faster and more stable.
    /// </summary>
    public class RMSNorm : BaseNormalization
    {
        public RMSNorm(int numFeatures, float eps = 1e-6f, bool useAffine = true)
            : base(numFeatures, eps, useAffine) { }

        protected override ITensor Normalize(ITensor input)
        {
            var rms = input.Pow(2).Mean(-1).Sqrt().Add(Eps).Sqrt();
            return input.Divide(rms);
        }

        protected override ITensor ComputeGradInput(ITensor input, ITensor gradOutput)
        {
            var rms = input.Pow(2).Mean(-1).Sqrt().Add(Eps).Sqrt();
            var normalized = input.Divide(rms);
            var N = Tensor.FromScalar((float)input.Shape[input.Shape.Rank - 1]);

            var gradNorm = gradOutput.Multiply(UseAffine ? Gamma : Tensor.Ones(input.Shape));

            // dL/dx = (gradNorm / rms) - (2 * x * mean(gradNorm * normalized) / (rms * N))
            var term1 = gradNorm.Divide(rms);
            var meanGrad = gradNorm.Multiply(normalized).Mean(-1);
            var term2 = input.Multiply(N.Divide(2f)).Multiply(meanGrad).Divide(rms);
            return term1.Subtract(term2);
        }
    }
}