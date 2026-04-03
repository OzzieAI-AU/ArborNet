using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace ArborNet.Layers.Normalization
{
    /// <summary>
    /// Layer Normalization normalizes across the feature dimension for each sample independently.
    /// No running stats - purely layer-wise (ideal for Transformers).
    /// </summary>
    public class LayerNorm : BaseNormalization
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LayerNorm"/> class.
        /// </summary>
        /// <param name="normalizedShape">Shape of the features to normalize (typically [-1] for last dim).</param>
        /// <param name="eps">Epsilon for stability. Default: 1e-5f.</param>
        /// <param name="useAffine">Enable gamma/beta. Default: true.</param>
        public LayerNorm(int[] normalizedShape, float eps = 1e-5f, bool useAffine = true)
            : base(new TensorShape(normalizedShape).Aggregate(1, (a, b) => a * b), eps, useAffine) { }

        protected override ITensor Normalize(ITensor input)
        {
            var mean = input.Mean(-1);
            var var_ = input.Subtract(mean).Pow(2).Mean(-1);
            var std = var_.Add(Eps).Sqrt();
            return input.Subtract(mean).Divide(std);
        }

        protected override ITensor ComputeGradInput(ITensor input, ITensor gradOutput)
        {
            var mean = input.Mean(-1);
            var var_ = input.Subtract(mean).Pow(2).Mean(-1);
            var std = var_.Add(Eps).Sqrt();
            var normalized = input.Subtract(mean).Divide(std);

            var N = Tensor.FromScalar((float)input.Shape[input.Shape.Rank - 1]);
            var ivar = std.Pow(-1);

            var gradNorm = gradOutput.Multiply(UseAffine ? Gamma : Tensor.Ones(input.Shape));

            // dL/dmean = sum(gradNorm * normalized * (-ivar)) / N
            var dL_dmean = gradNorm.Multiply(normalized).Multiply(ivar.Negate()).Sum(-1).Divide(N);

            // dL/dvar = sum(gradNorm * normalized * (-0.5 * ivar^3) * (input - mean)) / N
            var dL_dvar = gradNorm.Multiply(normalized).Multiply(input.Subtract(mean))
                                         .Multiply(ivar.Pow(3).Multiply(-0.5f)).Sum(-1).Divide(N);

            // dL/dx = gradNorm * ivar + (2 * (x - mean) / N) * (dL_dmean * ivar + dL_dvar * ivar^3 * (-0.5))
            var term1 = gradNorm.Multiply(ivar);
            var dx_mean = input.Subtract(mean).Multiply(N.Divide(2f));
            var term2 = dx_mean.Multiply(dL_dmean.Multiply(ivar));
            var term3 = dx_mean.Multiply(dL_dvar.Multiply(ivar.Pow(3).Multiply(-0.5f)));
            return term1.Add(term2).Add(term3);
        }
    }
}