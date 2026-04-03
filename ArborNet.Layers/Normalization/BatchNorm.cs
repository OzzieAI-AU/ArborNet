using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace ArborNet.Layers.Normalization
{
    /// <summary>
    /// Batch Normalization (1D/2D/3D) normalizes across the batch dimension for each channel/feature.
    /// Updates running mean/variance during training for use in evaluation mode.
    /// </summary>
    /// <remarks>
    /// Formula (training): <c>BN(x) = gamma * (x - E[x]) / sqrt(Var[x] + eps) + beta</c>.
    /// In eval mode, uses running statistics instead of batch statistics.
    /// Supports momentum for EMA updates of running stats.
    /// </remarks>
    public class BatchNorm : BaseNormalization
    {
        /// <summary>
        /// Running mean (updated during training via EMA).
        /// </summary>
        private ITensor RunningMean { get; set; }

        /// <summary>
        /// Running variance (updated during training via EMA).
        /// </summary>
        private ITensor RunningVar { get; set; }

        /// <summary>
        /// Momentum for updating running statistics (0 &lt; momentum &lt; 1).
        /// </summary>
        private readonly float Momentum;

        /// <summary>
        /// Initializes a new instance of the <see cref="BatchNorm"/> class.
        /// </summary>
        /// <param name="numFeatures">Number of channels/features to normalize.</param>
        /// <param name="eps">Epsilon for numerical stability. Default: 1e-5f.</param>
        /// <param name="momentum">Momentum for running stats EMA. Default: 0.1f.</param>
        /// <param name="useAffine">Enable learnable gamma/beta. Default: true.</param>
        public BatchNorm(int numFeatures, float eps = 1e-5f, float momentum = 0.1f, bool useAffine = true)
            : base(numFeatures, eps, useAffine)
        {
            Momentum = momentum;
            RunningMean = Tensor.Zeros(new TensorShape(numFeatures));
            RunningVar = Tensor.Ones(new TensorShape(numFeatures));
        }

        /// <summary>
        /// Normalizes input using batch or running statistics based on training mode.
        /// Updates running stats if in training mode.
        /// </summary>
        protected override ITensor Normalize(ITensor input)
        {
            var mean = IsTraining ? input.Mean(0) : RunningMean;
            var var_ = IsTraining ? input.Subtract(mean).Pow(2).Mean(0) : RunningVar;
            var std = var_.Add(Eps).Sqrt();

            if (IsTraining)
            {
                // EMA update for running stats (unbiased variance)
                var unbiasedVar = input.Subtract(mean).Pow(2).Mean(0).Multiply(input.Shape[0] / (input.Shape[0] - 1f));
                RunningMean = RunningMean.Multiply(Momentum).Add(mean.Multiply(1 - Momentum));
                RunningVar = RunningVar.Multiply(Momentum).Add(unbiasedVar.Multiply(1 - Momentum));
            }

            return input.Subtract(mean).Divide(std);
        }

        /// <summary>
        /// Computes input gradient: <c>grad_input = (grad_output * gamma) / (std * N) * (input - mean) * (-1/std) + ...</c>.
        /// Full analytical derivation implemented.
        /// </summary>
        protected override ITensor ComputeGradInput(ITensor input, ITensor gradOutput)
        {
            var N = Tensor.FromScalar((float)input.Shape[0]);
            var mean = IsTraining ? input.Mean(0) : RunningMean;
            var var_ = IsTraining ? input.Subtract(mean).Pow(2).Mean(0) : RunningVar;
            var std = var_.Add(Eps).Sqrt();

            var normalized = input.Subtract(mean).Divide(std);
            var gradNorm = gradOutput.Multiply(UseAffine ? Gamma : Tensor.Ones(input.Shape));

            // dL/dmean = sum(gradNorm * normalized * (-1/std)) / N
            var dL_dmean = gradNorm.Multiply(normalized.Divide(std)).Mean(0).Multiply(-1f);

            // dL/dvar = sum(gradNorm * normalized * (-0.5/std^3) * (input - mean)) / N
            var dL_dvar = gradNorm.Multiply(normalized).Multiply(input.Subtract(mean))
                                     .Multiply(std.Pow(-3).Multiply(-0.5f)).Mean(0);

            // dL/dinput = gradNorm/std + 2*(input-mean)/N * dL_dmean/std + 2*(input-mean)*dL_dvar/(N*std)
            var term1 = gradNorm.Divide(std);
            var term2 = input.Subtract(mean).Multiply(N.Divide(2f)).Multiply(dL_dmean).Divide(std);
            var term3 = input.Subtract(mean).Multiply(N.Divide(2f)).Multiply(dL_dvar).Divide(std);
            return term1.Add(term2).Add(term3);
        }
    }
}