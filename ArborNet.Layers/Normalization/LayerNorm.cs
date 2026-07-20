// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Layers.Normalization
{

    #region Using Statements:

    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    using System.Text;
    /// <summary>
    /// Represents a Layer Normalization (LayerNorm) layer.
    /// Layer Normalization normalizes across the feature dimension for each sample independently.
    /// Unlike Batch Normalization, it does not maintain running statistics, making it purely layer-wise
    /// and highly suitable for sequence models such as Transformers.
    /// </summary>
    /// <remarks>
    /// Mathematically, Layer Normalization computes the mean and variance over the specified 
    /// normalization shape (usually the last dimension) for each individual sample in a batch:
    /// <para>
    /// <c>y = ((x - mean) / sqrt(variance + epsilon)) * gamma + beta</c>
    /// </para>
    /// where <c>gamma</c> and <c>beta</c> are learnable parameter vectors of the same shape 
    /// as the normalized dimensions (if <c>useAffine</c> is enabled).
    /// </remarks>
    /// <seealso cref="BaseNormalization" />

    #endregion

    public class LayerNorm : BaseNormalization
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LayerNorm"/> class.
        /// </summary>
        /// <param name="normalizedShape">Shape of the features to normalize (typically [-1] for last dim).</param>
        /// <param name="eps">Epsilon for stability. Default: 1e-5f.</param>
        /// <param name="useAffine">Enable gamma/beta. Default: true.</param>
        public LayerNorm(int[] normalizedShape, float eps = 1e-5f, bool useAffine = true)
            : base(new TensorShape(normalizedShape).TotalElements, eps, useAffine) { }
        /// <summary>
        /// Normalizes the input tensor across its feature dimension during the forward pass of the neural network.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> containing the batch data to be normalized.</param>
        /// <returns>A new <see cref="ITensor"/> containing the normalized and optionally scaled/shifted values.</returns>
        /// <remarks>
        /// The normalization operation calculates the mean and variance across the last dimension (represented by index -1):
        /// <para>
        /// <c>mean = Mean(input, dim: -1)</c><br/>
        /// <c>var = Mean((input - mean)^2, dim: -1)</c><br/>
        /// <c>std = Sqrt(var + epsilon)</c><br/>
        /// <c>output = (input - mean) / std</c>
        /// </para>
        /// This ensures that the elements along the designated normalization dimensions have a mean of 0 and a variance of 1.
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown if the <paramref name="input"/> tensor is null.</exception>

        protected override ITensor Normalize(ITensor input)
        {
            var mean = input.Mean(-1);
            var var_ = input.Subtract(mean).Pow(2).Mean(-1);
            var std = var_.Add(Eps).Sqrt();
            return input.Subtract(mean).Divide(std);
        }
        /// <summary>
        /// Computes the gradient of the loss function with respect to the input tensor during the backward pass (backpropagation).
        /// </summary>
        /// <param name="input">The original input <see cref="ITensor"/> that was supplied to the forward pass.</param>
        /// <param name="gradOutput">The gradient of the loss with respect to the output of this layer (<c>dL/dy</c>).</param>
        /// <returns>The gradient of the loss with respect to the input of this layer (<c>dL/dx</c>).</returns>
        /// <remarks>
        /// This method calculates the analytical gradients using the chain rule over the Layer Normalization equations,
        /// taking into account the impact of the mean and variance calculations on the input elements:
        /// <para>
        /// 1. Computes the forward pass statistics (mean, variance, and standard deviation) to reconstruct the normalized input.<br/>
        /// 2. Scales the output gradient (<paramref name="gradOutput"/>) by the learnable parameter <c>Gamma</c> (or a tensor of ones if affine transforms are disabled).<br/>
        /// 3. Backpropagates through the mean and variance calculations using intermediate gradient accumulators (<c>dL_dmean</c> and <c>dL_dvar</c>).<br/>
        /// 4. Assembles the final input gradient (<c>dL/dx</c>) to pass back to the previous layers.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="input"/> or <paramref name="gradOutput"/> is null.</exception>

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