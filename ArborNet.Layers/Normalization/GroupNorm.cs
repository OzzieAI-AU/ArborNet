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
    using System.Linq;
    /// <summary>
    /// Production-grade Group Normalization with a fully differentiable analytical backward pass.
    /// Divides channels into groups and normalizes independently within each group.
    /// </summary>
    /// <remarks>
    /// This layer implements Group Normalization (GN) as described in the paper:
    /// "Group Normalization" by Yuxin Wu and Kaiming He (https://arxiv.org/abs/1803.08494).
    /// Group Normalization acts as an alternative to Batch Normalization by dividing channels into groups 
    /// and calculating mean and variance statistics within each group, making it independent of batch size.
    /// </remarks>

    #endregion

    public class GroupNorm : BaseNormalization
    {
        private readonly int _numChannels;
        private readonly int _numGroups;
        private readonly int _channelsPerGroup;

        // Cached values for backward pass
        private ITensor? _cachedNormalized;
        private ITensor? _cachedRstd;

        public GroupNorm(int numChannels, int numGroups, float eps = 1e-5f, bool useAffine = true)
            : base(numChannels, eps, useAffine)
        {
            if (numChannels % numGroups != 0)
                throw new ArgumentException("numChannels must be divisible by numGroups");

            _numChannels = numChannels;
            _numGroups = numGroups;
            _channelsPerGroup = numChannels / numGroups;
        }
        /// <summary>
        /// Executes the forward pass of Group Normalization by dividing channels into groups and normalizing each group independently.
        /// </summary>
        /// <param name="input">The input tensor to normalize. Assumed to have dimensions starting with [Batch, Channels, ...].</param>
        /// <returns>A new <see cref="ITensor"/> containing the group-normalized activations.</returns>

        protected override ITensor Normalize(ITensor input)
        {
            int batch = input.Shape[0];
            // Uses Linq's Skip extension on IEnumerable<int> cleanly
            int spatialVolume = input.Shape.Skip(2).Aggregate(1, (a, b) => a * b);

            // Reshape input to isolate groups: [N, G, C_per_G, Spatial]
            var reshaped = input.Reshape(batch, _numGroups, _channelsPerGroup, spatialVolume);

            // Calculate mean and variance across the channels-per-group and spatial axes (dimensions 2 and 3)
            var mean = reshaped.Mean(new[] { 2, 3 }, keepDims: true);
            var variance = reshaped.Subtract(mean).Pow(2).Mean(new[] { 2, 3 }, keepDims: true);

            _cachedRstd = variance.Add(Eps).Sqrt().Pow(-1f);
            _cachedNormalized = reshaped.Subtract(mean).Multiply(_cachedRstd);

            // Reshape back to input's original shape
            return _cachedNormalized.Reshape(input.Shape.Dimensions);
        }
        /// <summary>
        /// Computes the gradient of the loss with respect to the input tensor using an analytical differentiable backward pass.
        /// </summary>
        /// <param name="input">The original input tensor from the forward pass.</param>
        /// <param name="gradOutput">The gradient of the loss with respect to the output of this layer.</param>
        /// <returns>The gradient of the loss with respect to the input tensor.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the backward pass is invoked before running the forward pass.</exception>

        protected override ITensor ComputeGradInput(ITensor input, ITensor gradOutput)
        {
            if (_cachedNormalized == null || _cachedRstd == null)
                throw new InvalidOperationException("Backward pass called before forward pass.");

            int batch = input.Shape[0];
            int spatialVolume = input.Shape.Skip(2).Aggregate(1, (a, b) => a * b);
            float groupElementsVolume = _channelsPerGroup * spatialVolume;

            // Reshape gradients and activations to match GroupNorm groups: [N, G, C_per_G, Spatial]
            var dY = gradOutput.Reshape(batch, _numGroups, _channelsPerGroup, spatialVolume);
            var X_hat = _cachedNormalized;

            // Apply scaling parameter (Gamma) if affine is enabled
            if (UseAffine)
            {
                // Reshape Gamma to align with C_per_G: [1, G, C_per_G, 1]
                var gammaReshaped = Gamma.Reshape(1, _numGroups, _channelsPerGroup, 1);
                dY = dY.Multiply(gammaReshaped);
            }

            // Analytical derivatives of GroupNorm:
            // sum_dx_hat = sum(dY * X_hat) across Group axes
            var sum_dY_xhat = dY.Multiply(X_hat).Sum(new[] { 2, 3 }, keepDims: true);
            var sum_dY = dY.Sum(new[] { 2, 3 }, keepDims: true);

            // dL/dX = (1 / (N_elements_per_group * std)) * [ N_elements_per_group * dY - sum_dY - X_hat * sum_dY_xhat ]
            var term1 = dY.Multiply(groupElementsVolume);
            var term2 = sum_dY;
            var term3 = X_hat.Multiply(sum_dY_xhat);

            var dX_reshaped = term1.Subtract(term2).Subtract(term3)
                                  .Multiply(_cachedRstd)
                                  .Divide(groupElementsVolume);

            return dX_reshaped.Reshape(input.Shape.Dimensions);
        }
    }
}