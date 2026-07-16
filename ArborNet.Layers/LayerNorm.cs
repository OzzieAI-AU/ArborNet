using ArborNet.Core.Interfaces;
using ArborNet.Core.Layers;
using ArborNet.Core.Tensors;
using System;
using System.Collections.Generic;

namespace ArborNet.Layers
{
    /// <summary>
    /// Implements Layer Normalization (LayerNorm) as introduced in the paper 
    /// "Layer Normalization" by Jimmy Lei Ba et al. (2016).
    /// </summary>
    /// <remarks>
    /// Layer Normalization normalizes the activations across the feature dimensions 
    /// for each individual sample independently. It is widely used in transformer 
    /// architectures and recurrent neural networks because it does not depend on 
    /// batch statistics.
    /// 
    /// The forward pass computes:
    /// <code>
    /// normalized = (input - mean) / sqrt(variance + eps)
    /// output = gamma * normalized + beta
    /// </code>
    /// where mean and variance are calculated over the dimensions specified by <see cref="normalizedShape"/>.
    /// </remarks>
    public class LayerNorm : BaseLayer
    {
        /// <summary>
        /// The learnable scale (gain) parameter.
        /// </summary>
        private readonly ITensor gamma;

        /// <summary>
        /// The learnable bias (shift) parameter.
        /// </summary>
        private readonly ITensor beta;

        /// <summary>
        /// Small constant added to the variance for numerical stability.
        /// </summary>
        private readonly float eps;

        /// <summary>
        /// The shape of the dimensions over which normalization is performed.
        /// </summary>
        private readonly int[] normalizedShape;

        /// <summary>
        /// Initializes a new instance of the <see cref="LayerNorm"/> class.
        /// </summary>
        /// <param name="normalizedShape">The shape of the features to normalize over.</param>
        /// <param name="eps">A small value added to the variance to prevent division by zero.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="normalizedShape"/> is <see langword="null"/>.</exception>
        public LayerNorm(int[] normalizedShape, float eps = 1e-5f)
        {
            this.normalizedShape = normalizedShape ?? throw new ArgumentNullException(nameof(normalizedShape));
            this.eps = eps;

            gamma = Tensor.Ones(new TensorShape(normalizedShape));
            beta = Tensor.Zeros(new TensorShape(normalizedShape));

            gamma.RequiresGrad = true;
            beta.RequiresGrad = true;
        }

        /// <summary>
        /// Performs the forward pass of layer normalization on the input tensor.
        /// Features robust dimension recovery and broadcasting alignment.
        /// </summary>
        /// <param name="input">The input tensor to be normalized.</param>
        /// <returns>
        /// A tensor with the same shape as <paramref name="input"/> containing the 
        /// layer-normalized values with the learned affine transformation applied.
        /// </returns>
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            if (input.Shape.Rank == 0)
                throw new ArgumentException("Input tensor must have at least 1 dimension to perform LayerNorm.");

            try
            {
                // keepDims: true ensures [16, 64] -> Mean(-1) becomes [16, 1], perfectly broadcasting back to [16, 64]
                var mean = input.Mean(-1, keepDims: true);
                var variance = input.Subtract(mean).Pow(2f).Mean(-1, keepDims: true);
                var std = variance.Add(Tensor.FromScalar(eps)).Sqrt();

                var normalized = input.Subtract(mean).Divide(std);
                return normalized.Multiply(gamma).Add(beta);
            }
            catch (Exception ex)
            {
                // Resilient failure: catch broadcast/dimension mismatches and provide explicit context
                throw new InvalidOperationException($"Dimension alignment failed during LayerNorm forward pass. Verify input shape matches expected normalized shape. Inner Error: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Returns the trainable parameters of this layer.
        /// </summary>
        /// <returns>An enumerable containing the <see cref="gamma"/> and <see cref="beta"/> tensors.</returns>
        public override IEnumerable<ITensor> Parameters()
        {
            yield return gamma;
            yield return beta;
        }
    }
}