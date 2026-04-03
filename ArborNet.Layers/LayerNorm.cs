using ArborNet.Core.Interfaces;
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
        /// </summary>
        /// <param name="input">The input tensor to be normalized.</param>
        /// <returns>
        /// A tensor with the same shape as <paramref name="input"/> containing the 
        /// layer-normalized values with the learned affine transformation applied.
        /// </returns>
        /// <remarks>
        /// Normalization is performed over the last dimension. The implementation 
        /// follows the standard LayerNorm formula using mean and variance computed 
        /// across the specified normalized dimensions.
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            var mean = input.Mean(-1);
            var variance = input.Subtract(mean).Pow(2f).Mean(-1);
            var std = variance.Add(Tensor.FromScalar(eps)).Sqrt();
            var normalized = input.Subtract(mean).Divide(std);
            return normalized.Multiply(gamma).Add(beta);
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