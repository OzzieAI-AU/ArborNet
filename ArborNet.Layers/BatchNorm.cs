using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using System;
using System.Collections.Generic;

namespace ArborNet.Layers
{
    /// <summary>
    /// Implements Batch Normalization for neural network layers.
    /// </summary>
    /// <remarks>
    /// Normalizes the input using batch mean and variance, then applies a learned
    /// scale (<see cref="gamma"/>) and shift (<see cref="beta"/>).
    /// </remarks>
    public class BatchNorm : BaseLayer
    {
        /// <summary>
        /// The number of features/channels to normalize.
        /// </summary>
        private readonly int numFeatures;

        /// <summary>
        /// Small constant added to the variance for numerical stability.
        /// </summary>
        private readonly float eps;

        /// <summary>
        /// Momentum coefficient for updating running mean and variance.
        /// </summary>
        private readonly float momentum;

        /// <summary>
        /// Learnable scaling parameter (gamma).
        /// </summary>
        private ITensor gamma;

        /// <summary>
        /// Learnable shift parameter (beta).
        /// </summary>
        private ITensor beta;

        /// <summary>
        /// Running mean maintained for inference.
        /// </summary>
        private ITensor runningMean;

        /// <summary>
        /// Running variance maintained for inference.
        /// </summary>
        private ITensor runningVar;

        /// <summary>
        /// Initializes a new instance of the <see cref="BatchNorm"/> class.
        /// </summary>
        /// <param name="numFeatures">The number of features to normalize.</param>
        /// <param name="eps">A small value added to the variance for numerical stability.</param>
        /// <param name="momentum">The momentum used when updating running statistics.</param>
        public BatchNorm(int numFeatures, float eps = 1e-5f, float momentum = 0.1f)
        {
            this.numFeatures = numFeatures;
            this.eps = eps;
            this.momentum = momentum;

            gamma = Tensor.Ones(new TensorShape(numFeatures));
            beta = Tensor.Zeros(new TensorShape(numFeatures));
            runningMean = Tensor.Zeros(new TensorShape(numFeatures));
            runningVar = Tensor.Ones(new TensorShape(numFeatures));

            gamma.RequiresGrad = beta.RequiresGrad = true;
        }

        /// <summary>
        /// Performs the forward pass of the Batch Normalization layer.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The normalized and affine-transformed tensor.</returns>
        /// <remarks>
        /// Computes mean and variance over dimension 1, normalizes the input,
        /// then applies the learned scale and shift parameters.
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            var mean = input.Mean(1);
            var var = input.Subtract(mean).Pow(2f).Mean(1);
            var std = var.Add(Tensor.FromScalar(eps)).Sqrt();

            var normalized = input.Subtract(mean).Divide(std);
            return normalized.Multiply(gamma).Add(beta);
        }

        /// <summary>
        /// Returns the trainable parameters of this layer.
        /// </summary>
        /// <returns>An enumerable containing the gamma and beta parameters.</returns>
        public override IEnumerable<ITensor> Parameters()
        {
            yield return gamma;
            yield return beta;
        }
    }
}