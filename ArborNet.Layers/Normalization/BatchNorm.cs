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

    using System;
    using System.Collections.Generic;
    using System.Linq;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Core.Devices;
    /// <summary>
    /// Represents a Batch Normalization (BatchNorm) layer, which normalizes the activations of a neural network 
    /// over a mini-batch to improve stability and accelerate training.
    /// This layer maintains running statistics (mean and variance) during training, which are utilized 
    /// for normalization during inference.
    /// </summary>

    #endregion

    public sealed class BatchNorm : BaseNormalization
    {
        /// <summary>
        /// Gets the running mean computed during the training phase.
        /// </summary>
        /// <value>
        /// The running mean tensor.
        /// </value>
        public ITensor RunningMean { get; private set; }
        /// <summary>
        /// Gets the running variance computed during the training phase.
        /// </summary>
        /// <value>
        /// The running variance tensor.
        /// </value>
        public ITensor RunningVar { get; private set; }
        private readonly float Momentum;
        private readonly int _numFeatures;

        public BatchNorm(int numFeatures, float eps = 1e-5f, float momentum = 0.1f, bool useAffine = true)
            : base(numFeatures, eps, useAffine)
        {
            Momentum = momentum;
            _numFeatures = numFeatures;
            RunningMean = Tensor.Zeros(new TensorShape(numFeatures));
            RunningVar = Tensor.Ones(new TensorShape(numFeatures));
        }
        /// <summary>
        /// Moves this layer, its running statistics, and its learned parameters to the target device.
        /// </summary>
        /// <param name="targetDevice">The destination execution device.</param>

        public override void To(Device targetDevice)
        {
            base.To(targetDevice); // Migrates Gamma and Beta automatically
            RunningMean = RunningMean.To(targetDevice);
            RunningVar = RunningVar.To(targetDevice);
        }
        /// <summary>
        /// Performs the forward pass normalization on the input tensor.
        /// During training, calculates the batch mean and variance and updates the running statistics.
        /// During inference, applies the computed running mean and running variance for normalization.
        /// </summary>
        /// <param name="input">The input tensor to be normalized.</param>
        /// <returns>The normalized tensor of the same shape as the input.</returns>

        protected override ITensor Normalize(ITensor input)
        {
            int rank = input.Shape.Rank;
            int[] reduceAxes = GetReduceAxes(rank);

            int[] broadcastShape = Enumerable.Repeat(1, rank).ToArray();
            broadcastShape[1] = _numFeatures;

            ITensor mean;
            ITensor var_;

            if (IsTraining)
            {
                mean = input.Mean(reduceAxes, keepDims: true);
                var_ = input.Subtract(mean).Pow(2).Mean(reduceAxes, keepDims: true);

                // Flatten running stats back to 1D [numFeatures] for update
                var flatMean = mean.Reshape(_numFeatures);
                var flatVar = var_.Reshape(_numFeatures);

                // Update EMA on the correct device
                RunningMean = RunningMean.Multiply(1f - Momentum).Add(flatMean.Multiply(Momentum));
                RunningVar = RunningVar.Multiply(1f - Momentum).Add(flatVar.Multiply(Momentum));
            }
            else
            {
                mean = RunningMean.Reshape(broadcastShape);
                var_ = RunningVar.Reshape(broadcastShape);
            }

            var std = var_.Add(Eps).Sqrt();
            return input.Subtract(mean).Divide(std);
        }
        /// <summary>
        /// Computes the gradient of the loss with respect to the input tensor (backward pass).
        /// </summary>
        /// <param name="input">The original input tensor from the forward pass.</param>
        /// <param name="gradOutput">The gradient of the loss with respect to the output of this layer.</param>
        /// <returns>The gradient of the loss with respect to the input.</returns>

        protected override ITensor ComputeGradInput(ITensor input, ITensor gradOutput)
        {
            int rank = input.Shape.Rank;
            int[] reduceAxes = GetReduceAxes(rank);

            int[] broadcastShape = Enumerable.Repeat(1, rank).ToArray();
            broadcastShape[1] = _numFeatures;

            var mean = IsTraining ? input.Mean(reduceAxes, keepDims: true) : RunningMean.Reshape(broadcastShape);
            var var_ = IsTraining ? input.Subtract(mean).Pow(2).Mean(reduceAxes, keepDims: true) : RunningVar.Reshape(broadcastShape);
            var std = var_.Add(Eps).Sqrt();

            var normalized = input.Subtract(mean).Divide(std);
            var gradNorm = gradOutput.Multiply(UseAffine ? Gamma.Reshape(broadcastShape) : Tensor.Ones(input.Shape, input.Device));

            float numElementsToReduce = 1f;
            foreach (int axis in reduceAxes) numElementsToReduce *= input.Shape[axis];

            var N = Tensor.FromScalar(numElementsToReduce, input.Device);

            var sumGradNorm = gradNorm.Sum(reduceAxes, keepDims: true);
            var sumGradNormNorm = gradNorm.Multiply(normalized).Sum(reduceAxes, keepDims: true);

            var term1 = gradNorm.Multiply(N);
            var term2 = sumGradNorm;
            var term3 = normalized.Multiply(sumGradNormNorm);

            var num = term1.Subtract(term2).Subtract(term3);
            var den = N.Multiply(std);

            return num.Divide(den);
        }
        /// <summary>
        /// Computes the reduction axes for calculating statistics based on the input tensor's rank.
        /// Maps the dimensions that should be normalized across, skipping the feature dimension (axis 1).
        /// </summary>
        /// <param name="rank">The dimensional rank of the input tensor.</param>
        /// <returns>An array of axes to be reduced.</returns>

        private int[] GetReduceAxes(int rank)
        {
            return rank switch
            {
                2 => new[] { 0 },
                3 => new[] { 0, 2 },
                4 => new[] { 0, 2, 3 },
                5 => new[] { 0, 2, 3, 4 },
                _ => Enumerable.Range(0, rank).Where(i => i != 1).ToArray()
            };
        }
    }
}
