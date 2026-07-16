using System;
using System.Collections.Generic;
using System.Linq;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Devices;

namespace ArborNet.Layers.Normalization
{
    public sealed class BatchNorm : BaseNormalization
    {
        public ITensor RunningMean { get; private set; }
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
        public override void To(Device targetDevice)
        {
            base.To(targetDevice); // Migrates Gamma and Beta automatically
            RunningMean = RunningMean.To(targetDevice);
            RunningVar = RunningVar.To(targetDevice);
        }

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
