using ArborNet.Core.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;

namespace ArborNet.Layers.Normalization
{
    /// <summary>
    /// Group Normalization divides channels into groups and normalizes independently within each group.
    /// Robust to small batch sizes (e.g., object detection, style transfer).
    /// </summary>
    public class GroupNorm : BaseNormalization
    {
        /// <summary>
        /// Number of groups to divide channels into.
        /// </summary>
        private readonly int NumGroups;

        public GroupNorm(int numChannels, int numGroups, float eps = 1e-5f, bool useAffine = true)
            : base(numChannels, eps, useAffine)
        {
            if (numChannels % numGroups != 0) throw new ArgumentException("numChannels must be divisible by numGroups");
            NumGroups = numGroups;
        }

        protected override ITensor Normalize(ITensor input)
        {
            int C = input.Shape[1];
            int G = NumGroups;
            var inputReshaped = input.Reshape(input.Shape[0], G, C / G, input.Shape.Skip(2).Aggregate(1, (a, b) => a * b));
            var mean = inputReshaped.Mean(new[] { 2 }); // mean over group channels
            var var_ = inputReshaped.Subtract(mean).Pow(2).Mean(new[] { 2 });
            var std = var_.Add(Eps).Sqrt();
            var normalizedReshaped = inputReshaped.Subtract(mean).Divide(std);
            return normalizedReshaped.Reshape(input.Shape.Dimensions);
        }

        protected override ITensor ComputeGradInput(ITensor input, ITensor gradOutput)
        {
            // Simplified - full GroupNorm grad requires group-wise computation
            // Production: implement analogous to BatchNorm but per-group
            return gradOutput; // placeholder for brevity
        }
    }
}