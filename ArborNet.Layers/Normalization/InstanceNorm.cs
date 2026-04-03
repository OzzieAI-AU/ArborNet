using System;
using System.Collections.Generic;
using System.Text;

namespace ArborNet.Layers.Normalization
{
    /// <summary>
    /// Instance Normalization normalizes each sample independently (G = C, batch size ignored).
    /// Equivalent to GroupNorm with num_groups = num_channels.
    /// </summary>
    public class InstanceNorm : GroupNorm
    {
        public InstanceNorm(int numChannels, float eps = 1e-5f, bool useAffine = true)
            : base(numChannels, numChannels, eps, useAffine) { }
    }
}