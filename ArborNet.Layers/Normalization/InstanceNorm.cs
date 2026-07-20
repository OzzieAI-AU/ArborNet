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
    using System.Text;
    /// <summary>
    /// Represents an Instance Normalization layer.
    /// </summary>
    /// <remarks>
    /// Instance Normalization normalizes each sample and channel independently (batch size ignored).
    /// This is equivalent to <see cref="GroupNorm"/> where the number of groups is equal to the number of channels (G = C).
    /// It is commonly used in tasks like image style transfer and generative adversarial networks (GANs).
    /// </remarks>

    #endregion

    public class InstanceNorm : GroupNorm
    {
        public InstanceNorm(int numChannels, float eps = 1e-5f, bool useAffine = true)
            : base(numChannels, numChannels, eps, useAffine) { }
    }
}