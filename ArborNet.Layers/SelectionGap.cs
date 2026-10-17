using System;
using System.Collections.Generic;
using System.Linq;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Models;
using ArborNet.Core.Tensors;
using ArborNet.Layers;

namespace ArborNet.Layers
{
    /// <summary>
    /// Not a Knowledge Density raise. Not inserted into the scored graph.
    /// Benchmark run 9533bc5cf56f491892948396c4746db8 held KD-Rms-v1 at 76.0
    /// because ArborNet.Layers/Linear.cs and ArborNet.Layers/Embedding.cs were
    /// not selected. Do not approve this file as an architecture change.
    /// Do not construct Linear or Embedding here. Do not call X.Rsqrt or Pow.
    /// </summary>
    public static class SelectionGap
    {
        public const double HeldScore = 76.0;
        public const string BaselineRunId = "45bfaa3d1e1440b69bbdb569aa21b271";
        public const string ThisRunId = "9533bc5cf56f491892948396c4746db8";

        public static bool TryBookDensity()
        {
            return false;
        }
    }
}
