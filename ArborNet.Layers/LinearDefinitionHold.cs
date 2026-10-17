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
    /// Read-only hold. Not a layer insert and not a Knowledge Density raise.
    /// Benchmark run 125196dcef7445a694748cde2421690e did not select
    /// ArborNet.Layers/Linear.cs, so ctor, bias, Parameters() length/shape,
    /// and Forward stay UNVERIFIED. Call sites are not a definition.
    /// Score holds at 76.0 on run 45bfaa3d1e1440b69bbdb569aa21b271.
    /// </summary>
    public static class LinearDefinitionHold
    {
        public const string BaselineRunId = "45bfaa3d1e1440b69bbdb569aa21b271";
        public const double BaselineScore = 76.0;
        public const string ThisRunId = "125196dcef7445a694748cde2421690e";

        /// <summary>False until Linear.cs ctor and Parameters() are quoted from that file.</summary>
        public const bool LinearDefinitionQuoted = false;

        /// <summary>
        /// Refuses a shared-down stored-param booking. Does not allocate weights,
        /// does not construct Linear, and does not change active FLOPs.
        /// </summary>
        public static bool TryBookSharedDown(out string reason)
        {
            reason = "REFUSE: ArborNet.Layers/Linear.cs ctor, bias, Parameters() length/shape, and Forward are UNVERIFIED. Do not invent Linear(int,int). A shared down projection is not inserted. dP is not booked. Score holds at 76.0. Not a Knowledge Density raise.";
            return false;
        }
    }
}
