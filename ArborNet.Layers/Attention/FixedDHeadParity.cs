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
    /// Fixed-dModel head-count parity gate.
    /// Not a layer. Not inserted into RmsFfnMicro. Not a Knowledge Density raise.
    /// Does not construct MultiHeadAttention, does not allocate bias, and does not book F or P_norm.
    /// X.Rsqrt and Pow(-0.5) are not used.
    /// </summary>
    public static class FixedDHeadParity
    {
        public const string BaselineRunId = "45bfaa3d1e1440b69bbdb569aa21b271";
        public const double HeldScore = 76.0;

        public static bool TryBookDensity(
            bool vehicleFlopsQuoted,
            bool parametersQuoted,
            int parameterLengthBefore,
            int parameterLengthAfter,
            out string reason)
        {
            if (!vehicleFlopsQuoted || !parametersQuoted)
            {
                reason = "UNVERIFIED: vehicle F or Parameters() is unquoted. Refuse insert. Do not book |dF|/F, F, or P_norm.";
                return false;
            }

            if (parameterLengthAfter >= parameterLengthBefore)
            {
                reason = "No shorter Parameters() list. Fixed-D head count is parity, not a stored-param drop.";
                return false;
            }

            reason = "Shorter Parameters() is not booked by this file. A later tool must score it inside |dF|/F <= 0.08.";
            return false;
        }
    }
}
