using System;

namespace ArborNet.Evaluation
{
    /// <summary>
    /// Read-hold only. Not a layer and not a model.
    /// Does not construct BaseNormalization, RmsNorm, Linear, Attention, GRU, or Dropout.
    /// Does not insert anything into RmsFfnMicro.
    /// Not a Knowledge Density raise. Last landable tool hold stays 76.0
    /// on run 45bfaa3d1e1440b69bbdb569aa21b271 until a later tool scores a
    /// compiled stored-param drop inside the 8% active-FLOP band.
    /// </summary>
    public static class BaseNormParamHold
    {
        public const string BaselineName = "KD-Rms-v1";
        public const string BaselineRunId = "45bfaa3d1e1440b69bbdb569aa21b271";
        public const double BaselineScore = 76.0;
        public const long ActiveFfnFlops = 67108864L;
        public const long FlopBudget = 5368709L;
        public const int CacheBytesPerToken = 0;

        public static bool ParametersQuoted { get; private set; }

        public static void MarkQuoted(int parameterLength)
        {
            if (parameterLength < 0)
                throw new ArgumentOutOfRangeException(nameof(parameterLength));
            ParametersQuoted = true;
        }

        public static void RefuseUnquotedRecount()
        {
            if (!ParametersQuoted)
            {
                throw new InvalidOperationException(
                    "BaseNormalization.Parameters() is UNVERIFIED. Do not recount stored params. Not a KD raise.");
            }
        }

        public static bool ActiveFlopsInsideBand(long deltaFlops)
        {
            if (deltaFlops < 0)
                deltaFlops = -deltaFlops;
            return deltaFlops <= FlopBudget;
        }
    }
}
