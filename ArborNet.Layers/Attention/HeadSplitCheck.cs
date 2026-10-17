using System;

namespace ArborNet.Layers
{
    /// <summary>
    /// Head-split precheck only. Not inserted into a scored graph.
    /// Not a Knowledge Density raise. Does not construct MultiHeadAttention,
    /// Linear, or Embedding, and does not allocate weights.
    /// </summary>
    public static class HeadSplitCheck
    {
        public readonly struct Split
        {
            public Split(bool ok, int dHead, string reason)
            {
                Ok = ok;
                DHead = dHead;
                Reason = reason ?? string.Empty;
            }

            public bool Ok { get; }
            public int DHead { get; }
            public string Reason { get; }
        }

        /// <summary>
        /// Rejects numHeads &lt;= 0 and dModel &lt;= 0 before division.
        /// A passing split is not a skipped MatMul and not a param drop.
        /// </summary>
        public static Split Evaluate(int dModel, int numHeads)
        {
            if (numHeads <= 0)
            {
                return new Split(false, 0, "numHeads must be > 0");
            }

            if (dModel <= 0)
            {
                return new Split(false, 0, "dModel must be > 0");
            }

            if (dModel % numHeads != 0)
            {
                return new Split(false, 0, "dModel must be divisible by numHeads");
            }

            return new Split(true, dModel / numHeads, string.Empty);
        }
    }
}
