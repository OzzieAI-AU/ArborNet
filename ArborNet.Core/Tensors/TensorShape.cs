using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;

namespace ArborNet.Core.Tensors
{
    /// <summary>
    /// Represents the shape of a multi-dimensional tensor, encapsulating its dimensions,
    /// rank, and total element count.
    /// </summary>
    public class TensorShape : IEnumerable<int>
    {
        public int[] Dimensions { get; }
        public int Rank => Dimensions.Length;
        public int TotalElements { get; }
        public bool IsScalar => Rank == 0 || (Rank == 1 && Dimensions[0] == 1);

        public TensorShape(params int[] dimensions)
        {
            if (dimensions == null || dimensions.Length == 0)
            {
                Dimensions = Array.Empty<int>();
                TotalElements = 1;
                return;
            }
            foreach (var d in dimensions)
                if (d < 0) throw new ArgumentException("Dimension cannot be negative");
            Dimensions = (int[])dimensions.Clone();
            TotalElements = Dimensions.Aggregate(1, (a, b) => a * b);
        }

        public int this[int index] => Dimensions[index];

        public bool Equals(TensorShape? other)
        {
            if (other is null) return false;
            if (Rank != other.Rank) return false;
            return Dimensions.SequenceEqual(other.Dimensions);
        }

        public override bool Equals(object? obj) => obj is TensorShape s && Equals(s);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                foreach (var dim in Dimensions)
                    hash = hash * 31 + dim;
                return hash;
            }
        }

        public override string ToString() => $"[{string.Join(", ", Dimensions)}]";

        public bool IsCompatibleWithBroadcast(TensorShape other)
        {
            if (other is null) return false;

            var a = Dimensions.Reverse().ToArray();
            var b = other.Dimensions.Reverse().ToArray();
            int maxLen = Math.Max(a.Length, b.Length);

            for (int i = 0; i < maxLen; i++)
            {
                int da = i < a.Length ? a[i] : 1;
                int db = i < b.Length ? b[i] : 1;

                if (da != db && da != 1 && db != 1)
                    return false;
            }
            return true;
        }

        public TensorShape BroadcastTo(TensorShape other)
        {
            if (!IsCompatibleWithBroadcast(other))
                throw new ArgumentException($"Shapes are not broadcast compatible: {this} vs {other}");
            var a = Dimensions.Reverse().ToList();
            var b = other.Dimensions.Reverse().ToList();
            var result = new List<int>();
            int maxLen = Math.Max(a.Count, b.Count);
            for (int i = 0; i < maxLen; i++)
            {
                int da = i < a.Count ? a[i] : 1;
                int db = i < b.Count ? b[i] : 1;
                result.Add(Math.Max(da, db));
            }
            result.Reverse();
            return new TensorShape(result.ToArray());
        }

        public TensorShape Clone() => new TensorShape(Dimensions.ToArray());

        /// <summary>
        /// Renamed to avoid hiding Linq's Enumerable.Skip extension method.
        /// </summary>
        public TensorShape SkipLast(int v)
        {
            if (v < 0)
                throw new ArgumentOutOfRangeException(nameof(v), v, "Skip count cannot be negative.");

            if (Rank == 0)
                throw new InvalidOperationException("Cannot skip elements on a scalar tensor shape (rank 0).");

            int lastDimSize = Dimensions[Rank - 1];

            if (v >= lastDimSize)
                throw new ArgumentOutOfRangeException(nameof(v), v,
                    $"Cannot skip {v} elements when the last dimension only has {lastDimSize} elements.");

            int[] newDimensions = (int[])Dimensions.Clone();
            newDimensions[Rank - 1] = lastDimSize - v;

            return new TensorShape(newDimensions);
        }

        // IEnumerable Implementation for Clean Linq Chaining
        public IEnumerator<int> GetEnumerator() => ((IEnumerable<int>)Dimensions).GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => Dimensions.GetEnumerator();
    }
}