// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Tensors
{

    #region Using Statements:

    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;
    /// <summary>
    /// Represents the immutable shape of a multi-dimensional tensor, encapsulating its dimensions,
    /// rank, and total element count. Fully supports modern C# index and range syntax.
    /// </summary>
    /// <remarks>
    /// This class is designed to handle tensor shapes and facilitate operations such as
    /// broadcasting, dimension access, and collection enumeration.
    /// </remarks>

    #endregion

    public sealed class TensorShape : IEnumerable<int>, IEquatable<TensorShape>
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

            Dimensions = dimensions; // Avoid .Clone() to save GC allocation

            int total = 1;
            for (int i = 0; i < dimensions.Length; i++)
            {
                if (dimensions[i] < 0) throw new ArgumentException("Dimension cannot be negative");
                total *= dimensions[i];
            }
            TotalElements = total;
        }

        public int this[int index] => Dimensions[index];
        public int this[Index index] => Dimensions[index.GetOffset(Rank)];

        public bool Equals(TensorShape? other)
        {
            if (ReferenceEquals(this, other)) return true;
            if (other is null || Rank != other.Rank) return false;
            return Dimensions.SequenceEqual(other.Dimensions);
        }

        public override bool Equals(object? obj) => obj is TensorShape s && Equals(s);

        public override int GetHashCode()
        {
            int hash = 17;
            for (int i = 0; i < Dimensions.Length; i++) hash = hash * 31 + Dimensions[i];
            return hash;
        }

        public override string ToString() => $"[{string.Join(", ", Dimensions)}]";

        public bool IsCompatibleWithBroadcast(TensorShape other)
        {
            if (other is null) return false;
            int maxLen = Math.Max(Rank, other.Rank);
            for (int i = 0; i < maxLen; i++)
            {
                int da = i < Rank ? Dimensions[Rank - 1 - i] : 1;
                int db = i < other.Rank ? other.Dimensions[other.Rank - 1 - i] : 1;
                if (da != db && da != 1 && db != 1) return false;
            }
            return true;
        }

        public TensorShape BroadcastTo(TensorShape other)
        {
            if (!IsCompatibleWithBroadcast(other))
                throw new ArgumentException($"Shapes are not broadcast compatible: {this} vs {other}");

            int maxLen = Math.Max(Rank, other.Rank);
            int[] result = new int[maxLen];

            for (int i = 0; i < maxLen; i++)
            {
                int da = i < Rank ? Dimensions[Rank - 1 - i] : 1;
                int db = i < other.Rank ? other.Dimensions[other.Rank - 1 - i] : 1;
                result[maxLen - 1 - i] = Math.Max(da, db);
            }
            return new TensorShape(result);
        }

        // TensorShape is immutable, so Clone() can safely just return itself (Massive GC saving)
        public TensorShape Clone() => this;

        public TensorShape SkipLast(int v)
        {
            if (v < 0) throw new ArgumentOutOfRangeException(nameof(v));
            if (Rank == 0) throw new InvalidOperationException("Cannot skip on scalar.");
            if (v >= Dimensions[Rank - 1]) throw new ArgumentOutOfRangeException(nameof(v));

            int[] newDimensions = (int[])Dimensions.Clone();
            newDimensions[Rank - 1] -= v;
            return new TensorShape(newDimensions);
        }

        public IEnumerator<int> GetEnumerator() => ((IEnumerable<int>)Dimensions).GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => Dimensions.GetEnumerator();
    }
}