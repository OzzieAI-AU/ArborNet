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

    public class TensorShape : IEnumerable<int>
    {
        /// <summary>
        /// Gets the array containing the size of each dimension in the tensor shape.
        /// </summary>
        /// <value>
        /// An array of integers representing the size of each dimension.
        /// </value>
        public int[] Dimensions { get; }
        /// <summary>
        /// Gets the number of dimensions (rank) of the tensor shape.
        /// </summary>
        /// <value>
        /// The rank of the tensor, corresponding to the length of the <see cref="Dimensions"/> array.
        /// </value>

        public int Rank => Dimensions.Length;
        /// <summary>
        /// Gets the total number of elements represented by this shape.
        /// </summary>
        /// <value>
        /// The product of all dimensions. For scalar shapes (rank 0), this is 1.
        /// </value>

        public int TotalElements { get; }
        /// <summary>
        /// Gets a value indicating whether this shape represents a scalar.
        /// </summary>
        /// <value>
        /// <see langword="true"/> if the shape represents a scalar (rank is 0, or rank is 1 and the single dimension is 1); otherwise, <see langword="false"/>.
        /// </value>

        public bool IsScalar => Rank == 0 || (Rank == 1 && Dimensions[0] == 1);

        /// <summary>
        /// Initializes a new instance of the <see cref="TensorShape"/> class with the specified dimensions.
        /// </summary>
        /// <param name="dimensions">An array of integers representing the size of each dimension.</param>
        /// <exception cref="ArgumentException">Thrown when any of the dimensions is negative.</exception>
        public TensorShape(params int[] dimensions)
        {
            if (dimensions == null || dimensions.Length == 0)
            {
                Dimensions = Array.Empty<int>();
                TotalElements = 1;
                return;
            }
            foreach (var d in dimensions)
            {
                if (d < 0) throw new ArgumentException("Dimension cannot be negative");
            }
            Dimensions = (int[])dimensions.Clone();
            TotalElements = Dimensions.Aggregate(1, (a, b) => a * b);
        }

        /// <summary>
        /// Gets the size of the dimension at the specified zero-based index.
        /// </summary>
        /// <param name="index">The zero-based index of the dimension.</param>
        /// <returns>The size of the dimension at the specified index.</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown when <paramref name="index"/> is outside the bounds of the dimensions array.</exception>
        public int this[int index] => Dimensions[index];

        /// <summary>
        /// Support indexer for modern C# index syntax (e.g., shape[^1]).
        /// </summary>
        /// <param name="index">The index, supporting standard index and index-from-end syntax.</param>
        /// <returns>The size of the dimension at the specified index.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the computed index is out of bounds.</exception>
        public int this[Index index] => Dimensions[index.GetOffset(Rank)];
        /// <summary>
        /// Determines whether the specified <see cref="TensorShape"/> is equal to the current <see cref="TensorShape"/>.
        /// </summary>
        /// <param name="other">The <see cref="TensorShape"/> to compare with the current <see cref="TensorShape"/>.</param>
        /// <returns><see langword="true"/> if the specified shape has the same rank and dimensions as the current shape; otherwise, <see langword="false"/>.</returns>

        public bool Equals(TensorShape? other)
        {
            if (other is null) return false;
            if (Rank != other.Rank) return false;
            return Dimensions.SequenceEqual(other.Dimensions);
        }
        /// <summary>
        /// Determines whether the specified object is equal to the current <see cref="TensorShape"/>.
        /// </summary>
        /// <param name="obj">The object to compare with the current <see cref="TensorShape"/>.</param>
        /// <returns><see langword="true"/> if the specified object is a <see cref="TensorShape"/> and is equal to the current shape; otherwise, <see langword="false"/>.</returns>

        public override bool Equals(object? obj) => obj is TensorShape s && Equals(s);
        /// <summary>
        /// Serves as the default hash function.
        /// </summary>
        /// <returns>A hash code for the current <see cref="TensorShape"/>.</returns>

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
        /// <summary>
        /// Returns a string representation of the tensor shape.
        /// </summary>
        /// <returns>A string representation in the format "[dim1, dim2, ...]".</returns>

        public override string ToString() => $"[{string.Join(", ", Dimensions)}]";
        /// <summary>
        /// Determines whether the current shape is compatible for broadcasting with another shape.
        /// </summary>
        /// <remarks>
        /// Broadcasting is compatible if for each dimension starting from the trailing dimension,
        /// the dimension sizes are equal, or one of them is equal to 1.
        /// </remarks>
        /// <param name="other">The other <see cref="TensorShape"/> to evaluate compatibility with.</param>
        /// <returns><see langword="true"/> if the shapes can be broadcast together; otherwise, <see langword="false"/>. If <paramref name="other"/> is null, returns <see langword="false"/>.</returns>

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
        /// <summary>
        /// Broadcasts the current shape to match the target compatible <see cref="TensorShape"/>.
        /// </summary>
        /// <param name="other">The target shape to broadcast to.</param>
        /// <returns>A new <see cref="TensorShape"/> representing the broadcasted shape.</returns>
        /// <exception cref="ArgumentException">Thrown when the current shape is not compatible with the target shape for broadcasting, or when <paramref name="other"/> is null.</exception>

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
        /// <summary>
        /// Creates a deep copy of the current <see cref="TensorShape"/>.
        /// </summary>
        /// <returns>A new <see cref="TensorShape"/> instance with identical dimensions.</returns>

        public TensorShape Clone() => new TensorShape(Dimensions.ToArray());
        /// <summary>
        /// Reduces the size of the last dimension by the specified amount.
        /// </summary>
        /// <remarks>
        /// This method is renamed from a standard skip implementation to avoid hiding Linq's <see cref="Enumerable.Skip"/> extension method.
        /// </remarks>
        /// <param name="v">The amount by which to decrease the last dimension's size.</param>
        /// <returns>A new <see cref="TensorShape"/> with the adjusted last dimension.</returns>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown when <paramref name="v"/> is negative, or greater than or equal to the size of the last dimension.
        /// </exception>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the tensor shape represents a scalar (rank 0).
        /// </exception>

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
        /// <summary>
        /// Returns an enumerator that iterates through the dimensions of the tensor shape.
        /// </summary>
        /// <returns>An enumerator for the dimensions.</returns>

        public IEnumerator<int> GetEnumerator() => ((IEnumerable<int>)Dimensions).GetEnumerator();
        /// <summary>
        /// Returns an enumerator that iterates through the dimensions of the tensor shape.
        /// </summary>
        /// <returns>An enumerator for the dimensions.</returns>

        IEnumerator IEnumerable.GetEnumerator() => Dimensions.GetEnumerator();
    }
}