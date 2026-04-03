using System;
using System.Collections.Generic;
using System.Linq;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;

namespace ArborNet.Core.Tensors
{
    /// <summary>
    /// Represents the shape of a multi-dimensional tensor, encapsulating its dimensions,
    /// rank, and total element count. Provides utilities for shape comparison,
    /// broadcasting compatibility, and broadcasting operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// TensorShape is immutable in concept but exposes its internal dimension array.
    /// The shape is used throughout the tensor system to validate operations and 
    /// determine memory layout.
    /// </para>
    /// <para>
    /// A scalar is represented as either an empty dimension list (rank 0) or a single
    /// dimension of size 1.
    /// </para>
    /// </remarks>
    public class TensorShape
    {
        /// <summary>
        /// Gets the dimensions of the tensor.
        /// </summary>
        /// <value>
        /// An array containing the size of each dimension. 
        /// The returned array is a reference to the internal storage and should not be modified.
        /// </value>
        public int[] Dimensions { get; }

        /// <summary>
        /// Gets the rank (number of dimensions) of the tensor.
        /// </summary>
        /// <value>The length of the <see cref="Dimensions"/> array.</value>
        public int Rank => Dimensions.Length;

        /// <summary>
        /// Gets the total number of elements represented by this shape.
        /// </summary>
        /// <value>The product of all dimension values. Returns 1 for scalar shapes.</value>
        public int TotalElements { get; }

        /// <summary>
        /// Gets a value indicating whether this shape represents a scalar value.
        /// </summary>
        /// <value>
        /// <c>true</c> if the shape has rank 0 or is a 1-dimensional shape with a single element;
        /// otherwise, <c>false</c>.
        /// </value>
        public bool IsScalar => Rank == 0 || (Rank == 1 && Dimensions[0] == 1);

        /// <summary>
        /// Initializes a new instance of the <see cref="TensorShape"/> class.
        /// </summary>
        /// <param name="dimensions">The size of each dimension. Can be null or empty to represent a scalar.</param>
        /// <exception cref="ArgumentException">Thrown if any dimension is negative.</exception>
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

        /// <summary>
        /// Gets the dimension value at the specified index.
        /// </summary>
        /// <param name="index">The zero-based index of the dimension.</param>
        /// <returns>The size of the dimension at the specified index.</returns>
        public int this[int index] => Dimensions[index];

        /// <summary>
        /// Determines whether the specified <see cref="TensorShape"/> is equal to the current instance.
        /// </summary>
        /// <param name="other">The <see cref="TensorShape"/> to compare with this instance.</param>
        /// <returns>
        /// <c>true</c> if the shapes have the same rank and identical dimension values; otherwise, <c>false</c>.
        /// </returns>
        public bool Equals(TensorShape? other)
        {
            if (other is null) return false;
            if (Rank != other.Rank) return false;
            return Dimensions.SequenceEqual(other.Dimensions);
        }

        /// <summary>
        /// Determines whether the specified object is equal to the current <see cref="TensorShape"/>.
        /// </summary>
        /// <param name="obj">The object to compare with the current instance.</param>
        /// <returns>
        /// <c>true</c> if the specified object is a <see cref="TensorShape"/> and is equal to this instance;
        /// otherwise, <c>false</c>.
        /// </returns>
        public override bool Equals(object? obj) => obj is TensorShape s && Equals(s);

        /// <summary>
        /// Returns the hash code for this instance.
        /// </summary>
        /// <returns>A 32-bit signed integer hash code calculated from the dimensions.</returns>
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
        /// <returns>A string in the format "[dim1, dim2, ..., dimN]".</returns>
        public override string ToString() => $"[{string.Join(", ", Dimensions)}]";

        /// <summary>
        /// Determines whether this shape is compatible with the specified shape under broadcasting rules.
        /// </summary>
        /// <param name="other">The shape to check compatibility with.</param>
        /// <returns>
        /// <c>true</c> if the two shapes are broadcast compatible; otherwise, <c>false</c>.
        /// </returns>
        /// <remarks>
        /// Two shapes are compatible if, when aligned from the trailing dimensions, 
        /// for each dimension pair: the sizes are equal, or one of them is 1.
        /// </remarks>
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
        /// Computes the resulting shape when this shape is broadcasted to match the target shape.
        /// </summary>
        /// <param name="other">The target shape to broadcast to.</param>
        /// <returns>A new <see cref="TensorShape"/> representing the broadcasted dimensions.</returns>
        /// <exception cref="ArgumentException">
        /// Thrown when the shapes are not broadcast compatible.
        /// </exception>
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
        /// Creates a new <see cref="TensorShape"/> that is a copy of the current instance.
        /// </summary>
        /// <returns>A new independent <see cref="TensorShape"/> with identical dimensions.</returns>
        public TensorShape Clone() => new TensorShape(Dimensions.ToArray());

        /// <summary>
        /// Skips the first <paramref name="v"/> elements along the last dimension of the tensor shape
        /// and returns a new shape representing the sliced view.
        /// <para>
        /// This operation does not copy data — it only adjusts the shape metadata.
        /// In a full tensor implementation this would typically return a view (ITensor) into the original data.
        /// </para>
        /// </summary>
        /// <param name="v">The number of elements to skip from the beginning of the last dimension.</param>
        /// <returns>A new <see cref="TensorShape"/> with the last dimension reduced by <paramref name="v"/>.</returns>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown when <paramref name="v"/> is negative or exceeds the size of the last dimension.
        /// </exception>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the shape is scalar (rank 0) and cannot be sliced.
        /// </exception>
        public TensorShape Skip(int v)
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
        /// Computes the aggregate value (typically the product) of all dimensions in the shape,
        /// starting with the supplied initial value.
        /// <para>
        /// This is the standard reduction used throughout the library for reshape operations,
        /// e.g. <c>shape.Skip(2).Aggregate(1, (a, b) => a * b)</c>.
        /// </para>
        /// </summary>
        /// <typeparam name="T">Type of the accumulator (usually <see cref="int"/>).</typeparam>
        /// <param name="initialValue">The starting value for the aggregation (normally 1 when computing product).</param>
        /// <param name="aggregator">The function applied to each dimension (e.g. <c>(a, b) => a * b</c>).</param>
        /// <returns>The final aggregated integer value.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="aggregator"/> is null.</exception>
        public int Aggregate<T>(T initialValue, Func<T, int, T> aggregator)
        {
            if (aggregator == null)
                throw new ArgumentNullException(nameof(aggregator), "Aggregation function cannot be null.");

            T result = initialValue;

            foreach (int dimension in Dimensions)
            {
                result = aggregator(result, dimension);
            }

            return Convert.ToInt32(result);
        }
    }
}