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
    using System.Linq;
    /// <summary>
    /// Encapsulates the metadata—including shape, strides, and memory offset—required to represent 
    /// and manipulate zero-copy views (such as slices, transpositions, and reshapes) of multi-dimensional tensor data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This metadata-only representation allows multiple tensor views to share the same underlying physical 
    /// memory buffer without duplicating data.
    /// </para>
    /// <para>
    /// Strides dictate the physical memory step size (in elements) required to traverse to the next element 
    /// along each respective dimension. A contiguous, row-major layout (C-style) will have strides computed 
    /// in decreasing order based on the dimensions.
    /// </para>
    /// </remarks>

    #endregion

    public sealed class TensorLayout : IEquatable<TensorLayout>
    {
        /// <summary>
        /// Gets the shape (dimensions) of the tensor layout.
        /// </summary>
        /// <value>
        /// A <see cref="TensorShape"/> instance detailing the size and configuration of each dimension of the tensor.
        /// </value>
        public TensorShape Shape { get; }
        /// <summary>
        /// Gets the memory stride array for each dimension of the tensor.
        /// </summary>
        /// <value>
        /// An array of integers where each element specifies the linear stride (in elements) 
        /// required to step from one element to the next along that dimension.
        /// </value>

        public int[] Strides { get; }
        /// <summary>
        /// Gets the starting memory offset of this tensor view relative to the start of the underlying storage buffer.
        /// </summary>
        /// <value>
        /// An integer index representing the element-wise start position in the shared data array.
        /// </value>

        public int Offset { get; }
        /// <summary>
        /// Gets the rank (number of dimensions) of the tensor layout.
        /// </summary>
        /// <value>
        /// An integer indicating the dimensionality. Equivalent to <c>Shape.Rank</c>.
        /// </value>

        public int Rank => Shape.Rank;
        /// <summary>
        /// Gets the total number of elements represented by this tensor layout.
        /// </summary>
        /// <value>
        /// An integer representing the capacity. Equivalent to <c>Shape.TotalElements</c>.
        /// </value>

        public int TotalElements => Shape.TotalElements;

        /// <summary>
        /// Initializes a new instance of the <see cref="TensorLayout"/> class with contiguous strides.
        /// </summary>
        /// <param name="shape">The shape of the tensor defining its dimensions.</param>
        /// <param name="offset">The memory offset in elements from the start of the buffer. Defaults to 0.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="shape"/> is <see langword="null"/>.</exception>
        public TensorLayout(TensorShape shape, int offset = 0)
        {
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
            Offset = offset;
            Strides = ComputeContiguousStrides(shape.Dimensions);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TensorLayout"/> class with explicit strides and offset.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="strides">The strides for each dimension, specifying elements to skip to reach the next element in that dimension.</param>
        /// <param name="offset">The memory offset in elements.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="shape"/> or <paramref name="strides"/> is <see langword="null"/>.</exception>
        public TensorLayout(TensorShape shape, int[] strides, int offset)
        {
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
            Strides = strides ?? throw new ArgumentNullException(nameof(strides));
            Offset = offset;
        }
        /// <summary>
        /// Computes contiguous, row-major (C-style) strides for the given dimension sizes.
        /// </summary>
        /// <param name="dimensions">An array of integers representing the size of each dimension.</param>
        /// <returns>An array of strides corresponding to a contiguous memory layout for the dimensions.</returns>

        private static int[] ComputeContiguousStrides(int[] dimensions)
        {
            int[] strides = new int[dimensions.Length];
            int currentStride = 1;
            for (int i = dimensions.Length - 1; i >= 0; i--)
            {
                strides[i] = currentStride;
                currentStride *= dimensions[i];
            }
            return strides;
        }
        /// <summary>
        /// Computes the linear memory index in the underlying one-dimensional flat array for the given multidimensional coordinates.
        /// </summary>
        /// <param name="indices">An array of coordinates representing the index along each dimension.</param>
        /// <returns>The calculated flat 1D index within the physical storage buffer.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="indices"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when the length of <paramref name="indices"/> does not match the <see cref="Rank"/> of this layout.</exception>
        /// <exception cref="IndexOutOfRangeException">Thrown when any coordinate in <paramref name="indices"/> is less than zero or greater than or equal to the size of its corresponding dimension.</exception>

        public int GetLinearIndex(int[] indices)
        {
            if (indices.Length != Rank)
                throw new ArgumentException("Indices dimensions must match layout rank.");

            int index = Offset;
            for (int i = 0; i < Rank; i++)
            {
                if (indices[i] < 0 || indices[i] >= Shape.Dimensions[i])
                    throw new IndexOutOfRangeException($"Index {indices[i]} out of bounds for dimension {i} (size {Shape.Dimensions[i]}).");
                index += indices[i] * Strides[i];
            }
            return index;
        }
        /// <summary>
        /// Creates a sliced view of the current tensor layout without copying any underlying data.
        /// </summary>
        /// <param name="slices">
        /// An array of slicing parameters, where each tuple contains:
        /// <list type="bullet">
        /// <item>
        /// <term><c>start</c></term>
        /// <description>The inclusive start index of the slice on this dimension.</description>
        /// </item>
        /// <item>
        /// <term><c>end</c></term>
        /// <description>The exclusive end index of the slice on this dimension. Use <c>-1</c> to slice to the very end of the dimension.</description>
        /// </item>
        /// <item>
        /// <term><c>step</c></term>
        /// <description>The step size (stride factor) of the slice. If set to <c>0</c>, a default step size of <c>1</c> is applied.</description>
        /// </item>
        /// </list>
        /// </param>
        /// <returns>A new <see cref="TensorLayout"/> that describes the sliced subset of this layout.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="slices"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when the number of slices does not match the <see cref="Rank"/> of the layout.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the slice range or step is invalid (e.g., negative indices, out-of-bounds start/end, or end is smaller than start).</exception>

        public TensorLayout Slice(params (int start, int end, int step)[] slices)
        {
            if (slices.Length != Rank)
                throw new ArgumentException("Slice dimensions must match layout rank.");

            int[] newDims = new int[Rank];
            int[] newStrides = new int[Rank];
            int newOffset = Offset;

            for (int i = 0; i < Rank; i++)
            {
                int dimSize = Shape.Dimensions[i];
                int start = slices[i].start;
                int end = slices[i].end == -1 ? dimSize : slices[i].end;
                int step = slices[i].step == 0 ? 1 : slices[i].step;

                if (start < 0 || start >= dimSize || end < start || end > dimSize)
                    throw new ArgumentOutOfRangeException($"Invalid slice range [{start}:{end}:{step}] for dimension of size {dimSize}.");

                newDims[i] = ((end - start - 1) / step) + 1;
                newStrides[i] = Strides[i] * step;
                newOffset += start * Strides[i];
            }

            return new TensorLayout(new TensorShape(newDims), newStrides, newOffset);
        }
        /// <summary>
        /// Compares the current <see cref="TensorLayout"/> with another for structural and metadata equality.
        /// </summary>
        /// <param name="other">The other <see cref="TensorLayout"/> to evaluate.</param>
        /// <returns>
        /// <see langword="true"/> if <paramref name="other"/> is not <see langword="null"/> and possesses the exact same shape, strides, and memory offset; 
        /// otherwise, <see langword="false"/>.
        /// </returns>

        public bool Equals(TensorLayout? other)
        {
            if (other is null) return false;
            return Shape.Equals(other.Shape) && Strides.SequenceEqual(other.Strides) && Offset == other.Offset;
        }
        /// <summary>
        /// Compares the current <see cref="TensorLayout"/> with an object to determine equality.
        /// </summary>
        /// <param name="obj">The object to compare with the current layout.</param>
        /// <returns>
        /// <see langword="true"/> if <paramref name="obj"/> is a <see cref="TensorLayout"/> and matches the current instance's 
        /// shape, strides, and offset; otherwise, <see langword="false"/>.
        /// </returns>

        public override bool Equals(object? obj) => obj is TensorLayout other && Equals(other);
        /// <summary>
        /// Computes a hash code for the current <see cref="TensorLayout"/> instance.
        /// </summary>
        /// <returns>A 32-bit signed integer hash code computed from the shape, strides, and offset.</returns>

        public override int GetHashCode() => HashCode.Combine(Shape, Strides, Offset);
    }
}