using System;
using System.Linq;

namespace ArborNet.Core.Tensors
{
    /// <summary>
    /// Encapsulates shape, strides, and memory offset metadata for zero-copy tensor views.
    /// </summary>
    public sealed class TensorLayout : IEquatable<TensorLayout>
    {
        public TensorShape Shape { get; }
        public int[] Strides { get; }
        public int Offset { get; }

        public int Rank => Shape.Rank;
        public int TotalElements => Shape.TotalElements;

        public TensorLayout(TensorShape shape, int offset = 0)
        {
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
            Offset = offset;
            Strides = ComputeContiguousStrides(shape.Dimensions);
        }

        public TensorLayout(TensorShape shape, int[] strides, int offset)
        {
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
            Strides = strides ?? throw new ArgumentNullException(nameof(strides));
            Offset = offset;
        }

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
        /// Calculates the absolute memory index for a set of N-dimensional coordinates.
        /// </summary>
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
        /// Returns a slice layout without copying the underlying tensor data.
        /// </summary>
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

        public bool Equals(TensorLayout? other)
        {
            if (other is null) return false;
            return Shape.Equals(other.Shape) && Strides.SequenceEqual(other.Strides) && Offset == other.Offset;
        }

        public override bool Equals(object? obj) => obj is TensorLayout other && Equals(other);

        public override int GetHashCode() => HashCode.Combine(Shape, Strides, Offset);
    }
}