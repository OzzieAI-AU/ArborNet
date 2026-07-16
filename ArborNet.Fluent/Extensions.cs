using System;
using ArborNet.Core.Interfaces;
using ArborNet.Fluent;

namespace ArborNet.Fluent
{
    /// <summary>
    /// Provides extension methods for seamless, safe conversion between native ArborNet ITensor 
    /// primitives and the expressive X fluent wrappers.
    /// </summary>
    public static class TensorFluentExtensions
    {
        /// <summary>
        /// Safely converts a native ITensor into a fluent X wrapper.
        /// Useful when concluding native ITensor math and returning to a fluent pipeline.
        /// </summary>
        /// <param name="tensor">The native ITensor to wrap.</param>
        /// <returns>An instance of the X fluent API wrapper.</returns>
        /// <exception cref="ArgumentNullException">Thrown if the input tensor is null.</exception>
        public static X ToX(this ITensor tensor)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor), "Cannot convert a null ITensor to an X fluent wrapper.");
            }

            // X.Of is the elegant factory method defined in your X.cs class
            return X.Of(tensor);
        }

        /// <summary>
        /// Safely extracts the native ITensor from a fluent X wrapper.
        /// Useful when dropping down from a fluent script into native Core layers.
        /// </summary>
        /// <param name="x">The X fluent API wrapper.</param>
        /// <returns>The underlying native ITensor.</returns>
        /// <exception cref="ArgumentNullException">Thrown if the input wrapper is null.</exception>
        public static ITensor ToTensor(this X x)
        {
            if (x == null)
            {
                throw new ArgumentNullException(nameof(x), "Cannot extract an ITensor from a null X wrapper.");
            }

            // Directly access the underlying tensor property
            return x.Tensor;
        }
    }
}