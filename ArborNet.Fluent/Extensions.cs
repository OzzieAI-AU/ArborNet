// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Fluent
{

    #region Using Statements:

    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Fluent;
    /// <summary>
    /// Provides extension methods for seamless, safe conversion between native ArborNet <see cref="ITensor"/> 
    /// primitives, structural wrapper classes, and the expressive <see cref="X"/> fluent wrappers.
    /// </summary>
    /// <remarks>
    /// These extension methods simplify transitions between low-level tensor representations and 
    /// the fluent API syntax, enabling cleaner and more readable deep learning model definitions.
    /// </remarks>

    #endregion

    public static class TensorFluentExtensions
    {
        /// <summary>
        /// Converts a native <see cref="ITensor"/> into a fluent <see cref="X"/> wrapper.
        /// </summary>
        /// <param name="tensor">The native tensor instance to wrap.</param>
        /// <returns>A new <see cref="X"/> fluent wrapper instance containing the specified tensor.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="tensor"/> is <see langword="null"/>.</exception>
        public static X ToX(this ITensor tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor), "Cannot convert a null ITensor to an X fluent wrapper.");

            return X.Of(tensor);
        }
        /// <summary>
        /// Safely extracts the native <see cref="ITensor"/> from a fluent <see cref="X"/> wrapper.
        /// </summary>
        /// <param name="x">The fluent API wrapper instance.</param>
        /// <returns>The underlying <see cref="ITensor"/> instance managed by the wrapper.</returns>
        /// <exception cref="ArgumentNullException">Thrown when the fluent wrapper <paramref name="x"/> is <see langword="null"/>.</exception>

        public static ITensor ToTensor(this X x)
        {
            if (x == null)
                throw new ArgumentNullException(nameof(x), "Cannot extract an ITensor from a null X wrapper.");

            return x.Tensor;
        }
        /// <summary>
        /// Unwraps the fluent wrapper and converts the underlying tensor into a concrete <see cref="Tensor"/> implementation.
        /// </summary>
        /// <param name="x">The fluent API wrapper instance.</param>
        /// <returns>The unwrapped concrete <see cref="Tensor"/> implementation.</returns>
        /// <exception cref="ArgumentNullException">Thrown when the fluent wrapper <paramref name="x"/> is <see langword="null"/>.</exception>
        /// <exception cref="InvalidCastException">Thrown if the unwrapped tensor cannot be cast to the concrete <see cref="Tensor"/> type.</exception>

        public static Tensor ToConcreteTensor(this X x)
        {
            if (x == null)
                throw new ArgumentNullException(nameof(x), "Cannot convert null X to a concrete Tensor.");

            return (Tensor)Tensor.Unwrap(x.Tensor);
        }
        /// <summary>
        /// Converts a fluent <see cref="X"/> instance into a trainable <see cref="Variable"/> with autograd tracking capabilities.
        /// </summary>
        /// <param name="x">The fluent API wrapper instance.</param>
        /// <param name="requiresGrad">Determines whether this variable tracks gradients during backpropagation. Defaults to <see langword="true"/>.</param>
        /// <returns>A new <see cref="Variable"/> tracking the underlying tensor.</returns>
        /// <exception cref="ArgumentNullException">Thrown when the fluent wrapper <paramref name="x"/> is <see langword="null"/>.</exception>

        public static Variable ToVariable(this X x, bool requiresGrad = true)
        {
            if (x == null)
                throw new ArgumentNullException(nameof(x), "Cannot convert null X to a Variable.");

            return new Variable(x.Tensor, requiresGrad);
        }
    }
}