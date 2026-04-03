using System;
using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Layers;

namespace ArborNet.Models
{
    /// <summary>
    /// Provides utility methods and classes for working with models in ArborNet.
    /// </summary>
    /// <remarks>
    /// This static factory class offers convenient methods for creating common model architectures
    /// without requiring direct instantiation of concrete implementations. It serves as the primary
    /// entry point for model construction within the ArborNet framework.
    /// </remarks>
    public static class Models
    {
        /// <summary>
        /// Creates a new Sequential model with the specified layers.
        /// </summary>
        /// <param name="layers">The layers to include in the sequential model.</param>
        /// <returns>A new Sequential model instance.</returns>
        /// <remarks>
        /// The returned model is a linear stack of layers where each layer's output feeds directly
        /// into the next layer. Layers are processed in the exact order they appear in the provided
        /// collection. This is the most common architecture for feed-forward neural networks.
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="layers"/> is null.</exception>
        public static IModel CreateSequential(IEnumerable<ILayer> layers)
        {
            return new Sequential(layers);
        }
    }
}