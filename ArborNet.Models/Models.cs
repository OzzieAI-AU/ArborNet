// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Models
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Interfaces;
    using ArborNet.Layers;
    /// <summary>
    /// Provides utility methods and static factory functions for constructing various machine learning model architectures within the ArborNet framework.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This static class serves as the primary entry point and creational factory for model instantiation.
    /// By abstracting the concrete implementation classes (such as <see cref="Sequential"/>), it promotes loose coupling
    /// and adheres to the Factory design pattern.
    /// </para>
    /// <para>
    /// This class is thread-safe and stateless, as it only exposes pure factory methods.
    /// </para>
    /// </remarks>
    /// <example>
    /// The following example demonstrates how to use the <see cref="Models"/> factory to instantiate a sequential neural network:
    /// <code>
    /// using ArborNet.Models;
    /// using ArborNet.Layers;
    /// using ArborNet.Core.Interfaces;
    /// using System.Collections.Generic;
    /// 
    /// var layers = new List&lt;ILayer&gt;
    /// {
    ///     new Dense(784, 128),
    ///     new Activation(ActivationType.Relu),
    ///     new Dense(128, 10)
    /// };
    /// 
    /// IModel myModel = Models.CreateSequential(layers);
    /// </code>
    /// </example>

    #endregion

    public static class Models
    {
        /// <summary>
        /// Creates and initializes a new instance of a sequential model containing the specified sequence of layers.
        /// </summary>
        /// <param name="layers">An enumerable collection of <see cref="ILayer"/> instances representing the ordered execution pipeline of the network.</param>
        /// <returns>A fully constructed <see cref="IModel"/> configured with the provided layer sequence.</returns>
        /// <remarks>
        /// <para>
        /// A sequential model represents a linear stack of layers. During forward propagation, inputs pass through the layers
        /// in the exact order they are specified within the <paramref name="layers"/> collection.
        /// </para>
        /// <para>
        /// Developers must ensure that the output dimensions of each layer match the input dimensions of the subsequent layer to avoid runtime shape mismatch errors during evaluation or training.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when the <paramref name="layers"/> parameter is <see langword="null"/>.</exception>
        /// <example>
        /// <code>
        /// var sequentialModel = Models.CreateSequential(new ILayer[] 
        /// {
        ///     new Dense(10, 5),
        ///     new Softmax()
        /// });
        /// </code>
        /// </example>
        public static IModel CreateSequential(IEnumerable<ILayer> layers)
        {
            return new Sequential(layers);
        }
    }
}