// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Tests
{

    #region Using Statements:

    using ArborNet.Core;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Provides an abstract base implementation for test fixtures within the ArborNet testing suite.
    /// </summary>
    /// <remarks>
    /// This class serves as a foundation for unit and integration tests, offering standardized utility methods 
    /// to instantiate tensors on the host CPU. By inheriting from <see cref="TestBase"/>, test classes gain 
    /// streamlined access to factory methods for generating random, zero-initialized, and one-initialized tensors.
    /// </remarks>

    #endregion

    public abstract class TestBase
    {

        /// <summary>
        /// The CPU device instance used for creating test tensors.
        /// </summary>
        protected readonly Device Cpu = Device.CPU;
        /// <summary>
        /// Creates and initializes a new <see cref="Tensor"/> populated with random values drawn from a uniform distribution.
        /// </summary>
        /// <param name="shape">A variable number of integers representing the dimensions (axes) of the tensor.</param>
        /// <returns>A new <see cref="Tensor"/> instance allocated on the CPU with the specified shape and containing random uniform data.</returns>
        /// <exception cref="System.ArgumentNullException">Thrown when the <paramref name="shape"/> parameter is <see langword="null"/>.</exception>
        /// <exception cref="System.ArgumentException">Thrown when the <paramref name="shape"/> contains invalid dimensions, such as negative values.</exception>
        /// <remarks>
        /// The tensor is initialized on the host CPU using the <see cref="Device.CPU"/> instance. The underlying factory 
        /// method returns a generic tensor type which is explicitly cast to <see cref="Tensor"/>.
        /// </remarks>

        protected Tensor RandomTensor(params int[] shape)
    => (Tensor)Tensor.Rand(new TensorShape(shape), device: Cpu);
        /// <summary>
        /// Creates and initializes a new <see cref="Tensor"/> populated entirely with ones (1.0).
        /// </summary>
        /// <param name="shape">A variable number of integers representing the dimensions (axes) of the tensor.</param>
        /// <returns>A new <see cref="Tensor"/> instance allocated on the CPU with the specified shape and containing all ones.</returns>
        /// <exception cref="System.ArgumentNullException">Thrown when the <paramref name="shape"/> parameter is <see langword="null"/>.</exception>
        /// <exception cref="System.ArgumentException">Thrown when the <paramref name="shape"/> contains invalid dimensions, such as negative values.</exception>
        /// <remarks>
        /// This utility method is ideal for setting up identity operations, scaling biases, or initial gradient state tensors 
        /// during test scenarios.
        /// </remarks>

        protected Tensor Ones(params int[] shape)
    => (Tensor)Tensor.Ones(new TensorShape(shape), device: Cpu);
        /// <summary>
        /// Creates and initializes a new <see cref="Tensor"/> populated entirely with zeros (0.0).
        /// </summary>
        /// <param name="shape">A variable number of integers representing the dimensions (axes) of the tensor.</param>
        /// <returns>A new <see cref="Tensor"/> instance allocated on the CPU with the specified shape and containing all zeros.</returns>
        /// <exception cref="System.ArgumentNullException">Thrown when the <paramref name="shape"/> parameter is <see langword="null"/>.</exception>
        /// <exception cref="System.ArgumentException">Thrown when the <paramref name="shape"/> contains invalid dimensions, such as negative values.</exception>
        /// <remarks>
        /// This utility method is typically used to initialize bias vectors, accumulator tensors, or to verify zero-state 
        /// operations within network layers.
        /// </remarks>

        protected Tensor Zeros(params int[] shape)
    => (Tensor)Tensor.Zeros(new TensorShape(shape), device: Cpu);
    }
}