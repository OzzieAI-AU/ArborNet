// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Holonomic
{

    #region Using Statements:

    using System;
    using System.Numerics;
    /// <summary>
    /// Represents a dense layer of Holonomic Neurons within the ArborNet neural network framework.
    /// </summary>
    /// <remarks>
    /// This layer manages a collection of individual <see cref="HolonomicNeuron"/> instances
    /// and coordinates the forward propagation of complex-valued signals. It acts as a fully connected (dense)
    /// layer where every input is distributed to every neuron in the layer, enabling high-dimensional
    /// complex state transformations.
    /// </remarks>

    #endregion

    public class HolonomicLayer
    {
        /// <summary>
        /// The array of holonomic neurons comprising this dense layer.
        /// </summary>
        /// <remarks>
        /// This array is initialized during construction and remains read-only for the lifetime of the layer
        /// to ensure structural integrity during training and inference operations.
        /// </remarks>
        private readonly HolonomicNeuron[] _neurons;

        /// <summary>
        /// Initializes a new instance of the <see cref="HolonomicLayer"/> class with the specified configuration.
        /// </summary>
        /// <param name="inputSize">The dimensionality of the complex input vector expected by each neuron.</param>
        /// <param name="neuronCount">The number of holonomic neurons to instantiate in this layer.</param>
        /// <param name="fractalDepth">The recursion depth of the fractal structure defining the internal states of the neurons.</param>
        /// <param name="seed">The seed value used to initialize the pseudo-random number generator for weight initialization. Defaults to 42.</param>
        /// <remarks>
        /// This constructor instantiates the specified number of <see cref="HolonomicNeuron"/> objects,
        /// passing a single <see cref="Random"/> instance seeded with the provided value to ensure reproducible 
        /// weight initialization across all constituent neurons in the layer.
        /// </remarks>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown if <paramref name="neuronCount"/>, <paramref name="inputSize"/>, or <paramref name="fractalDepth"/> is less than or equal to zero.
        /// </exception>
        public HolonomicLayer(int inputSize, int neuronCount, int fractalDepth, int seed = 42)
        {
            _neurons = new HolonomicNeuron[neuronCount];
            Random rand = new Random(seed);

            for (int i = 0; i < neuronCount; i++)
            {
                _neurons[i] = new HolonomicNeuron(inputSize, fractalDepth, rand);
            }
        }
        /// <summary>
        /// Executes the forward propagation step for the entire layer, processing the input vector through all constituent neurons.
        /// </summary>
        /// <param name="inputs">An array of <see cref="Complex"/> numbers representing the incoming signals to the layer.</param>
        /// <returns>An array of <see cref="Complex"/> numbers representing the activation outputs of each neuron in this layer.</returns>
        /// <remarks>
        /// Each neuron in the layer processes the same <paramref name="inputs"/> vector independently.
        /// The resulting output array has a length equal to the number of neurons configured in this layer.
        /// </remarks>
        /// <exception cref="ArgumentNullException">
        /// Thrown when the <paramref name="inputs"/> array is <see langword="null"/>.
        /// </exception>
        /// <exception cref="ArgumentException">
        /// Thrown when the length of the <paramref name="inputs"/> array does not match the expected input size configured during initialization.
        /// </exception>

        public Complex[] Forward(Complex[] inputs)
        {
            Complex[] outputs = new Complex[_neurons.Length];
            for (int i = 0; i < _neurons.Length; i++)
            {
                outputs[i] = _neurons[i].Forward(inputs);
            }
            return outputs;
        }
    }
}