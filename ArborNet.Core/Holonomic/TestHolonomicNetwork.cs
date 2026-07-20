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
    using System.Collections.Generic;
    using System.Numerics;
    using System.Text;
    /// <summary>
    /// Provides a test harness and execution entry point to demonstrate, validate, and benchmark 
    /// the functionality of the Holonomic Fractal Network.
    /// </summary>
    /// <remarks>
    /// This class showcases the lifecycle of a holonomic operation, including layer initialization, 
    /// complex wave input generation using polar coordinates, forward propagation, and the analysis of 
    /// the output complex phase space.
    /// </remarks>

    #endregion

    class TestHolonomicNetwork
    {
        /// <summary>
        /// Executes a demonstration run of the Holonomic Fractal Network.
        /// </summary>
        /// <remarks>
        /// The demonstration performs the following sequential actions:
        /// <list type="bullet">
        /// <item>
        /// <description>Initializes a single <c>HolonomicLayer</c> with specified parameters (input size, neuron count, and internal fractal recursion depth).</description>
        /// </item>
        /// <item>
        /// <description>Synthesizes randomized input data, transforming pseudo-random signals into complex polar coordinates representing wave-encoded states.</description>
        /// </item>
        /// <item>
        /// <description>Executes the forward propagation pass of the layer to compute the complex output wave states.</description>
        /// </item>
        /// <item>
        /// <description>Logs the resulting amplitudes (magnitudes) and phases (angles in radians) of the inputs and outputs to the console.</description>
        /// </item>
        /// </list>
        /// </remarks>
        static void Test()
        {

            Console.WriteLine("Initializing Holonomic Fractal Network...");

            int inputSize = 4;
            int neuronCount = 3;
            int fractalDepth = 5; // The neuron will recurse 5 times internally

            // Create a single Holonomic Layer
            HolonomicLayer layer = new HolonomicLayer(inputSize, neuronCount, fractalDepth);

            // Create dummy input data (e.g., encoded sensor data converted to waves)
            Random rand = new Random();
            Complex[] inputs = new Complex[inputSize];
            for (int i = 0; i < inputSize; i++)
            {
                inputs[i] = Complex.FromPolarCoordinates(rand.NextDouble(), rand.NextDouble() * Math.PI);
            }

            Console.WriteLine("\n--- Input Waves (Amplitude ∠ Phase) ---");
            foreach (var input in inputs)
            {
                Console.WriteLine($"{input.Magnitude:F4} ∠ {input.Phase:F4} rad");
            }

            // Run the forward pass
            Complex[] outputs = layer.Forward(inputs);

            Console.WriteLine("\n--- Output Fractal States (Amplitude ∠ Phase) ---");
            for (int i = 0; i < outputs.Length; i++)
            {
                Console.WriteLine($"Neuron {i}: {outputs[i].Magnitude:F4} ∠ {outputs[i].Phase:F4} rad");
            }

            Console.WriteLine("\nNotice how the output is a complex wave state. This allows the network to chain holonomic layers together endlessly without losing phase data.");
        }
    }
}