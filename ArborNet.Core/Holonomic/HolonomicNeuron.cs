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
    /// Represents a single Holonomic Fractal Neuron (HFN).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Unlike classical artificial neurons that compute a scalar dot product followed by a static,
    /// real-valued activation function, the <see cref="HolonomicNeuron"/> operates entirely within the 
    /// complex domain (<see cref="Complex"/>). It models quantum-like or holographic wave interference 
    /// patterns in its first phase, followed by a non-linear dynamical system (chaotic attractor/fractal unfolding) 
    /// in its second phase.
    /// </para>
    /// <para>
    /// Phase 1 (Holographic Interference): Computes the complex inner product of inputs and weights. 
    /// Because complex multiplication naturally scales amplitudes and adds phases, this phase simulates 
    /// constructive and destructive wave interference.
    /// </para>
    /// <para>
    /// Phase 2 (Fractal Unfolding): Uses the resulting interference value as a control parameter (analogous 
    /// to the constant 'c' in Julia or Mandelbrot sets) within a recursive transcendental mapping function, 
    /// generating a localized trajectory in the complex plane. The final state of this trajectory forms the output.
    /// </para>
    /// </remarks>

    #endregion

    public class HolonomicNeuron
    {
        /// <summary>
        /// Gets the collection of complex weights used to compute the holographic interference pattern.
        /// Each weight represents a complex wave with amplitude and phase.
        /// </summary>
        /// <value>
        /// An array of <see cref="Complex"/> numbers representing the synaptic weights of the neuron, 
        /// where each weight encodes both an amplitude (magnitude) and a spatial phase (angle).
        /// </value>
        public Complex[] Weights { get; private set; }
        /// <summary>
        /// Gets the recursive complex weight used to generate the internal fractal geometry during the unfolding phase.
        /// </summary>
        /// <value>
        /// A <see cref="Complex"/> multiplier acting as the feedback parameter in the recursive mapping function.
        /// This scaling factor determines the contraction, rotation, and stability of the underlying attractor.
        /// </value>

        public Complex InternalWeight { get; private set; }
        /// <summary>
        /// Gets the number of recursive iterations (depth) performed during the fractal unfolding phase.
        /// </summary>
        /// <value>
        /// An <see cref="int"/> representing the number of iterations executed within the dynamical system recurrence loop.
        /// Higher values result in deeper fractal resolution and potentially chaotic attractor states.
        /// </value>

        public int FractalDepth { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="HolonomicNeuron"/> class with randomized weights and phase angles.
        /// </summary>
        /// <param name="inputSize">The expected dimensionality of the incoming complex input vector.</param>
        /// <param name="fractalDepth">The recursion depth (iterations) for the internal chaotic attractor.</param>
        /// <param name="rand">The random number generator used to initialize weights using polar coordinates.</param>
        /// <exception cref="NullReferenceException">Thrown if the provided <paramref name="rand"/> instance is null.</exception>
        /// <exception cref="OverflowException">Thrown if <paramref name="inputSize"/> is negative.</exception>
        public HolonomicNeuron(int inputSize, int fractalDepth, Random rand)
        {
            Weights = new Complex[inputSize];
            FractalDepth = fractalDepth;

            // Initialize weights as complex waves using polar coordinates (Amplitude and Phase)
            for (int i = 0; i < inputSize; i++)
            {
                double amplitude = rand.NextDouble();              // Radius
                double phase = rand.NextDouble() * 2 * Math.PI;    // Angle (0 to 2π)
                Weights[i] = Complex.FromPolarCoordinates(amplitude, phase);
            }

            // Initialize the internal fractal weight
            InternalWeight = Complex.FromPolarCoordinates(rand.NextDouble(), rand.NextDouble() * 2 * Math.PI);
        }
        /// <summary>
        /// Computes the forward pass of the holonomic neuron, simulating holographic interference followed by fractal unfolding.
        /// </summary>
        /// <param name="inputs">The input vector of complex waves representing incoming signals.</param>
        /// <returns>A <see cref="Complex"/> value representing the final state of the fractal system after unfolding.</returns>
        /// <remarks>
        /// <para>
        /// The forward evaluation consists of two distinct mathematical steps:
        /// </para>
        /// <para>
        /// <b>Step 1: Holographic Interference (Psi)</b>
        /// <br/>
        /// The input vector and the weight vector undergo a complex dot product:
        /// <c>Psi = Sum(inputs[i] * Weights[i])</c>. This simulates wave front superposition.
        /// </para>
        /// <para>
        /// <b>Step 2: Attractor Unfolding</b>
        /// <br/>
        /// The system starts at an initial state of <c>z_0 = 0</c>, and iteratively updates according to:
        /// <c>z_{t+1} = Tanh(InternalWeight * z_t + Psi)</c>.
        /// The hyperbolic tangent activation function (<c>ComplexActivations.Tanh</c>) constrains the growth, 
        /// mapping the system into a bounded chaotic attractor.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown when the length of the <paramref name="inputs"/> array does not match the configured weight size.</exception>
        /// <exception cref="NullReferenceException">Thrown when the <paramref name="inputs"/> array is null.</exception>

        public Complex Forward(Complex[] inputs)
        {
            if (inputs.Length != Weights.Length)
                throw new ArgumentException("Input size must match weight size.");

            Complex psi = Complex.Zero;
            for (int i = 0; i < inputs.Length; i++)
            {
                // Complex multiplication automatically handles the addition of phases 
                // and the multiplication of amplitudes, perfectly simulating wave interference.
                psi += inputs[i] * Weights[i];
            }

            Complex z = Complex.Zero; // Initial state Z_0 = 0

            for (int t = 0; t < FractalDepth; t++)
            {
                // The Dynamical System: Z_{t+1} = \sigma(W_internal * Z_t + Psi)
                // Psi acts as the constant 'c' (similar to the Mandelbrot set equation)
                z = ComplexActivations.Tanh((InternalWeight * z) + psi);
            }

            // The final state of the fractal is the output of the neuron
            return z;
        }
    }
}