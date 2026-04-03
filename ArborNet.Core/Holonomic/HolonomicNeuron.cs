using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace ArborNet.Core.Holonomic
{

    /// <summary>
    /// Represents a single Holonomic Fractal Neuron.
    /// Instead of a scalar dot product, it computes the interference of complex waves,
    /// followed by a recursive fractal unfolding.
    /// </summary>
    public class HolonomicNeuron
    {
        public Complex[] Weights { get; private set; }

        // The recursive weight used to generate the fractal geometry inside the neuron
        public Complex InternalWeight { get; private set; }

        // How many times the internal state recurses (fractal depth)
        public int FractalDepth { get; private set; }

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
        /// Computes the forward pass of the holonomic neuron.
        /// </summary>
        public Complex Forward(Complex[] inputs)
        {
            if (inputs.Length != Weights.Length)
                throw new ArgumentException("Input size must match weight size.");

            // ----------------------------------------------------------------
            // Phase 1: Holographic Interference Pattern (Psi)
            // ----------------------------------------------------------------
            Complex psi = Complex.Zero;
            for (int i = 0; i < inputs.Length; i++)
            {
                // Complex multiplication automatically handles the addition of phases 
                // and the multiplication of amplitudes, perfectly simulating wave interference.
                psi += inputs[i] * Weights[i];
            }

            // ----------------------------------------------------------------
            // Phase 2: Fractal Unfolding (Recursive Generation)
            // ----------------------------------------------------------------
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