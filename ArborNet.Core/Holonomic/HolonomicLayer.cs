using System;
using System.Numerics;

namespace ArborNet.Core.Holonomic
{
    /// <summary>
    /// Provides non-linear activation functions for Complex numbers.
    /// </summary>
    public static class ComplexActivations
    {
        /// <summary>
        /// Complex Hyperbolic Tangent. Bounds both the real and imaginary parts, 
        /// serving as an effective non-linearity for wave-interference neural networks.
        /// </summary>
        public static Complex Tanh(Complex z)
        {
            return Complex.Tanh(z);
        }
    }

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

    /// <summary>
    /// A dense layer of Holonomic Neurons.
    /// </summary>
    public class HolonomicLayer
    {
        private readonly HolonomicNeuron[] _neurons;

        public HolonomicLayer(int inputSize, int neuronCount, int fractalDepth, int seed = 42)
        {
            _neurons = new HolonomicNeuron[neuronCount];
            Random rand = new Random(seed);

            for (int i = 0; i < neuronCount; i++)
            {
                _neurons[i] = new HolonomicNeuron(inputSize, fractalDepth, rand);
            }
        }

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

    class Program
    {
        static void Main(string[] args)
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