using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace ArborNet.Core.Holonomic
{

    class TestHolonomicNetwork
    {

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