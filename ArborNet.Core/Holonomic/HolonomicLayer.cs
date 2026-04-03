using System;
using System.Numerics;

namespace ArborNet.Core.Holonomic
{

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
}