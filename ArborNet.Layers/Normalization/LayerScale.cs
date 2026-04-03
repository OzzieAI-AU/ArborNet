using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace ArborNet.Layers.Normalization
{

    /// <summary>
    /// LayerScale: Simple affine scaling layer (gamma * input). Used in some modern architectures.
    /// </summary>
    public class LayerScale : BaseLayer
    {
        private readonly ITensor gamma;

        public LayerScale(int numFeatures, float initScale = 1e-2f)
        {
            gamma = Tensor.FromScalar(initScale); // , new TensorShape(new[] { numFeatures })
            gamma.RequiresGrad = true;
        }

        public override ITensor Forward(ITensor input)
        {
            return input.Multiply(gamma);
        }

        public override IEnumerable<ITensor> Parameters()
        {
            yield return gamma;
        }
    }
}