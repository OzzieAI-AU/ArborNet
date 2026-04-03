using System;
using System.Collections.Generic;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Functional;

namespace ArborNet.Layers
{
    public class Linear : BaseLayer
    {
        private ITensor weight;
        private ITensor bias;
        private readonly Device device;

        public Linear(int inFeatures, int outFeatures, Device device = null)
        {
            this.device = device ?? Device.CPU;
            weight = Initializers.XavierUniform(new TensorShape(inFeatures, outFeatures), this.device);
            bias = Tensor.Zeros(new TensorShape(outFeatures), this.device);
            weight.RequiresGrad = true;
            bias.RequiresGrad = true;
        }

        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var output = input.MatMul(weight).Add(bias);

            if (input.RequiresGrad || weight.RequiresGrad || bias.RequiresGrad)
            {
                var capturedInput = input;
                var capturedWeight = weight;
                var capturedBias = bias;

                output.GradFn = gradOutput =>
                {
                    ITensor gradInput = null;
                    ITensor gradWeight = null;
                    ITensor gradBias = null;

                    if (capturedWeight.RequiresGrad)
                    {
                        gradWeight = capturedInput.Transpose(new[] { 1, 0 }).MatMul(gradOutput);
                        AccumulateGrad(capturedWeight.Grad, gradWeight, g => capturedWeight.Grad = g);
                    }

                    if (capturedBias.RequiresGrad)
                    {
                        gradBias = gradOutput.Sum(0);
                        AccumulateGrad(capturedBias.Grad, gradBias, g => capturedBias.Grad = g);
                    }

                    if (capturedInput.RequiresGrad)
                    {
                        gradInput = gradOutput.MatMul(capturedWeight.Transpose(new[] { 1, 0 }));
                        AccumulateGrad(capturedInput.Grad, gradInput, g => capturedInput.Grad = g);
                    }

                    return gradInput ?? gradOutput;
                };
            }

            return output;
        }

        private void AccumulateGrad(ITensor currentGrad, ITensor delta, Action<ITensor> setter)
        {
            if (delta == null) return;

            if (currentGrad == null)
            {
                setter(delta.Clone());
            }
            else
            {
                setter(currentGrad.Add(delta));
            }
        }

        public override IEnumerable<ITensor> Parameters()
        {
            yield return weight;
            yield return bias;
        }
    }
}