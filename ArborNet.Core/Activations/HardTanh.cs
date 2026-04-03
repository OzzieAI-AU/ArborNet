using ArborNet.Activations;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace ArborNet.Core.Activations
{


    /// <summary>
    /// Production-grade HardTanh activation.
    /// HardTanh(x) = max(-1, min(1, x))
    /// </summary>
    public class HardTanh : BaseActivation
    {
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var output = input.Clip(-1f, 1f);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var mask = input.GreaterThan(Tensor.FromScalar(-1f, input.Device))
                                   .Multiply(input.LessEqual(Tensor.FromScalar(1f, input.Device)));
                    return gradOutput.Multiply(mask);
                };
            }

            return output;
        }
    }
}