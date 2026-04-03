using ArborNet.Activations;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace ArborNet.Core.Activations
{


    /// <summary>
    /// Production-grade Softsign activation.
    /// Softsign(x) = x / (1 + |x|)
    /// </summary>
    public class Softsign : BaseActivation
    {
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var output = input.Divide(Tensor.Ones(input.Shape, input.Device).Add(input.Abs()));

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var denom = Tensor.Ones(input.Shape, input.Device).Add(input.Abs());
                    return gradOutput.Multiply(Tensor.Ones(input.Shape, input.Device).Divide(denom.Multiply(denom)));
                };
            }

            return output;
        }
    }
}