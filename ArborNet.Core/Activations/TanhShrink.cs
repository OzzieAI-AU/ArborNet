using ArborNet.Activations;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace ArborNet.Core.Activations
{


    /// <summary>
    /// Production-grade TanhShrink activation.
    /// TanhShrink(x) = x - tanh(x)
    /// </summary>
    public class TanhShrink : BaseActivation
    {
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var output = input.Subtract(new Tanh().Forward(input));

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var tanh = new Tanh().Forward(input);
                    return gradOutput.Multiply(Tensor.Ones(input.Shape, input.Device).Subtract(tanh.Multiply(tanh)));
                };
            }

            return output;
        }
    }
}