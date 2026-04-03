using System;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Activations
{

    /// <summary>
    /// Production-grade HardSigmoid activation.
    /// HardSigmoid(x) = max(0, min(1, (x + 3)/6))
    /// </summary>
    public class HardSigmoid : BaseActivation
    {
        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var output = input.Add(3f).Divide(6f).Clip(0f, 1f);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var mask = input.GreaterThan(Tensor.FromScalar(-3f, input.Device))
                                   .Multiply(input.LessEqual(Tensor.FromScalar(3f, input.Device)));
                    return gradOutput.Multiply(mask.Divide(6f));
                };
            }

            return output;
        }
    }
}