using System;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Activations
{


    /// <summary>
    /// Production-grade SELU activation.
    /// SELU(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
    /// </summary>
    public class SELU : BaseActivation
    {
        private const float Alpha = 1.6732632423543772848170429916717f;
        private const float Scale = 1.0507009873554804934193349852946f;

        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var positive = input.GreaterThan(Tensor.Zeros(input.Shape, input.Device));
            var expPart = input.Exp().Subtract(Tensor.Ones(input.Shape, input.Device)).Multiply(Alpha);
            var output = input.Multiply(positive)
                              .Add(expPart.Multiply(positive.LogicalNot()))
                              .Multiply(Scale);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var mask = positive;
                    var seluGrad = output.Divide(Scale).Add(Tensor.FromScalar(Alpha, input.Device)).Multiply(positive.LogicalNot());
                    return gradOutput.Multiply(mask.Add(seluGrad));
                };
            }

            return output;
        }
    }
}