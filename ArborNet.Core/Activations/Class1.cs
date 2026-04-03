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