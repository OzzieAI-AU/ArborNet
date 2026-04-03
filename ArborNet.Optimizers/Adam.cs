using System;
using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Optimizers
{
    public class Adam : IOptimizer
    {
        public double LearningRate { get; set; }

        private readonly double beta1;
        private readonly double beta2;
        private readonly double eps;
        private readonly double weightDecay;

        private int timestep;
        private readonly Dictionary<ITensor, (ITensor m, ITensor v)> state = new();

        public Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weightDecay = 0.0)
        {
            LearningRate = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
            this.weightDecay = weightDecay;
        }

        public void Step(IEnumerable<ITensor> parameters)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));

            timestep++;

            foreach (var param in parameters)
            {
                if (param == null || !param.RequiresGrad || param.Grad == null) continue;

                var grad = param.Grad;
                if (weightDecay > 0)
                    grad = grad.Add(param.Multiply(weightDecay));

                if (!state.TryGetValue(param, out var s))
                {
                    s.m = Tensor.Zeros(param.Shape, param.Device);
                    s.v = Tensor.Zeros(param.Shape, param.Device);
                    state[param] = s;
                }

                var m = s.m.Multiply(beta1).Add(grad.Multiply(1 - beta1));
                var v = s.v.Multiply(beta2).Add(grad.Multiply(grad).Multiply(1 - beta2));

                state[param] = (m, v);

                var mHat = m.Divide(1 - Math.Pow(beta1, timestep));
                var vHat = v.Divide(1 - Math.Pow(beta2, timestep));

                var denom = vHat.Sqrt().Add(Tensor.FromScalar((float)eps, param.Device));
                var update = mHat.Divide(denom).Multiply((float)LearningRate);

                var newValue = param.Subtract(update);
                param.SetData(newValue.ToArray());
            }
        }

        public void ZeroGrad(IEnumerable<ITensor> parameters)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));

            foreach (var param in parameters)
            {
                if (param != null && param.RequiresGrad)
                {
                    param.Grad = Tensor.Zeros(param.Shape, param.Device);
                }
            }
        }
    }
}