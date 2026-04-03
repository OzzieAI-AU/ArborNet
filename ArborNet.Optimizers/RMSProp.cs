using System;
using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Optimizers
{
    public class RMSProp : IOptimizer
    {
        public double LearningRate { get; set; }

        private readonly float _alpha;
        private readonly float _epsilon;

        private readonly Dictionary<ITensor, ITensor> _v = new();

        public RMSProp(double learningRate = 0.001, float alpha = 0.99f, float epsilon = 1e-8f)
        {
            LearningRate = learningRate;
            _alpha = alpha;
            _epsilon = epsilon;
        }

        public void Step(IEnumerable<ITensor> parameters)
        {
            foreach (var param in parameters)
            {
                if (param.Grad == null) continue;

                if (!_v.TryGetValue(param, out var v))
                {
                    v = Tensor.Zeros(param.Shape, param.Device);
                    _v[param] = v;
                }

                var gradSq = param.Grad.Multiply(param.Grad);
                v = v.Multiply(_alpha).Add(gradSq.Multiply(1 - _alpha));
                _v[param] = v;

                var denom = v.Sqrt().Add(Tensor.FromScalar(_epsilon, param.Device));
                var update = param.Grad.Divide(denom).Multiply((float)LearningRate);

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