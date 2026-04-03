using System;
using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Optimizers
{
    public class AdamW : IOptimizer
    {
        public double LearningRate { get; set; }

        private readonly double _beta1;
        private readonly double _beta2;
        private readonly double _eps;
        private readonly double _weightDecay;

        private int _timestep;
        private readonly Dictionary<ITensor, (ITensor m, ITensor v)> _state = new();

        public AdamW(double learningRate = 1e-3, double beta1 = 0.9, double beta2 = 0.999,
                    double eps = 1e-8, double weightDecay = 0.0)
        {
            LearningRate = learningRate;
            _beta1 = beta1;
            _beta2 = beta2;
            _eps = eps;
            _weightDecay = weightDecay;
        }

        public void Step(IEnumerable<ITensor> parameters)
        {
            _timestep++;

            foreach (var p in parameters)
            {
                if (p == null || !p.RequiresGrad || p.Grad == null) continue;

                var grad = p.Grad;

                if (_weightDecay > 0)
                    grad = grad.Add(p.Multiply(_weightDecay));

                if (!_state.TryGetValue(p, out var state))
                {
                    state.m = Tensor.Zeros(p.Shape, p.Device);
                    state.v = Tensor.Zeros(p.Shape, p.Device);
                    _state[p] = state;
                }

                var (m, v) = state;

                m = m.Multiply(_beta1).Add(grad.Multiply(1 - _beta1));
                v = v.Multiply(_beta2).Add(grad.Multiply(grad).Multiply(1 - _beta2));

                var mHat = m.Divide(1 - Math.Pow(_beta1, _timestep));
                var vHat = v.Divide(1 - Math.Pow(_beta2, _timestep));

                var denom = vHat.Sqrt().Add(Tensor.FromScalar((float)_eps, p.Device));
                var update = mHat.Divide(denom).Multiply((float)LearningRate);

                var updated = p.Subtract(update);
                p.SetData(updated.ToArray());

                _state[p] = (m, v);
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