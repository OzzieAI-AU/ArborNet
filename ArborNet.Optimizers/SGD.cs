using System;
using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Optimizers
{
    public class SGD : IOptimizer
    {
        public double LearningRate { get; set; }
        public double Momentum { get; set; }
        public double WeightDecay { get; set; }

        private readonly Dictionary<ITensor, ITensor> velocity = new();

        public SGD(double learningRate = 0.01, double momentum = 0.0, double weightDecay = 0.0)
        {
            LearningRate = learningRate;
            Momentum = momentum;
            WeightDecay = weightDecay;
        }

        public void Step(IEnumerable<ITensor> parameters)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));

            foreach (var param in parameters)
            {
                if (param == null || !param.RequiresGrad || param.Grad == null) continue;

                var grad = param.Grad;
                if (WeightDecay > 0)
                    grad = grad.Add(param.Multiply(WeightDecay));

                ITensor update;
                if (Momentum > 0)
                {
                    if (!velocity.TryGetValue(param, out var v))
                    {
                        v = Tensor.Zeros(param.Shape, param.Device);
                        velocity[param] = v;
                    }

                    v = v.Multiply(Momentum).Add(grad);
                    velocity[param] = v;
                    update = v;
                }
                else
                {
                    update = grad;
                }

                var scaledUpdate = update.Multiply(LearningRate);
                var newData = param.Subtract(scaledUpdate);
                param.SetData(newData.ToArray());
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