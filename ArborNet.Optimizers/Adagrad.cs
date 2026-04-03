using System;
using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Optimizers
{
    public class Adagrad : IOptimizer
    {
        public double LearningRate { get; set; }

        private readonly double epsilon;

        private readonly Dictionary<ITensor, ITensor> accumulatedSquares = new();

        public Adagrad(double learningRate = 0.01, double epsilon = 1e-10)
        {
            LearningRate = learningRate;
            this.epsilon = epsilon;
        }

        public void Step(IEnumerable<ITensor> parameters)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));

            foreach (var param in parameters)
            {
                if (param == null || !param.RequiresGrad || param.Grad == null) continue;

                var grad = param.Grad;

                if (!accumulatedSquares.TryGetValue(param, out var accum))
                {
                    accum = Tensor.Zeros(param.Shape, param.Device);
                    accumulatedSquares[param] = accum;
                }

                accum = accum.Add(grad.Multiply(grad));
                accumulatedSquares[param] = accum;

                var denom = accum.Sqrt().Add(Tensor.FromScalar((float)epsilon, param.Device));
                var update = grad.Divide(denom).Multiply((float)LearningRate);

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