using ArborNet.Activations;
using ArborNet.Core;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Fluent;
using ArborNet.Layers;
using ArborNet.Losses;
using ArborNet.Models;
using ArborNet.Optimizers;
using FluentAssertions;
using System;
using Xunit;

namespace ArborNet.Tests.Models
{
    public class ModelIntegrationTests : TestBase
    {
        [Fact]
        public void SimpleSequential_TrainableAndConverges()
        {
            var model = new Sequential(new ILayer[]
            {
                new Linear(4, 32, Cpu),
                new ActivationLayer(new ReLU()),
                new Linear(32, 1, Cpu)
            });

            var x = Tensor.Rand(new TensorShape(64, 4), Cpu);

            var y = x.Slice(new (int, int, int)[] { (0, 64, 1), (0, 1, 1) })
                     .Multiply(0.6f)
                     .Add(Tensor.Randn(new TensorShape(64, 1), Cpu).Multiply(0.15f));

            var optimizer = new Adam(learningRate: 0.01f);
            var lossFn = new MSE();

            float finalLoss = 0f;
            const int maxEpochs = 300;

            for (int i = 0; i < maxEpochs; i++)
            {
                var pred = model.Forward(x);
                var loss = lossFn.Forward(pred, y);

                loss.Backward();
                optimizer.Step(model.Parameters());
                optimizer.ZeroGrad(model.Parameters());

                if (i == maxEpochs - 1)
                    finalLoss = loss.ToScalar();
            }

            finalLoss.Should().BeLessThan(0.5f, "Model should show clear learning on linear pattern");
        }

        [Fact]
        public void Fluent_X_API_Chain_Works()
        {
            var result = X.Rand(32, 8)
                .Linear(16)
                .ReLU()
                .Linear(4)
                .GELU();

            result.Tensor.Shape.Dimensions.Should().BeEquivalentTo(new[] { 32, 4 });
            result.Tensor.ToArray().Should().NotContain(float.NaN);
        }
    }
}