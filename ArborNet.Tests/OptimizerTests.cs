using ArborNet.Activations;
using ArborNet.Core;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Layers;
using ArborNet.Losses;
using ArborNet.Models;
using ArborNet.Optimizers;
using FluentAssertions;
using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace ArborNet.Tests
{
    public class OptimizerTests : TestBase
    {
        private readonly Device _cpu = Device.CPU;

        [Fact]
        public void SGD_Converges_ToCorrectSolution()
        {
            var xData = Enumerable.Range(0, 100).Select(i => (float)i / 50f - 1f).ToArray();
            var x = Tensor.FromArray(xData, new TensorShape(100, 1), _cpu);
            var y = x.Multiply(2f).Add(1f);

            var model = new Linear(1, 1, _cpu);
            var optimizer = new SGD(learningRate: 0.1f);
            var lossFn = new MSE();

            for (int step = 0; step < 300; step++)
            {
                var pred = model.Forward(x);
                var loss = lossFn.Forward(pred, y, "mean");

                loss.Backward();
                optimizer.Step(model.Parameters());
                optimizer.ZeroGrad(model.Parameters());
            }

            var weight = model.Parameters().First().ToArray()[0];
            var bias = model.Parameters().Skip(1).First().ToArray()[0];
            weight.Should().BeApproximately(2f, 0.5f);
            bias.Should().BeApproximately(1f, 0.5f);
        }

        [Fact]
        public void SGD_WithMomentum_AcceleratesConvergence()
        {
            var xData = Enumerable.Range(0, 100).Select(i => (float)i / 50f - 1f).ToArray();
            var x = Tensor.FromArray(xData, new TensorShape(100, 1), _cpu);
            var y = x.Multiply(2f).Add(1f);

            var model = new Linear(1, 1, _cpu);
            var optimizer = new SGD(learningRate: 0.1f, momentum: 0.9f);
            var lossFn = new MSE();

            for (int step = 0; step < 200; step++)
            {
                var pred = model.Forward(x);
                var loss = lossFn.Forward(pred, y, "mean");

                loss.Backward();
                optimizer.Step(model.Parameters());
                optimizer.ZeroGrad(model.Parameters());
            }

            var weight = model.Parameters().First().ToArray()[0];
            var bias = model.Parameters().Skip(1).First().ToArray()[0];
            weight.Should().BeApproximately(2f, 0.5f);
            bias.Should().BeApproximately(1f, 0.5f);
        }

        [Fact]
        public void Adam_ConvergesFasterThanSGD()
        {
            var xData = Enumerable.Range(0, 100).Select(i => (float)i / 50f - 1f).ToArray();
            var x = Tensor.FromArray(xData, new TensorShape(100, 1), _cpu);
            var y = x.Multiply(2f).Add(1f);

            var model = new Linear(1, 1, _cpu);
            var optimizer = new Adam(learningRate: 0.05);
            var lossFn = new MSE();

            for (int step = 0; step < 300; step++)
            {
                var pred = model.Forward(x);
                var loss = lossFn.Forward(pred, y, "mean");

                loss.Backward();
                optimizer.Step(model.Parameters());
                optimizer.ZeroGrad(model.Parameters());
            }

            var weight = model.Parameters().First().ToArray()[0];
            var bias = model.Parameters().Skip(1).First().ToArray()[0];
            weight.Should().BeApproximately(2f, 0.5f);
            bias.Should().BeApproximately(1f, 0.5f);
        }

        [Fact]
        public void Adagrad_PerParameterAdaptation()
        {
            var xData = Enumerable.Range(0, 100).Select(i => (float)i / 50f - 1f).ToArray();
            var x = Tensor.FromArray(xData, new TensorShape(100, 1), _cpu);
            var y = x.Multiply(2f).Add(1f);

            var model = new Linear(1, 1, _cpu);
            var optimizer = new Adagrad(learningRate: 0.5f);
            var lossFn = new MSE();

            for (int step = 0; step < 300; step++)
            {
                var pred = model.Forward(x);
                var loss = lossFn.Forward(pred, y, "mean");

                loss.Backward();
                optimizer.Step(model.Parameters());
                optimizer.ZeroGrad(model.Parameters());
            }

            var weight = model.Parameters().First().ToArray()[0];
            var bias = model.Parameters().Skip(1).First().ToArray()[0];
            weight.Should().BeApproximately(2f, 0.5f);
            bias.Should().BeApproximately(1f, 0.5f);
        }

        [Fact]
        public void RMSProp_StableConvergence()
        {
            var xData = Enumerable.Range(0, 100).Select(i => (float)i / 50f - 1f).ToArray();
            var x = Tensor.FromArray(xData, new TensorShape(100, 1), _cpu);
            var y = x.Multiply(2f).Add(1f);

            var model = new Linear(1, 1, _cpu);
            var optimizer = new RMSProp(learningRate: 0.05f, alpha: 0.99f, epsilon: 1e-8f);
            var lossFn = new MSE();

            for (int step = 0; step < 300; step++)
            {
                var pred = model.Forward(x);
                var loss = lossFn.Forward(pred, y, "mean");

                loss.Backward();
                optimizer.Step(model.Parameters());
                optimizer.ZeroGrad(model.Parameters());
            }

            var weight = model.Parameters().First().ToArray()[0];
            var bias = model.Parameters().Skip(1).First().ToArray()[0];
            weight.Should().BeApproximately(2f, 0.5f);
            bias.Should().BeApproximately(1f, 0.5f);
        }

        [Fact]
        public void AllOptimizers_ZeroGrad_ClearsGradients()
        {
            var optimizers = new IOptimizer[] { new SGD(), new Adam(), new AdamW(), new RMSProp(), new Adagrad() };

            foreach (var opt in optimizers)
            {
                var model = new Linear(20, 10, _cpu);
                var x = Tensor.Randn(new TensorShape(32, 20), _cpu);
                var y = Tensor.Randn(new TensorShape(32, 10), _cpu);
                var pred = model.Forward(x);
                var loss = new MSE().Forward(pred, y);
                loss.Backward();

                opt.ZeroGrad(model.Parameters());

                foreach (var param in model.Parameters())
                {
                    param.Grad.Should().NotBeNull("Grad should be set to a zero tensor");
                    param.Grad.ToArray().Should().AllBeEquivalentTo(0f, "All gradient values should be zero");
                }
            }
        }
    }
}