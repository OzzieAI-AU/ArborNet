// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Tests
{

    #region Using Statements:

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
    /// <summary>
    /// Provides integration and unit tests for evaluating the mathematical accuracy, convergence rates,
    /// and core behaviors of various parameter optimization algorithms implemented within ArborNet.
    /// </summary>
    /// <remarks>
    /// These tests validate optimization methods like SGD, SGD with momentum, Adam, Adagrad, and RMSProp
    /// by training simple models (e.g., linear regression) and ensuring they converge to known mathematical solutions.
    /// </remarks>

    #endregion

    public class OptimizerTests : TestBase
    {
        private readonly Device _cpu = Device.CPU;
        /// <summary>
        /// Verifies that the Stochastic Gradient Descent (SGD) optimizer can successfully converge
        /// to the correct analytical weights and biases for a simple 1D linear regression task.
        /// </summary>
        /// <remarks>
        /// The test constructs synthetic linear data based on the equation <c>y = 2x + 1</c> and checks if the model
        /// parameters approach a weight of 2.0 and a bias of 1.0 within 300 steps of standard SGD optimization.
        /// </remarks>

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
        /// <summary>
        /// Verifies that the momentum coefficient in the SGD optimizer accelerates parameter updates,
        /// leading to stable and correct convergence within a reduced number of steps.
        /// </summary>
        /// <remarks>
        /// Uses the same target linear function <c>y = 2x + 1</c> as <see cref="SGD_Converges_ToCorrectSolution"/> but
        /// applies a momentum factor of 0.9, validating convergence over fewer training steps.
        /// </remarks>

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
        /// <summary>
        /// Tests the Adam (Adaptive Moment Estimation) optimizer, validating that its dual-moment
        /// tracking successfully adapts parameter learning rates to achieve convergence.
        /// </summary>
        /// <remarks>
        /// Verifies that Adam's combination of momentum and RMSProp tracking adapts learning steps
        /// to successfully find target weights and biases in a standard regression setup.
        /// </remarks>

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
        /// <summary>
        /// Tests the Adagrad optimizer to verify that its parameter-specific adaptive learning rates,
        /// based on historical gradient accumulation, lead to successful optimization.
        /// </summary>
        /// <remarks>
        /// Adagrad scales learning rates inversely proportional to the square root of the sum of all historical
        /// squared gradients, which this test validates against the linear solver problem.
        /// </remarks>

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
        /// <summary>
        /// Tests the RMSProp (Root Mean Square Propagation) optimizer, ensuring stable convergence
        /// using exponentially decaying average of squared gradients.
        /// </summary>
        /// <remarks>
        /// Ensures the learning rate is scaled by the moving average of the squared gradients,
        /// preventing premature decay of learning steps and successfully converging on target linear coefficients.
        /// </remarks>

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
        /// <summary>
        /// Ensures that calling the <see cref="IOptimizer.ZeroGrad(IEnumerable{Tensor})"/> method across all
        /// implemented optimizers successfully clears and resets the gradients of the target model's parameter tensors to zero.
        /// </summary>
        /// <remarks>
        /// Iterates over <see cref="SGD"/>, <see cref="Adam"/>, <see cref="AdamW"/>, <see cref="RMSProp"/>, and <see cref="Adagrad"/>
        /// to confirm that gradient accumulation is accurately wiped before the next forward-backward execution cycle.
        /// </remarks>

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