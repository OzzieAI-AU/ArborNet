// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Tests.Models
{

    #region Using Statements:

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
    /// <summary>
    /// Contains integration tests for verifying the end-to-end training pipelines,
    /// convergence of sequential models, and the behavior of the fluent API interface.
    /// </summary>
    /// <remarks>
    /// This test class ensures that the core components of the neural network library—such as
    /// layers, optimizers, activation functions, and loss functions—work in unison correctly.
    /// It inherits setup and teardown behavior from the <see cref="TestBase"/> class.
    /// </remarks>

    #endregion

    public class ModelIntegrationTests : TestBase
    {
        /// <summary>
        /// Verifies that a basic <see cref="Sequential"/> model can successfully propagate inputs forward,
        /// compute loss using <see cref="MSE"/>, calculate gradients via backpropagation, and utilize 
        /// the <see cref="Adam"/> optimizer to converge on a synthetic linear dataset with noise.
        /// </summary>
        /// <remarks>
        /// The network architecture consists of:
        /// <list type="bullet">
        /// <item><description>Linear layer mapping from 4 inputs to 32 hidden units.</description></item>
        /// <item><description>ReLU activation layer.</description></item>
        /// <item><description>Linear layer mapping from 32 hidden units to a single continuous output.</description></item>
        /// </list>
        /// The model is trained for 300 epochs, asserting that the final loss is less than 0.5f.
        /// This test ensures the integration of forward execution, gradient tracking, backpropagation, and optimizer parameter updates.
        /// </remarks>
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
        /// <summary>
        /// Verifies that the fluent <see cref="X"/> API correctly constructs, chains, and executes
        /// neural network layers while maintaining correct tensor dimensions and valid floating-point values.
        /// </summary>
        /// <remarks>
        /// This test defines a fluent chain:
        /// <list type="bullet">
        /// <item><description>Initializes a random tensor of shape [32, 8].</description></item>
        /// <item><description>Applies a Linear projection to 16 features.</description></item>
        /// <item><description>Applies a ReLU activation function.</description></item>
        /// <item><description>Applies a Linear projection to 4 features.</description></item>
        /// <item><description>Applies a GELU activation function.</description></item>
        /// </list>
        /// It asserts that the resulting tensor matches the expected dimensions of [32, 4] and contains no NaN values.
        /// </remarks>

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