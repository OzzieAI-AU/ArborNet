// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Tests.Activations
{

    #region Using Statements:

    using ArborNet.Activations;
    using ArborNet.Core;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Fluent;
    using FluentAssertions;
    using Xunit;
    /// <summary>
    /// Provides comprehensive unit and integration tests for activation functions within the ArborNet framework.
    /// </summary>
    /// <remarks>
    /// This test suite validates the mathematical correctness of both forward and backward execution passes,
    /// ensures numerical stability under randomized input distributions across all standard activations,
    /// and verifies seamless integration with the fluent API syntax.
    /// </remarks>

    #endregion

    public class ActivationTests : TestBase
    {
        /// <summary>
        /// Verifies that the Rectified Linear Unit (ReLU) activation function computes correct values
        /// for both its forward and backward (gradient) passes.
        /// </summary>
        /// <remarks>
        /// This test constructs a 1-dimensional tensor containing negative, zero, and positive values:
        /// <c>[-2.0, -1.0, 0.0, 1.0, 2.0]</c>.
        /// <para>
        /// <b>Forward Pass:</b> Evaluates <c>f(x) = max(0, x)</c>, asserting that negative inputs map to zero
        /// while positive inputs remain unchanged.
        /// </para>
        /// <para>
        /// <b>Backward Pass:</b> Supplying an upstream gradient of all ones, the test verifies that the local
        /// subgradient is correctly computed as <c>1.0</c> for <c>x &gt; 0</c> and <c>0.0</c> for <c>x &lt;= 0</c>.
        /// </para>
        /// </remarks>
        [Fact]
        public void ReLU_ForwardAndBackward_Correct()
        {
            var x = Tensor.FromArray(new[] { -2f, -1f, 0f, 1f, 2f }, new TensorShape(5), Cpu);
            x.RequiresGrad = true;

            var relu = new ReLU();
            var y = relu.Forward(x);

            y.ToArray().Should().BeEquivalentTo(new[] { 0f, 0f, 0f, 1f, 2f });

            y.Backward(Tensor.Ones(y.Shape, Cpu));

            x.Grad!.ToArray().Should().BeEquivalentTo(new[] { 0f, 0f, 0f, 1f, 1f });
        }
        /// <summary>
        /// Asserts the numerical stability of all supported activation functions in the framework
        /// when evaluated against randomized, normally distributed inputs.
        /// </summary>
        /// <remarks>
        /// This test loops through instances of <see cref="IActivation"/> including:
        /// <list type="bullet">
        /// <item><description><see cref="ReLU"/></description></item>
        /// <item><description><see cref="Sigmoid"/></description></item>
        /// <item><description><see cref="Tanh"/></description></item>
        /// <item><description><see cref="Gelu"/></description></item>
        /// <item><description><see cref="Mish"/></description></item>
        /// <item><description><see cref="ELU"/></description></item>
        /// <item><description><see cref="LeakyReLU"/></description></item>
        /// <item><description><see cref="Softplus"/></description></item>
        /// </list>
        /// Each activation processes a 1D tensor of 100 elements sampled from a standard normal distribution <c>N(0, 1)</c>.
        /// The resulting tensors are validated to confirm the absence of <see cref="float.NaN"/>, 
        /// <see cref="float.PositiveInfinity"/>, or <see cref="float.NegativeInfinity"/>.
        /// </remarks>

        [Fact]
        public void AllActivations_NumericallyStable()
        {
            var activations = new IActivation[]
            {
                new ReLU(), new Sigmoid(), new Tanh(), new Gelu(), new Mish(),
                new ELU(), new LeakyReLU(), new Softplus()
            };

            foreach (var act in activations)
            {
                var x = Tensor.Randn(new TensorShape(100), Cpu);
                var y = act.Forward(x);
                y.ToArray().Should().NotContain(float.NaN);
                y.ToArray().Should().NotContain(float.PositiveInfinity);
                y.ToArray().Should().NotContain(float.NegativeInfinity);
            }
        }
        /// <summary>
        /// Validates that activation functions can be chained sequentially using the Fluent API.
        /// </summary>
        /// <remarks>
        /// This integration test constructs an initial random tensor and chains multiple operations:
        /// <c>x.ReLU().GELU().Tanh()</c>. It verifies that the pipeline compiles, executes successfully,
        /// and produces a mathematically stable output tensor containing valid floating-point numbers.
        /// </remarks>

        [Fact]
        public void FluentActivations_Work()
        {
            var x = X.Randn(10);
            var y = x.ReLU().GELU().Tanh();
            y.Tensor.ToArray().Should().NotContain(float.NaN);
        }
    }
}