using ArborNet.Activations;
using ArborNet.Core;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Fluent;
using FluentAssertions;
using Xunit;

namespace ArborNet.Tests.Activations
{
    /// <summary>
    /// Unit tests for activation functions in ArborNet.
    /// Verifies correctness of forward and backward passes, numerical stability across all activations,
    /// and integration with the fluent API.
    /// </summary>
    public class ActivationTests : TestBase
    {
        /// <summary>
        /// Verifies that the ReLU activation correctly computes the forward pass (max(0, x))
        /// and backward pass (subgradient of 1 for x > 0, 0 otherwise) on a 1D tensor
        /// containing negative, zero, and positive values.
        /// </summary>
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
        /// Ensures all implemented activation functions produce numerically stable outputs.
        /// Tests ReLU, Sigmoid, Tanh, GELU, Mish, ELU, LeakyReLU, and Softplus on random normal inputs (100 elements),
        /// confirming absence of NaN or infinite values in forward pass results.
        /// </summary>
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
        /// Validates the fluent API chaining of activation functions (ReLU → GELU → Tanh)
        /// on a 1D random normal tensor, ensuring the composed output contains no NaN values.
        /// </summary>
        [Fact]
        public void FluentActivations_Work()
        {
            var x = X.Randn(10);
            var y = x.ReLU().GELU().Tanh();
            y.Tensor.ToArray().Should().NotContain(float.NaN);
        }
    }
}