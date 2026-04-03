using ArborNet.Core;
using ArborNet.Core.Backends;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Fluent;
using FluentAssertions;
using Xunit;

namespace ArborNet.Tests
{
    /// <summary>
    /// Unit tests for core Tensor functionality, including arithmetic operations, autograd, broadcasting, and fluent API.
    /// </summary>
    public class TensorTests : TestBase
    {
        /// <summary>
        /// Verifies that basic arithmetic operations (addition, multiplication, subtraction) on 2D tensors produce the expected results.
        /// </summary>
        [Fact]
        public void BasicArithmetic_CorrectResults()
        {
            var a = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f }, new TensorShape(2, 2), Cpu);
            var b = Tensor.FromArray(new[] { 5f, 6f, 7f, 8f }, new TensorShape(2, 2), Cpu);

            var add = a.Add(b);
            var mul = a.Multiply(b);
            var sub = a.Subtract(b);

            add.ToArray().Should().BeEquivalentTo(new[] { 6f, 8f, 10f, 12f });
            mul.ToArray().Should().BeEquivalentTo(new[] { 5f, 12f, 21f, 32f });
            sub.ToArray().Should().BeEquivalentTo(new[] { -4f, -4f, -4f, -4f });
        }

        /// <summary>
        /// Verifies that automatic differentiation (autograd) correctly computes gradients for a chain of multiplication and addition operations.
        /// </summary>
        [Fact]
        public void Autograd_AddAndMultiply_CorrectGradients()
        {
            var a = Tensor.FromArray(new[] { 2f, 3f }, new TensorShape(2), Cpu);
            var b = Tensor.FromArray(new[] { 4f, 5f }, new TensorShape(2), Cpu);
            a.RequiresGrad = b.RequiresGrad = true;

            var c = a.Multiply(b).Add(Tensor.FromScalar(1f, Cpu));
            c.Backward();

            a.Grad!.ToArray().Should().BeEquivalentTo(new[] { 4f, 5f });
            b.Grad!.ToArray().Should().BeEquivalentTo(new[] { 2f, 3f });
        }

        /// <summary>
        /// Verifies that broadcasting correctly handles addition between tensors of compatible but different shapes (2x1 and 1x2 effectively).
        /// </summary>
        [Fact]
        public void Broadcasting_WorksCorrectly()
        {
            var a = Tensor.Ones(new TensorShape(2, 1), Cpu);
            var b = Tensor.FromArray(new[] { 10f, 20f }, new TensorShape(2), Cpu);

            var result = a.Add(b);
            result.ToArray().Should().BeEquivalentTo(new[] { 11f, 21f, 11f, 21f });
        }

        /// <summary>
        /// Verifies that the fluent API (X) correctly supports chained operations including random initialization, addition, multiplication, and ReLU activation.
        /// </summary>
        [Fact]
        public void FluentApi_X_Works()
        {
            var result = X.Rand(4, 5)
                .Add(5.0f)
                .Multiply(2.0).ReLU();

            result.ToArray().Should().NotContain(float.NaN);
            result.Shape.Dimensions.Should().BeEquivalentTo(new[] { 4, 5 });
        }
    }
}