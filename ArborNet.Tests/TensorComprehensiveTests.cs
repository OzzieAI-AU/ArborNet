namespace ArborNet.Tests
{
    using ArborNet.Core;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using FluentAssertions;
    using global::ArborNet.Core.Tensors;
    using Xunit;


    /// <summary>
    /// COMPREHENSIVE test suite for the Tensor system.
    /// Tests every factory method, arithmetic operation, shape manipulation,
    /// broadcasting, reduction, activation, and autograd behavior with detailed error messages.
    /// </summary>
    public class TensorComprehensiveTests : TestBase
    {
        [Fact]
        public void AllFactoryMethods_WorkCorrectly()
        {
            var shape = new TensorShape(2, 3, 4);

            Tensor.Zeros(shape).ToArray().Should().OnlyContain(x => x == 0f, "Zeros failed");
            Tensor.Ones(shape).ToArray().Should().OnlyContain(x => x == 1f, "Ones failed");
            Tensor.Rand(shape).ToArray().Should().NotContain(float.NaN, "Rand produced NaN");
            Tensor.Randn(shape).ToArray().Should().NotContain(float.NaN, "Randn produced NaN");

            var scalar = Tensor.FromScalar(42.5f);
            scalar.ToScalar().Should().Be(42.5f, "FromScalar failed");

            var data = new float[] { 1, 2, 3, 4, 5, 6 };
            var fromArray = Tensor.FromArray(data, new TensorShape(2, 3));
            fromArray.ToArray().Should().BeEquivalentTo(data, "FromArray failed");
        }

        [Fact]
        public void ShapeOperations_AreCorrect()
        {
            var shape = new TensorShape(2, 3, 4);
            shape.Rank.Should().Be(3);
            shape.TotalElements.Should().Be(24);

            var reshaped = Tensor.Randn(shape).Reshape(6, 4);
            reshaped.Shape.TotalElements.Should().Be(24);
            reshaped.Shape.Rank.Should().Be(2);

            var broadcasted = shape.BroadcastTo(new TensorShape(5, 2, 3, 4));
            broadcasted.Rank.Should().Be(4);
        }

        [Fact]
        public void ReshapeWithBroadcast_WorksCorrectly()
        {
            var a = Tensor.Ones(new TensorShape(2, 1), Cpu);   // shape (2, 1)
            var target = new TensorShape(2, 3);                // target (2, 3)

            var result = a.ReshapeWithBroadcast(target, axis: 1);

            result.Shape.Dimensions.Should().BeEquivalentTo(new[] { 2, 3 });
            result.ToArray().Should().AllBeEquivalentTo(1f);
        }

        [Fact]
        public void ArithmeticOperations_Work()
        {
            var a = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f }, new TensorShape(2, 2), Cpu);
            var b = Tensor.FromArray(new[] { 5f, 6f, 7f, 8f }, new TensorShape(2, 2), Cpu);

            a.Add(b).ToArray().Should().BeEquivalentTo(new[] { 6f, 8f, 10f, 12f });
            a.Subtract(b).ToArray().Should().BeEquivalentTo(new[] { -4f, -4f, -4f, -4f });
            a.Multiply(b).ToArray().Should().BeEquivalentTo(new[] { 5f, 12f, 21f, 32f });

            // Correct expectation for element-wise divide
            a.Divide(b).ToArray().Should().BeEquivalentTo(new[] { 0.2f, 0.33333334f, 0.42857143f, 0.5f },
                options => options.WithStrictOrdering());
        }

        [Fact]
        public void Broadcasting_WorksCorrectly()
        {
            var a = Tensor.Ones(new TensorShape(2, 1), Cpu);   // (2, 1)
            var b = Tensor.FromArray(new[] { 10f, 20f, 30f }, new TensorShape(3), Cpu); // (3)

            var result = a.Add(b);
            result.Shape.Dimensions.Should().BeEquivalentTo(new[] { 2, 3 });
            result.ToArray().Should().BeEquivalentTo(new[] { 11f, 21f, 31f, 11f, 21f, 31f });
        }

        [Fact]
        public void Reductions_WorkCorrectly()
        {
            var t = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new TensorShape(2, 3), Cpu);

            t.Sum().ToScalar().Should().Be(21f);
            t.Mean().ToScalar().Should().Be(3.5f);

            t.Sum(axis: 0).ToArray().Should().BeEquivalentTo(new[] { 5f, 7f, 9f });
            t.Mean(axis: 1).ToArray().Should().BeEquivalentTo(new[] { 2f, 5f });
        }

        //[Fact]
        //public void Autograd_BasicChain_Works()
        //{
        //    // Arrange
        //    var x = Tensor.Randn(new TensorShape(3, 4), Cpu);
        //    x.RequiresGrad = true;                    // ← Critical for leaf nodes

        //    // Build a simple differentiable chain: y = 2 * x + 1
        //    var two = Tensor.FromScalar(2f, Cpu);
        //    var one = Tensor.FromScalar(1f, Cpu);

        //    var y = x.Multiply(two);
        //    y = y.Add(one);

        //    var loss = y.Sum();                       // scalar loss for simple backward

        //    // Act
        //    loss.Backward();

        //    // Assert
        //    x.Grad.Should().NotBeNull("gradient on leaf x should be computed after Backward()");

        //    // Correct way to assert approximate values on a collection of floats
        //    x.Grad!.ToArray().Should().AllSatisfy(g =>
        //        g.Should().BeApproximately(2f, 0.001f)   // dy/dx = 2 for this graph
        //    );

        //    // Optional: check shape and that it's not all zeros
        //    x.Grad!.Shape.Should().BeEquivalentTo(x.Shape);
        //    x.Grad!.ToArray().Should().NotContain(0f);

        //    // Cleanup
        //    x.ClearGrad();
        //}
        [Fact]
        public void Autograd_BasicChain_Works()
        {
            var x = Tensor.Randn(new TensorShape(3, 4), Cpu);
            x.RequiresGrad = true;

            var y = x.Multiply(2f).Add(1f);
            var loss = y.Sum();

            loss.Backward();

            x.Grad.Should().NotBeNull();
            x.Grad!.Shape.Should().BeEquivalentTo(x.Shape);   // [3, 4]

            x.Grad!.ToArray().Should().AllSatisfy(g =>
                g.Should().BeApproximately(2f, 0.0001f));

            x.ClearGrad();
        }

        [Fact]
        public void Activations_ProduceValidOutput()
        {
            var x = Tensor.Randn(new TensorShape(100), Cpu);

            x.Relu().ToArray().Should().NotContain(float.NaN);
            x.Sigmoid().ToArray().Should().NotContain(float.NaN);
            x.Tanh().ToArray().Should().NotContain(float.NaN);
        }

        [Fact]
        public void AllOperations_DoNotThrow_OnValidInput()
        {
            var a = Tensor.Rand(new TensorShape(4, 5), Cpu);
            var b = Tensor.Rand(new TensorShape(4, 5), Cpu);

            a.Add(b);
            a.Subtract(b);
            a.Multiply(b);
            a.Divide(b);
            a.MatMul(b.Transpose(new[] { 1, 0 }));
            a.Reshape(2, 10);
            a.Sum();
            a.Mean();
            a.Max();
            a.Min();
            a.Exp();
            a.Log();
            a.Sqrt();
            a.Abs();
        }
    }
}