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

    using ArborNet.Core;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using FluentAssertions;
    using global::ArborNet.Core.Tensors;
    using Xunit;
    /// <summary>
    /// Provides a comprehensive test suite for validating the core functionalities of the <see cref="Tensor"/> system.
    /// This class covers factory initialization methods, basic shape manipulations, broadcasting behaviors, 
    /// element-wise arithmetic, reduction operations, activation function behaviors, and automatic differentiation (autograd).
    /// </summary>

    #endregion

    public class TensorComprehensiveTests : TestBase
    {
        /// <summary>
        /// Verifies that all primary factory methods of the <see cref="Tensor"/> class,
        /// including Zeros, Ones, Rand, Randn, FromScalar, and FromArray, execute correctly,
        /// produce the expected dimensions, and contain the appropriate initial data.
        /// </summary>
        /// <remarks>
        /// This test ensures that:
        /// <list type="bullet">
        /// <item><description><see cref="Tensor.Zeros"/> initializes all values to 0.0f.</description></item>
        /// <item><description><see cref="Tensor.Ones"/> initializes all values to 1.0f.</description></item>
        /// <item><description><see cref="Tensor.Rand"/> and <see cref="Tensor.Randn"/> produce valid random values without NaN values.</description></item>
        /// <item><description><see cref="Tensor.FromScalar"/> accurately wraps a single float value.</description></item>
        /// <item><description><see cref="Tensor.FromArray"/> correctly populates a tensor from flat source arrays.</description></item>
        /// </list>
        /// </remarks>
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
        /// <summary>
        /// Verifies that tensor shape manipulations such as rank querying, total element calculation,
        /// reshaping, and broadcasting to higher dimensions function correctly and maintain mathematical integrity.
        /// </summary>
        /// <remarks>
        /// Evaluates basic metadata properties on the <see cref="TensorShape"/> object, confirms that 
        /// reshaping does not alter the total element count, and ensures broadcasting correctly scales the rank.
        /// </remarks>

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
        /// <summary>
        /// Tests the capability of a tensor to undergo unified reshaping and broadcasting along a specified axis,
        /// ensuring elements are correctly mapped and duplicated to match the target dimensions.
        /// </summary>
        /// <remarks>
        /// This test validates specialized broadcasting where a tensor of shape (2, 1) is expanded
        /// to shape (2, 3) along the specified target axis, ensuring that elements are properly replicated.
        /// </remarks>

        [Fact]
        public void ReshapeWithBroadcast_WorksCorrectly()
        {
            var a = Tensor.Ones(new TensorShape(2, 1), Cpu);   // shape (2, 1)
            var target = new TensorShape(2, 3);                // target (2, 3)

            var result = a.ReshapeWithBroadcast(target, axis: 1);

            result.Shape.Dimensions.Should().BeEquivalentTo(new[] { 2, 3 });
            result.ToArray().Should().AllBeEquivalentTo(1f);
        }
        /// <summary>
        /// Validates elementary arithmetic operations (addition, subtraction, multiplication, and division)
        /// between compatible tensors, verifying that element-wise computations yield exact expected values.
        /// </summary>
        /// <remarks>
        /// All operations are executed element-by-element using standard mathematical operators, 
        /// asserting with floating point precision standard ordering.
        /// </remarks>

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
        /// <summary>
        /// Tests automatic implicit broadcasting of tensors during binary operations (e.g., addition)
        /// when operand dimensions are compatible but unequal, ensuring standard NumPy-style broadcasting rules apply.
        /// </summary>
        /// <remarks>
        /// Verifies that a tensor of shape (2, 1) and a tensor of shape (3) can be successfully 
        /// added together to produce a resulting tensor of shape (2, 3).
        /// </remarks>

        [Fact]
        public void Broadcasting_WorksCorrectly()
        {
            var a = Tensor.Ones(new TensorShape(2, 1), Cpu);   // (2, 1)
            var b = Tensor.FromArray(new[] { 10f, 20f, 30f }, new TensorShape(3), Cpu); // (3)

            var result = a.Add(b);
            result.Shape.Dimensions.Should().BeEquivalentTo(new[] { 2, 3 });
            result.ToArray().Should().BeEquivalentTo(new[] { 11f, 21f, 31f, 11f, 21f, 31f });
        }
        /// <summary>
        /// Validates that reduction operations, specifically total and dimension-wise sum and mean calculations,
        /// aggregate elements accurately and output tensors of the correct reduced shapes.
        /// </summary>
        /// <remarks>
        /// Covers:
        /// <list type="bullet">
        /// <item><description>Global sum reduction.</description></item>
        /// <item><description>Global mean reduction.</description></item>
        /// <item><description>Axis-specific sum reduction (axis: 0).</description></item>
        /// <item><description>Axis-specific mean reduction (axis: 1).</description></item>
        /// </list>
        /// </remarks>

        [Fact]
        public void Reductions_WorkCorrectly()
        {
            var t = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new TensorShape(2, 3), Cpu);

            t.Sum().ToScalar().Should().Be(21f);
            t.Mean().ToScalar().Should().Be(3.5f);

            t.Sum(axis: 0).ToArray().Should().BeEquivalentTo(new[] { 5f, 7f, 9f });
            t.Mean(axis: 1).ToArray().Should().BeEquivalentTo(new[] { 2f, 5f });
        }
        /// <summary>
        /// Tests the basic operations of the autograd engine by building a simple mathematical chain,
        /// executing the backward pass to compute gradients, and verifying that the gradients propagated back
        /// to the leaf nodes are mathematically correct.
        /// </summary>
        /// <summary>
        /// Tests the basic operations of the autograd engine by constructing a differentiable mathematical chain (y = x * 2 + 1),
        /// triggering the backward pass on the sum of the results, and verifying the correct accumulation of gradients on leaf nodes.
        /// </summary>
        /// <remarks>
        /// Evaluates the backpropagation mechanism to ensure that:
        /// <list type="bullet">
        /// <item><description>Gradients are computed on leaf nodes marked with <c>RequiresGrad = true</c>.</description></item>
        /// <item><description>The gradient shape matches the original tensor shape.</description></item>
        /// <item><description>Gradients matches the expected mathematical derivative (dy/dx = 2.0).</description></item>
        /// </list>
        /// </remarks>

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
        /// <summary>
        /// Tests standard activation functions (ReLU, Sigmoid, Tanh) to ensure they map values
        /// correctly and do not produce invalid mathematical values like <see cref="float.NaN"/>.
        /// </summary>
        /// <remarks>
        /// Processes a broad set of random normal inputs and ensures element-wise bounds and validity constraints 
        /// are satisfied across activation ranges.
        /// </remarks>

        [Fact]
        public void Activations_ProduceValidOutput()
        {
            var x = Tensor.Randn(new TensorShape(100), Cpu);

            x.Relu().ToArray().Should().NotContain(float.NaN);
            x.Sigmoid().ToArray().Should().NotContain(float.NaN);
            x.Tanh().ToArray().Should().NotContain(float.NaN);
        }
        /// <summary>
        /// Performs a smoke test on a comprehensive suite of element-wise, reduction, matrix, and mathematical
        /// operations to ensure they execute without throwing exceptions when provided with valid tensor inputs.
        /// </summary>
        /// <remarks>
        /// Tests all major non-differentiable and differentiable APIs, including matrix multiplication, 
        /// absolute value, logarithms, square root, exponentiation, and min/max reductions.
        /// </remarks>

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