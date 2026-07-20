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
    using ArborNet.Core.Backends;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Fluent;
    using FluentAssertions;
    using Xunit;
    /// <summary>
    /// Contains comprehensive unit tests for core <see cref="Tensor"/> functionality within the ArborNet framework.
    /// </summary>
    /// <remarks>
    /// This test suite validates fundamental mathematical operations, automatic differentiation (autograd),
    /// tensor broadcasting rules, and the fluent syntax API. All test operations are verified on the 
    /// CPU device backend to ensure correct core logic before hardware acceleration.
    /// </remarks>
    /// <seealso cref="Tensor"/>
    /// <seealso cref="X"/>

    #endregion

    public class TensorTests : TestBase
    {
        /// <summary>
        /// Verifies that basic element-wise arithmetic operations (addition, multiplication, and subtraction) 
        /// on 2D tensors produce the expected numeric results.
        /// </summary>
        /// <remarks>
        /// This test instantiates two 2x2 matrices on the CPU and performs fundamental element-wise math:
        /// <list type="bullet">
        /// <item><description>Addition: <c>A + B</c></description></item>
        /// <item><description>Multiplication: <c>A * B</c></description></item>
        /// <item><description>Subtraction: <c>A - B</c></description></item>
        /// </list>
        /// </remarks>
        /// <seealso cref="Tensor.Add(Tensor)"/>
        /// <seealso cref="Tensor.Multiply(Tensor)"/>
        /// <seealso cref="Tensor.Subtract(Tensor)"/>
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
        /// Verifies that automatic differentiation (autograd) correctly computes and propagates gradients 
        /// for a chained sequence of element-wise multiplication and scalar addition.
        /// </summary>
        /// <remarks>
        /// This test constructs the computation graph <c>c = (a * b) + 1</c> and executes <see cref="Tensor.Backward()"/>.
        /// The analytical gradients are verified as:
        /// <list type="bullet">
        /// <item><description><c>dc/da = b</c></description></item>
        /// <item><description><c>dc/db = a</c></description></item>
        /// </list>
        /// </remarks>
        /// <seealso cref="Tensor.Backward()"/>
        /// <seealso cref="Tensor.RequiresGrad"/>
        /// <seealso cref="Tensor.Grad"/>

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
        /// Verifies that shape broadcasting correctly expands incompatible tensor shapes 
        /// to compatible dimensions during binary element-wise arithmetic operations.
        /// </summary>
        /// <remarks>
        /// This test performs addition between a 2x1 column tensor and a 1D vector of size 2. 
        /// The broadcasting engine is expected to project both operands into a 2x2 shape to complete the operation.
        /// </remarks>
        /// <seealso cref="Tensor.Add(Tensor)"/>

        [Fact]
        public void Broadcasting_WorksCorrectly()
        {
            var a = Tensor.Ones(new TensorShape(2, 1), Cpu);
            var b = Tensor.FromArray(new[] { 10f, 20f }, new TensorShape(2), Cpu);

            var result = a.Add(b);
            result.ToArray().Should().BeEquivalentTo(new[] { 11f, 21f, 11f, 21f });
        }
        /// <summary>
        /// Verifies that the fluent api utility <see cref="X"/> successfully builds and executes 
        /// a chained tensor computation pipeline containing random initialization, addition, multiplication, and a ReLU activation.
        /// </summary>
        /// <remarks>
        /// This test ensures that the fluent API produces mathematically valid outputs (non-NaN) 
        /// and correctly preserves the intended final shape of the tensor throughout the execution pipeline.
        /// </remarks>
        /// <seealso cref="X"/>

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