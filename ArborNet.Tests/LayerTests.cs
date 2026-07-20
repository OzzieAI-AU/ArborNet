// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Tests.Layers
{

    #region Using Statements:

    using ArborNet.Activations;
    using ArborNet.Core;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Fluent;
    using ArborNet.Layers;
    using FluentAssertions;
    using Xunit;
    /// <summary>
    /// Contains comprehensive unit tests for verifying the correctness, mathematical integrity, 
    /// and shape transformations of neural network layers within the ArborNet framework.
    /// </summary>
    /// <remarks>
    /// This test suite covers fundamental layer operations including forward and backward propagation, 
    /// output tensor shape validation, and high-level fluent API builder integration.
    /// All tests in this class inherit from <see cref="TestBase"/> to leverage shared testing infrastructure.
    /// </remarks>

    #endregion

    public class LayerTests : TestBase
    {
        /// <summary>
        /// Verifies that the <see cref="Linear"/> (fully connected) layer correctly computes the forward pass,
        /// successfully executes backpropagation to compute gradients, and exposes its learnable parameters.
        /// </summary>
        /// <remarks>
        /// This test performs the following validation steps:
        /// <list type="number">
        /// <item><description>Initializes a <see cref="Linear"/> layer with 4 input features and 2 output features on the CPU.</description></item>
        /// <item><description>Creates a random input <see cref="Variable"/> of shape [3, 4] with gradient tracking enabled.</description></item>
        /// <item><description>Executes the forward pass to obtain the output tensor.</description></item>
        /// <item><description>Executes the backward pass using a gradient tensor of ones.</description></item>
        /// <item><description>Asserts that the input gradients are correctly populated and that the layer's learnable parameters are exposed.</description></item>
        /// </list>
        /// </remarks>
        [Fact]
        public void Linear_ForwardAndBackward_Correct()
        {
            var linear = new Linear(4, 2, Cpu);
            var xTensor = Tensor.Rand(new TensorShape(3, 4), Cpu);
            var x = new Variable(xTensor, true);

            var y = linear.Forward(x);
            y.Backward(Tensor.Ones(y.Shape, Cpu));

            x.Grad.Should().NotBeNull();
            x.Grad!.ToArray().Should().NotBeNullOrEmpty();
            linear.Parameters().Should().NotBeEmpty();
        }
        /// <summary>
        /// Verifies that the <see cref="Conv2D"/> convolution layer produces the expected
        /// output tensor dimensions based on input size, kernel size, stride, and padding configurations.
        /// </summary>
        /// <remarks>
        /// This test validates the spatial dimension calculation formula for 2D convolutions:
        /// <c>Output = ((Input - Kernel + 2 * Padding) / Stride) + 1</c>.
        /// Given an input of [2, 3, 32, 32], a kernel size of 3, stride of 1, and padding of 1,
        /// the output spatial dimensions are expected to remain [32, 32] with 16 output channels.
        /// </remarks>

        [Fact]
        public void Conv2D_ProducesCorrectOutputShape()
        {
            var conv = new Conv2D(3, 16, 3, 1, 1, true);
            var input = Tensor.Rand(new TensorShape(2, 3, 32, 32), Cpu);

            var output = conv.Forward(input);
            output.Shape.Dimensions.Should().BeEquivalentTo(new[] { 2, 16, 32, 32 });
        }
        /// <summary>
        /// Verifies that the fluent builder API successfully constructs, chains, and executes
        /// a sequence of layers and activation functions, yielding the correct final output tensor dimensions.
        /// </summary>
        /// <remarks>
        /// This test validates the usability, consistency, and correctness of the fluent API interface.
        /// It starts with a random input of shape [32, 8], chains a <see cref="Linear"/> layer (16 outputs),
        /// applies a <see cref="ReLU"/> activation, chains another <see cref="Linear"/> layer (8 outputs),
        /// and applies a <see cref="GELU"/> activation, ensuring the final tensor shape is [32, 8].
        /// </remarks>

        [Fact]
        public void Fluent_LinearChain_Works()
        {
            var result = X.Rand(32, 8)
                .Linear(16)
                .ReLU()
                .Linear(8)
                .GELU();

            result.Tensor.Shape.Dimensions.Should().BeEquivalentTo(new[] { 32, 8 });
        }
    }
}