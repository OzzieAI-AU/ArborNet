using ArborNet.Activations;
using ArborNet.Core;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Fluent;
using ArborNet.Layers;
using FluentAssertions;
using Xunit;

namespace ArborNet.Tests.Layers
{
    public class LayerTests : TestBase
    {
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

        [Fact]
        public void Conv2D_ProducesCorrectOutputShape()
        {
            var conv = new Conv2D(3, 16, 3, 1, 1, true);
            var input = Tensor.Rand(new TensorShape(2, 3, 32, 32), Cpu);

            var output = conv.Forward(input);
            output.Shape.Dimensions.Should().BeEquivalentTo(new[] { 2, 16, 32, 32 });
        }

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