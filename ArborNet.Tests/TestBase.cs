using ArborNet.Core;
using ArborNet.Core.Devices;
using ArborNet.Core.Tensors;

namespace ArborNet.Tests
{
    /// <summary>
    /// Abstract base class for test fixtures, providing utility methods to create tensors on the CPU device.
    /// </summary>
    public abstract class TestBase
    {
 
        /// <summary>
        /// The CPU device instance used for creating test tensors.
        /// </summary>
        protected readonly Device Cpu = Device.CPU;

        /// <summary>
        /// Creates a tensor filled with random values from a uniform distribution on the CPU device.
        /// </summary>
        /// <param name="shape">The dimensions of the tensor, specified as variable integer arguments.</param>
        /// <returns>A new <see cref="Tensor"/> instance with the specified shape filled with random values.</returns>
        protected Tensor RandomTensor(params int[] shape)
            => (Tensor)Tensor.Rand(new TensorShape(shape), device: Cpu);

        /// <summary>
        /// Creates a tensor filled with ones on the CPU device.
        /// </summary>
        /// <param name="shape">The dimensions of the tensor, specified as variable integer arguments.</param>
        /// <returns>A new <see cref="Tensor"/> instance with the specified shape filled with ones.</returns>
        protected Tensor Ones(params int[] shape)
            => (Tensor)Tensor.Ones(new TensorShape(shape), device: Cpu);

        /// <summary>
        /// Creates a tensor filled with zeros on the CPU device.
        /// </summary>
        /// <param name="shape">The dimensions of the tensor, specified as variable integer arguments.</param>
        /// <returns>A new <see cref="Tensor"/> instance with the specified shape filled with zeros.</returns>
        protected Tensor Zeros(params int[] shape)
            => (Tensor)Tensor.Zeros(new TensorShape(shape), device: Cpu);
    }
}