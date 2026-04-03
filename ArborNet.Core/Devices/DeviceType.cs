namespace ArborNet.Core.Devices
{
    /// <summary>
    /// Represents the type of compute device that can be used for operations within ArborNet.
    /// </summary>
    /// <remarks>
    /// This enumeration is used throughout the framework to specify the target hardware
    /// for computational workloads, enabling seamless switching between CPU and GPU acceleration.
    /// </remarks>
    public enum DeviceType
    {
        /// <summary>
        /// The system's central processing unit (CPU).
        /// </summary>
        /// <remarks>
        /// Provides broad compatibility but typically offers lower parallel performance
        /// compared to GPU devices for deep learning and high-throughput workloads.
        /// </remarks>
        CPU,

        /// <summary>
        /// A CUDA-enabled NVIDIA GPU device.
        /// </summary>
        /// <remarks>
        /// Utilizes the NVIDIA CUDA platform for hardware-accelerated computations.
        /// </remarks>
        CUDA,

        /// <summary>
        /// A ROCm-enabled AMD GPU device.
        /// </summary>
        /// <remarks>
        /// Utilizes the AMD ROCm platform for hardware-accelerated computations.
        /// </remarks>
        ROCm
    }
}