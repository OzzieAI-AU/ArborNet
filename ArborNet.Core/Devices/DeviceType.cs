// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Devices
{



    #region
    #endregion


    /// <summary>
    /// Specifies the type of computational device used for executing operations within the ArborNet framework.
    /// </summary>
    /// <remarks>
    /// This enumeration is utilized throughout the system to target specific hardware backends, 
    /// enabling seamless runtime switching between standard host execution (CPU) and various 
    /// hardware-accelerated graphics processing units (GPUs).
    /// </remarks>
    public enum DeviceType
    {
        /// <summary>
        /// The system's Central Processing Unit (CPU).
        /// </summary>
        /// <remarks>
        /// Serves as the default fallback device. Provides maximum compatibility and direct access 
        /// to host system memory, but generally yields lower throughput for highly parallelized 
        /// tensor and deep learning workloads compared to dedicated accelerator hardware.
        /// </remarks>
        CPU,

        /// <summary>
        /// An NVIDIA Graphics Processing Unit (GPU) utilizing the CUDA platform.
        /// </summary>
        /// <remarks>
        /// Enables hardware-accelerated computations by targeting NVIDIA's Compute Unified Device Architecture. 
        /// Requires a compatible NVIDIA GPU and the appropriate CUDA driver/runtime environment.
        /// </remarks>
        CUDA,

        /// <summary>
        /// An AMD Graphics Processing Unit (GPU) utilizing the ROCm platform.
        /// </summary>
        /// <remarks>
        /// Enables hardware-accelerated computations by targeting AMD's Radeon Open Compute platform. 
        /// Requires a compatible AMD GPU and the ROCm runtime environment.
        /// </remarks>
        ROCm
    }
}