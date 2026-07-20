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

    #region Using Statements:

    using System;
    /// <summary>
    /// Represents a computational execution and memory allocation device within the ArborNet framework.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This class acts as a unified abstraction over different hardware backends, including central processing units (<see cref="DeviceType.CPU"/>) 
    /// and graphics processing units from NVIDIA (<see cref="DeviceType.CUDA"/>) and AMD (<see cref="DeviceType.ROCm"/>).
    /// </para>
    /// <para>
    /// Instances of this class are immutable. To target a specific hardware accelerator, use the predefined static instances 
    /// (e.g., <see cref="CPU"/>, <see cref="CUDA"/>, <see cref="ROCm"/>) or construct custom instances via the factory methods or constructor.
    /// </para>
    /// </remarks>
    /// <example>
    /// <para>
    /// The following example demonstrates how to reference the default CPU device and create a custom CUDA device:
    /// </para>
    /// <code language="csharp">
    /// // Reference the default CPU device
    /// Device cpuDevice = Device.CPU;
    /// 
    /// // Create a CUDA device targeting GPU index 1
    /// Device cudaDevice = Device.Cuda(1);
    /// 
    /// // Use implicit conversion from an integer (maps to CUDA for positive indices)
    /// Device implicitDevice = 2; // Equivalent to Device.Cuda(2)
    /// </code>
    /// </example>

    #endregion

    public class Device
    {
        /// <summary>
        /// Gets the hardware architecture type of this device.
        /// </summary>
        /// <value>
        /// A <see cref="DeviceType"/> representing the hardware backend (e.g., CPU, CUDA, or ROCm).
        /// </value>
        public DeviceType Type { get; }
        /// <summary>
        /// Gets the zero-based physical index/identifier of the device.
        /// </summary>
        /// <value>
        /// An integer indicating the specific device ID. For <see cref="DeviceType.CPU"/>, this is always 0.
        /// </value>
        /// <remarks>
        /// This ID corresponds to the system-level index of the accelerator card (e.g., GPU index 0, 1, etc.).
        /// During instantiation, negative ID values are automatically clamped to 0.
        /// </remarks>

        public int Id { get; }

        /// <summary>
        /// Gets a predefined static instance representing the system's central processing unit (CPU).
        /// </summary>
        public static readonly Device CPU = new Device(DeviceType.CPU, 0);

        /// <summary>
        /// Gets a predefined static instance representing the primary NVIDIA CUDA GPU device (index 0).
        /// </summary>
        public static readonly Device CUDA = new Device(DeviceType.CUDA, 0);

        /// <summary>
        /// Gets a predefined static instance representing the primary AMD ROCm GPU device (index 0).
        /// </summary>
        public static readonly Device ROCm = new Device(DeviceType.ROCm, 0);

        /// <summary>
        /// Initializes a new instance of the <see cref="Device"/> class with the specified device type and identifier.
        /// </summary>
        /// <param name="type">The hardware architecture type of the device.</param>
        /// <param name="id">The zero-based hardware index. If a negative value is provided, it will be clamped to 0.</param>
        public Device(DeviceType type, int id = 0)
        {
            Type = type;
            Id = Math.Max(0, id);
        }
        /// <summary>
        /// Factory method to create a <see cref="Device"/> instance representing an NVIDIA CUDA GPU with the specified index.
        /// </summary>
        /// <param name="deviceId">The zero-based CUDA device identifier. Defaults to 0.</param>
        /// <returns>A new <see cref="Device"/> configured for CUDA execution on the specified device index.</returns>

        public static Device Cuda(int deviceId = 0) => new Device(DeviceType.CUDA, deviceId);
        /// <summary>
        /// Factory method to create a <see cref="Device"/> instance representing an AMD ROCm GPU with the specified index.
        /// </summary>
        /// <param name="deviceId">The zero-based ROCm device identifier. Defaults to 0.</param>
        /// <returns>A new <see cref="Device"/> configured for ROCm execution on the specified device index.</returns>

        public static Device Rocm(int deviceId = 0) => new Device(DeviceType.ROCm, deviceId);

        /// <summary>
        /// Performs an implicit conversion from a device identifier integer to a <see cref="Device"/> instance.
        /// </summary>
        /// <param name="deviceId">The device identifier. A value of 0 maps to <see cref="CPU"/>, and positive values map to a CUDA device with that ID.</param>
        /// <returns>A <see cref="Device"/> instance matching the specified conversion rule.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="deviceId"/> is less than 0.</exception>
        public static implicit operator Device(int deviceId)
        {
            if (deviceId == 0) return CPU;
            if (deviceId > 0) return new Device(DeviceType.CUDA, deviceId);
            throw new ArgumentOutOfRangeException(nameof(deviceId));
        }
        /// <summary>
        /// Returns a string representation of the current device.
        /// </summary>
        /// <returns>
        /// A string representing the device. Returns "CPU" for CPU devices, and the backend name suffixed with the ID (e.g., "CUDA:0") for accelerators.
        /// </returns>

        public override string ToString() => Type switch
        {
            DeviceType.CPU => "CPU",
            DeviceType.CUDA => $"CUDA:{Id}",
            DeviceType.ROCm => $"ROCm:{Id}",
            _ => "Unknown"
        };
        /// <summary>
        /// Determines whether the specified object is equal to the current device.
        /// </summary>
        /// <param name="obj">The object to compare with the current device.</param>
        /// <returns>
        /// <see langword="true"/> if the specified object is a <see cref="Device"/> and has the same <see cref="Type"/> and <see cref="Id"/>; otherwise, <see langword="false"/>.
        /// </returns>
        /// <remarks>
        /// This method performs value-based equality comparison based on the device's hardware type and hardware index.
        /// </remarks>

        public override bool Equals(object? obj) => obj is Device other && Type == other.Type && Id == other.Id;
        /// <summary>
        /// Serves as the default hash function.
        /// </summary>
        /// <returns>A hash code for the current device, computed from its <see cref="Type"/> and <see cref="Id"/>.</returns>

        public override int GetHashCode() => HashCode.Combine(Type, Id);
        /// <summary>
        /// Gets a value indicating whether this device represents the CPU.
        /// </summary>
        /// <returns><see langword="true"/> if the device type is <see cref="DeviceType.CPU"/>; otherwise, <see langword="false"/>.</returns>

        public bool IsCpu() => Type == DeviceType.CPU;
        /// <summary>
        /// Gets a value indicating whether this device represents an NVIDIA CUDA GPU.
        /// </summary>
        /// <returns><see langword="true"/> if the device type is <see cref="DeviceType.CUDA"/>; otherwise, <see langword="false"/>.</returns>

        public bool IsCuda() => Type == DeviceType.CUDA;
        /// <summary>
        /// Gets a value indicating whether this device represents an AMD ROCm GPU.
        /// </summary>
        /// <returns><see langword="true"/> if the device type is <see cref="DeviceType.ROCm"/>; otherwise, <see langword="false"/>.</returns>

        public bool IsRocm() => Type == DeviceType.ROCm;
    }
}