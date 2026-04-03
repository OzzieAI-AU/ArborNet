using System;

namespace ArborNet.Core.Devices
{
    /// <summary>
    /// Represents a computational device in the ArborNet framework.
    /// Supports CPU, CUDA (NVIDIA GPUs), and ROCm (AMD GPUs) with optional device indexing.
    /// </summary>
    /// <remarks>
    /// This class is used throughout the framework to specify target hardware for operations,
    /// memory allocation, and execution. It provides both predefined static instances and
    /// factory methods for creating device references.
    /// </remarks>
    public class Device
    {
        /// <summary>
        /// Gets the type of the device.
        /// </summary>
        public DeviceType Type { get; }

        /// <summary>
        /// Gets the identifier of the device.
        /// </summary>
        /// <remarks>
        /// For <see cref="DeviceType.CPU"/>, this value is always 0.
        /// Device IDs are zero-based and clamped to non-negative values.
        /// </remarks>
        public int Id { get; }

        /// <summary>
        /// Represents the CPU device.
        /// </summary>
        public static readonly Device CPU = new Device(DeviceType.CPU, 0);

        /// <summary>
        /// Represents the default CUDA device (device ID 0).
        /// </summary>
        public static readonly Device CUDA = new Device(DeviceType.CUDA, 0);

        /// <summary>
        /// Represents the default ROCm device (device ID 0).
        /// </summary>
        public static readonly Device ROCm = new Device(DeviceType.ROCm, 0);

        /// <summary>
        /// Initializes a new instance of the <see cref="Device"/> class.
        /// </summary>
        /// <param name="type">The type of the device.</param>
        /// <param name="id">The device identifier. Negative values are clamped to zero.</param>
        public Device(DeviceType type, int id = 0)
        {
            Type = type;
            Id = Math.Max(0, id);
        }

        /// <summary>
        /// Creates a CUDA device with the specified device identifier.
        /// </summary>
        /// <param name="deviceId">The zero-based CUDA device identifier.</param>
        /// <returns>A <see cref="Device"/> instance representing the specified CUDA device.</returns>
        public static Device Cuda(int deviceId = 0) => new Device(DeviceType.CUDA, deviceId);

        /// <summary>
        /// Creates a ROCm device with the specified device identifier.
        /// </summary>
        /// <param name="deviceId">The zero-based ROCm device identifier.</param>
        /// <returns>A <see cref="Device"/> instance representing the specified ROCm device.</returns>
        public static Device Rocm(int deviceId = 0) => new Device(DeviceType.ROCm, deviceId);

        /// <summary>
        /// Performs an implicit conversion from an integer to a <see cref="Device"/>.
        /// </summary>
        /// <param name="deviceId">The device identifier to convert.</param>
        /// <returns>
        /// A <see cref="Device"/> instance. Zero returns <see cref="CPU"/>, 
        /// positive values return a CUDA device.
        /// </returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="deviceId"/> is negative.</exception>
        public static implicit operator Device(int deviceId)
        {
            if (deviceId == 0) return CPU;
            if (deviceId > 0) return new Device(DeviceType.CUDA, deviceId);
            throw new ArgumentOutOfRangeException(nameof(deviceId));
        }

        /// <summary>
        /// Returns a string that represents the current device.
        /// </summary>
        /// <returns>A human-readable string representation of the device.</returns>
        public override string ToString() => Type switch
        {
            DeviceType.CPU => "CPU",
            DeviceType.CUDA => $"CUDA:{Id}",
            DeviceType.ROCm => $"ROCm:{Id}",
            _ => "Unknown"
        };

        /// <summary>
        /// Determines whether the specified object is equal to the current <see cref="Device"/>.
        /// </summary>
        /// <param name="obj">The object to compare with the current device.</param>
        /// <returns>
        /// <c>true</c> if the specified object is a <see cref="Device"/> with the same type and ID; 
        /// otherwise, <c>false</c>.
        /// </returns>
        public override bool Equals(object? obj) => obj is Device other && Type == other.Type && Id == other.Id;

        /// <summary>
        /// Returns the hash code for this device.
        /// </summary>
        /// <returns>A hash code based on the device type and identifier.</returns>
        public override int GetHashCode() => HashCode.Combine(Type, Id);

        /// <summary>
        /// Determines whether this instance represents a CPU device.
        /// </summary>
        /// <returns><c>true</c> if this device is a CPU; otherwise, <c>false</c>.</returns>
        public bool IsCpu() => Type == DeviceType.CPU;

        /// <summary>
        /// Determines whether this instance represents a CUDA device.
        /// </summary>
        /// <returns><c>true</c> if this device is a CUDA device; otherwise, <c>false</c>.</returns>
        public bool IsCuda() => Type == DeviceType.CUDA;

        /// <summary>
        /// Determines whether this instance represents a ROCm device.
        /// </summary>
        /// <returns><c>true</c> if this device is a ROCm device; otherwise, <c>false</c>.</returns>
        public bool IsRocm() => Type == DeviceType.ROCm;
    }
}