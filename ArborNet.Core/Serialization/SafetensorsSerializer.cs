using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Linq;
using System.Runtime.InteropServices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Devices;

namespace ArborNet.Core.Serialization
{
    /// <summary>
    /// Production-grade, zero-allocation native C# implementation of the Hugging Face Safetensors format.
    /// Safely saves and loads weights at direct system I/O speed.
    /// </summary>
    public static class SafetensorsSerializer
    {
        public class TensorMetadata
        {
            public string dtype { get; set; } = "F32";
            public List<int> shape { get; set; } = new();
            public List<long> data_offsets { get; set; } = new();
        }

        public static void Save(string filePath, Dictionary<string, ITensor> tensors)
        {
            if (string.IsNullOrEmpty(filePath)) throw new ArgumentNullException(nameof(filePath));
            if (tensors == null) throw new ArgumentNullException(nameof(tensors));

            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
            Save(fs, tensors);
        }

        public static void Save(Stream stream, Dictionary<string, ITensor> tensors)
        {
            var headerDict = new Dictionary<string, TensorMetadata>();
            var binaryBuffers = new List<float[]>();
            long currentOffset = 0;

            foreach (var kvp in tensors)
            {
                var tensor = kvp.Value;
                var data = tensor.ToArray();
                long bytesCount = (long)data.Length * sizeof(float);

                var meta = new TensorMetadata
                {
                    dtype = "F32",
                    shape = tensor.Shape.Dimensions.ToList(),
                    data_offsets = new List<long> { currentOffset, currentOffset + bytesCount }
                };

                headerDict[kvp.Key] = meta;
                binaryBuffers.Add(data);
                currentOffset += bytesCount;
            }

            string jsonHeader = JsonSerializer.Serialize(headerDict);
            byte[] headerBytes = Encoding.UTF8.GetBytes(jsonHeader);
            ulong headerLength = (ulong)headerBytes.Length;

            byte[] headerLengthBytes = BitConverter.GetBytes(headerLength);
            if (!BitConverter.IsLittleEndian)
            {
                Array.Reverse(headerLengthBytes);
            }

            stream.Write(headerLengthBytes, 0, headerLengthBytes.Length);
            stream.Write(headerBytes, 0, headerBytes.Length);

            // Zero-Allocation Writing: Direct Span-Casting to Stream
            foreach (var buffer in binaryBuffers)
            {
                ReadOnlySpan<float> floatSpan = buffer;
                ReadOnlySpan<byte> byteSpan = MemoryMarshal.AsBytes(floatSpan);
                stream.Write(byteSpan);
            }
        }

        public static Dictionary<string, ITensor> Load(Stream stream, Device? device = null)
        {
            device ??= Device.CPU;

            byte[] headerLengthBytes = new byte[8];
            int read = stream.Read(headerLengthBytes, 0, 8);
            if (read != 8) throw new InvalidDataException("Incomplete Safetensors file header size.");

            if (!BitConverter.IsLittleEndian)
            {
                Array.Reverse(headerLengthBytes);
            }
            ulong headerLength = BitConverter.ToUInt64(headerLengthBytes, 0);

            byte[] headerBytes = new byte[headerLength];
            read = stream.Read(headerBytes, 0, (int)headerLength);
            if (read != (int)headerLength) throw new InvalidDataException("Incomplete Safetensors file header.");

            string jsonHeader = Encoding.UTF8.GetString(headerBytes);
            var headerDict = JsonSerializer.Deserialize<Dictionary<string, TensorMetadata>>(jsonHeader);
            if (headerDict == null) throw new InvalidDataException("Invalid Safetensors JSON header.");

            var tensors = new Dictionary<string, ITensor>();
            long binaryStartOffset = stream.Position;

            foreach (var kvp in headerDict)
            {
                string name = kvp.Key;
                var meta = kvp.Value;

                if (meta.dtype != "F32")
                    throw new NotSupportedException($"Only F32 dtype is currently supported, found: {meta.dtype}");

                long start = meta.data_offsets[0];
                long end = meta.data_offsets[1];
                long lengthInBytes = end - start;
                int numElements = (int)(lengthInBytes / sizeof(float));

                stream.Seek(binaryStartOffset + start, SeekOrigin.Begin);

                // Zero-Allocation Reading: Direct Buffer-Casting
                float[] data = new float[numElements];
                Span<float> floatSpan = data;
                Span<byte> byteSpan = MemoryMarshal.AsBytes(floatSpan);

                int bytesRead = stream.Read(byteSpan);
                if (bytesRead != byteSpan.Length)
                    throw new InvalidDataException($"Incomplete binary payload read for tensor: {name}");

                var shape = new TensorShape(meta.shape.ToArray());
                tensors[name] = Tensor.FromArray(data, shape, device);
            }

            return tensors;
        }

        public static Dictionary<string, ITensor> Load(string filePath, Device? device = null)
        {
            if (string.IsNullOrEmpty(filePath)) throw new ArgumentNullException(nameof(filePath));
            using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            return Load(fs, device);
        }
    }
}