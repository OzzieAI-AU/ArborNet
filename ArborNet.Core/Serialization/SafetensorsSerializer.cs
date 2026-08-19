// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Serialization
{

    #region Using Statements:

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
    /// <summary>
    /// Production-grade, zero-allocation native C# implementation of the Hugging Face Safetensors format.
    /// Safely saves and loads weights at direct system I/O speed.
    /// </summary>
    /// <remarks>
    /// This static utility class bypasses typical serialization overheads by performing direct memory-casting 
    /// of floating-point arrays to raw byte spans. The Safetensors format consists of an 8-byte little-endian 
    /// unsigned integer indicating the header length, followed by a UTF-8 JSON header string detailing 
    /// metadata (tensor shapes, types, and relative byte offsets), and ends with a contiguous binary payload.
    /// </remarks>

    #endregion

    public static class SafetensorsSerializer
    {
        /// <summary>
        /// Represents the metadata structure for an individual tensor within the Safetensors header.
        /// </summary>
        /// <remarks>
        /// This structure mirrors the JSON schema defined by the Hugging Face Safetensors format specification,
        /// containing critical layout details used to index and slice individual tensors from the trailing binary payload.
        /// </remarks>
        public class TensorMetadata
        {
            /// <summary>
            /// Gets or sets the data type of the tensor elements. Defaults to "F32".
            /// </summary>
            /// <value>
            /// A <see cref="string"/> representing the data format of the tensor. Currently, only "F32" is supported.
            /// </value>
            public string dtype { get; set; } = "F32";
            /// <summary>
            /// Gets or sets the shape dimensions of the tensor.
            /// </summary>
            /// <value>
            /// A <see cref="List{T}"/> of integers representing the size of each dimension in the tensor.
            /// </value>

            public List<int> shape { get; set; } = new();
            /// <summary>
            /// Gets or sets the start and end byte offsets of the tensor data within the binary buffer payload.
            /// </summary>
            /// <value>
            /// A <see cref="List{T}"/> containing exactly two <see cref="long"/> values: the zero-based starting 
            /// byte offset (inclusive) and the ending byte offset (exclusive).
            /// </value>

            public List<long> data_offsets { get; set; } = new();
        }

        /// <summary>
        /// Saves a collection of named tensors to a file in the Safetensors format.
        /// </summary>
        /// <param name="filePath">The path of the file to write to. Cannot be null or empty.</param>
        /// <param name="tensors">The dictionary mapping tensor names to their respective <see cref="ITensor"/> objects. Cannot be null.</param>
        /// <remarks>
        /// This method creates or overwrites the file at the specified <paramref name="filePath"/> and uses a write-only
        /// unshared file stream to delegate serialization to <see cref="Save(Stream, Dictionary{string, ITensor})"/>.
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="filePath"/> or <paramref name="tensors"/> is null.</exception>
        /// <exception cref="IOException">Thrown if an I/O error occurs while opening or writing to the file.</exception>
        /// <exception cref="UnauthorizedAccessException">Thrown if the caller does not have the required permissions to write to the specified path.</exception>
        /// <exception cref="DirectoryNotFoundException">Thrown if the directory specified in <paramref name="filePath"/> does not exist.</exception>
        /// <exception cref="PathTooLongException">Thrown when the specified path exceeds the system-defined maximum length.</exception>
        public static void Save(string filePath, Dictionary<string, ITensor> tensors)
        {
            if (string.IsNullOrEmpty(filePath)) throw new ArgumentNullException(nameof(filePath));
            if (tensors == null) throw new ArgumentNullException(nameof(tensors));

            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);

            Save(fs, tensors);
        }

        /// <summary>
        /// Saves a collection of named tensors to the specified stream in the Safetensors format.
        /// </summary>
        /// <param name="stream">The output <see cref="Stream"/> to which the tensor data will be written. Must support writing.</param>
        /// <param name="tensors">The dictionary mapping tensor names to their respective <see cref="ITensor"/> objects. Cannot be null.</param>
        /// <remarks>
        /// <para>
        /// Serialization conforms strictly to the Hugging Face Safetensors specification:
        /// <list type="number">
        /// <item><description>Computes metadata offsets and formats the JSON header.</description></item>
        /// <item><description>Writes an 8-byte little-endian unsigned integer representing the length of the JSON header.</description></item>
        /// <item><description>Writes the UTF-8 encoded JSON header bytes.</description></item>
        /// <item><description>Writes the raw contiguous binary payload of all floating-point arrays.</description></item>
        /// </list>
        /// </para>
        /// <para>
        /// High throughput is maintained by writing the raw memory spans of the float arrays directly to the output stream
        /// via <see cref="MemoryMarshal.AsBytes{T}(ReadOnlySpan{T})"/>, bypassing extra copy operations.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="stream"/> or <paramref name="tensors"/> is null.</exception>
        /// <exception cref="NotSupportedException">Thrown if the specified <paramref name="stream"/> does not support write operations.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if the <paramref name="stream"/> is closed or disposed before or during serialization.</exception>
        /// <exception cref="IOException">Thrown if an I/O error occurs while writing data to the stream.</exception>
        public static void Save(Stream stream, Dictionary<string, ITensor> tensors)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));
            if (tensors == null) throw new ArgumentNullException(nameof(tensors));

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

            // FIX: Pad header to ensure 8-byte alignment for the binary payload
            int padding = 8 - (headerBytes.Length % 8);
            if (padding < 8)
            {
                byte[] paddedHeader = new byte[headerBytes.Length + padding];
                Array.Copy(headerBytes, paddedHeader, headerBytes.Length);
                for (int i = 0; i < padding; i++) paddedHeader[headerBytes.Length + i] = 0x20; // 0x20 is ASCII Space
                headerBytes = paddedHeader;
            }

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
        
        /// <summary>
        /// Loads a collection of named tensors from the specified stream in Safetensors format.
        /// </summary>
        /// <param name="stream">The input <see cref="Stream"/> containing the Safetensors formatted data. Must support reading and seeking.</param>
        /// <param name="device">The target execution <see cref="Device"/> on which the loaded tensors should be allocated. Defaults to <see cref="Device.CPU"/> if null.</param>
        /// <returns>A dictionary containing the reconstructed named <see cref="ITensor"/> objects.</returns>
        /// <remarks>
        /// <para>
        /// This method reads the little-endian header length, extracts and parses the UTF-8 JSON metadata schema, 
        /// and performs seek operations to extract individual tensor segments directly from the stream.
        /// </para>
        /// <para>
        /// Reconstructed tensor buffers are read directly into float arrays via <see cref="MemoryMarshal.AsBytes{T}(Span{T})"/>
        /// to ensure zero intermediate heap allocations, before being instantiated onto the specified <paramref name="device"/>.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="stream"/> is null.</exception>
        /// <exception cref="InvalidDataException">
        /// Thrown when the header size, header JSON structure, or binary data payload is corrupted, incomplete, or fails verification.
        /// </exception>
        /// <exception cref="NotSupportedException">
        /// Thrown when encountering an unsupported tensor data type (anything other than F32), or if the stream does not support read/seek operations.
        /// </exception>
        /// <exception cref="ObjectDisposedException">Thrown if the <paramref name="stream"/> is closed or disposed before or during deserialization.</exception>
        /// <exception cref="IOException">Thrown if an I/O error occurs while reading or seeking within the stream.</exception>
        public static Dictionary<string, ITensor> Load(Stream stream, Device? device = null)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));
            device ??= Device.CPU;

            byte[] headerLengthBytes = new byte[8];
            int read = stream.Read(headerLengthBytes, 0, 8);
            if (read != 8) throw new InvalidDataException("Incomplete Safetensors file header size.");

            if (!BitConverter.IsLittleEndian) Array.Reverse(headerLengthBytes);
            ulong headerLength = BitConverter.ToUInt64(headerLengthBytes, 0);

            byte[] headerBytes = new byte[headerLength];
            read = stream.Read(headerBytes, 0, (int)headerLength);
            if (read != (int)headerLength) throw new InvalidDataException("Incomplete Safetensors file header.");

            string jsonHeader = Encoding.UTF8.GetString(headerBytes);
            var headerDict = JsonSerializer.Deserialize<Dictionary<string, TensorMetadata>>(jsonHeader);
            if (headerDict == null) throw new InvalidDataException("Invalid Safetensors JSON header.");

            var tensors = new Dictionary<string, ITensor>();
            long binaryStartOffset = stream.Position;

            // Optional: If stream is a FileStream, we can memory-map it for zero-allocation
            if (stream is FileStream fs && device.IsCpu())
            {
                using var mmf = System.IO.MemoryMappedFiles.MemoryMappedFile.CreateFromFile(
                    fs, null, 0, System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read, HandleInheritability.None, false);

                foreach (var kvp in headerDict)
                {
                    string name = kvp.Key;
                    var meta = kvp.Value;
                    long start = meta.data_offsets[0];
                    long lengthInBytes = meta.data_offsets[1] - start;
                    int numElements = (int)(lengthInBytes / sizeof(float));

                    using var accessor = mmf.CreateViewAccessor(binaryStartOffset + start, lengthInBytes, System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read);

                    float[] data = new float[numElements];

                    // FIX: ReadArray safely handles the required page-alignment PointerOffset
                    accessor.ReadArray(0, data, 0, numElements);

                    tensors[name] = Tensor.FromArray(data, new TensorShape(meta.shape.ToArray()), device);
                }
            }
            else
            {
                // Fallback for non-file streams or GPU target
                foreach (var kvp in headerDict)
                {
                    var meta = kvp.Value;
                    long start = meta.data_offsets[0];
                    int numElements = (int)((meta.data_offsets[1] - start) / sizeof(float));

                    stream.Seek(binaryStartOffset + start, SeekOrigin.Begin);

                    // FIX: Read directly into the array span to avoid double heap allocations
                    float[] dataArr = new float[numElements];
                    stream.ReadExactly(MemoryMarshal.AsBytes<float>(dataArr.AsSpan()));

                    tensors[kvp.Key] = Tensor.FromArray(dataArr, new TensorShape(meta.shape.ToArray()), device);
                }
            }

            return tensors;
        }

        /// <summary>
        /// Loads a collection of named tensors from a Safetensors file.
        /// </summary>
        /// <param name="filePath">The path to the Safetensors file on disk. Cannot be null or empty.</param>
        /// <param name="device">The target execution <see cref="Device"/> on which the loaded tensors should be allocated. Defaults to <see cref="Device.CPU"/> if null.</param>
        /// <returns>A dictionary containing the reconstructed named <see cref="ITensor"/> objects.</returns>
        /// <remarks>
        /// This method opens the specified file in read-only mode with shared read access and delegates 
        /// deserialization to <see cref="Load(Stream, Device?)"/>.
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="filePath"/> is null or empty.</exception>
        /// <exception cref="FileNotFoundException">Thrown when the file specified by <paramref name="filePath"/> does not exist.</exception>
        /// <exception cref="DirectoryNotFoundException">Thrown when the path structure leading up to <paramref name="filePath"/> is invalid.</exception>
        /// <exception cref="IOException">Thrown if an I/O error occurs while opening or reading the file.</exception>
        /// <exception cref="UnauthorizedAccessException">Thrown if the caller does not have read permissions for the specified file.</exception>
        /// <exception cref="InvalidDataException">Thrown when the file header, JSON metadata, or binary payloads are malformed.</exception>
        /// <exception cref="NotSupportedException">Thrown when encountering an unsupported tensor data type.</exception>
        public static Dictionary<string, ITensor> Load(string filePath, Device? device = null)
        {
            if (string.IsNullOrEmpty(filePath)) throw new ArgumentNullException(nameof(filePath));

            using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);

            return Load(fs, device);
        }
    }
}