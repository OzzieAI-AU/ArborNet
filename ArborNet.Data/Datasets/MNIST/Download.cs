// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Data.Datasets.MNIST
{

    #region Using Statements:

    using System;
    using System.IO;
    using System.IO.Compression;
    using System.Net.Http;
    using System.Threading.Tasks;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Provides comprehensive asynchronous utility methods to download, extract, parse, and load the MNIST dataset.
    /// </summary>
    /// <remarks>
    /// The MNIST database (Modified National Institute of Standards and Technology database) contains 70,000 grayscale 
    /// images of handwritten digits (0 through 9), each sized 28x28 pixels. The dataset is split into:
    /// <list type="bullet">
    /// <item><description>A training set of 60,000 samples.</description></item>
    /// <item><description>A test set of 10,000 samples.</description></item>
    /// </list>
    /// This utility class automates the retrieval of binary gz archives from the official servers, decompresses them locally, 
    /// parses the custom IDX binary file formats, and maps the loaded values to optimized <see cref="ITensor"/> objects.
    /// </remarks>

    #endregion

    public static class Download
    {
        /// <summary>
        /// The base URL where the official MNIST dataset files are hosted.
        /// </summary>
        private const string BaseUrl = "http://yann.lecun.com/exdb/mnist/";
        /// <summary>
        /// The filenames of the four gzipped MNIST dataset files (training images, training labels, 
        /// test images, and test labels).
        /// </summary>
        private static readonly string[] Files = {
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        };
        /// <summary>
        /// Downloads the MNIST dataset if not already locally cached, extracts the gzipped files, and loads the data into tensors and arrays.
        /// </summary>
        /// <param name="dataDir">The relative or absolute file directory path where the dataset is stored or should be downloaded. Defaults to "data/MNIST".</param>
        /// <returns>
        /// A task representing the asynchronous operation, wrapping a tuple containing:
        /// <list type="table">
        /// <listheader>
        /// <term>Element Type</term>
        /// <description>Description</description>
        /// </listheader>
        /// <item>
        /// <term><see cref="ITensor"/> <c>TrainImages</c></term>
        /// <description>A flattened tensor of shape [60000, 784] containing pixel values normalized to the range [0.0f, 1.0f].</description>
        /// </item>
        /// <item>
        /// <term><see cref="T:int[]"/> <c>TrainLabels</c></term>
        /// <description>An array containing 60,000 integer target labels corresponding to each training image.</description>
        /// </item>
        /// <item>
        /// <term><see cref="ITensor"/> <c>TestImages</c></term>
        /// <description>A flattened tensor of shape [10000, 784] containing pixel values normalized to the range [0.0f, 1.0f].</description>
        /// </item>
        /// <item>
        /// <term><see cref="T:int[]"/> <c>TestLabels</c></term>
        /// <description>An array containing 10,000 integer target labels corresponding to each testing image.</description>
        /// </item>
        /// </list>
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="dataDir"/> is null or empty.</exception>
        /// <exception cref="HttpRequestException">Thrown if a network request error occurs during dataset download.</exception>
        /// <exception cref="IOException">Thrown if an I/O error occurs while creating directories or accessing files.</exception>

        public static async Task<(ITensor TrainImages, int[] TrainLabels, ITensor TestImages, int[] TestLabels)> GetDatasetAsync(string dataDir = "data/MNIST")
        {
            Directory.CreateDirectory(dataDir);

            var tasks = new Task[4];
            for (int i = 0; i < 4; i++)
            {
                string fileName = Files[i];
                string localGzPath = Path.Combine(dataDir, fileName);
                string extractedPath = Path.Combine(dataDir, fileName.Replace(".gz", ""));
                tasks[i] = DownloadAndExtractAsync($"{BaseUrl}{fileName}", localGzPath, extractedPath);
            }

            await Task.WhenAll(tasks);

            // Load the data
            string trainImagesPath = Path.Combine(dataDir, "train-images-idx3-ubyte");
            string trainLabelsPath = Path.Combine(dataDir, "train-labels-idx1-ubyte");
            string testImagesPath = Path.Combine(dataDir, "t10k-images-idx3-ubyte");
            string testLabelsPath = Path.Combine(dataDir, "t10k-labels-idx1-ubyte");

            var trainImages = LoadImages(trainImagesPath);
            var trainLabels = LoadLabels(trainLabelsPath);
            var testImages = LoadImages(testImagesPath);
            var testLabels = LoadLabels(testLabelsPath);

            return (trainImages, trainLabels, testImages, testLabels);
        }
        /// <summary>
        /// Asynchronously downloads a compressed gzip archive from the specified URL and decompresses it to the target file path.
        /// </summary>
        /// <param name="url">The source URL from which to download the compressed archive file.</param>
        /// <param name="gzPath">The local destination file path for the downloaded gzip archive.</param>
        /// <param name="extractedPath">The final destination path where the decompressed data will be written.</param>
        /// <returns>A <see cref="Task"/> representing the asynchronous operation.</returns>
        /// <exception cref="HttpRequestException">Thrown if the remote server returns a non-success HTTP status code.</exception>
        /// <exception cref="IOException">Thrown if a read or write operation fails during the file download or decompression phases.</exception>

        private static async Task DownloadAndExtractAsync(string url, string gzPath, string extractedPath)
        {
            if (!File.Exists(extractedPath))
            {
                using (var client = new HttpClient())
                {
                    using (var response = await client.GetAsync(url))
                    {
                        response.EnsureSuccessStatusCode();
                        using (var fs = new FileStream(gzPath, FileMode.Create))
                        {
                            await response.Content.CopyToAsync(fs);
                        }
                    }
                }

                using (var gzStream = new GZipStream(File.OpenRead(gzPath), CompressionMode.Decompress))
                {
                    using (var fs = new FileStream(extractedPath, FileMode.Create))
                    {
                        await gzStream.CopyToAsync(fs);
                    }
                }
            }
        }
        /// <summary>
        /// Reads and decodes a local binary file formatted in the MNIST IDX3-ubyte layout representing image byte matrices.
        /// </summary>
        /// <param name="path">The local file path pointing to the decompressed IDX3 image byte file.</param>
        /// <returns>An <see cref="ITensor"/> containing the flattened grayscale image pixel values normalized to the range [0.0f, 1.0f].</returns>
        /// <exception cref="FileNotFoundException">Thrown if the file specified by <paramref name="path"/> cannot be located.</exception>
        /// <exception cref="InvalidDataException">Thrown if the file does not have the correct magic number structure or is corrupt.</exception>

        private static ITensor LoadImages(string path)
        {

            using (var br = new BinaryReader(File.OpenRead(path)))
            {
                int magic = br.ReadInt32BigEndian();
                int numImages = br.ReadInt32BigEndian();
                int rows = br.ReadInt32BigEndian();
                int cols = br.ReadInt32BigEndian();

                float[] data = new float[numImages * rows * cols];
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = br.ReadByte() / 255.0f;
                }

                return Tensor.FromArray(data, new TensorShape(numImages, rows * cols), Device.CPU);
            }
        }
        /// <summary>
        /// Reads and decodes a local binary file formatted in the MNIST IDX1-ubyte layout representing single-byte label values.
        /// </summary>
        /// <param name="path">The local file path pointing to the decompressed IDX1 label byte file.</param>
        /// <returns>An array of integers representing class indices (values from 0 to 9) corresponding to each input sample.</returns>
        /// <exception cref="FileNotFoundException">Thrown if the file specified by <paramref name="path"/> cannot be located.</exception>
        /// <exception cref="InvalidDataException">Thrown if the file does not contain correct sequence lengths or values.</exception>

        private static int[] LoadLabels(string path)
        {
            using (var br = new BinaryReader(File.OpenRead(path)))
            {
                int magic = br.ReadInt32BigEndian();
                int numLabels = br.ReadInt32BigEndian();

                int[] labels = new int[numLabels];
                for (int i = 0; i < numLabels; i++)
                {
                    labels[i] = br.ReadByte();
                }

                return labels;
            }
        }
        /// <summary>
        /// Extension method for <see cref="BinaryReader"/> to safely decode a 32-bit signed integer written in big-endian byte order.
        /// </summary>
        /// <param name="br">The binary reader to read bytes from.</param>
        /// <returns>The decoded 32-bit signed integer converted to little-endian representation.</returns>
        /// <remarks>
        /// This method is required because the standard .NET BinaryReader interprets integers as little-endian by default, 
        /// whereas standard IDX format specification files store multibyte integer data fields exclusively in big-endian layout.
        /// </remarks>
        /// <exception cref="EndOfStreamException">Thrown if the stream reaches its end before reading the requested 4 bytes.</exception>

        private static int ReadInt32BigEndian(this BinaryReader br)
        {
            byte[] bytes = br.ReadBytes(4);
            Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
    }
}