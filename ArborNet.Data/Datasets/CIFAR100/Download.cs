// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Data.Datasets.CIFAR100
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Net.Http;
    using System.Threading.Tasks;
    using ArborNet.Core.Tensors;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Functional;
    /// <summary>
    /// Provides utility methods for downloading and loading the CIFAR-100 dataset
    /// in its Python tar.gz format into ArborNet tensors and label arrays.
    /// </summary>
    /// <remarks>
    /// The CIFAR-100 dataset consists of 60,000 32x32 color images in 100 classes, with 600 images per class. 
    /// There are 500 training images and 100 testing images per class.
    /// This helper automates the retrieval of the raw binaries and parses the dataset format into ready-to-use float tensors.
    /// </remarks>

    #endregion

    public static class Download
    {
        /// <summary>
        /// The base URL for the official CIFAR dataset repository.
        /// </summary>
        private const string BaseUrl = "https://www.cs.toronto.edu/~kriz/";

        /// <summary>
        /// The filename of the CIFAR-100 Python version archive.
        /// </summary>
        private const string FileName = "cifar-100-python.tar.gz";

        /// <summary>
        /// The complete download URL for the CIFAR-100 dataset archive.
        /// </summary>
        private const string Url = BaseUrl + FileName;
        /// <summary>
        /// Downloads the CIFAR-100 dataset archive asynchronously from the official URL to a specified destination directory.
        /// </summary>
        /// <param name="destinationPath">The local directory path where the dataset archive will be saved.</param>
        /// <returns>A <see cref="Task"/> that represents the asynchronous download operation.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="destinationPath"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when <paramref name="destinationPath"/> is empty or contains invalid path characters.</exception>
        /// <exception cref="HttpRequestException">Thrown when the HTTP request fails or returns an unsuccessful status code.</exception>
        /// <exception cref="IOException">Thrown when writing to the file system fails.</exception>
        /// <exception cref="UnauthorizedAccessException">Thrown when access to the destination path is denied.</exception>

        public static async Task DownloadDatasetAsync(string destinationPath)
        {
            string filePath = Path.Combine(destinationPath, FileName);

            using (var httpClient = new HttpClient())
            {
                var response = await httpClient.GetAsync(Url);
                response.EnsureSuccessStatusCode();

                using (var contentStream = await response.Content.ReadAsStreamAsync())
                using (var fileStream = File.Create(filePath))
                {
                    await contentStream.CopyToAsync(fileStream);
                }
            }
        }
        /// <summary>
        /// Loads the CIFAR-100 training and test data from the extracted binary batch files in the specified path.
        /// </summary>
        /// <param name="extractedPath">The folder path containing the extracted CIFAR batch files (<c>data_batch_1</c> to <c>data_batch_5</c> and <c>test_batch</c>).</param>
        /// <returns>
        /// A tuple containing:
        /// <list type="bullet">
        ///   <item>
        ///     <term><c>trainData</c></term>
        ///     <description>An <see cref="ITensor"/> representing normalized training images of shape (50000, 3072).</description>
        ///   </item>
        ///   <item>
        ///     <term><c>trainLabels</c></term>
        ///     <description>An array of 50,000 integer training labels.</description>
        ///   </item>
        ///   <item>
        ///     <term><c>testData</c></term>
        ///     <description>An <see cref="ITensor"/> representing normalized test images of shape (10000, 3072).</description>
        ///   </item>
        ///   <item>
        ///     <term><c>testLabels</c></term>
        ///     <description>An array of 10,000 integer test labels.</description>
        ///   </item>
        /// </list>
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="extractedPath"/> is null.</exception>
        /// <exception cref="DirectoryNotFoundException">Thrown when <paramref name="extractedPath"/> cannot be found.</exception>
        /// <exception cref="FileNotFoundException">Thrown when any expected binary batch file is missing.</exception>
        /// <exception cref="IOException">Thrown when reading the dataset file fails.</exception>

        public static (ITensor trainData, int[] trainLabels, ITensor testData, int[] testLabels) LoadDataset(string extractedPath)
        {
            List<float[]> trainImages = new List<float[]>();
            List<int> trainLabels = new List<int>();
            for (int i = 1; i <= 5; i++)
            {
                string batchFile = Path.Combine(extractedPath, $"data_batch_{i}");
                var (images, labels) = LoadBatch(batchFile);
                trainImages.AddRange(images);
                trainLabels.AddRange(labels);
            }

            string testFile = Path.Combine(extractedPath, "test_batch");
            var (testImages, testLabelsList) = LoadBatch(testFile);

            int numTrain = trainImages.Count;
            int numTest = testImages.Count;
            int imageSize = 32 * 32 * 3;

            float[] trainDataFlat = new float[numTrain * imageSize];
            for (int i = 0; i < numTrain; i++)
            {
                Array.Copy(trainImages[i], 0, trainDataFlat, i * imageSize, imageSize);
            }

            float[] testDataFlat = new float[numTest * imageSize];
            for (int i = 0; i < numTest; i++)
            {
                Array.Copy(testImages[i], 0, testDataFlat, i * imageSize, imageSize);
            }

            ITensor trainData = Ops.FromArray(trainDataFlat, new TensorShape(numTrain, imageSize));
            ITensor testData = Ops.FromArray(testDataFlat, new TensorShape(numTest, imageSize));

            return (trainData, trainLabels.ToArray(), testData, testLabelsList.ToArray());
        }
        /// <summary>
        /// Loads and parses a single CIFAR-100 binary batch file.
        /// </summary>
        /// <param name="filePath">The full file path to the binary batch file.</param>
        /// <returns>
        /// A tuple containing:
        /// <list type="bullet">
        ///   <item>
        ///     <term><c>images</c></term>
        ///     <description>A list of float arrays, each representing a flattened, normalized image (3072 features, normalized to [0, 1]).</description>
        ///   </item>
        ///   <item>
        ///     <term><c>labels</c></term>
        ///     <description>A list of corresponding integer labels.</description>
        ///   </item>
        /// </list>
        /// </returns>
        /// <remarks>
        /// This method skips file metadata using predefined offsets and scales byte channel values (0-255) to floating point values (0.0-1.0).
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="filePath"/> is null.</exception>
        /// <exception cref="FileNotFoundException">Thrown when the target file does not exist.</exception>
        /// <exception cref="EndOfStreamException">Thrown if reading beyond the end of the binary file stream.</exception>
        /// <exception cref="IOException">Thrown on stream access failures.</exception>

        private static (List<float[]> images, List<int> labels) LoadBatch(string filePath)
        {
            List<float[]> images = new List<float[]>();
            List<int> labels = new List<int>();

            using (var reader = new BinaryReader(File.OpenRead(filePath)))
            {
                reader.BaseStream.Position = 0;

                reader.BaseStream.Seek(8, SeekOrigin.Begin);

                for (int i = 0; i < 10000; i++)
                {
                    labels.Add(reader.ReadInt32());
                }

                reader.BaseStream.Seek(10000 * 4 + 16, SeekOrigin.Begin);

                for (int i = 0; i < 10000; i++)
                {
                    byte[] imageBytes = reader.ReadBytes(3072);
                    float[] image = new float[3072];
                    for (int j = 0; j < 3072; j++)
                    {
                        image[j] = imageBytes[j] / 255.0f;
                    }
                    images.Add(image);
                }
            }

            return (images, labels);
        }
    }
}