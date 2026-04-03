using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using ArborNet.Core.Tensors;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Functional;

namespace ArborNet.Data.Datasets.CIFAR100
{
    /// <summary>
    /// Provides methods for downloading and loading the CIFAR-100 dataset
    /// in its Python tar.gz format into ArborNet tensors and label arrays.
    /// </summary>
    /// <remarks>
    /// The CIFAR-100 dataset contains 100 classes with 600 images per class (500 training, 100 test).
    /// This utility downloads the official archive from the University of Toronto and parses
    /// the binary batch files into normalized image tensors of shape (N, 3072).
    /// </remarks>
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
        /// Downloads the CIFAR-100 dataset archive from the official source.
        /// </summary>
        /// <param name="destinationPath">The directory where the dataset archive will be saved.</param>
        /// <returns>A task that represents the asynchronous download operation.</returns>
        /// <exception cref="HttpRequestException">Thrown when the HTTP request fails or returns a non-success status code.</exception>
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
        /// Loads the CIFAR-100 training and test data from the extracted batch files.
        /// </summary>
        /// <param name="extractedPath">The path to the directory containing the extracted batch files (data_batch_1 through data_batch_5 and test_batch).</param>
        /// <returns>
        /// A tuple containing:
        /// <list type="bullet">
        ///   <item><c>trainData</c>: A tensor of shape (50000, 3072) containing normalized training images.</item>
        ///   <item><c>trainLabels</c>: An array of 50000 integer training labels.</item>
        ///   <item><c>testData</c>: A tensor of shape (10000, 3072) containing normalized test images.</item>
        ///   <item><c>testLabels</c>: An array of 10000 integer test labels.</item>
        /// </list>
        /// </returns>
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
        /// Loads a single batch file containing 10,000 images and their labels from the CIFAR-100 dataset.
        /// </summary>
        /// <param name="filePath">The full path to the binary batch file to load.</param>
        /// <returns>A tuple containing a list of normalized image arrays (each of length 3072) and a list of corresponding labels.</returns>
        /// <remarks>
        /// Images are normalized by dividing each byte value by 255.0f, resulting in float values in the range [0, 1].
        /// The method uses specific byte offsets to navigate the binary format of the CIFAR batch files.
        /// </remarks>
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