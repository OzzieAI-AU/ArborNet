using System;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Threading.Tasks;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Data.Datasets.MNIST
{
    /// <summary>
    /// Provides functionality to download and load the MNIST dataset.
    /// MNIST consists of handwritten digit images (28x28 pixels) and corresponding labels (0-9).
    /// The dataset is split into training (60,000 samples) and test (10,000 samples) sets.
    /// </summary>
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
        /// Downloads the MNIST dataset if not already present, extracts it, and loads it into tensors and arrays.
        /// Returns training images as a tensor of shape [60000, 784] (flattened), training labels as int array,
        /// test images as tensor of shape [10000, 784], and test labels as int array.
        /// Images are normalized to [0, 1] as floats.
        /// </summary>
        /// <param name="dataDir">Directory to store the dataset files. Defaults to "data/MNIST".</param>
        /// <returns>A tuple containing train images, train labels, test images, and test labels.</returns>
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
        /// Downloads a gzipped file from the URL and extracts it to the specified path if not already extracted.
        /// </summary>
        /// <param name="url">The complete URL of the gzipped file to download.</param>
        /// <param name="gzPath">The local filesystem path where the downloaded .gz file is saved.</param>
        /// <param name="extractedPath">The local filesystem path where the decompressed file is saved.</param>
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
        /// Loads images from the IDX3 file format into a tensor.
        /// Returns a tensor of shape [numImages, 784] with pixel values normalized to [0, 1].
        /// </summary>
        /// <param name="path">The path to the IDX3-ubyte image file.</param>
        /// <returns>A <see cref="ITensor"/> of shape [numImages, 784] containing normalized pixel values in the range [0, 1].</returns>
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
        /// Loads labels from the IDX1 file format into an int array.
        /// </summary>
        /// <param name="path">The path to the IDX1-ubyte label file.</param>
        /// <returns>An array of integer labels corresponding to the images.</returns>
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
        /// Extension method to read a 32-bit integer in big-endian format.
        /// </summary>
        /// <param name="br">The <see cref="BinaryReader"/> to read the bytes from.</param>
        /// <returns>The 32-bit integer value read using big-endian byte order.</returns>
        /// <remarks>
        /// MNIST files store multi-byte integers in big-endian format. This method ensures correct
        /// interpretation regardless of the host system's endianness.
        /// </remarks>
        private static int ReadInt32BigEndian(this BinaryReader br)
        {
            byte[] bytes = br.ReadBytes(4);
            Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
    }
}