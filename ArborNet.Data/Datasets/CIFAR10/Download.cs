using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

namespace ArborNet.Data.Datasets.CIFAR10
{
    /// <summary>
    /// Provides functionality to download the CIFAR10 dataset.
    /// CIFAR10 consists of 60,000 32x32 color images in 10 classes (50,000 training, 10,000 test).
    /// Downloads the binary tar.gz file from the official source.
    /// </summary>
    /// <remarks>
    /// This is a static utility class responsible for downloading the official binary 
    /// distribution of the CIFAR-10 dataset from the University of Toronto.
    /// The dataset is provided as a single compressed tar.gz archive containing 
    /// 10 binary batch files (5 training batches and 1 test batch).
    /// This class only handles the download; extraction and parsing must be performed separately.
    /// </remarks>
    public static class Download
    {
        /// <summary>
        /// The official URL for the CIFAR-10 binary dataset archive.
        /// </summary>
        private const string Url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

        /// <summary>
        /// The filename of the downloaded CIFAR-10 binary archive.
        /// </summary>
        private const string FileName = "cifar-10-binary.tar.gz";

        /// <summary>
        /// Downloads the CIFAR10 dataset asynchronously to the specified destination path.
        /// </summary>
        /// <param name="destinationPath">The directory where the dataset file will be saved.</param>
        /// <returns>A task representing the asynchronous download operation.</returns>
        /// <exception cref="ArgumentException">Thrown if the destination path is invalid.</exception>
        /// <exception cref="HttpRequestException">Thrown if the download fails.</exception>
        /// <remarks>
        /// Creates the destination directory if it does not exist.
        /// Uses streaming to efficiently download the file without loading it entirely into memory.
        /// Writes a confirmation message to the console upon successful completion.
        /// The downloaded file is a tar.gz archive that must be extracted before use.
        /// </remarks>
        public static async Task DownloadDatasetAsync(string destinationPath)
        {
            if (string.IsNullOrWhiteSpace(destinationPath))
            {
                throw new ArgumentException("Destination path cannot be null or empty.", nameof(destinationPath));
            }

            if (!Directory.Exists(destinationPath))
            {
                Directory.CreateDirectory(destinationPath);
            }

            string filePath = Path.Combine(destinationPath, FileName);

            using (HttpClient client = new HttpClient())
            {
                try
                {
                    using (HttpResponseMessage response = await client.GetAsync(Url, HttpCompletionOption.ResponseHeadersRead))
                    {
                        response.EnsureSuccessStatusCode();

                        using (Stream contentStream = await response.Content.ReadAsStreamAsync())
                        {
                            using (FileStream fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None))
                            {
                                await contentStream.CopyToAsync(fileStream);
                            }
                        }
                    }

                    Console.WriteLine($"CIFAR10 dataset downloaded successfully to: {filePath}");
                }
                catch (HttpRequestException ex)
                {
                    throw new HttpRequestException($"Failed to download CIFAR10 dataset: {ex.Message}", ex);
                }
            }
        }
    }
}