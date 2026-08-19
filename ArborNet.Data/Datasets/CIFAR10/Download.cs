// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Data.Datasets.CIFAR10
{

    #region Using Statements:

    using System;
    using System.IO;
    using System.Net.Http;
    using System.Security.Cryptography;
    using System.Threading.Tasks;
    /// <summary>
    /// Provides utility functionality to download the CIFAR-10 dataset archive from official sources.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The CIFAR-10 dataset consists of 60,000 32x32 color images categorized into 10 mutually exclusive classes.
    /// The dataset is divided into 50,000 training images and 10,000 test images.
    /// </para>
    /// <para>
    /// This static class manages the remote acquisition of the binary version of the dataset (<c>cifar-10-binary.tar.gz</c>)
    /// hosted by the University of Toronto. It handles local directory validation, directory creation, and streaming 
    /// network I/O to safely write the dataset to disk. Extraction and binary parsing are handled by separate components.
    /// </para>
    /// </remarks>
    /// <example>
    /// The following example demonstrates how to call the download utility:
    /// <code>
    /// string targetDirectory = "./datasets/cifar10";
    /// await Download.DownloadDatasetAsync(targetDirectory);
    /// </code>
    /// </example>

    #endregion

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
        /// Downloads the CIFAR-10 dataset archive asynchronously and saves it to the specified destination path.
        /// </summary>
        /// <param name="destinationPath">The local directory path where the dataset archive will be downloaded and saved.</param>
        /// <returns>A <see cref="Task"/> representing the asynchronous download and file write operation.</returns>
        /// <exception cref="ArgumentException">Thrown when <paramref name="destinationPath"/> is null, empty, or consists only of white-space characters.</exception>
        /// <exception cref="UnauthorizedAccessException">Thrown when the application lacks permission to create the target directory or write to the destination file.</exception>
        /// <exception cref="PathTooLongException">Thrown when the computed destination file path exceeds the system's maximum allowed path length.</exception>
        /// <exception cref="DirectoryNotFoundException">Thrown when the specified destination path is invalid or resides on an unmapped drive.</exception>
        /// <exception cref="IOException">Thrown when an I/O error occurs during directory creation, file writing, or network streaming.</exception>
        /// <exception cref="HttpRequestException">Thrown when the network request fails, returning a non-successful status code, or during client transmission issues.</exception>
        /// <remarks>
        /// <para>
        /// If the specified <paramref name="destinationPath"/> does not exist, this method will attempt to automatically create
        /// the directory structure.
        /// </para>
        /// <para>
        /// This method uses <see cref="HttpCompletionOption.ResponseHeadersRead"/> to begin streaming the payload directly to disk. 
        /// This approach optimizes memory consumption, preventing the entire archive from being loaded into the system's RAM.
        /// </para>
        /// </remarks>
        /// <example>
        /// <code>
        /// try
        /// {
        ///     await Download.DownloadDatasetAsync(@"C:\MLData\CIFAR10\");
        ///     Console.WriteLine("Download complete.");
        /// }
        /// catch (Exception ex)
        /// {
        ///     Console.WriteLine($"Download failed: {ex.Message}");
        /// }
        /// </code>
        /// </example>
        public static async Task DownloadDatasetAsync(string destinationPath)
        {

            // Known official hash for CIFAR-10 binary payload
            const string ExpectedMd5 = "c58f30108f718f92721af3b95e74349a";

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
                    // 1. Stream the download to disk
                    using (HttpResponseMessage response = await client.GetAsync(Url, HttpCompletionOption.ResponseHeadersRead))
                    {
                        response.EnsureSuccessStatusCode();

                        using (Stream contentStream = await response.Content.ReadAsStreamAsync())
                        using (FileStream fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None))
                        {
                            await contentStream.CopyToAsync(fileStream);
                        }
                    }

                    // 2. Post-Download Validation
                    string hashString;
                    using (var md5 = MD5.Create())
                    using (var stream = File.OpenRead(filePath))
                    {
                        var hashBytes = md5.ComputeHash(stream);
                        hashString = BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
                    } // Stream is disposed and file lock released here

                    // 3. Verify Checksum
                    if (hashString != ExpectedMd5)
                    {
                        File.Delete(filePath); // Purge corrupted file safely
                        throw new InvalidDataException(
                            $"Checksum verification failed. Expected MD5 '{ExpectedMd5}', got '{hashString}'. " +
                            "The downloaded dataset was corrupted and has been deleted.");
                    }

                    Console.WriteLine($"CIFAR10 dataset downloaded and verified successfully to: {filePath}");
                }
                catch (HttpRequestException ex)
                {
                    throw new HttpRequestException($"Failed to download CIFAR10 dataset: {ex.Message}", ex);
                }
            }
        }
    }
}