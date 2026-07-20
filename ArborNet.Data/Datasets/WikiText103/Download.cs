// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Data.Datasets.WikiText103
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.IO.Compression;
    using System.Linq;
    using System.Net.Http;
    using System.Threading.Tasks;
    using ArborNet.Data;
    /// <summary>
    /// Provides utility methods to asynchronously download, extract, and load the WikiText-103 dataset 
    /// for language modeling tasks within the ArborNet machine learning pipeline.
    /// </summary>
    /// <remarks>
    /// WikiText-103 is a large-scale language modeling dataset extracted from high-quality, verified 
    /// Good and Featured articles on Wikipedia. It consists of training, validation, and testing splits, 
    /// retaining original casing, punctuation, and numbers.
    /// </remarks>

    #endregion

    public static class Download
    {
        /// <summary>
        /// The base URL from which the WikiText-103 dataset ZIP archive is downloaded.
        /// </summary>
        private const string BaseUrl = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip";
        /// <summary>
        /// The name of the dataset, used as the ZIP filename (without .zip) and the name of the extracted folder.
        /// </summary>
        private const string DatasetName = "wikitext-103-v1";
        /// <summary>
        /// The local directory path (relative to <paramref name="dataDir"/>) where the dataset will be stored.
        /// </summary>
        private const string LocalDirectory = "Datasets/WikiText103";
        /// <summary>
        /// Downloads the compressed WikiText-103 dataset ZIP archive asynchronously if not already present on disk, 
        /// and extracts its contents to the designated dataset directory.
        /// </summary>
        /// <param name="dataDir">The base directory where dataset storage folders are created. Defaults to the current directory (".").</param>
        /// <returns>A <see cref="Task"/> that represents the asynchronous download and extraction operation.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="dataDir"/> is <see langword="null"/>.</exception>
        /// <exception cref="HttpRequestException">Thrown when the network request to download the dataset fails.</exception>
        /// <exception cref="IOException">Thrown when file writing, directory creation, or ZIP extraction fails due to file system issues.</exception>
        /// <exception cref="UnauthorizedAccessException">Thrown when the process lacks permissions to write to the requested file paths.</exception>
        /// <remarks>
        /// This method checks for the existence of the expected extracted dataset directory before starting. 
        /// If the directory exists, it returns immediately to avoid redundant bandwidth consumption.
        /// </remarks>

        public static async Task DownloadDatasetAsync(string dataDir = ".")
        {
            string datasetDir = Path.Combine(dataDir, LocalDirectory);
            string zipPath = Path.Combine(datasetDir, $"{DatasetName}.zip");
            string extractedDir = Path.Combine(datasetDir, DatasetName);

            if (Directory.Exists(extractedDir))
            {
                Console.WriteLine("WikiText-103 dataset already downloaded and extracted.");
                return;
            }

            Directory.CreateDirectory(datasetDir);
            using (HttpClient client = new HttpClient())
            {
                Console.WriteLine("Downloading WikiText-103 dataset...");
                byte[] data = await client.GetByteArrayAsync(BaseUrl);
                await File.WriteAllBytesAsync(zipPath, data);
                Console.WriteLine("Download complete.");
            }

            Console.WriteLine("Extracting WikiText-103 dataset...");
            ZipFile.ExtractToDirectory(zipPath, datasetDir);
            Console.WriteLine("Extraction complete.");
        }
        /// <summary>
        /// Loads the training split of the WikiText-103 dataset from the local file system.
        /// </summary>
        /// <param name="dataDir">The base directory where the dataset is stored. Defaults to the current directory (".").</param>
        /// <returns>A <see cref="List{T}"/> of strings containing the raw tokens/lines of the training partition.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="dataDir"/> is <see langword="null"/>.</exception>
        /// <exception cref="FileNotFoundException">Thrown when the training file <c>wiki.train.tokens</c> cannot be found. Ensure <see cref="DownloadDatasetAsync"/> has run successfully.</exception>
        /// <exception cref="IOException">Thrown when an I/O error occurs while reading the training file.</exception>
        /// <exception cref="UnauthorizedAccessException">Thrown when the caller does not have read permissions for the file.</exception>

        public static List<string> LoadTrain(string dataDir = ".")
        {
            string filePath = Path.Combine(dataDir, LocalDirectory, DatasetName, "wiki.train.tokens");
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"Train file not found at {filePath}. Ensure the dataset is downloaded.");
            }

            return File.ReadAllLines(filePath).ToList();
        }
        /// <summary>
        /// Loads the validation split of the WikiText-103 dataset from the local file system.
        /// </summary>
        /// <param name="dataDir">The base directory where the dataset is stored. Defaults to the current directory (".").</param>
        /// <returns>A <see cref="List{T}"/> of strings containing the raw tokens/lines of the validation partition.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="dataDir"/> is <see langword="null"/>.</exception>
        /// <exception cref="FileNotFoundException">Thrown when the validation file <c>wiki.valid.tokens</c> cannot be found. Ensure <see cref="DownloadDatasetAsync"/> has run successfully.</exception>
        /// <exception cref="IOException">Thrown when an I/O error occurs while reading the validation file.</exception>
        /// <exception cref="UnauthorizedAccessException">Thrown when the caller does not have read permissions for the file.</exception>

        public static List<string> LoadValid(string dataDir = ".")
        {
            string filePath = Path.Combine(dataDir, LocalDirectory, DatasetName, "wiki.valid.tokens");
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"Validation file not found at {filePath}. Ensure the dataset is downloaded.");
            }

            return File.ReadAllLines(filePath).ToList();
        }
        /// <summary>
        /// Loads the testing split of the WikiText-103 dataset from the local file system.
        /// </summary>
        /// <param name="dataDir">The base directory where the dataset is stored. Defaults to the current directory (".").</param>
        /// <returns>A <see cref="List{T}"/> of strings containing the raw tokens/lines of the testing partition.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="dataDir"/> is <see langword="null"/>.</exception>
        /// <exception cref="FileNotFoundException">Thrown when the testing file <c>wiki.test.tokens</c> cannot be found. Ensure <see cref="DownloadDatasetAsync"/> has run successfully.</exception>
        /// <exception cref="IOException">Thrown when an I/O error occurs while reading the testing file.</exception>
        /// <exception cref="UnauthorizedAccessException">Thrown when the caller does not have read permissions for the file.</exception>

        public static List<string> LoadTest(string dataDir = ".")
        {
            string filePath = Path.Combine(dataDir, LocalDirectory, DatasetName, "wiki.test.tokens");
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"Test file not found at {filePath}. Ensure the dataset is downloaded.");
            }

            return File.ReadAllLines(filePath).ToList();
        }
        /// <summary>
        /// Loads all dataset partitions (Train, Validation, and Test) of the WikiText-103 corpus simultaneously.
        /// </summary>
        /// <param name="dataDir">The base directory where the dataset is stored. Defaults to the current directory (".").</param>
        /// <returns>
        /// A named tuple containing:
        /// <list type="bullet">
        /// <item><description><c>Train</c>: A list of strings comprising the training split.</description></item>
        /// <item><description><c>Valid</c>: A list of strings comprising the validation split.</description></item>
        /// <item><description><c>Test</c>: A list of strings comprising the testing split.</description></item>
        /// </list>
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="dataDir"/> is <see langword="null"/>.</exception>
        /// <exception cref="FileNotFoundException">Thrown when one or more of the token partition files cannot be found locally.</exception>
        /// <exception cref="IOException">Thrown when an general I/O error occurs accessing any of the files.</exception>
        /// <exception cref="UnauthorizedAccessException">Thrown when permissions prevent accessing dataset files.</exception>

        public static (List<string> Train, List<string> Valid, List<string> Test) LoadAll(string dataDir = ".")
        {
            return (LoadTrain(dataDir), LoadValid(dataDir), LoadTest(dataDir));
        }
    }
}