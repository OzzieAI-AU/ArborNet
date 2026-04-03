using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using ArborNet.Data;

namespace ArborNet.Data.Datasets.WikiText103
{
    /// <summary>
    /// Handles downloading and loading the WikiText-103 dataset for language modeling.
    /// WikiText-103 is a large-scale language modeling dataset extracted from Wikipedia articles.
    /// It consists of train, validation, and test splits.
    /// </summary>
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
        /// Downloads the WikiText-103 dataset if not already present locally.
        /// </summary>
        /// <param name="dataDir">The base directory to store the dataset.</param>
        /// <returns>A task representing the asynchronous download operation.</returns>
        /// <remarks>
        /// If the extracted directory already exists, the operation returns immediately without downloading.
        /// The dataset is downloaded using <see cref="HttpClient"/> and extracted using <see cref="ZipFile.ExtractToDirectory(string,string)"/>.
        /// All intermediate directories are created automatically.
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
        /// Loads the train split of the WikiText-103 dataset.
        /// </summary>
        /// <param name="dataDir">The base directory where the dataset is stored.</param>
        /// <returns>A list of strings, where each string is a line from the train file.</returns>
        /// <exception cref="FileNotFoundException">
        /// Thrown when the train file <c>wiki.train.tokens</c> does not exist in the expected location.
        /// </exception>
        /// <remarks>
        /// Reads the entire file using <see cref="File.ReadAllLines(string)"/> and converts the result to a <see cref="List{T}"/>.
        /// </remarks>
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
        /// Loads the validation split of the WikiText-103 dataset.
        /// </summary>
        /// <param name="dataDir">The base directory where the dataset is stored.</param>
        /// <returns>A list of strings, where each string is a line from the validation file.</returns>
        /// <exception cref="FileNotFoundException">
        /// Thrown when the validation file <c>wiki.valid.tokens</c> does not exist in the expected location.
        /// </exception>
        /// <remarks>
        /// Reads the entire file using <see cref="File.ReadAllLines(string)"/> and converts the result to a <see cref="List{T}"/>.
        /// </remarks>
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
        /// Loads the test split of the WikiText-103 dataset.
        /// </summary>
        /// <param name="dataDir">The base directory where the dataset is stored.</param>
        /// <returns>A list of strings, where each string is a line from the test file.</returns>
        /// <exception cref="FileNotFoundException">
        /// Thrown when the test file <c>wiki.test.tokens</c> does not exist in the expected location.
        /// </exception>
        /// <remarks>
        /// Reads the entire file using <see cref="File.ReadAllLines(string)"/> and converts the result to a <see cref="List{T}"/>.
        /// </remarks>
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
        /// Loads all splits of the WikiText-103 dataset.
        /// </summary>
        /// <param name="dataDir">The base directory where the dataset is stored.</param>
        /// <returns>A tuple containing lists for train, validation, and test data.</returns>
        /// <remarks>
        /// Convenience method that invokes <see cref="LoadTrain(string)"/>, <see cref="LoadValid(string)"/>, 
        /// and <see cref="LoadTest(string)"/> and returns the results as a named tuple.
        /// </remarks>
        /// <exception cref="FileNotFoundException">
        /// Thrown if any of the dataset split files are missing.
        /// </exception>
        public static (List<string> Train, List<string> Valid, List<string> Test) LoadAll(string dataDir = ".")
        {
            return (LoadTrain(dataDir), LoadValid(dataDir), LoadTest(dataDir));
        }
    }
}