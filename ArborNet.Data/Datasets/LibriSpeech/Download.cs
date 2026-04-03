using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;

namespace ArborNet.Data.Datasets.LibriSpeech
{
    /// <summary>
    /// Provides functionality to download, extract, and load the LibriSpeech corpus.
    /// </summary>
    /// <remarks>
    /// LibriSpeech is a public domain ASR corpus derived from LibriVox audiobooks.
    /// This class handles downloading subsets from OpenSLR, extracting the tar.gz archives,
    /// and parsing the transcripts into usable (audio, transcript) pairs.
    /// </remarks>
    public class Download
    {
        /// <summary>
        /// The HTTP client used for downloading dataset archives.
        /// </summary>
        private readonly HttpClient _httpClient;

        /// <summary>
        /// Base URL for LibriSpeech resources on the OpenSLR website.
        /// </summary>
        private const string BaseUrl = "https://www.openslr.org/resources/12";

        /// <summary>
        /// Initializes a new instance of the <see cref="Download"/> class.
        /// </summary>
        /// <remarks>
        /// Creates a new <see cref="HttpClient"/> instance internally.
        /// </remarks>
        public Download()
        {
            _httpClient = new HttpClient();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Download"/> class with a provided HttpClient.
        /// </summary>
        /// <param name="httpClient">The <see cref="HttpClient"/> instance to use for downloads.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="httpClient"/> is null.</exception>
        public Download(HttpClient httpClient)
        {
            _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
        }

        /// <summary>
        /// Downloads and extracts the specified LibriSpeech subset.
        /// </summary>
        /// <param name="subset">The subset to download (e.g., "train-clean-100", "dev-clean").</param>
        /// <param name="destinationPath">The path where to extract the dataset.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task DownloadAndExtractAsync(string subset, string destinationPath)
        {
            if (string.IsNullOrWhiteSpace(subset))
                throw new ArgumentException("Subset cannot be null or empty.", nameof(subset));
            if (string.IsNullOrWhiteSpace(destinationPath))
                throw new ArgumentException("Destination path cannot be null or empty.", nameof(destinationPath));

            string fileName = $"{subset}.tar.gz";
            string downloadUrl = $"{BaseUrl}/{fileName}";
            string localFilePath = Path.Combine(Path.GetTempPath(), fileName);

            try
            {
                // Download the file
                using (var response = await _httpClient.GetAsync(downloadUrl, HttpCompletionOption.ResponseHeadersRead))
                {
                    response.EnsureSuccessStatusCode();
                    using (var fileStream = new FileStream(localFilePath, FileMode.Create, FileAccess.Write, FileShare.None))
                    {
                        await response.Content.CopyToAsync(fileStream);
                    }
                }

                // Extract the tar.gz file
                ExtractTarGz(localFilePath, destinationPath);
            }
            finally
            {
                // Clean up the downloaded file
                if (File.Exists(localFilePath))
                    File.Delete(localFilePath);
            }
        }

        /// <summary>
        /// Downloads and extracts multiple LibriSpeech subsets.
        /// </summary>
        /// <param name="subsets">The list of subsets to download.</param>
        /// <param name="destinationPath">The path where to extract the datasets.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task DownloadAndExtractMultipleAsync(IEnumerable<string> subsets, string destinationPath)
        {
            if (subsets == null)
                throw new ArgumentNullException(nameof(subsets));
            if (string.IsNullOrWhiteSpace(destinationPath))
                throw new ArgumentException("Destination path cannot be null or empty.", nameof(destinationPath));

            var tasks = subsets.Select(subset => DownloadAndExtractAsync(subset, destinationPath)).ToArray();
            await Task.WhenAll(tasks);
        }

        /// <summary>
        /// Extracts a .tar.gz archive to the specified destination directory.
        /// </summary>
        /// <param name="tarGzFilePath">The full path to the downloaded .tar.gz file.</param>
        /// <param name="destinationPath">The directory where the archive should be extracted.</param>
        /// <exception cref="FileNotFoundException">Thrown when the specified tar.gz file does not exist.</exception>
        private void ExtractTarGz(string tarGzFilePath, string destinationPath)
        {
            if (!File.Exists(tarGzFilePath))
                throw new FileNotFoundException("The tar.gz file does not exist.", tarGzFilePath);

            Directory.CreateDirectory(destinationPath);

            // Decompress gz
            string tarFilePath = Path.Combine(Path.GetTempPath(), Path.GetFileNameWithoutExtension(tarGzFilePath));
            using (var gzStream = new GZipStream(File.OpenRead(tarGzFilePath), CompressionMode.Decompress))
            using (var tarStream = File.Create(tarFilePath))
            {
                gzStream.CopyTo(tarStream);
            }

            try
            {
                // Extract tar (simplified, as .NET doesn't have built-in tar extraction)
                // In a real implementation, you might need a library like SharpZipLib or similar
                // For now, assume the tar is extracted manually or use an external tool
                // Placeholder: Move the tar file to destination (not accurate)
                // Actually, since .NET Core 3.0 has System.Formats.Tar, but to keep it simple, assume extraction is done
                // For completeness, if using .NET 8.0, we can use System.Formats.Tar
                ExtractTar(tarFilePath, destinationPath);
            }
            finally
            {
                if (File.Exists(tarFilePath))
                    File.Delete(tarFilePath);
            }
        }

        /// <summary>
        /// Extracts the contents of a TAR archive using <see cref="System.Formats.Tar.TarReader"/>.
        /// </summary>
        /// <param name="tarFilePath">The path to the extracted .tar file.</param>
        /// <param name="destinationPath">The root directory to extract the archive contents into.</param>
        private void ExtractTar(string tarFilePath, string destinationPath)
        {
            // Using System.Formats.Tar for extraction (available in .NET 8.0)
            using (var tarStream = File.OpenRead(tarFilePath))
            {
                var reader = new System.Formats.Tar.TarReader(tarStream);
                System.Formats.Tar.TarEntry entry;
                while ((entry = reader.GetNextEntry()) != null)
                {
                    if (entry.EntryType == System.Formats.Tar.TarEntryType.RegularFile)
                    {
                        string entryPath = Path.Combine(destinationPath, entry.Name);
                        Directory.CreateDirectory(Path.GetDirectoryName(entryPath));
                        using (var entryStream = entry.DataStream)
                        using (var fileStream = File.Create(entryPath))
                        {
                            entryStream.CopyTo(fileStream);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Loads the dataset from the extracted path.
        /// Returns a list of (audioFilePath, transcript) tuples.
        /// </summary>
        /// <param name="extractedPath">The path where the dataset was extracted.</param>
        /// <param name="subset">The subset name.</param>
        /// <returns>A list of data samples.</returns>
        public List<(string AudioFilePath, string Transcript)> LoadDataset(string extractedPath, string subset)
        {
            if (string.IsNullOrWhiteSpace(extractedPath))
                throw new ArgumentException("Extracted path cannot be null or empty.", nameof(extractedPath));
            if (string.IsNullOrWhiteSpace(subset))
                throw new ArgumentException("Subset cannot be null or empty.", nameof(subset));

            var samples = new List<(string, string)>();
            string subsetPath = Path.Combine(extractedPath, "LibriSpeech", subset);

            if (!Directory.Exists(subsetPath))
                throw new DirectoryNotFoundException($"Subset directory not found: {subsetPath}");

            foreach (var speakerDir in Directory.EnumerateDirectories(subsetPath))
            {
                foreach (var chapterDir in Directory.EnumerateDirectories(speakerDir))
                {
                    string transcriptFile = Path.Combine(chapterDir, $"{Path.GetFileName(chapterDir)}.trans.txt");
                    if (File.Exists(transcriptFile))
                    {
                        var transcripts = File.ReadAllLines(transcriptFile)
                            .Select(line => line.Split(' ', 2))
                            .Where(parts => parts.Length == 2)
                            .ToDictionary(parts => parts[0], parts => parts[1]);

                        foreach (var audioFile in Directory.EnumerateFiles(chapterDir, "*.flac"))
                        {
                            string audioId = Path.GetFileNameWithoutExtension(audioFile);
                            if (transcripts.TryGetValue(audioId, out string transcript))
                            {
                                samples.Add((audioFile, transcript));
                            }
                        }
                    }
                }
            }

            return samples;
        }

        /// <summary>
        /// Disposes the HttpClient.
        /// </summary>
        public void Dispose()
        {
            _httpClient?.Dispose();
        }
    }
}