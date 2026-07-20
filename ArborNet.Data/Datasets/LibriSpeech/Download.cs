// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Data.Datasets.LibriSpeech
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.IO.Compression;
    using System.Linq;
    using System.Net.Http;
    using System.Threading.Tasks;
    /// <summary>
    /// Provides comprehensive functionality to download, extract, and load the LibriSpeech corpus.
    /// </summary>
    /// <remarks>
    /// LibriSpeech is a public domain Automatic Speech Recognition (ASR) corpus derived from LibriVox audiobooks.
    /// This class handles downloading requested subsets (e.g., "train-clean-100", "dev-clean") directly from OpenSLR,
    /// extracting the downloaded tar.gz archives using GZip compression and TAR reader streams, and parsing the 
    /// associated transcript files to construct structured (audio file path, transcript text) pairs.
    /// </remarks>

    #endregion

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
        /// Asynchronously downloads the specified LibriSpeech dataset subset archive and extracts its contents to the target directory.
        /// </summary>
        /// <param name="subset">The identifier of the dataset subset to download (e.g., "train-clean-100", "dev-clean").</param>
        /// <param name="destinationPath">The local path where the contents of the subset should be extracted.</param>
        /// <returns>A <see cref="Task"/> representing the asynchronous operation of downloading and extracting the dataset.</returns>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="subset"/> is null, empty, or consists only of white-space characters,
        /// or when <paramref name="destinationPath"/> is null, empty, or consists only of white-space characters.
        /// </exception>
        /// <exception cref="HttpRequestException">Thrown when an error occurs during the HTTP request or if the remote server returns a failure status code.</exception>
        /// <exception cref="IOException">Thrown when file system, stream write operations, or archive decompression fails.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when the underlying <see cref="HttpClient"/> has already been disposed.</exception>

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
        /// Asynchronously downloads and extracts multiple LibriSpeech dataset subsets in parallel.
        /// </summary>
        /// <param name="subsets">An enumerable collection of subset identifiers to download (e.g., "train-clean-100", "dev-clean").</param>
        /// <param name="destinationPath">The local root folder where all subset archives should be extracted.</param>
        /// <returns>A <see cref="Task"/> representing the completion of all download and extraction processes.</returns>
        /// <exception cref="ArgumentNullException">Thrown when the <paramref name="subsets"/> collection is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when <paramref name="destinationPath"/> is null, empty, or consists only of white-space characters.</exception>
        /// <exception cref="AggregateException">Thrown when one or more subset download or extraction tasks fail during parallel execution.</exception>

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
        /// Decompresses a <c>.tar.gz</c> compressed archive and extracts its tar file payload to the destination path.
        /// </summary>
        /// <remarks>
        /// This method decompresses the GZip wrapper to a temporary uncompressed <c>.tar</c> file, 
        /// invokes the <see cref="ExtractTar(string, string)"/> helper to extract its constituent entries, 
        /// and ensures the temporary file is deleted afterwards.
        /// </remarks>
        /// <param name="tarGzFilePath">The full path to the compressed <c>.tar.gz</c> file on disk.</param>
        /// <param name="destinationPath">The destination root folder where the archive's structure and files will be extracted.</param>
        /// <exception cref="FileNotFoundException">Thrown when the archive file specified in <paramref name="tarGzFilePath"/> does not exist.</exception>
        /// <exception cref="IOException">Thrown when file streaming, directory creation, or disk write operations encounter failures.</exception>

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
        /// Extracts the contents of a flat, uncompressed <c>.tar</c> archive file to the specified target directory.
        /// </summary>
        /// <remarks>
        /// This method utilizes the <see cref="System.Formats.Tar.TarReader"/> to read through the tape archive stream, 
        /// recreates the relative directory structure, and writes regular files to the disk.
        /// </remarks>
        /// <param name="tarFilePath">The local file path pointing to the uncompressed <c>.tar</c> archive file.</param>
        /// <param name="destinationPath">The destination root folder where the files should be written.</param>
        /// <exception cref="IOException">Thrown when an I/O error occurs while reading the tar file or writing entries to the file system.</exception>

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
        /// Recursively scans the extracted LibriSpeech directory structure, parsing transcripts and mapping them to their corresponding FLAC audio files.
        /// </summary>
        /// <remarks>
        /// This method assumes the standard LibriSpeech organizational hierarchy: 
        /// <c>[extractedPath]/LibriSpeech/[subset]/[speaker_id]/[chapter_id]/[speaker_id]-[chapter_id].trans.txt</c> 
        /// and corresponding <c>*.flac</c> audio files. It parses each transcription file and pairs the audio file paths with their respective transcript strings.
        /// </remarks>
        /// <param name="extractedPath">The root directory path where the LibriSpeech dataset structure is located.</param>
        /// <param name="subset">The specific dataset subset folder to parse (e.g., "train-clean-100").</param>
        /// <returns>A list of value tuples containing the full system path to the audio file and its matching transcription text.</returns>
        /// <exception cref="ArgumentException">Thrown when <paramref name="extractedPath"/> or <paramref name="subset"/> is null, empty, or consists only of white-space.</exception>
        /// <exception cref="DirectoryNotFoundException">Thrown when the resolved subset subdirectory path does not exist on disk.</exception>
        /// <exception cref="IOException">Thrown when reading files, parsing lines, or exploring directories encounters system-level errors.</exception>

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
        /// Releases all resources used by the current instance of the <see cref="Download"/> class.
        /// </summary>
        /// <remarks>
        /// This method disposes of the underlying <see cref="HttpClient"/> instance to free up network sockets and system resources.
        /// </remarks>

        public void Dispose()
        {
            _httpClient?.Dispose();
        }
    }
}