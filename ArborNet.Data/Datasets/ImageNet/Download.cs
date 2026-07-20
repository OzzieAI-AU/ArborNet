// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Data.Datasets.ImageNet
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Net.Http;
    using System.Threading.Tasks;
    using ArborNet.Core.Tensors;
    using ArborNet.Core.Interfaces;
    using SharpCompress.Archives;
    using SharpCompress.Archives.Tar;
    using SharpCompress.Common;
    using SharpCompress.Readers;
    using SixLabors.ImageSharp;
    using SixLabors.ImageSharp.PixelFormats;
    using SixLabors.ImageSharp.Processing;
    /// <summary>
    /// Provides utility methods for downloading, extracting, and preprocessing the ImageNet dataset into tensor representations.
    /// </summary>
    /// <remarks>
    /// Access to the official ImageNet download endpoints requires user registration and authentication at http://www.image-net.org/download-images.
    /// Before invoking the download methods, ensure <see cref="BaseUrl"/> is configured with valid, authenticated credentials.
    /// </remarks>

    #endregion

    public class Download
    {
        /// <summary>
        /// Reusable HTTP client for downloading dataset archives.
        /// </summary>
        private readonly HttpClient _httpClient = new HttpClient();

        /// <summary>
        /// Base URL for ImageNet dataset downloads.
        /// This value is a placeholder. After registering at image-net.org, replace with the authenticated download URLs.
        /// </summary>
        private const string BaseUrl = "http://www.image-net.org/download-images"; // Placeholder; requires registration and authentication
        /// <summary>
        /// Asynchronously downloads the standard training, validation, and test ImageNet tar files to the designated local directory.
        /// </summary>
        /// <param name="destinationPath">The local directory path where the downloaded tar files will be saved. The directory will be created if it does not exist.</param>
        /// <returns>A <see cref="Task"/> representing the asynchronous operation.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="destinationPath"/> is null or empty.</exception>
        /// <exception cref="HttpRequestException">Thrown when a network error occurs or the server returns an unsuccessful HTTP status code.</exception>

        public async Task DownloadDatasetAsync(string destinationPath)
        {
            Directory.CreateDirectory(destinationPath);
            var files = new[] { "ILSVRC2012_img_train.tar", "ILSVRC2012_img_val.tar", "ILSVRC2012_img_test_v10102019.tar" };
            foreach (var file in files)
            {
                var url = $"{BaseUrl}/{file}";
                var localPath = Path.Combine(destinationPath, file);
                await DownloadFileAsync(url, localPath);
            }
        }
        /// <summary>
        /// Downloads a specific file asynchronously from the provided URL and writes it directly to the local disk.
        /// </summary>
        /// <param name="url">The complete remote URL from which the resource will be retrieved.</param>
        /// <param name="localPath">The absolute local file path where the downloaded stream will be written.</param>
        /// <returns>A <see cref="Task"/> representing the asynchronous transfer operation.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="url"/> or <paramref name="localPath"/> is null.</exception>
        /// <exception cref="HttpRequestException">Thrown if the HTTP response indicates a non-success status code.</exception>
        /// <exception cref="IOException">Thrown if an I/O error occurs while creating or writing to the local file.</exception>

        private async Task DownloadFileAsync(string url, string localPath)
        {
            using var response = await _httpClient.GetAsync(url);

            response.EnsureSuccessStatusCode();
            using var fs = File.Create(localPath);
            await response.Content.CopyToAsync(fs);
        }
        /// <summary>
        /// Locates all tar archives within the source directory and extracts their contents to the specified destination path.
        /// </summary>
        /// <param name="sourcePath">The local directory containing the source .tar archive files.</param>
        /// <param name="extractPath">The target directory path where extracted file hierarchies will be written.</param>
        /// <exception cref="DirectoryNotFoundException">Thrown if the source directory does not exist.</exception>
        /// <exception cref="ArchiveException">Thrown if any archive is corrupted or has an invalid format.</exception>
        /// <exception cref="IOException">Thrown if disk I/O errors occur during the extraction process.</exception>

        public void ExtractDataset(string sourcePath, string extractPath)
        {
            Directory.CreateDirectory(extractPath);
            var tarFiles = Directory.GetFiles(sourcePath, "*.tar");
            foreach (var tarFile in tarFiles)
            {
                // FIXED: Use correct SharpCompress API with ReaderOptions for robustness
                using var archive = TarArchive.OpenArchive(tarFile, new ReaderOptions { LeaveStreamOpen = true });

                foreach (var entry in archive.Entries.Where(e => !e.IsDirectory))
                {
                    entry.WriteToDirectory(extractPath, new ExtractionOptions { ExtractFullPath = true, Overwrite = true });
                }
            }
        }
        /// <summary>
        /// Asynchronously loads an image, resizes it to the specified dimensions, and generates a corresponding tensor.
        /// </summary>
        /// <param name="imagePath">The absolute path to the local image file to load.</param>
        /// <param name="targetWidth">The desired width of the output image in pixels.</param>
        /// <param name="targetHeight">The desired height of the output image in pixels.</param>
        /// <returns>A task representing the asynchronous load operation, containing the initialized <see cref="ITensor"/> with a shape of [Height, Width, Channels].</returns>
        /// <exception cref="FileNotFoundException">Thrown when the file specified by <paramref name="imagePath"/> cannot be found.</exception>
        /// <exception cref="ArgumentException">Thrown when target dimensions are less than or equal to zero.</exception>

        public async Task<ITensor> LoadImageToTensorAsync(string imagePath, int targetWidth, int targetHeight)
        {
            using var image = await Image.LoadAsync<Rgb24>(imagePath);

            image.Mutate(x => x.Resize(targetWidth, targetHeight));
            var tensorData = new float[targetHeight, targetWidth, 3];
            float[] flatData = new float[targetHeight * targetWidth * 3];
            int idx = 0;
            for (int y = 0; y < targetHeight; y++)
                for (int x = 0; x < targetWidth; x++)
                    for (int c = 0; c < 3; c++)
                        flatData[idx++] = tensorData[y, x, c];

            return Tensor.FromArray(flatData, new TensorShape(targetHeight, targetWidth, 3));
        }
        /// <summary>
        /// Recursively searches the specified directory for JPEG files and loads them asynchronously into a list of tensors.
        /// </summary>
        /// <param name="directoryPath">The base directory path from which image files will be retrieved.</param>
        /// <param name="targetWidth">The target width in pixels for scaling each loaded image.</param>
        /// <param name="targetHeight">The target height in pixels for scaling each loaded image.</param>
        /// <returns>A task representing the asynchronous operation, returning a list of loaded <see cref="ITensor"/> objects.</returns>
        /// <exception cref="DirectoryNotFoundException">Thrown if the path provided in <paramref name="directoryPath"/> does not exist.</exception>
        /// <exception cref="UnauthorizedAccessException">Thrown if the application lacks permissions to access the directory or its contents.</exception>

        public async Task<List<ITensor>> LoadAllImagesAsync(string directoryPath, int targetWidth, int targetHeight)
        {
            var images = new List<ITensor>();
            var files = Directory.GetFiles(directoryPath, "*.JPEG", SearchOption.AllDirectories);
            foreach (var file in files)
            {
                var tensor = await LoadImageToTensorAsync(file, targetWidth, targetHeight);
                images.Add(tensor);
            }
            return images;
        }
    }
}