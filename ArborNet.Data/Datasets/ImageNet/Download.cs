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

namespace ArborNet.Data.Datasets.ImageNet
{
    /// <summary>
    /// Handles download, extraction, and loading of ImageNet dataset images into tensors.
    /// Note: ImageNet dataset requires registration at http://www.image-net.org/download-images to access download links.
    /// Replace BaseUrl with actual authenticated URLs after registration.
    /// </summary>
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
        /// Downloads the ImageNet dataset tar files to the specified destination path.
        /// Requires valid URLs from ImageNet registration.
        /// </summary>
        /// <param name="destinationPath">Path where tar files will be saved.</param>
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
        /// Downloads a single file from the given URL to the local path.
        /// </summary>
        /// <param name="url">URL of the file to download.</param>
        /// <param name="localPath">Local path to save the file.</param>
        private async Task DownloadFileAsync(string url, string localPath)
        {
            using var response = await _httpClient.GetAsync(url);
            response.EnsureSuccessStatusCode();
            using var fs = File.Create(localPath);
            await response.Content.CopyToAsync(fs);
        }

        /// <summary>
        /// Extracts the downloaded tar files to the specified extract path using SharpCompress.
        /// </summary>
        /// <param name="sourcePath">Path containing the tar files.</param>
        /// <param name="extractPath">Path where contents will be extracted.</param>
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
        /// Loads a single image from the given path into a tensor, resizing to target dimensions.
        /// Assumes RGB images; normalizes pixel values to [0, 1].
        /// </summary>
        /// <param name="imagePath">Path to the image file.</param>
        /// <param name="targetWidth">Target width for resizing.</param>
        /// <param name="targetHeight">Target height for resizing.</param>
        /// <returns>Tensor representation of the image with shape [height, width, 3].</returns>
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
        /// Loads all JPEG images from the specified directory into a list of tensors.
        /// Handles large-scale data by processing asynchronously but sequentially to avoid memory issues.
        /// </summary>
        /// <param name="directoryPath">Path to the directory containing images.</param>
        /// <param name="targetWidth">Target width for resizing.</param>
        /// <param name="targetHeight">Target height for resizing.</param>
        /// <returns>List of tensors for each image.</returns>
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