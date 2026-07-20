// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Data.Datasets.SQuAD
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Net.Http;
    using System.Text.Json;
    using System.Threading.Tasks;
    /// <summary>
    /// Provides functionality to download, store, and load the SQuAD (Stanford Question Answering Dataset) v1.1.
    /// </summary>
    /// <remarks>
    /// The SQuAD dataset consists of question-answer pairs derived from Wikipedia articles.
    /// This class handles downloading the official JSON files and flattening the hierarchical
    /// structure into a consumable list of <see cref="SquadEntry"/> objects.
    /// </remarks>

    #endregion

    public class Download
    {
        /// <summary>
        /// Base URL for the official SQuAD v1.1 dataset files.
        /// </summary>
        private const string BaseUrl = "https://rajpurkar.github.io/SQuAD-explorer/dataset/";

        /// <summary>
        /// Filename of the training dataset.
        /// </summary>
        private const string TrainFile = "train-v1.1.json";

        /// <summary>
        /// Filename of the development dataset.
        /// </summary>
        private const string DevFile = "dev-v1.1.json";

        /// <summary>
        /// The directory where dataset files are stored.
        /// </summary>
        private readonly string _dataDirectory;

        /// <summary>
        /// Initializes a new instance of the <see cref="Download"/> class.
        /// </summary>
        /// <param name="dataDirectory">The directory to store the dataset files. Defaults to "data/squad".</param>
        public Download(string dataDirectory = "data/squad")
        {
            _dataDirectory = dataDirectory;
            Directory.CreateDirectory(_dataDirectory);
        }
        /// <summary>
        /// Asynchronously downloads the SQuAD training and/or development datasets if they do not already exist in the local directory.
        /// </summary>
        /// <param name="downloadTrain">Indicates whether the training dataset should be downloaded. Default is <see langword="true"/>.</param>
        /// <param name="downloadDev">Indicates whether the development dataset should be downloaded. Default is <see langword="true"/>.</param>
        /// <returns>A <see cref="Task"/> representing the asynchronous operation.</returns>
        /// <exception cref="HttpRequestException">Thrown if the HTTP request fails due to network issues, DNS resolution, or server errors.</exception>
        /// <exception cref="IOException">Thrown if an error occurs while writing the files to disk.</exception>
        /// <exception cref="UnauthorizedAccessException">Thrown if write permission is denied for the destination directory.</exception>

        public async Task DownloadDatasetAsync(bool downloadTrain = true, bool downloadDev = true)
        {
            using var httpClient = new HttpClient();
            if (downloadTrain)
            {
                string trainUrl = BaseUrl + TrainFile;
                string trainPath = Path.Combine(_dataDirectory, TrainFile);
                if (!File.Exists(trainPath))
                {
                    await DownloadFileAsync(httpClient, trainUrl, trainPath);
                }
            }

            if (downloadDev)
            {
                string devUrl = BaseUrl + DevFile;
                string devPath = Path.Combine(_dataDirectory, DevFile);
                if (!File.Exists(devPath))
                {
                    await DownloadFileAsync(httpClient, devUrl, devPath);
                }
            }
        }
        /// <summary>
        /// Asynchronously downloads a file from the specified URL and writes it to the local filesystem.
        /// </summary>
        /// <param name="client">The <see cref="HttpClient"/> instance used to send the request.</param>
        /// <param name="url">The absolute URI of the file to download.</param>
        /// <param name="path">The local file path where the downloaded contents will be saved.</param>
        /// <returns>A <see cref="Task"/> representing the asynchronous download and write operation.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="client"/>, <paramref name="url"/>, or <paramref name="path"/> is null.</exception>
        /// <exception cref="HttpRequestException">Thrown if the HTTP response does not indicate success.</exception>
        /// <exception cref="IOException">Thrown if an I/O error occurs during streaming or writing the file.</exception>

        private async Task DownloadFileAsync(HttpClient client, string url, string path)
        {
            using var response = await client.GetAsync(url);

            response.EnsureSuccessStatusCode();
            using var stream = await response.Content.ReadAsStreamAsync();
            using var fileStream = File.Create(path);
            await stream.CopyToAsync(fileStream);
        }
        /// <summary>
        /// Loads and flattens the local SQuAD training dataset file.
        /// </summary>
        /// <returns>A list of <see cref="SquadEntry"/> objects containing the flattened training data.</returns>
        /// <exception cref="FileNotFoundException">Thrown when the training dataset file does not exist locally.</exception>
        /// <exception cref="JsonException">Thrown if the dataset file contains invalid JSON.</exception>
        /// <exception cref="IOException">Thrown if an I/O error occurs while reading the file.</exception>

        public List<SquadEntry> LoadTrainDataset()
        {
            string path = Path.Combine(_dataDirectory, TrainFile);
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Train dataset not found. Please download it first.");
            }
            return LoadDataset(path);
        }
        /// <summary>
        /// Loads and flattens the local SQuAD development dataset file.
        /// </summary>
        /// <returns>A list of <see cref="SquadEntry"/> objects containing the flattened development data.</returns>
        /// <exception cref="FileNotFoundException">Thrown when the development dataset file does not exist locally.</exception>
        /// <exception cref="JsonException">Thrown if the dataset file contains invalid JSON.</exception>
        /// <exception cref="IOException">Thrown if an I/O error occurs while reading the file.</exception>

        public List<SquadEntry> LoadDevDataset()
        {
            string path = Path.Combine(_dataDirectory, DevFile);
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Dev dataset not found. Please download it first.");
            }
            return LoadDataset(path);
        }
        /// <summary>
        /// Deserializes the SQuAD JSON dataset and flattens its hierarchical structure.
        /// </summary>
        /// <param name="path">The file path of the SQuAD JSON file to load.</param>
        /// <returns>A list of flattened <see cref="SquadEntry"/> objects extracted from the hierarchical dataset.</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="path"/> is null or empty.</exception>
        /// <exception cref="FileNotFoundException">Thrown if the file specified by <paramref name="path"/> does not exist.</exception>
        /// <exception cref="JsonException">Thrown if the JSON is invalid or does not match the target schema.</exception>
        /// <exception cref="IOException">Thrown if an error occurs while accessing or reading the file.</exception>

        private List<SquadEntry> LoadDataset(string path)
        {
            string json = File.ReadAllText(path);
            var squadData = JsonSerializer.Deserialize<SquadDataset>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            var entries = new List<SquadEntry>();
            foreach (var article in squadData.Data)
            {
                foreach (var paragraph in article.Paragraphs)
                {
                    foreach (var qa in paragraph.Qas)
                    {
                        entries.Add(new SquadEntry
                        {
                            Title = article.Title,
                            Context = paragraph.Context,
                            Question = qa.Question,
                            Id = qa.Id,
                            Answers = qa.Answers,
                            IsImpossible = qa.IsImpossible
                        });
                    }
                }
            }
            return entries;
        }
    }
    /// <summary>
    /// Represents the root schema of a SQuAD dataset JSON file.
    /// </summary>

    public class SquadDataset
    {
        /// <summary>
        /// Gets or sets the schema version of the SQuAD dataset (e.g., "1.1").
        /// </summary>
        /// <value>The string representing the version.</value>
        public string Version { get; set; }
        /// <summary>
        /// Gets or sets the collection of Wikipedia articles contained within the dataset.
        /// </summary>
        /// <value>A list of <see cref="SquadArticle"/> objects.</value>

        public List<SquadArticle> Data { get; set; }
    }
    /// <summary>
    /// Represents a Wikipedia article containing multiple context paragraphs.
    /// </summary>

    public class SquadArticle
    {
        /// <summary>
        /// Gets or sets the title of the Wikipedia article.
        /// </summary>
        /// <value>The title string.</value>
        public string Title { get; set; }
        /// <summary>
        /// Gets or sets the list of paragraphs associated with this article.
        /// </summary>
        /// <value>A list of <see cref="SquadParagraph"/> objects.</value>

        public List<SquadParagraph> Paragraphs { get; set; }
    }
    /// <summary>
    /// Represents a paragraph of text from an article and its associated questions.
    /// </summary>

    public class SquadParagraph
    {
        /// <summary>
        /// Gets or sets the raw passage or context text that contains answers to the questions.
        /// </summary>
        /// <value>The text context.</value>
        public string Context { get; set; }
        /// <summary>
        /// Gets or sets the collection of question-answer structures associated with this paragraph.
        /// </summary>
        /// <value>A list of <see cref="SquadQa"/> objects.</value>

        public List<SquadQa> Qas { get; set; }
    }
    /// <summary>
    /// Represents a question, its unique identifier, its candidate answers, and its solvability.
    /// </summary>

    public class SquadQa
    {
        /// <summary>
        /// Gets or sets the unique identifier for this specific question-answer element.
        /// </summary>
        /// <value>The unique ID string.</value>
        public string Id { get; set; }
        /// <summary>
        /// Gets or sets the question text.
        /// </summary>
        /// <value>The text of the question.</value>

        public string Question { get; set; }
        /// <summary>
        /// Gets or sets the collection of acceptable answers found within the context.
        /// </summary>
        /// <value>A list of <see cref="SquadAnswer"/> objects.</value>

        public List<SquadAnswer> Answers { get; set; }
        /// <summary>
        /// Gets or sets a value indicating whether the question cannot be answered from the provided context.
        /// </summary>
        /// <value><see langword="true"/> if the question is impossible to answer; otherwise, <see langword="false"/>.</value>

        public bool IsImpossible { get; set; }
    }
    /// <summary>
    /// Represents a localized answer span within a context paragraph.
    /// </summary>

    public class SquadAnswer
    {
        /// <summary>
        /// Gets or sets the text of the answer.
        /// </summary>
        /// <value>The text of the answer segment.</value>
        public string Text { get; set; }
        /// <summary>
        /// Gets or sets the zero-based character index where the answer starts in the context paragraph.
        /// </summary>
        /// <value>The character offset index.</value>

        public int AnswerStart { get; set; }
    }
    /// <summary>
    /// Represents a flattened, denormalized SQuAD entry combining the article title, context, question, and answers.
    /// </summary>
    /// <remarks>
    /// This model simplifies consumption in training pipelines by avoiding nested traversals.
    /// </remarks>

    public class SquadEntry
    {
        /// <summary>
        /// Gets or sets the title of the Wikipedia article.
        /// </summary>
        /// <value>The title string.</value>
        public string Title { get; set; }
        /// <summary>
        /// Gets or sets the raw passage or context text that contains answers to the questions.
        /// </summary>
        /// <value>The text context.</value>

        public string Context { get; set; }
        /// <summary>
        /// Gets or sets the question text.
        /// </summary>
        /// <value>The text of the question.</value>

        public string Question { get; set; }
        /// <summary>
        /// Gets or sets the unique identifier for this specific question-answer element.
        /// </summary>
        /// <value>The unique ID string.</value>

        public string Id { get; set; }
        /// <summary>
        /// Gets or sets the collection of acceptable answers found within the context.
        /// </summary>
        /// <value>A list of <see cref="SquadAnswer"/> objects.</value>

        public List<SquadAnswer> Answers { get; set; }
        /// <summary>
        /// Gets or sets a value indicating whether the question cannot be answered from the provided context.
        /// </summary>
        /// <value><see langword="true"/> if the question is impossible to answer; otherwise, <see langword="false"/>.</value>

        public bool IsImpossible { get; set; }
    }
}