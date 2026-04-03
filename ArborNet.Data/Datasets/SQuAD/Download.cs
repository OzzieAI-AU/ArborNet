using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;

namespace ArborNet.Data.Datasets.SQuAD
{
    /// <summary>
    /// Provides functionality to download and load the SQuAD (Stanford Question Answering Dataset) v1.1.
    /// </summary>
    /// <remarks>
    /// The SQuAD dataset consists of question-answer pairs derived from Wikipedia articles.
    /// This class handles downloading the official JSON files and flattening the hierarchical
    /// structure into a consumable list of <see cref="SquadEntry"/> objects.
    /// </remarks>
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
        /// Downloads the SQuAD train and/or development datasets if they do not already exist locally.
        /// </summary>
        /// <param name="downloadTrain">Whether to download the training dataset. Default is <c>true</c>.</param>
        /// <param name="downloadDev">Whether to download the development dataset. Default is <c>true</c>.</param>
        /// <returns>A task representing the asynchronous download operation.</returns>
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
        /// Downloads a file from the specified URL and saves it to the target path.
        /// </summary>
        /// <param name="client">The HTTP client used to perform the download.</param>
        /// <param name="url">The URL of the file to download.</param>
        /// <param name="path">The local filesystem path where the file will be saved.</param>
        /// <returns>A task representing the asynchronous file download and write operation.</returns>
        private async Task DownloadFileAsync(HttpClient client, string url, string path)
        {
            using var response = await client.GetAsync(url);
            response.EnsureSuccessStatusCode();
            using var stream = await response.Content.ReadAsStreamAsync();
            using var fileStream = File.Create(path);
            await stream.CopyToAsync(fileStream);
        }

        /// <summary>
        /// Loads the SQuAD training dataset from the local filesystem.
        /// </summary>
        /// <returns>A list of <see cref="SquadEntry"/> objects containing the training data.</returns>
        /// <exception cref="FileNotFoundException">Thrown when the train dataset file does not exist.</exception>
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
        /// Loads the SQuAD development dataset from the local filesystem.
        /// </summary>
        /// <returns>A list of <see cref="SquadEntry"/> objects containing the development data.</returns>
        /// <exception cref="FileNotFoundException">Thrown when the dev dataset file does not exist.</exception>
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
        /// Loads and deserializes a SQuAD JSON file, then flattens the hierarchical structure
        /// into a list of <see cref="SquadEntry"/> objects.
        /// </summary>
        /// <param name="path">The full path to the SQuAD JSON file.</param>
        /// <returns>A flattened list of dataset entries.</returns>
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
    /// Represents the root object of a SQuAD dataset JSON file.
    /// </summary>
    public class SquadDataset
    {
        /// <summary>
        /// Gets or sets the version of the SQuAD dataset format.
        /// </summary>
        public string Version { get; set; }

        /// <summary>
        /// Gets or sets the list of articles contained in the dataset.
        /// </summary>
        public List<SquadArticle> Data { get; set; }
    }

    /// <summary>
    /// Represents a Wikipedia article in the SQuAD dataset.
    /// </summary>
    public class SquadArticle
    {
        /// <summary>
        /// Gets or sets the title of the Wikipedia article.
        /// </summary>
        public string Title { get; set; }

        /// <summary>
        /// Gets or sets the list of paragraphs within the article.
        /// </summary>
        public List<SquadParagraph> Paragraphs { get; set; }
    }

    /// <summary>
    /// Represents a paragraph from a Wikipedia article containing context for questions.
    /// </summary>
    public class SquadParagraph
    {
        /// <summary>
        /// Gets or sets the paragraph text that serves as the context for questions.
        /// </summary>
        public string Context { get; set; }

        /// <summary>
        /// Gets or sets the list of question-answer pairs associated with this paragraph.
        /// </summary>
        public List<SquadQa> Qas { get; set; }
    }

    /// <summary>
    /// Represents a single question and its associated answers in the SQuAD dataset.
    /// </summary>
    public class SquadQa
    {
        /// <summary>
        /// Gets or sets the unique identifier for this question.
        /// </summary>
        public string Id { get; set; }

        /// <summary>
        /// Gets or sets the text of the question.
        /// </summary>
        public string Question { get; set; }

        /// <summary>
        /// Gets or sets the list of acceptable answers for this question.
        /// </summary>
        public List<SquadAnswer> Answers { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether the question is impossible to answer
        /// from the provided context.
        /// </summary>
        public bool IsImpossible { get; set; }
    }

    /// <summary>
    /// Represents a single answer span in the SQuAD dataset.
    /// </summary>
    public class SquadAnswer
    {
        /// <summary>
        /// Gets or sets the text of the answer.
        /// </summary>
        public string Text { get; set; }

        /// <summary>
        /// Gets or sets the character position where the answer begins within the context paragraph.
        /// </summary>
        public int AnswerStart { get; set; }
    }

    /// <summary>
    /// Represents a flattened SQuAD entry combining article, context, question, and answer information
    /// for easier consumption in machine learning pipelines.
    /// </summary>
    public class SquadEntry
    {
        /// <summary>
        /// Gets or sets the title of the article the context was taken from.
        /// </summary>
        public string Title { get; set; }

        /// <summary>
        /// Gets or sets the context paragraph used for the question.
        /// </summary>
        public string Context { get; set; }

        /// <summary>
        /// Gets or sets the question text.
        /// </summary>
        public string Question { get; set; }

        /// <summary>
        /// Gets or sets the unique identifier for this question-answer pair.
        /// </summary>
        public string Id { get; set; }

        /// <summary>
        /// Gets or sets the list of answers for the question.
        /// </summary>
        public List<SquadAnswer> Answers { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether the question is impossible to answer
        /// from the provided context.
        /// </summary>
        public bool IsImpossible { get; set; }
    }
}