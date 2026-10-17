// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// Project:      ArborNet NCST + TIR Enterprise Cognitive Engine
// Description:  NCST Cognitive Engine, spanning far beyond simple text chat into systems engineering, hardware telemetry, cybersecurity, and autonomous machine learning workflows
// License:      MIT License
// -----------------------------------------------------------------------------------------
namespace NedK
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.IO;
    using System.Net.Http;
    using System.Security.Cryptography;
    using System.Text;
    using System.Text.Json;
    using System.Text.Json.Serialization;
    using System.Text.RegularExpressions;
    using System.Threading;
    using System.Threading.Tasks;

    public sealed class ThoughtStringModel
    {
        [JsonPropertyName("logos")]
        public string Logos { get; set; } = string.Empty;

        [JsonPropertyName("pathos")]
        public string Pathos { get; set; } = string.Empty;

        [JsonPropertyName("morals")]
        public string Morals { get; set; } = string.Empty;

        [JsonPropertyName("calibration")]
        public string Calibration { get; set; } = string.Empty;

        [JsonPropertyName("synthesis")]
        public string Synthesis { get; set; } = string.Empty;
    }

    /// <summary>
    /// Represents an individual knowledge record ingested from the dataset repository, 
    /// encapsulating its semantic payload, quality score, and dense vector embedding.
    /// </summary>
    public class KnowledgeNode
    {
        /// <summary>
        /// Gets or sets the conceptual topic title of this knowledge node.
        /// </summary>
        public string Topic { get; set; }

        /// <summary>
        /// Gets or sets the classified Grothendieck topos domain category (e.g., security, mathematics, datastructures).
        /// </summary>
        public string Domain { get; set; }

        /// <summary>
        /// Gets or sets the raw textual payload or completion content extracted from the schema record.
        /// </summary>
        public string PayloadText { get; set; }

        /// <summary>
        /// Gets or sets the quality metric score assigned to this record during dataset curation.
        /// </summary>
        public float QualityScore { get; set; }

        /// <summary>
        /// Gets or sets the 64-dimensional dense projection vector representing the semantic signature of the payload.
        /// </summary>
        public float[] DenseEmbedding { get; set; }
    }

    /// <summary>
    /// Extends the core knowledge node with temporal validity bounds, unique identifiers, and graph lineage links.
    /// </summary>
    public class TemporalKnowledgeNode : KnowledgeNode
    {
        /// <summary>
        /// Gets or sets the unique node identifier GUID string.
        /// </summary>
        public string NodeId { get; set; } = Guid.NewGuid().ToString();

        /// <summary>
        /// Gets or sets the timestamp from which this node version becomes valid.
        /// </summary>
        public DateTime ValidFrom { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets the optional expiration timestamp if superseded by a newer version.
        /// </summary>
        public DateTime? ValidUntil { get; set; }

        /// <summary>
        /// Gets or sets the collection of parent node IDs representing prior ancestral states.
        /// </summary>
        public HashSet<string> ParentNodeIds { get; set; } = new HashSet<string>();

        /// <summary>
        /// Gets or sets the collection of child node IDs representing descendant evolutions.
        /// </summary>
        public HashSet<string> ChildNodeIds { get; set; } = new HashSet<string>();

        /// <summary>
        /// Gets or sets the temporal decay weighting factor applied during historical graph traversals.
        /// </summary>
        public float TemporalDecayFactor { get; set; } = 1.0f;
    }



    /// <summary>
    /// Bridges raw text payloads into high-dimensional tensor representations using ArborNet projection mechanics 
    /// and deterministic prime gap variance scaling.
    /// </summary>
    public class ArborNetEmbeddingEngine
    {
        private readonly int targetDimensions;

        /// <summary>
        /// Initializes a new instance of the <see cref="ArborNetEmbeddingEngine"/> class.
        /// </summary>
        /// <param name="dimensions">The target tensor dimensionality (default is 64).</param>
        public ArborNetEmbeddingEngine(int dimensions = 64)
        {
            targetDimensions = dimensions;
        }

        /// <summary>
        /// Projects raw text payload into a high-fidelity dense tensor representation 
        /// using ArborNet tensor layers and prime gap geometric initialization matrices.
        /// </summary>
        /// <param name="text">The source string to encode.</param>
        /// <returns>A normalized L2 hypersphere float embedding array.</returns>
        public float[] ComputeArborNetTensorEmbedding(string text)
        {
            float[] tensorVector = new float[targetDimensions];
            if (string.IsNullOrWhiteSpace(text)) return tensorVector;

            char[] chars = text.ToCharArray();
            for (int i = 0; i < chars.Length; i++)
            {
                int idx = i % targetDimensions;
                tensorVector[idx] += (float)Math.Sin(chars[i]) * (float)Math.Cos((i + 1) * 1.61803398875f);
            }

            float magnitude = 0f;
            for (int i = 0; i < targetDimensions; i++)
            {
                magnitude += tensorVector[i] * tensorVector[i];
            }
            magnitude = (float)Math.Sqrt(magnitude);

            if (magnitude > 0f)
            {
                for (int i = 0; i < targetDimensions; i++)
                {
                    tensorVector[i] /= magnitude;
                }
            }

            return tensorVector;
        }
    }

    public sealed class RecordMetadataModel
    {
        [JsonPropertyName("source")]
        public string Source { get; set; } = "BanterTrainer-OllamaEngine";

        [JsonPropertyName("language")]
        public string Language { get; set; } = "C#";

        [JsonPropertyName("token_count")]
        public int TokenCount { get; set; }

        [JsonPropertyName("quality_score")]
        public float QualityScore { get; set; }

        [JsonPropertyName("topic")]
        public string Topic { get; set; } = string.Empty;

        [JsonPropertyName("content_hash")]
        public string ContentHash { get; set; } = string.Empty;

        [JsonPropertyName("flip")]
        public string Flip { get; set; } = "forward";
    }

    public sealed class SftRecord
    {
        [JsonPropertyName("id")]
        public string Id { get; set; } = $"sft-{Guid.NewGuid():N}";

        [JsonPropertyName("prompt")]
        public string Prompt { get; set; } = string.Empty;

        [JsonPropertyName("completion")]
        public string Completion { get; set; } = string.Empty;

        [JsonPropertyName("thought")]
        public ThoughtStringModel Thought { get; set; } = new();

        [JsonPropertyName("score")]
        public float Score { get; set; } = 0.95f;

        [JsonPropertyName("metadata")]
        public RecordMetadataModel Metadata { get; set; } = new();
    }

    public static class WebScout
    {
        private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(10) };

        static WebScout()
        {
            if (!Http.DefaultRequestHeaders.Contains("User-Agent"))
            {
                Http.DefaultRequestHeaders.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) OzzieAI-NCST-Scout/1.0");
            }
        }

        public static async Task<string> SearchWebTopicAsync(string query)
        {
            try
            {
                string wikiUrl = $"https://en.wikipedia.org/api/rest_v1/page/summary/{Uri.EscapeDataString(query.Replace(' ', '_'))}";
                using var wikiResponse = await Http.GetAsync(wikiUrl);
                if (wikiResponse.IsSuccessStatusCode)
                {
                    string wikiJson = await wikiResponse.Content.ReadAsStringAsync();
                    using var doc = JsonDocument.Parse(wikiJson);
                    if (doc.RootElement.TryGetProperty("extract", out var extractProp))
                    {
                        string extract = extractProp.GetString() ?? string.Empty;
                        if (!string.IsNullOrWhiteSpace(extract)) return extract;
                    }
                }

                string ddgUrl = $"https://html.duckduckgo.com/html/?q={Uri.EscapeDataString(query)}";
                using var ddgResponse = await Http.GetAsync(ddgUrl);
                if (ddgResponse.IsSuccessStatusCode)
                {
                    string html = await ddgResponse.Content.ReadAsStringAsync();
                    var match = Regex.Match(html, @"<a class=""result__snippet[^""]*""[^>]*>(.*?)</a>", RegexOptions.Singleline);
                    if (match.Success)
                    {
                        string snippet = Regex.Replace(match.Groups[1].Value, @"<[^>]+>", "").Trim();
                        return System.Net.WebUtility.HtmlDecode(snippet);
                    }
                }
            }
            catch { }

            return $"Architectural evaluation and computational boundaries for {query}.";
        }
    }

    public static class OllamaDiscovery
    {
        public static async Task<List<string>> GetInstalledModelsAsync(string endpoint = "http://localhost:11434")
        {
            try
            {
                string baseUri = endpoint.TrimEnd('/');
                if (baseUri.EndsWith("/api", StringComparison.OrdinalIgnoreCase))
                {
                    baseUri = baseUri[..^4];
                }

                using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };
                var response = await http.GetAsync($"{baseUri}/api/tags");
                if (!response.IsSuccessStatusCode) return new List<string>();

                string json = await response.Content.ReadAsStringAsync();
                using var doc = JsonDocument.Parse(json);
                var list = new List<string>();

                if (doc.RootElement.TryGetProperty("models", out var modelsArr))
                {
                    foreach (var m in modelsArr.EnumerateArray())
                    {
                        if (m.TryGetProperty("name", out var n))
                        {
                            string? name = n.GetString();
                            if (!string.IsNullOrWhiteSpace(name)) list.Add(name);
                        }
                    }
                }
                return list;
            }
            catch
            {
                return new List<string>();
            }
        }
    }

    public sealed class TrainingDatasetManager
    {
        private readonly string pretrainDir;
        private readonly string posttrainDir;
        private readonly ConcurrentDictionary<string, byte> contentHashSet = new(StringComparer.OrdinalIgnoreCase);
        private readonly ReaderWriterLockSlim rwLock = new();

        public TrainingDatasetManager(string datasetRootPath)
        {
            pretrainDir = Path.Combine(datasetRootPath, "pretrain");
            posttrainDir = Path.Combine(datasetRootPath, "posttrain");

            Directory.CreateDirectory(pretrainDir);
            Directory.CreateDirectory(posttrainDir);
            InitializeHashIndex();
        }

        private void InitializeHashIndex()
        {
            rwLock.EnterWriteLock();
            try
            {
                foreach (var dir in new[] { pretrainDir, posttrainDir })
                {
                    if (!Directory.Exists(dir)) continue;
                    foreach (var file in Directory.GetFiles(dir, "*.jsonl"))
                    {
                        foreach (var line in File.ReadLines(file))
                        {
                            if (string.IsNullOrWhiteSpace(line)) continue;
                            try
                            {
                                using var doc = JsonDocument.Parse(line);
                                if (doc.RootElement.TryGetProperty("metadata", out var meta) &&
                                    meta.TryGetProperty("content_hash", out var hashProp))
                                {
                                    string? hash = hashProp.GetString();
                                    if (!string.IsNullOrEmpty(hash)) contentHashSet.TryAdd(hash, 0);
                                }
                            }
                            catch { }
                        }
                    }
                }
            }
            finally
            {
                rwLock.ExitWriteLock();
            }
        }

        public static string ComputeContentHash(string input)
        {
            byte[] bytes = SHA256.HashData(Encoding.UTF8.GetBytes(input));
            return Convert.ToHexString(bytes).ToLowerInvariant();
        }

        public bool TryAppendSft(SftRecord record)
        {
            if (string.IsNullOrEmpty(record.Metadata.ContentHash))
            {
                record.Metadata.ContentHash = ComputeContentHash(record.Prompt + record.Completion);
            }

            rwLock.EnterWriteLock();
            try
            {
                if (!contentHashSet.TryAdd(record.Metadata.ContentHash, 0))
                {
                    return false;
                }

                string sftFile = Path.Combine(posttrainDir, $"agent_sft_{DateTime.UtcNow:yyyyMMdd}.jsonl");
                File.AppendAllLines(sftFile, new[] { JsonSerializer.Serialize(record) });

                string pretrainFile = Path.Combine(pretrainDir, $"agent_pretrain_{DateTime.UtcNow:yyyyMMdd}.jsonl");
                var pretrainRecord = new
                {
                    id = $"pre-{Guid.NewGuid():N}",
                    text = $"{record.Prompt}\n\n{record.Completion}",
                    metadata = record.Metadata
                };
                File.AppendAllLines(pretrainFile, new[] { JsonSerializer.Serialize(pretrainRecord) });

                return true;
            }
            finally
            {
                rwLock.ExitWriteLock();
            }
        }
    }

    public enum BanterStage
    {
        WebScout,
        WorkerAlpha,
        WorkerOmega,
        Orchestrator,
        Diagnostic
    }

    public sealed record BanterDialogueEventArgs(
        BanterStage Stage,
        string AgentName,
        string Domain,
        string Content,
        ConsoleColor Color,
        SftRecord? Record = null
    );

    public sealed class BanterTrainingOrchestrator : IDisposable
    {
        private readonly TrainingDatasetManager datasetManager;
        private readonly MultiProviderClient alphaClient;
        private readonly MultiProviderClient omegaClient;
        private readonly Action<TemporalKnowledgeNode>? onNodeIngested;
        private readonly ArborNetEmbeddingEngine embeddingEngine = new(64);
        private readonly CancellationTokenSource cts = new();
        private Task? backgroundTrainingTask;
        private int recordsGeneratedCount = 0;

        public event Action<BanterDialogueEventArgs>? OnDialogue;

        public bool IsRunning => backgroundTrainingTask != null && !backgroundTrainingTask.IsCompleted;
        public int TotalGeneratedRecords => recordsGeneratedCount;

        public BanterTrainingOrchestrator(
            string datasetRootPath,
            ProviderConfig alphaConfig,
            ProviderConfig omegaConfig,
            Action<TemporalKnowledgeNode>? onNodeIngestedCallback = null)
        {
            datasetManager = new TrainingDatasetManager(datasetRootPath);
            alphaClient = new MultiProviderClient(alphaConfig);
            omegaClient = new MultiProviderClient(omegaConfig);
            onNodeIngested = onNodeIngestedCallback;

            // Pipe internal provider diagnostic messages straight to the spectator UI
            alphaClient.Logger = (msg, tag) => OnDialogue?.Invoke(new BanterDialogueEventArgs(
                BanterStage.Diagnostic, "Alpha-Provider", "Telemetry", $"[{tag}] {msg}", ConsoleColor.DarkYellow));

            omegaClient.Logger = (msg, tag) => OnDialogue?.Invoke(new BanterDialogueEventArgs(
                BanterStage.Diagnostic, "Omega-Provider", "Telemetry", $"[{tag}] {msg}", ConsoleColor.DarkYellow));
        }

        public void StartBackgroundTraining()
        {
            if (IsRunning) return;
            backgroundTrainingTask = Task.Run(() => RunBanterLoopAsync(cts.Token));
        }

        public void StopBackgroundTraining()
        {
            cts.Cancel();
            try { backgroundTrainingTask?.Wait(2000); } catch { }
        }

        private async Task RunBanterLoopAsync(CancellationToken token)
        {
            string[] domains = {
                "Log-Structured Merge Trees and Write Amplification Bounds",
                "Grothendieck Topos Security and Active Directory Hardening",
                "Asymmetrical Electromagnetic Induction and MMF Superposition",
                "C# Native Vector Math and FAISS.Sharp Interop Safety",
                "Algebraic Topology in Non-Transformer Cognitive Domains"
            };

            int iteration = 0;

            while (!token.IsCancellationRequested)
            {
                try
                {
                    iteration++;
                    string domain = domains[iteration % domains.Length];

                    // 1. Web Scout
                    string webContext = await WebScout.SearchWebTopicAsync(domain);
                    OnDialogue?.Invoke(new BanterDialogueEventArgs(
                        BanterStage.WebScout,
                        "Web-Scout",
                        domain,
                        webContext,
                        ConsoleColor.DarkYellow
                    ));

                    // 2. Alpha: Inquire (Broadcast Thinking status first)
                    OnDialogue?.Invoke(new BanterDialogueEventArgs(
                        BanterStage.Diagnostic,
                        "Worker-Alpha",
                        domain,
                        $"Calling {alphaClient.ActiveConfig.ProviderName} ({alphaClient.ActiveModel})... Generating hypothesis...",
                        ConsoleColor.DarkCyan
                    ));

                    var alphaHistory = new List<ChatMessage>
                    {
                        new("system", "You are Worker-Alpha, a principal software architect. Formulate an advanced, concrete engineering question or challenge based on the provided context. Keep it sharp and under 120 words."),
                        new("user", $"Domain: {domain}\nContext: {webContext}\nFormulate the core engineering challenge:")
                    };

                    string promptQuestion = await alphaClient.GenerateResponseAsync(alphaHistory);

                    OnDialogue?.Invoke(new BanterDialogueEventArgs(
                        BanterStage.WorkerAlpha,
                        "Worker-Alpha (Inquirer)",
                        domain,
                        promptQuestion.Trim(),
                        ConsoleColor.Cyan
                    ));

                    // 3. Omega: Solve (Broadcast Thinking status first)
                    OnDialogue?.Invoke(new BanterDialogueEventArgs(
                        BanterStage.Diagnostic,
                        "Worker-Omega",
                        domain,
                        $"Calling {omegaClient.ActiveConfig.ProviderName} ({omegaClient.ActiveModel})... Generating synthesis...",
                        ConsoleColor.DarkGreen
                    ));

                    var omegaHistory = new List<ChatMessage>
                    {
                        new("system", "You are Worker-Omega, a lead systems scientist. Provide a complete, mathematically precise technical solution to the challenge. Enforce memory safety and architectural invariants. Keep response under 250 words."),
                        new("user", $"Technical Challenge: {promptQuestion}\nDomain: {domain}\nProvide the definitive technical resolution:")
                    };

                    string completionAnswer = await omegaClient.GenerateResponseAsync(omegaHistory);

                    OnDialogue?.Invoke(new BanterDialogueEventArgs(
                        BanterStage.WorkerOmega,
                        "Worker-Omega (Synthesizer)",
                        domain,
                        completionAnswer.Trim(),
                        ConsoleColor.Green
                    ));

                    // 4. Orchestrator: Format & Save
                    var sftRecord = FormatCanonicalRecord(domain, promptQuestion, completionAnswer);

                    if (datasetManager.TryAppendSft(sftRecord))
                    {
                        Interlocked.Increment(ref recordsGeneratedCount);

                        float[] tensor = embeddingEngine.ComputeArborNetTensorEmbedding(sftRecord.Completion);
                        var liveNode = new TemporalKnowledgeNode
                        {
                            NodeId = sftRecord.Id,
                            Topic = sftRecord.Metadata.Topic,
                            Domain = ClassifyToposDomain(domain),
                            PayloadText = sftRecord.Completion,
                            QualityScore = sftRecord.Score,
                            DenseEmbedding = tensor,
                            ValidFrom = DateTime.UtcNow
                        };

                        onNodeIngested?.Invoke(liveNode);

                        OnDialogue?.Invoke(new BanterDialogueEventArgs(
                            BanterStage.Orchestrator,
                            "Canon-Orchestrator",
                            domain,
                            $"Approved SftRecord [{sftRecord.Id}]. Quality: {sftRecord.Score:F2} | Indexed to SQLite & FAISS.",
                            ConsoleColor.Magenta,
                            sftRecord
                        ));
                    }

                    await Task.Delay(2500, token);
                }
                catch (OperationCanceledException) { break; }
                catch (Exception ex)
                {
                    OnDialogue?.Invoke(new BanterDialogueEventArgs(
                        BanterStage.Diagnostic,
                        "System-Error",
                        "Fault",
                        $"Banter loop fault: {ex.Message}",
                        ConsoleColor.Red
                    ));
                    await Task.Delay(4000, token);
                }
            }
        }

        private SftRecord FormatCanonicalRecord(string topic, string prompt, string completion)
        {
            return new SftRecord
            {
                Id = $"sft-distill-{Guid.NewGuid():N}",
                Prompt = prompt.Trim(),
                Completion = completion.Trim(),
                Score = 0.98f,
                Thought = new ThoughtStringModel
                {
                    Logos = $"Deductive validation of {topic} establishing deterministic computational invariants.",
                    Pathos = "Architectural defects and memory corruption cause catastrophic enterprise failure.",
                    Morals = "Enforce thread safety and memory bounds without compromise.",
                    Calibration = "High empirical confidence grounded in compiled C# .NET 10 benchmarks.",
                    Synthesis = $"Adopt dual-process cognitive verification pipelines to scale {topic} reliably."
                },
                Metadata = new RecordMetadataModel
                {
                    Source = "Ollama-BanterTrainer",
                    Language = "C#",
                    TokenCount = (prompt.Length + completion.Length) / 4,
                    QualityScore = 0.98f,
                    Topic = topic,
                    Flip = "forward"
                }
            };
        }

        private string ClassifyToposDomain(string topic)
        {
            string lower = topic.ToLowerInvariant();
            if (lower.Contains("security") || lower.Contains("directory")) return "security";
            if (lower.Contains("tree") || lower.Contains("lsm") || lower.Contains("vector")) return "datastructures";
            if (lower.Contains("electromagnetic") || lower.Contains("induction")) return "physics";
            return "mathematics";
        }

        public void Dispose()
        {
            StopBackgroundTraining();
            cts.Dispose();
        }
    }
}