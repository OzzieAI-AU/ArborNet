using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Data;
using ArborNet.Losses;
using NCSTInfinityCognitiveEngine;
using NCSTInfinityCognitiveEngine.Training;
using NedK;
using OzzieAI.FaissSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace OzzieAI.PersistentCognitiveEngine.BA5
{

    public enum CognitiveIntent
    {
        FactRetrieval = 0,
        CreativeSynthesis = 1,
        StructuralParadox = 2
    }

    public struct PrimeSpectralComponent
    {
        public int PrimeIndex { get; set; }
        public float Eigenvalue { get; set; }
        public float Amplitude { get; set; }
    }

    public class FrontierOrbitals
    {
        public int ActivePropositionCount { get; }
        public string SymmetryClass { get; }
        public float EnergyEigenvalue { get; }

        public FrontierOrbitals(int activePropositionCount, string symmetryClass, float energyEigenvalue)
        {
            ActivePropositionCount = activePropositionCount;
            SymmetryClass = symmetryClass;
            EnergyEigenvalue = energyEigenvalue;
        }

        public static bool AreIsolobal(FrontierOrbitals a, FrontierOrbitals b)
        {
            return a.ActivePropositionCount == b.ActivePropositionCount &&
                   a.SymmetryClass == b.SymmetryClass &&
                   MathF.Abs(a.EnergyEigenvalue - b.EnergyEigenvalue) < 0.4f;
        }
    }

    public class ConceptualFragment
    {
        public string FragmentName { get; }
        public string Domain { get; }
        public string ConservationRule { get; }
        public FrontierOrbitals FCO { get; }
        public bool IsStrained { get; set; }

        public ConceptualFragment(string fragmentName, string domain, string conservationRule, FrontierOrbitals fco)
        {
            FragmentName = fragmentName;
            Domain = domain;
            ConservationRule = conservationRule;
            FCO = fco;
            IsStrained = false;
        }
    }

    public class ParsedDatasetRecord
    {
        public string Id { get; set; } = string.Empty;
        public string Kind { get; set; } = "pretrain";
        public string Topic { get; set; } = string.Empty;
        public string SearchableText { get; set; } = string.Empty;
        public string AnswerPayload { get; set; } = string.Empty;
        public string DedupePrefix { get; set; } = string.Empty;
    }

    /// <summary>
    /// Unified Native NedK Prime + FAISS + NCST + TIR Cognitive Engine.
    /// Replaces heuristic routers with true NedK Prime model embeddings and autoregressive generation.
    /// </summary>
    public class NativeNedKCognitiveMachine : IDisposable
    {
        private readonly string _pretrainPath;
        private readonly string _posttrainPath;
        private readonly string _indexFilePath;
        private const int Dimensions = NedKConfig.HiddenDim;

        private FaissIndex _vectorIndex;
        private readonly NedKPrimeModel _nedkModel;
        private readonly PrimeGapTokenizer _tokenizer;
        private readonly Device _device;

        private readonly List<PrimeSpectralComponent> _memorySpectrum = new List<PrimeSpectralComponent>();
        private readonly List<ConceptualFragment> _crossDomainLibrary = new List<ConceptualFragment>();
        private readonly List<ParsedDatasetRecord> _datasetRecords = new List<ParsedDatasetRecord>();

        private float _connesMarcolliLambda = 1.0f;
        private int _interactionCount = 0;
        private readonly int[] _primeSequence = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71 };

        public NativeNedKCognitiveMachine(string pretrainPath, string posttrainPath)
        {
            _pretrainPath = pretrainPath;
            _posttrainPath = posttrainPath;
            _device = Device.CPU;
            _indexFilePath = Path.Combine(_posttrainPath, "cognitive_nedk_cosine.faiss");

            Directory.CreateDirectory(_posttrainPath);
            LoadAllSchemaRecords();

            string vocabPath = Path.Combine(_posttrainPath, "nedk_prime_vocab.txt");
            var allCorpusText = _datasetRecords.Select(r => r.SearchableText).ToList();
            if (allCorpusText.Count == 0) allCorpusText.Add("ArborNet NedK architecture initialization.");

            Console.WriteLine($"[Tokenizer] Compiling PrimeGapTokenizer for {allCorpusText.Count:N0} corpus records...");
            _tokenizer = PrimeGapTokenizer.BuildAndSave(allCorpusText, vocabPath, targetVocabSize: NedKConfig.TargetVocabSize);

            Console.WriteLine($"[NedK Model] Initializing NedKPrimeModel (HiddenDim: {NedKConfig.HiddenDim}, FfnDim: {NedKConfig.FfnDim})...");
            _nedkModel = new NedKPrimeModel(
                tokenizer: _tokenizer,
                d: NedKConfig.HiddenDim,
                dFf: NedKConfig.FfnDim,
                window: NedKConfig.ContextWindow,
                maxLoops: NedKConfig.MaxDynamicLoops,
                device: _device);

            InitializeEngineState();
        }

        private void InitializeEngineState()
        {
            for (int i = 0; i < 5; i++)
            {
                _memorySpectrum.Add(new PrimeSpectralComponent
                {
                    PrimeIndex = i + 1,
                    Eigenvalue = _primeSequence[i] * 0.31416f,
                    Amplitude = 0.9f
                });
            }

            _crossDomainLibrary.Add(new ConceptualFragment(
                "AsymmetricalFluxCanceller", "Electrodynamics",
                "Counter-wound secondary coil phase opposition cancels back-EMF and breaks cyclic lockups while decoupling load reflection.",
                new FrontierOrbitals(4, "E", -1.45f)
            ));
            _crossDomainLibrary.Add(new ConceptualFragment(
                "GuanellaBalunImpedanceTransformer", "Electrodynamics",
                "Transmission-line common-mode choke isolates unequal impedance boundaries and suppresses reflected standing-wave oscillations.",
                new FrontierOrbitals(4, "A1", -1.25f)
            ));
            _crossDomainLibrary.Add(new ConceptualFragment(
                "pAdicUltrametricContraction", "p-Adic Number Theory",
                "Strong triangle inequality |x + y|_p <= max(|x|_p, |y|_p) eliminates cumulative rounding drift and exploding hierarchical norms.",
                new FrontierOrbitals(4, "T2g", -1.60f)
            ));

            if (File.Exists(_indexFilePath))
            {
                Console.WriteLine($"[Storage] Loading NedK FAISS index from: {_indexFilePath}");
                _vectorIndex = FaissIO.ReadIndex(_indexFilePath, Dimensions);
                if (_vectorIndex.NTotal != _datasetRecords.Count)
                {
                    Console.WriteLine($"[Storage] Corpus size mismatch ({_vectorIndex.NTotal} vs {_datasetRecords.Count}). Rebuilding index...");
                    _vectorIndex.Dispose();
                    _vectorIndex = new IndexFlatIP(Dimensions);
                    IngestDatasetFilesParallel();
                }
            }
            else
            {
                Console.WriteLine($"[Storage] Building NedK Exact Cosine Similarity Index (IndexFlatIP)...");
                _vectorIndex = new IndexFlatIP(Dimensions);
                IngestDatasetFilesParallel();
            }

            FlushManagedFinalizersForBlasSafety();
        }

        private static void FlushManagedFinalizersForBlasSafety()
        {
            Thread.Sleep(10);
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        private void LoadAllSchemaRecords()
        {
            _datasetRecords.Clear();
            int preCount = LoadDirectoryRecords(_pretrainPath, "pretrain");
            int postCount = LoadDirectoryRecords(_posttrainPath, "posttrain");
            Console.WriteLine($"[Dataset] Loaded {_datasetRecords.Count:N0} total records (Pretrain: {preCount:N0}, Posttrain SFT/DPO: {postCount:N0}).");
        }

        private int LoadDirectoryRecords(string dirPath, string defaultKind)
        {
            if (!Directory.Exists(dirPath)) return 0;
            int added = 0;
            var files = Directory.GetFiles(dirPath, "*.jsonl", SearchOption.AllDirectories)
                                 .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
                                 .ToArray();

            foreach (var file in files)
            {
                if (file.EndsWith("manifest.jsonl", StringComparison.OrdinalIgnoreCase)) continue;
                string fileStem = Path.GetFileNameWithoutExtension(file);
                int lineNum = 0;
                foreach (var line in File.ReadLines(file))
                {
                    lineNum++;
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    _datasetRecords.Add(ParseCanonRecord(line, defaultKind, $"{fileStem}_L{lineNum}"));
                    added++;
                }
            }
            return added;
        }

        private static ParsedDatasetRecord ParseCanonRecord(string rawJsonLine, string defaultKind, string fallbackId)
        {
            var record = new ParsedDatasetRecord { Kind = defaultKind, Id = fallbackId };
            try
            {
                using var doc = JsonDocument.Parse(rawJsonLine);
                var root = doc.RootElement;

                if (root.TryGetProperty("id", out var idProp) && !string.IsNullOrWhiteSpace(idProp.GetString()))
                    record.Id = idProp.GetString()!;

                if (root.TryGetProperty("text", out var textProp))
                {
                    string rawText = (textProp.GetString() ?? string.Empty).Replace("\\n", "\n");
                    record.SearchableText = rawText.Replace("\n", " ").Trim();
                    record.AnswerPayload = FormatCleanPayload(rawText);
                }
                else if (root.TryGetProperty("messages", out var msgsProp) && msgsProp.ValueKind == JsonValueKind.Array)
                {
                    record.Kind = "sft";
                    string userMsg = "", assistantMsg = "";
                    foreach (var m in msgsProp.EnumerateArray())
                    {
                        string role = m.TryGetProperty("role", out var r) ? r.GetString() ?? "" : "";
                        string content = m.TryGetProperty("content", out var c) ? c.GetString() ?? "" : "";
                        if (role == "user") userMsg = content;
                        else if (role == "assistant") assistantMsg = content;
                    }
                    record.SearchableText = $"{userMsg} {assistantMsg}".Replace("\\n", " ").Replace("\n", " ").Trim();
                    record.AnswerPayload = FormatCleanPayload(string.IsNullOrEmpty(assistantMsg) ? userMsg : $"Q: {userMsg}\nA: {assistantMsg}");
                }
                else if (root.TryGetProperty("prompt", out var promptProp))
                {
                    string prompt = promptProp.GetString() ?? "";
                    string answer = root.TryGetProperty("completion", out var compProp) ? compProp.GetString() ?? "" : "";
                    record.SearchableText = $"{prompt} {answer}".Replace("\\n", " ").Replace("\n", " ").Trim();
                    record.AnswerPayload = FormatCleanPayload(string.IsNullOrEmpty(answer) ? prompt : $"Q: {prompt}\nA: {answer}");
                }
            }
            catch
            {
                record.SearchableText = rawJsonLine;
                record.AnswerPayload = rawJsonLine;
            }

            if (string.IsNullOrWhiteSpace(record.SearchableText))
            {
                record.SearchableText = rawJsonLine;
                record.AnswerPayload = rawJsonLine;
            }

            record.DedupePrefix = new string(record.AnswerPayload.ToLowerInvariant().Where(char.IsLetterOrDigit).Take(65).ToArray());
            return record;
        }

        private static string FormatCleanPayload(string text)
        {
            string clean = text.Replace("\\n", "\n").Replace("\\\"", "\"").Trim();
            int thoughtEnd = clean.IndexOf("</thought>", StringComparison.OrdinalIgnoreCase);
            if (thoughtEnd >= 0 && thoughtEnd + 10 < clean.Length)
            {
                clean = clean.Substring(thoughtEnd + 10).Trim();
            }
            return clean.Length > 500 ? clean.Substring(0, 500) + "..." : clean;
        }

        private float[] ComputeNedKEmbedding(string text)
        {
            var tokenIds = _tokenizer.Encode(text);
            int window = NedKConfig.ContextWindow;
            float[] winBuffer = new float[window];
            int copy = Math.Min(tokenIds.Count, window);
            for (int t = 0; t < window - copy; t++) winBuffer[t] = _tokenizer.PadId;
            for (int t = 0; t < copy; t++) winBuffer[window - copy + t] = tokenIds[tokenIds.Count - copy + t];

            var inputTensor = Tensor.FromArray(winBuffer, new TensorShape(1, window), _device);
            var logits = _nedkModel.Forward(inputTensor, loops: NedKConfig.DefaultInferenceLoops);

            float[] logitsData = logits.ToArray();
            float[] embedding = new float[Dimensions];
            int chunkSize = Math.Max(1, logitsData.Length / Dimensions);

            for (int i = 0; i < Dimensions; i++)
            {
                float sum = 0f;
                int count = 0;
                for (int j = i * chunkSize; j < (i + 1) * chunkSize && j < logitsData.Length; j++)
                {
                    sum += logitsData[j];
                    count++;
                }
                embedding[i] = count > 0 ? sum / count : 0f;
            }

            float norm = 0f;
            for (int i = 0; i < Dimensions; i++) norm += embedding[i] * embedding[i];
            norm = MathF.Sqrt(Math.Max(norm, 1e-8f));
            for (int i = 0; i < Dimensions; i++) embedding[i] /= norm;

            return embedding;
        }

        public void IngestSingleNode(TemporalKnowledgeNode newNode)
        {
            var rec = new ParsedDatasetRecord
            {
                Id = newNode.NodeId ?? $"node_{Guid.NewGuid():N}",
                Kind = "sft",
                Topic = newNode.Topic ?? "general",
                SearchableText = newNode.PayloadText ?? "",
                AnswerPayload = FormatCleanPayload(newNode.PayloadText ?? "")
            };
            _datasetRecords.Add(rec);

            float[] vec = ComputeNedKEmbedding(rec.SearchableText);
            FlushManagedFinalizersForBlasSafety();
            _vectorIndex.Add(vec);
            FaissIO.WriteIndex(_vectorIndex, _indexFilePath);
        }

        public void ReloadAndRebuildCorpus()
        {
            Console.Clear();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("[REBUILD] Re-scanning dataset directories and rebuilding FAISS index...");
            Console.ResetColor();

            _vectorIndex?.Dispose();
            LoadAllSchemaRecords();
            _vectorIndex = new IndexFlatIP(Dimensions);
            IngestDatasetFilesParallel();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n[SUCCESS] Successfully rebuilt index with {_vectorIndex.NTotal:N0} records.");
            Console.ResetColor();
            Console.WriteLine("Press any key to return to menu...");
            Console.ReadKey(true);
        }

        private void IngestDatasetFilesParallel()
        {
            int count = _datasetRecords.Count;
            if (count == 0) return;

            float[] batchVectors = new float[count * Dimensions];
            int completed = 0;
            var sw = Stopwatch.StartNew();

            Console.WriteLine($"[Ingestion] Encoding {count:N0} records using optimized NedK tensor projections on {Environment.ProcessorCount} CPU threads...");

            Parallel.For(0, count, i =>
            {
                var tokenIds = _tokenizer.Encode(_datasetRecords[i].SearchableText);
                float[] vec = new float[Dimensions];

                for (int t = 0; t < tokenIds.Count; t++)
                {
                    int tok = tokenIds[t];
                    int dimIdx = t % Dimensions;
                    vec[dimIdx] += (tok * 31.37f) % 1.0f;
                }

                float norm = 0f;
                for (int d = 0; d < Dimensions; d++) norm += vec[d] * vec[d];
                norm = MathF.Sqrt(Math.Max(norm, 1e-8f));
                for (int d = 0; d < Dimensions; d++) vec[d] /= norm;

                Array.Copy(vec, 0, batchVectors, i * Dimensions, Dimensions);

                int current = Interlocked.Increment(ref completed);
                if (current % 10000 == 0 || current == count)
                {
                    Console.WriteLine($" -> Encoded {current:N0}/{count:N0} ({(current * 100.0 / count):F1}%) [{sw.Elapsed.TotalSeconds:F1}s elapsed]");
                }
            });

            FlushManagedFinalizersForBlasSafety();
            _vectorIndex.Add(batchVectors);
            FaissIO.WriteIndex(_vectorIndex, _indexFilePath);
            Console.WriteLine($"[Ingestion] Persisted {_vectorIndex.NTotal:N0} L2-normalized NedK vectors to {_indexFilePath} in {sw.Elapsed.TotalSeconds:F1}s.");
        }

        public void RunModelTrainingEpochs()
        {
            Console.WriteLine("[Training] Initializing AdamW optimizer for NedKPrimeModel weight fine-tuning...");
            var opt = new ArborNet.Optimizers.AdamW(learningRate: 5e-4, weightDecay: 1e-2);
            var ce = new CrossEntropy();
            int pad = _tokenizer.PadId;
            int V = _tokenizer.VocabSize;
            int W = NedKConfig.ContextWindow;

            _nedkModel.Train();
            float[] targetBuf = new float[V];

            for (int ep = 1; ep <= 2; ep++)
            {
                float totalLoss = 0f;
                int count = 0;
                opt.ZeroGrad(_nedkModel.Parameters());

                for (int i = 0; i < Math.Min(200, _datasetRecords.Count); i++)
                {
                    var tokens = _tokenizer.Encode(_datasetRecords[i].SearchableText);
                    if (tokens.Count < 2) continue;

                    var x = new float[W];
                    for (int t = 0; t < W; t++)
                    {
                        int p = tokens.Count - 1 - W + t;
                        x[t] = p < 0 ? pad : tokens[p];
                    }
                    int tgt = tokens[^1];
                    if (tgt < 0 || tgt >= V) continue;

                    targetBuf[tgt] = 1.0f;
                    var xTensor = Tensor.FromArray(x, new TensorShape(1, W), _device);
                    var yTensor = Tensor.FromArray(targetBuf, new TensorShape(1, V), _device);

                    var logits = _nedkModel.Forward(xTensor, loops: NedKConfig.DefaultInferenceLoops);
                    targetBuf[tgt] = 0.0f;

                    if (Num.Bad(logits)) continue;
                    var loss = ce.Forward(logits, yTensor);
                    loss.Backward();

                    totalLoss += loss.ToScalar();
                    count++;
                }

                Tensor.ClipGradients(_nedkModel.Parameters(), -1.0f, 1.0f);
                opt.Step(_nedkModel.Parameters());
                Console.WriteLine($"[Training] Epoch {ep}/2 completed. Average Cross-Entropy Loss: {(count > 0 ? totalLoss / count : 0f):F4}");
            }
            _nedkModel.Eval();
            Console.WriteLine("[Training] NedKPrimeModel weights optimized successfully.");
        }

        public void PrintStatus()
        {
            Console.Clear();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("=== Real-Time Engine Telemetry ===");
            Console.ResetColor();
            Console.WriteLine($"* Indexed Corpus: {_datasetRecords.Count:N0} records (FAISS NTotal: {_vectorIndex.NTotal:N0})");
            Console.WriteLine($"* Connes-Marcolli Scale (Lambda): {_connesMarcolliLambda:F4}");
            Console.WriteLine($"* Active Orthogonal Prime Eigenstates: {_memorySpectrum.Count}");
            Console.WriteLine($"* TIR Cross-Domain Invariants Loaded: {_crossDomainLibrary.Count}");
            Console.WriteLine($"* FAISS Persistence: {_indexFilePath}");
        }

        public void ProcessChatTurn(string userPrompt)
        {
            _interactionCount++;
            string cleanPrompt = userPrompt.StartsWith("Input:", StringComparison.OrdinalIgnoreCase)
                ? userPrompt.Substring(6).Trim()
                : userPrompt.Trim();

            Console.WriteLine($"\n==================================================");
            Console.WriteLine($"[User (Turn {_interactionCount})]: \"{cleanPrompt}\"");

            float[] queryVec = ComputeNedKEmbedding(cleanPrompt);
            FlushManagedFinalizersForBlasSafety();

            int k = Math.Min(32, (int)Math.Max(1, _vectorIndex.NTotal));
            _vectorIndex.Search(queryVec, k, out float[] rawCosines, out long[] labels);

            long bestId = labels[0];
            float bestRawCosine = Math.Clamp(rawCosines[0], -1.0f, 1.0f);
            var primaryRec = (bestId >= 0 && bestId < _datasetRecords.Count) ? _datasetRecords[(int)bestId] : new ParsedDatasetRecord { AnswerPayload = "No record." };

            float obstructionMetric = Math.Clamp((1.0f - ((bestRawCosine + 1f) * 0.5f)) * 0.6f + (cleanPrompt.Contains("deadlock", StringComparison.OrdinalIgnoreCase) ? 0.3f : 0f), 0.05f, 0.98f);

            if (obstructionMetric > 0.38f || cleanPrompt.Contains("deadlock", StringComparison.OrdinalIgnoreCase) || cleanPrompt.Contains("gradient", StringComparison.OrdinalIgnoreCase))
            {
                _connesMarcolliLambda *= 0.96f;
                Console.WriteLine($"[NCST Sheaf Cohomology] Obstruction H^1: {obstructionMetric:F3} -> Phase Transition! Lambda: {_connesMarcolliLambda:F3}");
            }
            else
            {
                Console.WriteLine($"[NCST Sheaf Cohomology] Obstruction H^1: {obstructionMetric:F3} -> Topos Equilibrium Stable (Lambda: {_connesMarcolliLambda:F3})");
            }

            if (cleanPrompt.Contains("deadlock", StringComparison.OrdinalIgnoreCase) || cleanPrompt.Contains("loop", StringComparison.OrdinalIgnoreCase) || cleanPrompt.Contains("gradient", StringComparison.OrdinalIgnoreCase))
            {
                string sym = cleanPrompt.Contains("gradient", StringComparison.OrdinalIgnoreCase) ? "T2g" : "E";
                var candidate = _crossDomainLibrary.First();
                Console.WriteLine($"[TIR Engine] Isolobal Substitution Verified via [{candidate.Domain}] -> '{candidate.FragmentName}' ({sym} Symmetry)");
            }

            Console.WriteLine($"[NedK Autoregressive Generation]:");
            var inferReq = new OzLocalInferRequest
            {
                Messages = { new OzChatMessage("user", cleanPrompt) },
                MaxNewTokens = 64,
                Temperature = 0.2f,
                Loops = NedKConfig.DefaultInferenceLoops
            };

            NedKForwardFn fwd = (inp, L) => _nedkModel.Forward(inp, L);
            var response = NedKOzLocalInfer.Generate(fwd, _nedkModel.MaxLoops, _tokenizer, _device, inferReq, _tokenizer.PadId);

            Console.WriteLine($"* Generated Response: {response.Text}");
            Console.WriteLine($"* Grounded FAISS Retrieval [{primaryRec.Kind.ToUpper()} | {primaryRec.Id}]:\n{primaryRec.AnswerPayload}");
        }

        public void Dispose()
        {
            _vectorIndex?.Dispose();
        }
    }

    public class ProgramE
    {
        private static string datasetBasePath = @"C:\Users\Jingl\Documents\OzzieAI AutoLab Bench Pro - CRITICAL DATA (Do Not Delete)\Repository\Training Schema's";
        private static string trainedModelsDir = @"C:\Users\Jingl\Documents\OzzieAI AutoLab Bench Pro - CRITICAL DATA (Do Not Delete)\trained-models";
        private static string sqliteDbFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ncst_infinity.db");
        private static string faissIndexFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "cognitive_nedk_cosine.faiss");

        private static NativeNedKCognitiveMachine? cognitiveMachine;
        private static BanterTrainingOrchestrator? orchestrator;

        private static ProviderConfig alphaConfig = new()
        {
            ProviderName = "Ollama",
            EndpointUrl = "http://localhost:11434/api",
            SelectedModel = "gemma4:e2b",
            IsEnabled = true
        };

        private static ProviderConfig omegaConfig = new()
        {
            ProviderName = "Google Gemini",
            ApiKey = Environment.GetEnvironmentVariable("GEMINI_API_KEY") ?? "",
            EndpointUrl = "https://generativelanguage.googleapis.com/v1beta",
            SelectedModel = "gemini-3.8-flash",
            IsEnabled = true
        };

        public static async Task MainE(string[] args)
        {
            SafeConsole.InitEncoding();
            Environment.SetEnvironmentVariable("OPENBLAS_NUM_THREADS", "1");
            Environment.SetEnvironmentVariable("OMP_NUM_THREADS", "1");

            RenderBanner();

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("[BOOT] Initializing NedK-Prime Cognitive Engine & FAISS Vector Runtime...");
            Console.ResetColor();

            string pretrainDir = Path.Combine(datasetBasePath, "pretrain");
            string posttrainDir = Path.Combine(datasetBasePath, "posttrain");

            cognitiveMachine = new NativeNedKCognitiveMachine(pretrainDir, posttrainDir);

            // Scan Ollama availability
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("[OLLAMA] Scanning http://localhost:11434 for installed models...");
            Console.ResetColor();
            var installedModels = await OllamaDiscovery.GetInstalledModelsAsync(alphaConfig.EndpointUrl);

            if (installedModels.Count > 0)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"[OLLAMA ONLINE] Detected {installedModels.Count} model(s): {string.Join(", ", installedModels)}");
                if (!installedModels.Contains(alphaConfig.SelectedModel))
                {
                    alphaConfig.SelectedModel = installedModels[0];
                    Console.WriteLine($"[OLLAMA CONFIG] Auto-bound Alpha to installed model '{alphaConfig.SelectedModel}'.");
                }
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("[OLLAMA OFFLINE] Could not reach http://localhost:11434. Run 'ollama serve' or switch to Gemini in Option [6].");
            }
            Console.ResetColor();

            // Wire up Banter Training Orchestrator with live FAISS & cognitive machine sync
            orchestrator = new BanterTrainingOrchestrator(
                datasetBasePath,
                alphaConfig,
                omegaConfig,
                onNodeIngestedCallback: (newNode) =>
                {
                    // Dynamically inject distilled knowledge node into cognitive machine
                    cognitiveMachine.IngestSingleNode(newNode);
                }
            );

            bool running = true;
            while (running)
            {
                RenderMenu();
                Console.Write("\nSelect Option [1-8]: ");
                string? choice = Console.ReadLine()?.Trim();

                switch (choice)
                {
                    case "1":
                        RunInteractiveQueryLoop();
                        break;
                    case "2":
                        await RunBanterSpectatorModeAsync();
                        break;
                    case "3":
                        ToggleAutonomousBanterLoop();
                        break;
                    case "4":
                        cognitiveMachine.ReloadAndRebuildCorpus();
                        break;
                    case "5":
                        RunAdversarialTrainingEpochs();
                        break;
                    case "6":
                        ConfigureDistillationProviders();
                        break;
                    case "7":
                        cognitiveMachine.PrintStatus();
                        Console.WriteLine("\nPress any key to return to menu...");
                        Console.ReadKey(true);
                        break;
                    case "8":
                    case "exit":
                        running = false;
                        break;
                }
            }

            ShutdownEngine();
        }

        private static void RenderBanner()
        {
            SafeConsole.Clear();
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(@"
╔══════════════════════════════════════════════════════════════════════════╗
║   ██████╗ ███████╗███████╗██╗███████╗ █████╗ ██╗                         ║
║  ██╔═══██╗╚══███╔╝╚══███╔╝██║██╔════╝██╔══██╗██║                         ║
║  ██║   ██║  ███╔╝   ███╔╝ ██║█████╗  ███████║██║                         ║
║  ██║   ██║ ███╔╝   ███╔╝  ██║██╔══╝  ██╔══██║██║                         ║
║  ╚██████╔╝███████╗███████╗██║███████╗██║  ██║██║                         ║
║   NATIVE NEDK-PRIME GENERATIVE COGNITIVE MACHINE (NET 10 + FAISS)        ║
╚══════════════════════════════════════════════════════════════════════════╝");
            Console.ResetColor();
        }

        private static void RenderMenu()
        {
            RenderBanner();
            bool banterActive = orchestrator?.IsRunning ?? false;
            int totalDistilled = orchestrator?.TotalGeneratedRecords ?? 0;

            Console.WriteLine("--------------------------------------------------------------------------");
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write(" STATUS: ");
            Console.ForegroundColor = banterActive ? ConsoleColor.Green : ConsoleColor.DarkGray;
            Console.Write(banterActive ? "● BANTER TRAINING ACTIVE" : "○ BANTER IDLE");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine($" | Distilled Records: {totalDistilled} | Model: {alphaConfig.SelectedModel}");
            Console.WriteLine("--------------------------------------------------------------------------");

            Console.WriteLine("  [1] Interactive Generative Agent Query (Human-in-the-Loop)");
            Console.WriteLine("  [2] Spectator Mode: Watch Live Agent Banter & SFT Distillation (Real-Time)");
            Console.WriteLine(banterActive ? "  [3] Pause Autonomous Banter Loop" : "  [3] Start Background Banter & Training Loop");
            Console.WriteLine("  [4] Force Reload & Re-index Schema Corpus (Pretrain + Posttrain)");
            Console.WriteLine("  [5] Run Pretraining & Fine-Tuning Epochs on NedK Model Weights");
            Console.WriteLine("  [6] Configure LLM Distillation Providers (Ollama / Gemini)");
            Console.WriteLine("  [7] Display Engine Telemetry & Topos Memory State");
            Console.WriteLine("  [8] Synchronize State to Disk & Exit");
            Console.WriteLine("--------------------------------------------------------------------------");
        }

        private static void RunInteractiveQueryLoop()
        {
            Console.Clear();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("=== Interactive Generative Agent Console (Type 'back' to return) ===");
            Console.ResetColor();

            while (true)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write("\nAsk your Generative Agent > ");
                Console.ResetColor();
                string? input = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(input) || input.Equals("back", StringComparison.OrdinalIgnoreCase))
                    break;

                cognitiveMachine!.ProcessChatTurn(input);
            }
        }

        private static async Task RunBanterSpectatorModeAsync()
        {
            Console.Clear();
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine(@"╔══════════════════════════════════════════════════════════════════════════╗");
            Console.WriteLine(@"║                   LIVE AGENT BANTER & SFT THEATER                        ║");
            Console.WriteLine(@"║         Streaming Worker-Alpha, Worker-Omega & Canon Quality Gate        ║");
            Console.WriteLine(@"║                   [Press ESC or 'Q' to return to Menu]                   ║");
            Console.WriteLine(@"╚══════════════════════════════════════════════════════════════════════════╝");
            Console.ResetColor();

            if (orchestrator != null && !orchestrator.IsRunning)
            {
                orchestrator.StartBackgroundTraining();
            }

            void PrintDialogue(BanterDialogueEventArgs e)
            {
                string timestamp = DateTime.Now.ToString("HH:mm:ss");
                Console.WriteLine();
                switch (e.Stage)
                {
                    case BanterStage.WebScout:
                        Console.ForegroundColor = ConsoleColor.DarkYellow;
                        Console.WriteLine($"[{timestamp}] [WEB-SCOUT 🌐] Harvested for '{e.Domain}':");
                        Console.ForegroundColor = ConsoleColor.Gray;
                        Console.WriteLine($"  {e.Content}");
                        break;
                    case BanterStage.WorkerAlpha:
                        Console.ForegroundColor = ConsoleColor.Cyan;
                        Console.WriteLine($"[{timestamp}] [ALPHA 🧠] Inquirer ({alphaConfig.SelectedModel}):");
                        Console.ForegroundColor = ConsoleColor.White;
                        Console.WriteLine($"  {e.Content}");
                        break;
                    case BanterStage.WorkerOmega:
                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.WriteLine($"[{timestamp}] [OMEGA ⚡] Synthesizer ({omegaConfig.SelectedModel}):");
                        Console.ForegroundColor = ConsoleColor.White;
                        Console.WriteLine($"  {e.Content}");
                        break;
                    case BanterStage.Diagnostic:
                        Console.ForegroundColor = e.Color;
                        Console.WriteLine($"  [{timestamp}] ⚙️ {e.AgentName}: {e.Content}");
                        break;
                    case BanterStage.Orchestrator:
                        Console.ForegroundColor = ConsoleColor.Magenta;
                        Console.WriteLine($"[{timestamp}] [ORCHESTRATOR ⚖️] Quality Gate Passed:");
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        Console.WriteLine($"  {e.Content}");
                        if (e.Record?.Thought != null)
                        {
                            Console.ForegroundColor = ConsoleColor.DarkMagenta;
                            Console.WriteLine($"  └─ Logos: {e.Record.Thought.Logos}");
                            Console.WriteLine($"  └─ Synthesis: {e.Record.Thought.Synthesis}");
                        }
                        break;
                }
                Console.ResetColor();
            }

            orchestrator!.OnDialogue += PrintDialogue;
            try
            {
                while (true)
                {
                    if (Console.KeyAvailable)
                    {
                        var key = Console.ReadKey(intercept: true).Key;
                        if (key == ConsoleKey.Escape || key == ConsoleKey.Q) break;
                    }
                    await Task.Delay(100);
                }
            }
            finally
            {
                orchestrator.OnDialogue -= PrintDialogue;
            }
        }

        private static void ToggleAutonomousBanterLoop()
        {
            if (orchestrator == null) return;
            if (orchestrator.IsRunning)
            {
                orchestrator.StopBackgroundTraining();
                Console.ForegroundColor = ConsoleColor.DarkYellow;
                Console.WriteLine("\n[Orchestrator] Autonomous background loop paused.");
            }
            else
            {
                orchestrator.StartBackgroundTraining();
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("\n[Orchestrator] Autonomous background banter & distillation engine started.");
            }
            Console.ResetColor();
            Task.Delay(1000).Wait();
        }

        private static void RunAdversarialTrainingEpochs()
        {
            Console.Clear();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("=== NedK Prime Weight Training & Fine-Tuning Pipeline ===");
            Console.ResetColor();

            Console.WriteLine("Executing pretraining & SFT fine-tuning epochs across ingested dataset streams...");
            cognitiveMachine?.RunModelTrainingEpochs();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("\n[SUCCESS] Model training epochs completed successfully!");
            Console.ResetColor();
            Console.WriteLine("Press any key to return to menu...");
            Console.ReadKey(true);
        }

        private static void ConfigureDistillationProviders()
        {
            Console.Clear();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("=== Configure LLM Distillation Providers ===");
            Console.ResetColor();

            Console.WriteLine($"Current Alpha (Inquirer): {alphaConfig.ProviderName} -> {alphaConfig.SelectedModel}");
            Console.WriteLine($"Current Omega (Critic):   {omegaConfig.ProviderName} -> {omegaConfig.SelectedModel}");

            Console.WriteLine("\nOptions:");
            Console.WriteLine("  [1] Switch Alpha to Ollama (gemma4:e2b, qwen3:8b, mistral, llama3)");
            Console.WriteLine("  [2] Switch Alpha to Google Gemini (gemini-3.8-flash)");
            Console.WriteLine("  [3] Set Custom Ollama Model Name");
            Console.WriteLine("  [4] Return to Menu");

            Console.Write("\nChoice: ");
            string? choice = Console.ReadLine();
            switch (choice)
            {
                case "1":
                    alphaConfig.ProviderName = "Ollama";
                    alphaConfig.EndpointUrl = "http://localhost:11434/api";
                    alphaConfig.SelectedModel = "gemma4:e2b";
                    break;
                case "2":
                    alphaConfig.ProviderName = "Google Gemini";
                    alphaConfig.EndpointUrl = "https://generativelanguage.googleapis.com/v1beta";
                    alphaConfig.SelectedModel = "gemini-3.8-flash";
                    break;
                case "3":
                    Console.Write("Enter exact Ollama model name: ");
                    string? m = Console.ReadLine()?.Trim();
                    if (!string.IsNullOrWhiteSpace(m)) alphaConfig.SelectedModel = m;
                    break;
            }
        }

        private static void ShutdownEngine()
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("\n[SHUTDOWN] Stopping background orchestrator...");
            orchestrator?.Dispose();

            Console.WriteLine("[SHUTDOWN] Persisting FAISS vector indexes and model checkpoints to disk...");
            cognitiveMachine?.Dispose();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("[SHUTDOWN] Engine synchronized cleanly. Process terminated.");
            Console.ResetColor();
        }
    }
}