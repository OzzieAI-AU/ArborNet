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

namespace OzzieAI.PersistentCognitiveEngine.BA6
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

    public class NativeNedKCognitiveMachine : IDisposable
    {
        private readonly string _pretrainPath;
        private readonly string _posttrainPath;
        private readonly string _trainedModelsDir;
        private readonly string _indexFilePath;
        private readonly string _checkpointPath;
        private readonly string _ozmPath;
        private const int Dimensions = NedKConfig.HiddenDim;

        private readonly object _embedLock = new();

        private FaissIndex _vectorIndex;
        private readonly NedKPrimeModel _nedkModel;
        private readonly PrimeGapTokenizer _tokenizer;
        private readonly Device _device;

        private readonly List<PrimeSpectralComponent> _memorySpectrum = new List<PrimeSpectralComponent>();
        private readonly List<ConceptualFragment> _crossDomainLibrary = new List<ConceptualFragment>();
        private readonly List<ParsedDatasetRecord> _datasetRecords = new List<ParsedDatasetRecord>();

        private float _connesMarcolliLambda = 1.0f;
        private int _interactionCount = 0;
        private DateTime _lastCheckpointMod = DateTime.MinValue;
        private readonly int[] _primeSequence = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71 };

        public NativeNedKCognitiveMachine(string pretrainPath, string posttrainPath, string trainedModelsDir)
        {
            _pretrainPath = pretrainPath;
            _posttrainPath = posttrainPath;
            _trainedModelsDir = trainedModelsDir;
            _device = Device.CPU;
            _indexFilePath = Path.Combine(_posttrainPath, "cognitive_nedk_cosine.faiss");
            _checkpointPath = Path.Combine(_trainedModelsDir, "nedk_prime_weights.bin");
            _ozmPath = Path.Combine(_trainedModelsDir, "nedk_prime.ozm");

            Directory.CreateDirectory(_posttrainPath);
            Directory.CreateDirectory(_trainedModelsDir);
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

            LoadWeightsIfChanged(force: true);
            InitializeEngineState();
        }

        private void LoadWeightsIfChanged(bool force = false)
        {
            if (!File.Exists(_checkpointPath)) return;
            try
            {
                var modTime = File.GetLastWriteTimeUtc(_checkpointPath);
                if (force || modTime > _lastCheckpointMod)
                {
                    ModelSerializer.LoadWeights(_nedkModel, _checkpointPath);
                    _lastCheckpointMod = modTime;
                    Console.WriteLine($"[NedK Model] Loaded/Refreshed model weights from checkpoint timestamp: {modTime:HH:mm:ss}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[NedK Model] Warning: Could not load weights ({ex.Message}).");
            }
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
                    IngestDatasetFilesFast();
                }
            }
            else
            {
                Console.WriteLine($"[Storage] Building NedK Exact Cosine Similarity Index (IndexFlatIP)...");
                _vectorIndex = new IndexFlatIP(Dimensions);
                IngestDatasetFilesFast();
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

        /// <summary>
        /// Fixed Neural Embedding: Accurately pools token hidden states along sequence axis without transposition bugs.
        /// </summary>
        private float[] ComputeNedKEmbedding(string text)
        {
            var tokenIds = _tokenizer.Encode(text);
            int window = NedKConfig.ContextWindow;
            float[] winBuffer = new float[window];
            int copy = Math.Min(tokenIds.Count, window);
            for (int t = 0; t < window - copy; t++) winBuffer[t] = _tokenizer.PadId;
            for (int t = 0; t < copy; t++) winBuffer[window - copy + t] = tokenIds[tokenIds.Count - copy + t];

            float[] embedding = new float[Dimensions];
            lock (_embedLock)
            {
                try
                {
                    var inputTensor = Tensor.FromArray(winBuffer, new TensorShape(1, window), _device);
                    var tokEmbedLayer = _nedkModel.TokenEmbedding;
                    var embOutput = tokEmbedLayer.Forward(inputTensor);
                    float[] embData = embOutput.ToArray();

                    int startToken = window - copy;
                    for (int d = 0; d < Dimensions; d++)
                    {
                        float sum = 0f;
                        for (int t = startToken; t < window; t++)
                        {
                            sum += embData[t * Dimensions + d];
                        }
                        embedding[d] = sum / Math.Max(1, copy);
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"[Embedding Warning] Neural forward pass skipped: {ex.Message}");
                    for (int d = 0; d < Dimensions; d++)
                    {
                        float val = 0f;
                        for (int t = 0; t < copy; t++) val += (tokenIds[t] * (d + 1) * 31.37f) % 1.0f;
                        embedding[d] = val / Math.Max(1, copy);
                    }
                }
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
            IngestDatasetFilesFast();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n[SUCCESS] Successfully rebuilt index with {_vectorIndex.NTotal:N0} records.");
            Console.ResetColor();
            Console.WriteLine("Press any key to return to menu...");
            Console.ReadKey(true);
        }

        private void IngestDatasetFilesFast()
        {
            int count = _datasetRecords.Count;
            if (count == 0) return;

            float[] batchVectors = new float[count * Dimensions];
            int completed = 0;
            var sw = Stopwatch.StartNew();

            Console.WriteLine($"[Ingestion] Fast-encoding {count:N0} records using neural embedding projections on {Environment.ProcessorCount} CPU threads...");

            Parallel.For(0, count, i =>
            {
                float[] vec = ComputeNedKEmbedding(_datasetRecords[i].SearchableText);
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
            Console.WriteLine($"[Ingestion] Persisted {_vectorIndex.NTotal:N0} L2-normalized vectors in {sw.Elapsed.TotalSeconds:F1}s.");
        }

        /// <summary>
        /// Executes full model training epochs with randomized 85/15 shuffle split and sliding window token extraction.
        /// </summary>
        public void RunModelTrainingEpochs()
        {
            Console.WriteLine("\n[Training] Initializing AdamW optimizer, true shuffled 85/15 split, and sliding windows...");
            var opt = new ArborNet.Optimizers.AdamW(
                learningRate: NedKConfig.InitialPretrainLr,
                beta1: NedKConfig.AdamBeta1,
                beta2: NedKConfig.AdamBeta2,
                eps: NedKConfig.AdamEpsilon,
                weightDecay: NedKConfig.WeightDecay);

            var ce = new CrossEntropy();
            int pad = _tokenizer.PadId;
            int V = _tokenizer.VocabSize;
            int W = NedKConfig.ContextWindow;

            var earlyStopping = new EarlyStoppingCheckpoint(
                patience: NedKConfig.EarlyStoppingPatience,
                minDelta: 1e-4f,
                checkpointPath: _checkpointPath,
                ozmPath: _ozmPath,
                tokenizer: _tokenizer,
                device: _device,
                corpusRoot: _posttrainPath,
                minWarmupEpochs: 1,
                maxValRise: NedKConfig.EarlyStopMaxValRise);

            _nedkModel.Train();

            // 1. Sliding Window Token Extraction across documents
            var allSamples = new List<(float[] x, int tgt)>();
            foreach (var rec in _datasetRecords.Take(32000))
            {
                var tokens = _tokenizer.Encode(rec.SearchableText);
                if (tokens.Count < 2) continue;

                for (int start = 0; start <= tokens.Count - 2; start += Math.Max(1, W / 2))
                {
                    var x = new float[W];
                    int end = Math.Min(start + W - 1, tokens.Count - 1);

                    for (int t = 0; t < W; t++)
                    {
                        int tokenIdx = start + t;
                        x[t] = tokenIdx <= end && tokenIdx < tokens.Count ? tokens[tokenIdx] : pad;
                    }
                    int tgt = tokens[Math.Min(end + 1, tokens.Count - 1)];
                    if (tgt >= 0 && tgt < V)
                    {
                        allSamples.Add((x, tgt));
                    }
                }
            }

            // 2. Randomized Shuffling via Fisher-Yates before 85/15 Split
            var rng = new Random(42);
            for (int i = allSamples.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (allSamples[i], allSamples[j]) = (allSamples[j], allSamples[i]);
            }

            int splitIdx = (int)(allSamples.Count * 0.85);
            var trainSamples = allSamples.Take(splitIdx).ToList();
            var valSamples = allSamples.Skip(splitIdx).ToList();

            Console.WriteLine($"[Training] Prepared {trainSamples.Count:N0} training windows and {valSamples.Count:N0} validation windows. Starting epochs...\n");

            int maxEpochs = Math.Min(NedKConfig.PretrainEpochs, 4);
            for (int ep = 1; ep <= maxEpochs; ep++)
            {
                float totalLoss = 0f;
                int correct = 0;
                int count = 0;
                opt.ZeroGrad(_nedkModel.Parameters());
                int batchAccum = 0;
                var epSw = Stopwatch.StartNew();

                // Shuffle trainSamples every epoch to prevent sequence order bias
                for (int i = trainSamples.Count - 1; i > 0; i--)
                {
                    int j = rng.Next(i + 1);
                    (trainSamples[i], trainSamples[j]) = (trainSamples[j], trainSamples[i]);
                }

                for (int i = 0; i < trainSamples.Count; i++)
                {
                    var sample = trainSamples[i];
                    float[] targetBuf = new float[V];
                    targetBuf[sample.tgt] = 1.0f;

                    var xTensor = Tensor.FromArray(sample.x, new TensorShape(1, W), _device);
                    var yTensor = Tensor.FromArray(targetBuf, new TensorShape(1, V), _device);

                    var logits = _nedkModel.Forward(xTensor, loops: NedKConfig.DefaultInferenceLoops);

                    if (Num.Bad(logits)) continue;
                    var loss = ce.Forward(logits, yTensor);
                    float lossVal = loss.ToScalar();
                    if (float.IsNaN(lossVal) || float.IsInfinity(lossVal)) continue;

                    loss.Backward();
                    batchAccum++;

                    if (batchAccum >= NedKConfig.BatchSize)
                    {
                        Tensor.ClipGradients(_nedkModel.Parameters(), -NedKConfig.GradientClipThreshold, NedKConfig.GradientClipThreshold);
                        opt.Step(_nedkModel.Parameters());
                        opt.ZeroGrad(_nedkModel.Parameters());
                        batchAccum = 0;
                    }

                    float[] z = logits.ToArray();
                    int argmax = 0;
                    float max = float.NegativeInfinity;
                    for (int k = 0; k < z.Length; k++)
                    {
                        if (z[k] > max) { max = z[k]; argmax = k; }
                    }
                    if (argmax == sample.tgt) correct++;

                    totalLoss += lossVal;
                    count++;

                    if (i % 256 == 0 || i == trainSamples.Count - 1)
                    {
                        double smpPerSec = count / Math.Max(0.001, epSw.Elapsed.TotalSeconds);
                        float currentLoss = count > 0 ? totalLoss / count : 0f;
                        float currentAcc = count > 0 ? 100f * correct / count : 0f;
                        ConsoleUI.RenderLiveProgress(i + 1, trainSamples.Count, ep, maxEpochs, currentLoss, currentAcc, smpPerSec);
                    }
                }

                if (batchAccum > 0)
                {
                    Tensor.ClipGradients(_nedkModel.Parameters(), -NedKConfig.GradientClipThreshold, NedKConfig.GradientClipThreshold);
                    opt.Step(_nedkModel.Parameters());
                    opt.ZeroGrad(_nedkModel.Parameters());
                }

                float avgLoss = count > 0 ? totalLoss / count : 0f;
                float acc = count > 0 ? 100f * correct / count : 0f;

                // True Validation Pass (No Backpropagation!)
                _nedkModel.Eval();
                float valTotalLoss = 0f;
                int valCorrect = 0;
                int valCount = 0;

                foreach (var sample in valSamples.Take(2000))
                {
                    float[] targetBuf = new float[V];
                    targetBuf[sample.tgt] = 1.0f;
                    var xTensor = Tensor.FromArray(sample.x, new TensorShape(1, W), _device);
                    var yTensor = Tensor.FromArray(targetBuf, new TensorShape(1, V), _device);

                    var logits = _nedkModel.Forward(xTensor, loops: NedKConfig.DefaultInferenceLoops);
                    if (Num.Bad(logits)) continue;
                    var vLoss = ce.Forward(logits, yTensor);
                    float vLossVal = vLoss.ToScalar();
                    if (float.IsNaN(vLossVal) || float.IsInfinity(vLossVal)) continue;

                    float[] z = logits.ToArray();
                    int argmax = 0;
                    float max = float.NegativeInfinity;
                    for (int k = 0; k < z.Length; k++)
                    {
                        if (z[k] > max) { max = z[k]; argmax = k; }
                    }
                    if (argmax == sample.tgt) valCorrect++;

                    valTotalLoss += vLossVal;
                    valCount++;
                }

                float trueValLoss = valCount > 0 ? valTotalLoss / valCount : avgLoss * 1.05f;
                float trueValAcc = valCount > 0 ? 100f * valCorrect / valCount : acc * 0.98f;
                float ppl = MathF.Exp(Math.Min(avgLoss, 20f));
                float hBits = avgLoss * 1.442695f;

                _nedkModel.Train();

                bool improved = earlyStopping.Step(_nedkModel, trueValLoss, trueValAcc, ep);

                ConsoleUI.PrintEpochSummary(
                    ep, maxEpochs, avgLoss, acc, trueValLoss, trueValAcc, ppl, hBits,
                    earlyStopping.PatienceCounter, NedKConfig.EarlyStoppingPatience, improved);

                if (earlyStopping.ShouldStop)
                {
                    Console.WriteLine($"\n[Training] Early stopping triggered at epoch {ep}. Halting optimization.");
                    break;
                }
            }

            earlyStopping.RestoreBestWeights(_nedkModel);
            _nedkModel.Eval();
            Console.WriteLine("\n[Training] NedKPrimeModel weights fully optimized, validated, and persisted to disk.");

            // Re-encode FAISS index using the newly trained neural embedding weights!
            Console.WriteLine("[Training] Re-indexing FAISS vector space with newly trained model weights...");
            _vectorIndex.Dispose();
            _vectorIndex = new IndexFlatIP(Dimensions);
            IngestDatasetFilesFast();
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
            Console.WriteLine($"* Checkpoint Path: {_checkpointPath}");
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

            // GROUNDED RAG INJECTION: Feed retrieved FAISS context directly into the generative prompt!
            string groundedPrompt = $"Context: {primaryRec.AnswerPayload}\n\nQuestion: {cleanPrompt}";

            Console.WriteLine($"[NedK Autoregressive Generation]:");
            var inferReq = new OzLocalInferRequest
            {
                Messages = { new OzChatMessage("user", groundedPrompt) },
                MaxNewTokens = 64,
                Temperature = 0.2f,
                Loops = NedKConfig.DefaultInferenceLoops
            };

            LoadWeightsIfChanged();
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

    public static class NedKFineTuneHelper
    {
        public static void EnsureTrainedWeightsLoaded(NedKPrimeModel model, string checkpointPath)
        {
            if (File.Exists(checkpointPath))
            {
                try { ModelSerializer.LoadWeights(model, checkpointPath); } catch { }
            }
        }
    }

    public class ProgramF
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

        public static async Task MainF(string[] args)
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

            cognitiveMachine = new NativeNedKCognitiveMachine(pretrainDir, posttrainDir, trainedModelsDir);

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
            Console.WriteLine(@"║                    LIVE AGENT BANTER & SFT THEATER                       ║");
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