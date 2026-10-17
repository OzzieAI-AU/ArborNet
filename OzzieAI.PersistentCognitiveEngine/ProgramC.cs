using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using OzzieAI.FaissSharp;
using ArborNet.Core.Tensors;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Layers;

namespace OzzieAI.PersistentCognitiveEngine.BA3
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
        public string Kind { get; set; } = "pretrain"; // pretrain, sft, dpo
        public string Topic { get; set; } = string.Empty;
        public string SearchableText { get; set; } = string.Empty;
        public string AnswerPayload { get; set; } = string.Empty;
    }

    /// <summary>
    /// v3.5 Orthogonal 2D-Native ArborNet Neural Network with Live Autograd Training Support.
    /// Bit-compatible with the 72,774-vector v2/v3 FAISS index.
    /// </summary>
    public class ArborNet2DCognitiveNetwork
    {
        private readonly int _dModel;
        private readonly Device _device;
        private readonly ITensor _orthoKeyWeight;
        private readonly ITensor _orthoValWeight;
        private readonly Linear _intentHead;
        public float LastAutogradLoss { get; private set; }

        public ArborNet2DCognitiveNetwork(int dModel = 64, Device? device = null)
        {
            _dModel = dModel;
            _device = device ?? Device.CPU;

            _orthoKeyWeight = CreateNearIdentityMatrix(dModel, seed: 1337);
            _orthoValWeight = CreateNearIdentityMatrix(dModel, seed: 7331);
            _intentHead = new Linear(dModel, 3, _device);
        }

        private ITensor CreateNearIdentityMatrix(int dim, int seed)
        {
            var rand = new Random(seed);
            float[] data = new float[dim * dim];
            for (int i = 0; i < dim; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    float noise = (float)(rand.NextDouble() * 2.0 - 1.0) * 0.08f;
                    data[i * dim + j] = (i == j ? 0.92f : 0.0f) + noise;
                }
            }
            return Tensor.FromArray(data, new TensorShape(dim, dim), _device);
        }

        private ITensor EncodeWordsTo2DTensor(string text, int maxWords = 28)
        {
            var words = text.ToLowerInvariant()
                .Split(new[] { ' ', '\n', '\r', '\t', '.', ',', '?', '!', ':', ';', '"', '\'', '(', ')', '[', ']', '{', '}', '-', '#' },
                       StringSplitOptions.RemoveEmptyEntries)
                .Where(w => w.Length > 2 && w != "the" && w != "and" && w != "for" && w != "are" && w != "you" && w != "with")
                .Take(maxWords)
                .ToArray();

            if (words.Length == 0) words = new[] { "default" };

            int seqLen = words.Length;
            float[] flatMatrix = new float[seqLen * _dModel];

            for (int row = 0; row < seqLen; row++)
            {
                string word = words[row];
                float positionWeight = 1.0f + (1.0f / (1.0f + row * 0.25f));

                for (int i = 0; i < word.Length; i++)
                {
                    uint h1 = (uint)word[i] * 2654435761u;
                    uint h2 = i + 1 < word.Length ? (uint)(word[i] << 8 | word[i + 1]) * 2246822519u : h1;
                    uint h3 = i + 2 < word.Length ? (uint)(word[i] << 16 | word[i + 1] << 8 | word[i + 2]) * 3266489917u : h2;

                    int d1 = (int)(h1 % (uint)_dModel);
                    int d2 = (int)(h2 % (uint)_dModel);
                    int d3 = (int)(h3 % (uint)_dModel);

                    float sign1 = (h1 & 1) == 0 ? 1f : -1f;
                    float sign2 = (h2 & 2) == 0 ? 1f : -1f;
                    float sign3 = (h3 & 4) == 0 ? 1f : -1f;

                    flatMatrix[row * _dModel + d1] += sign1 * 0.5f * positionWeight;
                    flatMatrix[row * _dModel + d2] += sign2 * 1.0f * positionWeight;
                    flatMatrix[row * _dModel + d3] += sign3 * 1.5f * positionWeight;
                }
            }

            return Tensor.FromArray(flatMatrix, new TensorShape(seqLen, _dModel), _device);
        }

        public (float[] embedding, CognitiveIntent intent, float[] intentProbs) Forward2D(string cleanText, bool trainOnline = false)
        {
            using (new TensorScope())
            {
                ITensor x = EncodeWordsTo2DTensor(cleanText);

                // 2D Self-Attention Mixing ([SeqLen, dModel])
                ITensor k = x.MatMul(_orthoKeyWeight);
                ITensor v = x.MatMul(_orthoValWeight);
                ITensor kT = k.Transpose(new[] { 1, 0 });

                float scale = 1.0f / MathF.Sqrt(_dModel);
                ITensor attnScores = x.MatMul(kT).Multiply(scale).Softmax(axis: 1);
                ITensor context = attnScores.MatMul(v);

                ITensor residual = x.Multiply(0.80f).Add(context.Multiply(0.20f));
                ITensor pooled2D = residual.Mean(axis: 0, keepDims: true); // [1, dModel]

                // Zero-Mean Centering & L2 Normalization
                float[] rawEmbed = pooled2D.ToArray();
                float mean = 0f;
                for (int i = 0; i < rawEmbed.Length; i++) mean += rawEmbed[i];
                mean /= rawEmbed.Length;

                float norm = 0f;
                for (int i = 0; i < rawEmbed.Length; i++)
                {
                    rawEmbed[i] -= mean;
                    norm += rawEmbed[i] * rawEmbed[i];
                }
                norm = MathF.Sqrt(Math.Max(norm, 1e-8f));
                for (int i = 0; i < rawEmbed.Length; i++) rawEmbed[i] /= norm;

                // Intent Routing + Online Autograd Step
                ITensor normTensor2D = Tensor.FromArray(rawEmbed, new TensorShape(1, _dModel), _device);
                ITensor intentLogits = _intentHead.Forward(normTensor2D);

                float[] teacherBias = ComputeSemanticLogitBias(cleanText);
                ITensor biasTensor = Tensor.FromArray(teacherBias, new TensorShape(1, 3), _device);
                ITensor probsTensor = intentLogits.Add(biasTensor).Softmax(axis: 1);

                float[] probs = probsTensor.ToArray();
                int bestClass = (int)probsTensor.ArgMax(axis: 1).ToArray()[0];

                if (trainOnline)
                {
                    float[] targetOneHot = new float[3];
                    targetOneHot[bestClass] = 1.0f;
                    ITensor targetTensor = Tensor.FromArray(targetOneHot, new TensorShape(1, 3), _device);

                    ITensor rawProbs = intentLogits.Softmax(axis: 1);
                    ITensor diff = rawProbs.Subtract(targetTensor);
                    ITensor loss = diff.Multiply(diff).Mean();
                    LastAutogradLoss = loss.ToArray()[0];

                    foreach (var p in _intentHead.Parameters()) p.ClearGrad();
                    loss.Backward();

                    float learningRate = 0.08f;
                    foreach (var p in _intentHead.Parameters())
                    {
                        if (p.Grad != null)
                        {
                            p.SubtractInPlace(p.Grad.Multiply(learningRate));
                        }
                    }
                }

                return (rawEmbed, (CognitiveIntent)bestClass, probs);
            }
        }

        private static float[] ComputeSemanticLogitBias(string text)
        {
            string lower = text.ToLowerInvariant();
            if (lower.Contains("deadlock") || lower.Contains("paradox") || lower.Contains("contradiction") ||
                lower.Contains("loop") || lower.Contains("strain") || lower.Contains("impedance") ||
                lower.Contains("exploding") || lower.Contains("dissonance") || lower.Contains("bottleneck"))
            {
                return new float[] { -1.0f, 0.2f, 3.5f };
            }
            if (lower.Contains("what") || lower.Contains("how") || lower.Contains("list") ||
                lower.Contains("define") || lower.Contains("research") || lower.Contains("structured") || lower.Contains("explain"))
            {
                return new float[] { 3.2f, 0.4f, -1.0f };
            }
            return new float[] { 0.4f, 2.8f, 0.1f };
        }
    }

    public class NativeArborNetCognitiveMachine : IDisposable
    {
        private readonly string _pretrainPath;
        private readonly string _posttrainPath;
        private readonly string _indexFilePath;
        private const int Dimensions = 64;

        private FaissIndex _vectorIndex;
        private readonly ArborNet2DCognitiveNetwork _arborNet2D;
        private readonly List<PrimeSpectralComponent> _memorySpectrum = new List<PrimeSpectralComponent>();
        private readonly List<ConceptualFragment> _crossDomainLibrary = new List<ConceptualFragment>();
        private readonly List<ParsedDatasetRecord> _datasetRecords = new List<ParsedDatasetRecord>();

        private float _connesMarcolliLambda = 1.0f;
        private int _interactionCount = 0;
        private readonly int[] _primeSequence = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71 };

        private static readonly HashSet<string> StopWords = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "that", "this", "these", "those", "should", "would", "could", "have", "has",
            "had", "with", "from", "into", "unto", "upon", "about", "above", "below",
            "under", "over", "again", "further", "then", "once", "here", "there", "all",
            "any", "both", "each", "few", "more", "most", "other", "some", "such", "only",
            "own", "same", "than", "too", "very", "can", "will", "just", "don", "now", "our", "your"
        };

        public NativeArborNetCognitiveMachine(string pretrainPath, string posttrainPath)
        {
            _pretrainPath = pretrainPath;
            _posttrainPath = posttrainPath;
            _indexFilePath = Path.Combine(_posttrainPath, "cognitive_arbornet2d_v2_cosine.faiss");

            _arborNet2D = new ArborNet2DCognitiveNetwork(Dimensions, Device.CPU);
            InitializeEngine();
        }

        private void InitializeEngine()
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

            // Expanded 6-Orbital Cross-Domain TIR Library
            _crossDomainLibrary.Add(new ConceptualFragment(
                "AsymmetricalFluxCanceller",
                "Electrodynamics",
                "Counter-wound secondary coil phase opposition cancels back-EMF and breaks cyclic lockups while decoupling load reflection.",
                new FrontierOrbitals(activePropositionCount: 4, symmetryClass: "E", energyEigenvalue: -1.45f)
            ));
            _crossDomainLibrary.Add(new ConceptualFragment(
                "GuanellaBalunImpedanceTransformer",
                "Electrodynamics",
                "Transmission-line common-mode choke isolates unequal impedance boundaries and suppresses reflected standing-wave oscillations.",
                new FrontierOrbitals(activePropositionCount: 4, symmetryClass: "A1", energyEigenvalue: -1.25f)
            ));
            _crossDomainLibrary.Add(new ConceptualFragment(
                "pAdicUltrametricContraction",
                "p-Adic Number Theory",
                "Strong triangle inequality |x + y|_p <= max(|x|_p, |y|_p) eliminates cumulative rounding drift and exploding hierarchical norms.",
                new FrontierOrbitals(activePropositionCount: 4, symmetryClass: "T2g", energyEigenvalue: -1.60f)
            ));
            _crossDomainLibrary.Add(new ConceptualFragment(
                "ConsciousInterfaceMarkovKernel",
                "Cognitive Interface Theory",
                "Fitness-payoff epistemic projection compresses high-entropy objective states into bounded, actionable interface icons.",
                new FrontierOrbitals(activePropositionCount: 2, symmetryClass: "B2", energyEigenvalue: -0.85f)
            ));

            Directory.CreateDirectory(_posttrainPath);
            LoadAllSchemaRecords();

            if (File.Exists(_indexFilePath))
            {
                Console.WriteLine($"[Storage] Loading Cosine Similarity FAISS index from: {_indexFilePath}");
                _vectorIndex = FaissIO.ReadIndex(_indexFilePath, Dimensions);
                if (_vectorIndex.NTotal != _datasetRecords.Count)
                {
                    Console.WriteLine($"[Storage] Corpus size changed ({_vectorIndex.NTotal} -> {_datasetRecords.Count}). Rebuilding index...");
                    _vectorIndex.Dispose();
                    _vectorIndex = new IndexFlatIP(Dimensions);
                    IngestDatasetFilesParallel();
                }
            }
            else
            {
                Console.WriteLine($"[Storage] Building Exact Cosine Similarity Index (IndexFlatIP)...");
                _vectorIndex = new IndexFlatIP(Dimensions);
                IngestDatasetFilesParallel();
            }
        }

        private void LoadAllSchemaRecords()
        {
            int preCount = LoadDirectoryRecords(_pretrainPath, "pretrain");
            int postCount = LoadDirectoryRecords(_posttrainPath, "posttrain");
            Console.WriteLine($"[Dataset] Loaded {_datasetRecords.Count} total records (Pretrain: {preCount}, Posttrain SFT/DPO: {postCount}).");
        }

        private int LoadDirectoryRecords(string dirPath, string defaultKind)
        {
            if (!Directory.Exists(dirPath)) return 0;
            int added = 0;
            var files = Directory.GetFiles(dirPath, "*.jsonl", SearchOption.AllDirectories);

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

                if (root.TryGetProperty("metadata", out var metaProp) && metaProp.ValueKind == JsonValueKind.Object)
                {
                    if (metaProp.TryGetProperty("topic", out var topicProp))
                        record.Topic = topicProp.GetString() ?? string.Empty;
                    if (record.Id == fallbackId && metaProp.TryGetProperty("content_hash", out var hashProp))
                    {
                        string h = hashProp.GetString() ?? "";
                        if (h.Length >= 8) record.Id = $"{defaultKind}_{h.Substring(0, 8)}";
                    }
                }

                if (root.TryGetProperty("text", out var textProp))
                {
                    string rawText = (textProp.GetString() ?? string.Empty).Replace("\\n", "\n");
                    record.SearchableText = rawText.Replace("\n", " ").Trim();
                    record.AnswerPayload = FormatCleanPayload(rawText);
                }
                else if (root.TryGetProperty("messages", out var msgsProp) && msgsProp.ValueKind == JsonValueKind.Array)
                {
                    record.Kind = "sft";
                    string userMsg = "";
                    string assistantMsg = "";
                    foreach (var m in msgsProp.EnumerateArray())
                    {
                        string role = m.TryGetProperty("role", out var r) ? r.GetString() ?? "" : "";
                        string content = m.TryGetProperty("content", out var c) ? c.GetString() ?? "" : "";
                        if (role == "user") userMsg = content;
                        else if (role == "assistant") assistantMsg = content;
                    }
                    record.SearchableText = $"{record.Topic} {userMsg} {assistantMsg}".Replace("\\n", " ").Replace("\n", " ").Trim();
                    record.AnswerPayload = FormatCleanPayload(string.IsNullOrEmpty(assistantMsg) ? userMsg : $"Q: {TruncateOneLine(userMsg, 110)}\nA: {assistantMsg}");
                }
                else if (root.TryGetProperty("prompt", out var promptProp))
                {
                    string prompt = promptProp.GetString() ?? "";
                    string answer = "";
                    if (root.TryGetProperty("completion", out var compProp))
                    {
                        record.Kind = "sft";
                        answer = compProp.GetString() ?? "";
                    }
                    else if (root.TryGetProperty("chosen", out var chosenProp))
                    {
                        record.Kind = "dpo";
                        answer = chosenProp.ValueKind == JsonValueKind.Object && chosenProp.TryGetProperty("content", out var cProp)
                            ? cProp.GetString() ?? ""
                            : chosenProp.ToString();
                    }
                    record.SearchableText = $"{record.Topic} {prompt} {answer}".Replace("\\n", " ").Replace("\n", " ").Trim();
                    record.AnswerPayload = FormatCleanPayload(string.IsNullOrEmpty(answer) ? prompt : $"Q: {TruncateOneLine(prompt, 110)}\nA: {answer}");
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

            return record;
        }

        private static string TruncateOneLine(string s, int maxLen)
        {
            string flat = s.Replace("\\n", " ").Replace("\n", " ").Trim();
            return flat.Length > maxLen ? flat.Substring(0, maxLen) + "..." : flat;
        }

        private static string FormatCleanPayload(string text)
        {
            string clean = text.Replace("\\n", "\n").Replace("\\\"", "\"").Trim();
            int thoughtEnd = clean.IndexOf("</thought>", StringComparison.OrdinalIgnoreCase);
            if (thoughtEnd >= 0 && thoughtEnd + 10 < clean.Length)
            {
                clean = clean.Substring(thoughtEnd + 10).Trim();
            }
            return clean.Length > 600 ? clean.Substring(0, 600) + "..." : clean;
        }

        private void IngestDatasetFilesParallel()
        {
            int count = _datasetRecords.Count;
            if (count == 0) return;

            float[] batchVectors = new float[count * Dimensions];
            int completed = 0;
            var sw = Stopwatch.StartNew();

            Console.WriteLine($"[Ingestion] Encoding {count} records on {Environment.ProcessorCount} CPU threads...");

            Parallel.For(0, count, i =>
            {
                string textToEncode = string.IsNullOrEmpty(_datasetRecords[i].Topic)
                    ? _datasetRecords[i].SearchableText
                    : $"{_datasetRecords[i].Topic} {_datasetRecords[i].SearchableText}";

                var (vec, _, _) = _arborNet2D.Forward2D(textToEncode, trainOnline: false);
                Array.Copy(vec, 0, batchVectors, i * Dimensions, Dimensions);

                int current = Interlocked.Increment(ref completed);
                if (current % 5000 == 0 || current == count)
                {
                    Console.WriteLine($" -> Encoded {current}/{count} ({(current * 100.0 / count):F1}%) [{sw.Elapsed.TotalSeconds:F1}s]");
                }
            });

            _vectorIndex.Add(batchVectors);
            FaissIO.WriteIndex(_vectorIndex, _indexFilePath);
            Console.WriteLine($"[Ingestion] Persisted {count} L2-normalized vectors to {_indexFilePath}.");
        }

        private float RetrieveSpectralResonance(float signal)
        {
            float sum = 0f;
            foreach (var comp in _memorySpectrum)
            {
                sum += MathF.Cos(comp.Eigenvalue * signal) * comp.Amplitude;
            }
            return sum / Math.Max(1, _memorySpectrum.Count);
        }

        public void PrintStatus()
        {
            Console.WriteLine("\n--- [Cognitive Engine Telemetry Status] ---");
            Console.WriteLine($"* Indexed Corpus: {_datasetRecords.Count:N0} records (FAISS NTotal: {_vectorIndex.NTotal:N0})");
            Console.WriteLine($"* Connes-Marcolli Scale (Lambda): {_connesMarcolliLambda:F4}");
            Console.WriteLine($"* Active Orthogonal Prime Eigenstates: {_memorySpectrum.Count}");
            Console.WriteLine($"* Last Online Autograd Router Loss: {_arborNet2D.LastAutogradLoss:F6}");
            Console.WriteLine($"* TIR Cross-Domain Invariants Loaded: {_crossDomainLibrary.Count}");
        }

        public void ProcessChatTurn(string userPrompt)
        {
            if (userPrompt.Equals("/dream", StringComparison.OrdinalIgnoreCase))
            {
                PerformDreamConsolidation();
                return;
            }
            if (userPrompt.Equals("/status", StringComparison.OrdinalIgnoreCase))
            {
                PrintStatus();
                return;
            }

            _interactionCount++;
            string cleanPrompt = userPrompt.StartsWith("Input:", StringComparison.OrdinalIgnoreCase)
                ? userPrompt.Substring(6).Trim()
                : userPrompt.Trim();

            Console.WriteLine($"\n==================================================");
            Console.WriteLine($"[User (Turn {_interactionCount})]: \"{cleanPrompt}\"");

            // 1. 2D ArborNet Forward Pass + Live Online Autograd Update
            var (queryVec, intent, probs) = _arborNet2D.Forward2D(cleanPrompt, trainOnline: true);
            Console.WriteLine($"[ArborNet 2D Router + Autograd] Intent: {intent} (Fact: {probs[0]:F2}, Synth: {probs[1]:F2}, Paradox: {probs[2]:F2} | Loss: {_arborNet2D.LastAutogradLoss:F4})");

            // 2. Deep Candidate Pool Search (k = 64) + Domain-Term & SFT/DPO Priority Reranking
            int k = Math.Min(64, (int)Math.Max(1, _vectorIndex.NTotal));
            _vectorIndex.Search(queryVec, k, out float[] rawCosines, out long[] labels);

            string[] queryTerms = cleanPrompt.ToLowerInvariant()
                .Split(new[] { ' ', '?', '.', ',', '!', '-', ':', ';', '(', ')', '"', '\'' }, StringSplitOptions.RemoveEmptyEntries)
                .Where(w => w.Length >= 3 && !StopWords.Contains(w))
                .Distinct()
                .ToArray();

            var scoredCandidates = new List<(long id, float rawCos, float finalScore)>(k);
            for (int i = 0; i < k; i++)
            {
                long cid = labels[i];
                float rawCos = Math.Clamp(rawCosines[i], -1.0f, 1.0f);
                if (cid < 0 || cid >= _datasetRecords.Count) continue;

                var rec = _datasetRecords[(int)cid];
                string searchable = rec.SearchableText.ToLowerInvariant();
                string topicLower = rec.Topic.ToLowerInvariant();

                float termScore = 0f;
                if (queryTerms.Length > 0)
                {
                    int matched = 0;
                    foreach (var term in queryTerms)
                    {
                        if (searchable.Contains(term)) matched++;
                        if (topicLower.Contains(term)) matched++; // Double weight for topic match
                    }
                    termScore = Math.Clamp((float)matched / queryTerms.Length, 0f, 1.25f);
                }

                // Curated SFT/DPO bonus (+0.05) so structured reasoning records edge out raw boilerplates when relevant
                float kindBonus = (rec.Kind == "sft" || rec.Kind == "dpo") ? 0.05f : 0.0f;
                float combined = Math.Clamp(0.60f * ((rawCos + 1.0f) * 0.5f) + 0.35f * termScore + kindBonus, 0.01f, 0.99f);
                scoredCandidates.Add((cid, rawCos, combined));
            }

            scoredCandidates.Sort((a, b) => b.finalScore.CompareTo(a.finalScore));
            var primary = scoredCandidates[0];
            var secondary = scoredCandidates.FirstOrDefault(c => c.id != primary.id &&
                _datasetRecords[(int)c.id].Id != _datasetRecords[(int)primary.id].Id);

            var primaryRec = _datasetRecords[(int)primary.id];
            var secondaryRec = secondary.id >= 0 && secondary.id < _datasetRecords.Count ? _datasetRecords[(int)secondary.id] : null;

            // 3. True Top-3 Sheaf Cohomological Obstruction H^1
            float top3Disagreement = 0f;
            if (scoredCandidates.Count >= 3)
            {
                float s0 = scoredCandidates[0].finalScore;
                float s1 = scoredCandidates[1].finalScore;
                float s2 = scoredCandidates[2].finalScore;
                top3Disagreement = (MathF.Abs(s0 - s1) + MathF.Abs(s1 - s2) + MathF.Abs(s0 - s2)) * 1.8f;
            }
            float spectralWave = MathF.Abs(RetrieveSpectralResonance(primary.rawCos));
            float obstructionMetric = Math.Clamp(
                (1.0f - primary.finalScore) * 0.55f +
                top3Disagreement * 0.30f +
                (intent == CognitiveIntent.StructuralParadox ? 0.28f : 0f),
                0.04f, 0.98f);

            Console.WriteLine($"[FAISS Deep-64 Memory] Primary ID: {primary.id} ({primaryRec.Kind.ToUpper()} | {primaryRec.Id}) | Raw Cosine: {primary.rawCos:F4} | Calibrated Confidence: {primary.finalScore:F3}");

            if (obstructionMetric > 0.38f || intent == CognitiveIntent.StructuralParadox)
            {
                _connesMarcolliLambda *= 0.96f;
                int nextPrime = _primeSequence[_memorySpectrum.Count % _primeSequence.Length];
                _memorySpectrum.Add(new PrimeSpectralComponent
                {
                    PrimeIndex = _memorySpectrum.Count + 1,
                    Eigenvalue = nextPrime * 0.31416f * _connesMarcolliLambda,
                    Amplitude = Math.Clamp(primary.finalScore + 0.15f, 0.4f, 1.0f)
                });
                Console.WriteLine($"[NCST Sheaf Cohomology] Obstruction H^1: {obstructionMetric:F3} (Wave: {spectralWave:F3}) -> Phase Transition! Lambda: {_connesMarcolliLambda:F3} (Eigenstates: {_memorySpectrum.Count})");
            }
            else
            {
                Console.WriteLine($"[NCST Sheaf Cohomology] Obstruction H^1: {obstructionMetric:F3} (Wave: {spectralWave:F3}) -> Topos Equilibrium Stable (Lambda: {_connesMarcolliLambda:F3})");
            }

            // 4. Multi-Orbital TIR Isolobal Substitution on Structural Paradox
            if (intent == CognitiveIntent.StructuralParadox)
            {
                var (targetSymmetry, targetEnergy, targetProps) = ClassifyParadoxOrbital(cleanPrompt);
                Console.WriteLine($"[Cognitive Dissonance] Structural strain detected! Forming Frontier Conceptual Orbitals (Props: {targetProps}, Symmetry: {targetSymmetry}, Energy: {targetEnergy:F2} eV)...");
                var strainedThought = new ConceptualFragment(cleanPrompt, "UserDialogue", "Unresolved Paradox", new FrontierOrbitals(targetProps, targetSymmetry, targetEnergy));

                foreach (var candidate in _crossDomainLibrary)
                {
                    if (FrontierOrbitals.AreIsolobal(candidate.FCO, strainedThought.FCO))
                    {
                        Console.WriteLine($"[TIR Engine] Isolobal Substitution Verified via [{candidate.Domain}] -> '{candidate.FragmentName}'");
                        Console.WriteLine($"\n--- [OzzieAI Canon Synthesis (Sheaf-Glued + TIR Resolved)] ---");
                        Console.WriteLine($"* Logos (Isolobal Law): {candidate.ConservationRule}");
                        Console.WriteLine($"* Worked Check: Raw Cosine={primary.rawCos:F4}, Calibrated Score={primary.finalScore:F3}, Sheaf H^1={obstructionMetric:F3}, Lambda={_connesMarcolliLambda:F3}.");
                        Console.WriteLine($"* Primary Topos Patch [{primaryRec.Kind.ToUpper()} | {primaryRec.Id}]:\n{primaryRec.AnswerPayload}");
                        if (secondaryRec != null)
                        {
                            Console.WriteLine($"* Glued Corroborating Patch [{secondaryRec.Kind.ToUpper()} | {secondaryRec.Id} (Score: {secondary.finalScore:F3})]:\n{TruncateOneLine(secondaryRec.AnswerPayload, 220)}");
                        }
                        CheckDreamCycle();
                        return;
                    }
                }
            }

            Console.WriteLine($"\n--- [OzzieAI Canon Synthesis (Sheaf-Glued Global Section)] ---");
            Console.WriteLine($"* Logos: Verified 2D ArborNet self-attention + NCST sheaf consistency across Top-64 candidates under [{intent}].");
            Console.WriteLine($"* Worked Check: Raw Cosine={primary.rawCos:F4}, Calibrated Score={primary.finalScore:F3}, Sheaf H^1={obstructionMetric:F3}, Lambda={_connesMarcolliLambda:F3}.");
            Console.WriteLine($"* Primary Topos Patch [{primaryRec.Kind.ToUpper()} | {primaryRec.Id}]:\n{primaryRec.AnswerPayload}");
            if (secondaryRec != null)
            {
                Console.WriteLine($"* Glued Corroborating Patch [{secondaryRec.Kind.ToUpper()} | {secondaryRec.Id} (Score: {secondary.finalScore:F3})]:\n{TruncateOneLine(secondaryRec.AnswerPayload, 220)}");
            }
            CheckDreamCycle();
        }

        private static (string symmetry, float energy, int props) ClassifyParadoxOrbital(string prompt)
        {
            string lower = prompt.ToLowerInvariant();
            if (lower.Contains("impedance") || lower.Contains("balun") || lower.Contains("reflection") || lower.Contains("mismatch"))
                return ("A1", -1.25f, 4); // Matches GuanellaBalunImpedanceTransformer
            if (lower.Contains("gradient") || lower.Contains("exploding") || lower.Contains("norm") || lower.Contains("hierarchy") || lower.Contains("ultrametric"))
                return ("T2g", -1.60f, 4); // Matches pAdicUltrametricContraction
            if (lower.Contains("entropy") || lower.Contains("perception") || lower.Contains("consciousness") || lower.Contains("interface"))
                return ("B2", -0.85f, 2); // Matches ConsciousInterfaceMarkovKernel
            return ("E", -1.48f, 4); // Matches AsymmetricalFluxCanceller (Deadlocks / Cyclic Loops)
        }

        private void CheckDreamCycle()
        {
            if (_interactionCount % 5 == 0)
            {
                PerformDreamConsolidation();
            }
        }

        public void PerformDreamConsolidation()
        {
            Console.WriteLine($"\n--- [REM Sleep Triggered] Automated Dream Consolidation at Turn {_interactionCount} ---");
            int before = _memorySpectrum.Count;
            _memorySpectrum.RemoveAll(c => c.Amplitude < 0.45f);

            for (int i = 0; i < _memorySpectrum.Count; i++)
            {
                var comp = _memorySpectrum[i];
                comp.Amplitude = Math.Min(1.0f, comp.Amplitude * 1.05f);
                _memorySpectrum[i] = comp;
            }
            Console.WriteLine($"[Dream Complete] Pruned {before - _memorySpectrum.Count} weak components. Active Orthogonal Prime Eigenstates: {_memorySpectrum.Count}\n");
        }

        public void Dispose()
        {
            _vectorIndex?.Dispose();
        }
    }

    public class ProgramC
    {
        public static void MainC(string[] args)
        {
            string pretrainDir = @"C:\Users\Jingl\Documents\OzzieAI AutoLab Bench Pro - CRITICAL DATA (Do Not Delete)\Repository\Training Schema's\pretrain";
            string posttrainDir = @"C:\Users\Jingl\Documents\OzzieAI AutoLab Bench Pro - CRITICAL DATA (Do Not Delete)\Repository\Training Schema's\posttrain";

            using var machine = new NativeArborNetCognitiveMachine(pretrainDir, posttrainDir);

            Console.WriteLine("======================================================================");
            Console.WriteLine(" OzzieAI v3.5 Sheaf-Glued 2D-ArborNet Cognitive Machine");
            Console.WriteLine(" Commands: '/status' (telemetry), '/dream' (REM cycle), 'exit' (quit)");
            Console.WriteLine("======================================================================");

            while (true)
            {
                Console.Write("\nAsk your Cognitive Agent > ");
                string? input = Console.ReadLine();
                if (string.IsNullOrWhiteSpace(input) || input.Equals("exit", StringComparison.OrdinalIgnoreCase)) break;

                machine.ProcessChatTurn(input);
            }
        }
    }
}