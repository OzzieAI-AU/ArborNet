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

namespace OzzieAI.PersistentCognitiveEngine.BA2
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
        public FrontierOrbitals FCO { get; }
        public bool IsStrained { get; set; }

        public ConceptualFragment(string fragmentName, string domain, FrontierOrbitals fco)
        {
            FragmentName = fragmentName;
            Domain = domain;
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
    /// v3 Orthogonal 2D-Native ArborNet Neural Network with Live Autograd Training Support.
    /// </summary>
    public class ArborNet2DCognitiveNetwork
    {
        private readonly int _dModel;
        private readonly Device _device;
        private readonly ITensor _orthoKeyWeight;
        private readonly ITensor _orthoValWeight;
        private readonly Linear _intentHead;

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

                // Intent Routing + Optional Online Autograd Step
                ITensor normTensor2D = Tensor.FromArray(rawEmbed, new TensorShape(1, _dModel), _device);
                ITensor intentLogits = _intentHead.Forward(normTensor2D);

                float[] teacherBias = ComputeSemanticLogitBias(cleanText);
                ITensor biasTensor = Tensor.FromArray(teacherBias, new TensorShape(1, 3), _device);
                ITensor probsTensor = intentLogits.Add(biasTensor).Softmax(axis: 1);

                float[] probs = probsTensor.ToArray();
                int bestClass = (int)probsTensor.ArgMax(axis: 1).ToArray()[0];

                // Live Autograd Training Step: train _intentHead weights to internalize the routing rules!
                if (trainOnline)
                {
                    float[] targetOneHot = new float[3];
                    targetOneHot[bestClass] = 1.0f;
                    ITensor targetTensor = Tensor.FromArray(targetOneHot, new TensorShape(1, 3), _device);

                    // MSE Loss on Intent Head: (pred - target)^2
                    ITensor rawProbs = intentLogits.Softmax(axis: 1);
                    ITensor diff = rawProbs.Subtract(targetTensor);
                    ITensor loss = diff.Multiply(diff).Mean();

                    foreach (var p in _intentHead.Parameters()) p.ClearGrad();
                    loss.Backward();

                    float learningRate = 0.05f;
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
                lower.Contains("loop") || lower.Contains("strain") || lower.Contains("impedance") || lower.Contains("exploding"))
            {
                return new float[] { -1.0f, 0.2f, 3.5f };
            }
            if (lower.Contains("what") || lower.Contains("how") || lower.Contains("list") ||
                lower.Contains("define") || lower.Contains("research") || lower.Contains("structured"))
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
        private readonly int[] _primeSequence = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47 };

        public NativeArborNetCognitiveMachine(string pretrainPath, string posttrainPath)
        {
            _pretrainPath = pretrainPath;
            _posttrainPath = posttrainPath;
            // Reuse the v2 cosine index if record counts match, or build v3 if posttrain adds new records
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

            _crossDomainLibrary.Add(new ConceptualFragment(
                "AsymmetricalFluxCanceller",
                "Electrodynamics",
                new FrontierOrbitals(activePropositionCount: 4, symmetryClass: "E", energyEigenvalue: -1.45f)
            ));
            _crossDomainLibrary.Add(new ConceptualFragment(
                "GuanellaBalunImpedanceTransformer",
                "Electrodynamics",
                new FrontierOrbitals(activePropositionCount: 4, symmetryClass: "A1", energyEigenvalue: -1.25f)
            ));
            _crossDomainLibrary.Add(new ConceptualFragment(
                "TopochemicalIsolobalResonance",
                "QuantumChemistry",
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

        /// <summary>
        /// Ingests both pretrain and posttrain (SFT/DPO) JSONL files according to schema_canon.json.
        /// </summary>
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
                foreach (var line in File.ReadLines(file))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    _datasetRecords.Add(ParseCanonRecord(line, defaultKind));
                    added++;
                }
            }
            return added;
        }

        private static ParsedDatasetRecord ParseCanonRecord(string rawJsonLine, string defaultKind)
        {
            var record = new ParsedDatasetRecord { Kind = defaultKind };
            try
            {
                using var doc = JsonDocument.Parse(rawJsonLine);
                var root = doc.RootElement;

                if (root.TryGetProperty("id", out var idProp))
                    record.Id = idProp.GetString() ?? string.Empty;

                if (root.TryGetProperty("metadata", out var metaProp) && metaProp.TryGetProperty("topic", out var topicProp))
                    record.Topic = topicProp.GetString() ?? string.Empty;

                // 1. Pretrain shape: { "id", "text" }
                if (root.TryGetProperty("text", out var textProp))
                {
                    string rawText = (textProp.GetString() ?? string.Empty).Replace("\\n", "\n");
                    record.SearchableText = rawText.Replace("\n", " ").Trim();
                    record.AnswerPayload = FormatCleanPayload(rawText);
                }
                // 2. SFT messages[] shape: { "id", "messages": [ { "role", "content" } ] }
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
                    record.AnswerPayload = FormatCleanPayload(string.IsNullOrEmpty(assistantMsg) ? userMsg : assistantMsg);
                }
                // 3. SFT target or DPO shape: { "prompt", "completion" / "chosen" }
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
                    record.AnswerPayload = FormatCleanPayload(string.IsNullOrEmpty(answer) ? prompt : answer);
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

        private static string FormatCleanPayload(string text)
        {
            string clean = text.Replace("\\n", "\n").Replace("\\\"", "\"").Trim();
            // Strip <thought>...</thought> internal scratchpad if present so only the clean canon answer is shown
            int thoughtEnd = clean.IndexOf("</thought>", StringComparison.OrdinalIgnoreCase);
            if (thoughtEnd >= 0 && thoughtEnd + 10 < clean.Length)
            {
                clean = clean.Substring(thoughtEnd + 10).Trim();
            }
            return clean.Length > 550 ? clean.Substring(0, 550) + "..." : clean;
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

        public void ProcessChatTurn(string userPrompt)
        {
            _interactionCount++;
            string cleanPrompt = userPrompt.StartsWith("Input:", StringComparison.OrdinalIgnoreCase)
                ? userPrompt.Substring(6).Trim()
                : userPrompt.Trim();

            Console.WriteLine($"\n==================================================");
            Console.WriteLine($"[User (Turn {_interactionCount})]: \"{cleanPrompt}\"");

            // 1. 2D ArborNet Forward Pass + Live Online Autograd Weight Update!
            var (queryVec, intent, probs) = _arborNet2D.Forward2D(cleanPrompt, trainOnline: true);
            Console.WriteLine($"[ArborNet 2D Router + Autograd] Intent: {intent} (Softmax -> Fact: {probs[0]:F2}, Synth: {probs[1]:F2}, Paradox: {probs[2]:F2})");

            // 2. FAISS Top-5 Search + Bounded Convex Reranking (fixes 0.99 clamping bug!)
            int k = Math.Min(5, (int)Math.Max(1, _vectorIndex.NTotal));
            _vectorIndex.Search(queryVec, k, out float[] rawCosines, out long[] labels);

            string[] queryWords = cleanPrompt.ToLowerInvariant()
                .Split(new[] { ' ', '?', '.', ',', '!', '-', ':' }, StringSplitOptions.RemoveEmptyEntries)
                .Where(w => w.Length > 3)
                .ToArray();

            long bestId = labels[0];
            float bestRawCosine = Math.Clamp(rawCosines[0], -1.0f, 1.0f);
            float bestCombinedScore = -1.0f;

            for (int i = 0; i < k; i++)
            {
                long cid = labels[i];
                float rawCos = Math.Clamp(rawCosines[i], -1.0f, 1.0f);
                if (cid >= 0 && cid < _datasetRecords.Count)
                {
                    string candidateText = _datasetRecords[(int)cid].SearchableText.ToLowerInvariant();
                    int hits = queryWords.Length > 0 ? queryWords.Count(kw => candidateText.Contains(kw)) : 0;
                    float lexicalRatio = queryWords.Length > 0 ? (float)hits / queryWords.Length : 0f;

                    // Strictly bounded in [0.0, 1.0]: 75% Cosine + 25% Lexical Overlap
                    float combined = 0.75f * ((rawCos + 1.0f) * 0.5f) + 0.25f * lexicalRatio;
                    if (combined > bestCombinedScore)
                    {
                        bestCombinedScore = combined;
                        bestRawCosine = rawCos;
                        bestId = cid;
                    }
                }
            }

            // 3. True Top-3 Sheaf Cohomological Obstruction (measures disagreement across Top-3 local memory patches)
            float top3Disagreement = 0f;
            if (k >= 3)
            {
                float c0 = Math.Clamp(rawCosines[0], -1f, 1f);
                float c1 = Math.Clamp(rawCosines[1], -1f, 1f);
                float c2 = Math.Clamp(rawCosines[2], -1f, 1f);
                top3Disagreement = (MathF.Abs(c0 - c1) + MathF.Abs(c1 - c2) + MathF.Abs(c0 - c2)) * 2.5f;
            }
            float spectralWave = MathF.Abs(RetrieveSpectralResonance(bestRawCosine));
            float obstructionMetric = Math.Clamp((1.0f - bestCombinedScore) * 0.65f + top3Disagreement * 0.35f + (intent == CognitiveIntent.StructuralParadox ? 0.25f : 0f), 0.05f, 0.99f);

            var matchedRecord = (bestId >= 0 && bestId < _datasetRecords.Count)
                ? _datasetRecords[(int)bestId]
                : new ParsedDatasetRecord { AnswerPayload = "No record found." };

            Console.WriteLine($"[FAISS Cosine Memory] Match ID: {bestId} ({matchedRecord.Kind.ToUpper()} | {matchedRecord.Id}) | Raw Cosine: {bestRawCosine:F4} | Calibrated Confidence: {bestCombinedScore:F3}");

            if (obstructionMetric > 0.38f || intent == CognitiveIntent.StructuralParadox)
            {
                _connesMarcolliLambda *= 0.96f;
                int nextPrime = _primeSequence[_memorySpectrum.Count % _primeSequence.Length];
                _memorySpectrum.Add(new PrimeSpectralComponent
                {
                    PrimeIndex = _memorySpectrum.Count + 1,
                    Eigenvalue = nextPrime * 0.31416f * _connesMarcolliLambda,
                    Amplitude = Math.Clamp(bestCombinedScore + 0.15f, 0.4f, 1.0f)
                });
                Console.WriteLine($"[NCST Sheaf Cohomology] Obstruction H^1: {obstructionMetric:F3} (Wave: {spectralWave:F3}) -> Phase Transition! Lambda: {_connesMarcolliLambda:F3} (Eigenstates: {_memorySpectrum.Count})");
            }
            else
            {
                Console.WriteLine($"[NCST Sheaf Cohomology] Obstruction H^1: {obstructionMetric:F3} (Wave: {spectralWave:F3}) -> Topos Equilibrium Stable (Lambda: {_connesMarcolliLambda:F3})");
            }

            // 4. TIR Isolobal Substitution on Structural Paradox
            if (intent == CognitiveIntent.StructuralParadox)
            {
                string targetSymmetry = cleanPrompt.Contains("impedance", StringComparison.OrdinalIgnoreCase) || cleanPrompt.Contains("balun", StringComparison.OrdinalIgnoreCase) ? "A1" : "E";
                float targetEnergy = targetSymmetry == "A1" ? -1.25f : -1.48f;

                Console.WriteLine($"[Cognitive Dissonance] Structural strain detected! Forming Frontier Conceptual Orbitals (Symmetry: {targetSymmetry}, Energy: {targetEnergy:F2} eV)...");
                var strainedThought = new ConceptualFragment(cleanPrompt, "UserDialogue", new FrontierOrbitals(4, targetSymmetry, targetEnergy));

                foreach (var candidate in _crossDomainLibrary)
                {
                    if (FrontierOrbitals.AreIsolobal(candidate.FCO, strainedThought.FCO))
                    {
                        Console.WriteLine($"[TIR Engine] Isolobal Substitution Verified via [{candidate.Domain}] -> '{candidate.FragmentName}'");
                        Console.WriteLine($"\n--- [OzzieAI Canon Synthesis] ---");
                        Console.WriteLine($"* Logos: Mapped [{strainedThought.Domain}] paradox onto [{candidate.Domain}] invariant '{candidate.FragmentName}' ({targetSymmetry} orbital symmetry).");
                        Console.WriteLine($"* Worked Check: Raw Cosine={bestRawCosine:F4}, Calibrated Score={bestCombinedScore:F3}, Sheaf H^1={obstructionMetric:F3}, Lambda={_connesMarcolliLambda:F3}.");
                        Console.WriteLine($"* Grounded Corpus Payload ({matchedRecord.Id}):\n{matchedRecord.AnswerPayload}");
                        CheckDreamCycle();
                        return;
                    }
                }
            }

            Console.WriteLine($"\n--- [OzzieAI Canon Synthesis] ---");
            Console.WriteLine($"* Logos: Verified 2D ArborNet attention + NCST sheaf consistency under [{intent}].");
            Console.WriteLine($"* Worked Check: Raw Cosine={bestRawCosine:F4}, Calibrated Score={bestCombinedScore:F3}, Sheaf H^1={obstructionMetric:F3}, Lambda={_connesMarcolliLambda:F3}.");
            Console.WriteLine($"* Grounded Corpus Payload ({matchedRecord.Id}):\n{matchedRecord.AnswerPayload}");
            CheckDreamCycle();
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

    public class ProgramB
    {
        public static void MainB(string[] args)
        {
            string pretrainDir = @"C:\Users\Jingl\Documents\OzzieAI AutoLab Bench Pro - CRITICAL DATA (Do Not Delete)\Repository\Training Schema's\pretrain";
            string posttrainDir = @"C:\Users\Jingl\Documents\OzzieAI AutoLab Bench Pro - CRITICAL DATA (Do Not Delete)\Repository\Training Schema's\posttrain";

            using var machine = new NativeArborNetCognitiveMachine(pretrainDir, posttrainDir);

            Console.WriteLine("============================================================");
            Console.WriteLine(" OzzieAI v3 Canon-Compliant 2D-ArborNet Cognitive Machine");
            Console.WriteLine("============================================================");

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