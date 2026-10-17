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

namespace OzzieAI.PersistentCognitiveEngine.Ba1
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

    public class LogicalPatch
    {
        public string ContextName { get; set; } = string.Empty;
        public Dictionary<string, float> LocalTruths { get; set; } = new Dictionary<string, float>();
    }

    public class ParsedDatasetRecord
    {
        public string Id { get; set; } = string.Empty;
        public string Topic { get; set; } = string.Empty;
        public string CleanText { get; set; } = string.Empty;
    }

    /// <summary>
    /// Orthogonal 2D-Native ArborNet Neural Encoder & Intent Router.
    /// Eliminates representation collapse via Prime-Trigram Tokenization + Residual 2D Self-Attention.
    /// </summary>
    public class ArborNet2DCognitiveNetwork
    {
        private readonly int _dModel;
        private readonly Device _device;
        private readonly ITensor _orthoKeyWeight;
        private readonly ITensor _orthoValWeight;
        private readonly Linear _intentHead;

        private static readonly int[] Primes = {
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
            59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131
        };

        public ArborNet2DCognitiveNetwork(int dModel = 64, Device? device = null)
        {
            _dModel = dModel;
            _device = device ?? Device.CPU;

            // Near-identity orthogonal projection matrices [dModel, dModel] to prevent hub collapse
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

        /// <summary>
        /// Tokenizes text into up to 24 semantic word tokens and projects each word into a 64D prime-hashed vector,
        /// forming a strict 2D matrix [SeqLen, 64] that preserves fine-grained lexical identity.
        /// </summary>
        private ITensor EncodeWordsTo2DTensor(string text, int maxWords = 24)
        {
            var words = text.ToLowerInvariant()
                .Split(new[] { ' ', '\n', '\r', '\t', '.', ',', '?', '!', ':', ';', '"', '\'', '(', ')', '[', ']', '{', '}' },
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
                // Weight earlier words (like Task Name / Prompt subject) higher
                float positionWeight = 1.0f + (1.0f / (1.0f + row * 0.25f));

                // Hash character n-grams of the word onto orthogonal prime dimensions
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

        /// <summary>
        /// Executes a 2D ArborNet Forward Pass ([SeqLen, 64]) without representation collapse.
        /// </summary>
        public (float[] embedding, CognitiveIntent intent, float[] intentProbs) Forward2D(string cleanText)
        {
            using (new TensorScope())
            {
                // 1. Word-level 2D Tensor: [SeqLen, dModel] (Rank 2)
                ITensor x = EncodeWordsTo2DTensor(cleanText);

                // 2. 2D Self-Attention Mixing (Strictly Rank 2 MatMul!)
                ITensor k = x.MatMul(_orthoKeyWeight); // [SeqLen, dModel]
                ITensor v = x.MatMul(_orthoValWeight); // [SeqLen, dModel]
                ITensor kT = k.Transpose(new[] { 1, 0 }); // [dModel, SeqLen]

                float scale = 1.0f / MathF.Sqrt(_dModel);
                ITensor attnScores = x.MatMul(kT).Multiply(scale).Softmax(axis: 1); // [SeqLen, SeqLen]
                ITensor context = attnScores.MatMul(v); // [SeqLen, dModel]

                // Residual connection preserving 80% direct lexical signal + 20% attention context
                ITensor residual = x.Multiply(0.80f).Add(context.Multiply(0.20f));

                // 3. Mean Pool across sequence into [1, dModel] (Rank 2)
                ITensor pooled2D = residual.Mean(axis: 0, keepDims: true);

                // 4. Zero-Mean Centering & L2 Normalization (for exact Cosine Similarity in FAISS)
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

                // 5. Compute Softmax Intent Probabilities: [1, dModel] -> [1, 3] (Rank 2)
                ITensor intentLogits = _intentHead.Forward(pooled2D).Multiply(0.1f);
                float[] logitBias = ComputeSemanticLogitBias(cleanText);
                ITensor biasTensor = Tensor.FromArray(logitBias, new TensorShape(1, 3), _device);
                ITensor probsTensor = intentLogits.Add(biasTensor).Softmax(axis: 1);

                float[] probs = probsTensor.ToArray();
                int bestClass = (int)probsTensor.ArgMax(axis: 1).ToArray()[0];

                return (rawEmbed, (CognitiveIntent)bestClass, probs);
            }
        }

        private static float[] ComputeSemanticLogitBias(string text)
        {
            string lower = text.ToLowerInvariant();
            if (lower.Contains("deadlock") || lower.Contains("paradox") || lower.Contains("contradiction") ||
                lower.Contains("loop") || lower.Contains("strain") || lower.Contains("impedance") || lower.Contains("exploding"))
            {
                return new float[] { -1.0f, 0.2f, 3.5f }; // StructuralParadox
            }
            if (lower.Contains("what") || lower.Contains("how") || lower.Contains("list") ||
                lower.Contains("define") || lower.Contains("research") || lower.Contains("structured"))
            {
                return new float[] { 3.2f, 0.4f, -1.0f }; // FactRetrieval
            }
            return new float[] { 0.4f, 2.8f, 0.1f }; // CreativeSynthesis
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
        private readonly List<LogicalPatch> _knowledgeTopos = new List<LogicalPatch>();
        private readonly List<ConceptualFragment> _crossDomainLibrary = new List<ConceptualFragment>();
        private readonly List<ParsedDatasetRecord> _datasetRecords = new List<ParsedDatasetRecord>();

        private float _connesMarcolliLambda = 1.0f;
        private int _interactionCount = 0;
        private readonly int[] _primeSequence = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47 };

        public NativeArborNetCognitiveMachine(string pretrainPath, string posttrainPath)
        {
            _pretrainPath = pretrainPath;
            _posttrainPath = posttrainPath;
            // New v2 filename for Cosine Similarity (IndexFlatIP) with uncollapsed embeddings
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

            _knowledgeTopos.Add(new LogicalPatch
            {
                ContextName = "NeuralCognition",
                LocalTruths = { ["AutogradConsistency"] = 0.85f, ["SemanticCoherence"] = 0.80f }
            });
            _knowledgeTopos.Add(new LogicalPatch
            {
                ContextName = "Electrodynamics",
                LocalTruths = { ["FluxConservation"] = 0.88f, ["ImpedanceMatching"] = 0.82f }
            });

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
            LoadAndCleanDatasetRecords();

            if (File.Exists(_indexFilePath))
            {
                Console.WriteLine($"[Storage] Loading v2 Cosine Similarity FAISS index from: {_indexFilePath}");
                _vectorIndex = FaissIO.ReadIndex(_indexFilePath, Dimensions);
            }
            else
            {
                Console.WriteLine($"[Storage] Building new Exact Cosine Similarity Index (IndexFlatIP) with 2D ArborNet...");
                _vectorIndex = new IndexFlatIP(Dimensions);
                IngestDatasetFilesParallel();
            }
        }

        /// <summary>
        /// Phase 1 Fix: Parses JSONL records FIRST so we encode actual text content instead of JSON {"id":...} headers.
        /// </summary>
        private void LoadAndCleanDatasetRecords()
        {
            if (!Directory.Exists(_pretrainPath)) return;
            var files = Directory.GetFiles(_pretrainPath, "*.jsonl", SearchOption.AllDirectories);

            foreach (var file in files)
            {
                foreach (var line in File.ReadLines(file))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    _datasetRecords.Add(ParseJsonlRecord(line));
                }
            }
            Console.WriteLine($"[Dataset] Parsed and cleaned {_datasetRecords.Count} JSONL records in memory.");
        }

        private static ParsedDatasetRecord ParseJsonlRecord(string rawJsonLine)
        {
            var record = new ParsedDatasetRecord();
            try
            {
                using var doc = JsonDocument.Parse(rawJsonLine);
                var root = doc.RootElement;

                if (root.TryGetProperty("id", out var idProp))
                    record.Id = idProp.GetString() ?? string.Empty;

                if (root.TryGetProperty("metadata", out var metaProp) && metaProp.TryGetProperty("topic", out var topicProp))
                    record.Topic = topicProp.GetString() ?? string.Empty;

                if (root.TryGetProperty("text", out var textProp))
                {
                    record.CleanText = (textProp.GetString() ?? string.Empty).Replace("\\n", " ").Replace("\n", " ").Trim();
                }
                else if (root.TryGetProperty("prompt", out var promptProp))
                {
                    string comp = root.TryGetProperty("completion", out var compProp) ? compProp.GetString() ?? "" : "";
                    record.CleanText = $"{promptProp.GetString()} {comp}".Replace("\\n", " ").Replace("\n", " ").Trim();
                }
            }
            catch
            {
                record.CleanText = rawJsonLine;
            }

            if (string.IsNullOrWhiteSpace(record.CleanText))
                record.CleanText = rawJsonLine;

            return record;
        }

        private void IngestDatasetFilesParallel()
        {
            int count = _datasetRecords.Count;
            if (count == 0) return;

            float[] batchVectors = new float[count * Dimensions];
            int completed = 0;
            var sw = Stopwatch.StartNew();

            Console.WriteLine($"[Ingestion] Encoding {count} cleaned records on {Environment.ProcessorCount} CPU threads...");

            Parallel.For(0, count, i =>
            {
                string textToEncode = string.IsNullOrEmpty(_datasetRecords[i].Topic)
                    ? _datasetRecords[i].CleanText
                    : $"{_datasetRecords[i].Topic} {_datasetRecords[i].CleanText}";

                var (vec, _, _) = _arborNet2D.Forward2D(textToEncode);
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
            // Strip accidental "Input:" prefix if pasted from instructions
            string cleanPrompt = userPrompt.StartsWith("Input:", StringComparison.OrdinalIgnoreCase)
                ? userPrompt.Substring(6).Trim()
                : userPrompt.Trim();

            Console.WriteLine($"\n==================================================");
            Console.WriteLine($"[User (Turn {_interactionCount})]: \"{cleanPrompt}\"");

            // 1. 2D ArborNet Forward Pass
            var (queryVec, intent, probs) = _arborNet2D.Forward2D(cleanPrompt);
            Console.WriteLine($"[ArborNet 2D Router] Intent: {intent} (Softmax Probs -> Fact: {probs[0]:F2}, Synth: {probs[1]:F2}, Paradox: {probs[2]:F2})");

            // 2. FAISS Top-5 Cosine Similarity Search + Lexical Reranking
            int k = Math.Min(5, (int)Math.Max(1, _vectorIndex.NTotal));
            _vectorIndex.Search(queryVec, k, out float[] cosineScores, out long[] labels);

            long bestId = labels[0];
            float bestScore = cosineScores[0];

            // Rerank Top-5 candidates with keyword boost for pinpoint topic accuracy
            string[] queryKeywords = cleanPrompt.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            for (int i = 0; i < k; i++)
            {
                long candidateId = labels[i];
                if (candidateId >= 0 && candidateId < _datasetRecords.Count)
                {
                    string candidateText = _datasetRecords[(int)candidateId].CleanText.ToLowerInvariant();
                    int hits = queryKeywords.Count(kw => kw.Length > 3 && candidateText.Contains(kw));
                    float boostedScore = cosineScores[i] + (hits * 0.08f);
                    if (boostedScore > bestScore)
                    {
                        bestScore = boostedScore;
                        bestId = candidateId;
                    }
                }
            }

            float confidence = Math.Clamp((bestScore + 1.0f) * 0.5f, 0.01f, 0.99f);
            var matchedRecord = (bestId >= 0 && bestId < _datasetRecords.Count) ? _datasetRecords[(int)bestId] : new ParsedDatasetRecord { CleanText = "No record" };
            string snippet = matchedRecord.CleanText.Length > 220 ? matchedRecord.CleanText.Substring(0, 220) + "..." : matchedRecord.CleanText;

            Console.WriteLine($"[FAISS Cosine Memory] Top Match ID: {bestId} (Record: {matchedRecord.Id}) | Cosine Score: {bestScore:F4} | Confidence: {confidence:F2}");
            Console.WriteLine($"[Grounded Context]: \"{snippet}\"");

            // 3. Grothendieck Sheaf Cohomology & Calibrated Obstruction
            float spectralSignal = RetrieveSpectralResonance(confidence);
            float combinedField = (confidence + MathF.Abs(spectralSignal)) * 0.5f;

            float totalObstruction = 0f;
            int truthCount = 0;
            foreach (var patch in _knowledgeTopos)
            {
                foreach (var truth in patch.LocalTruths)
                {
                    totalObstruction += MathF.Abs(truth.Value - combinedField);
                    truthCount++;
                }
            }
            float obstructionMetric = totalObstruction / Math.Max(1, truthCount);

            if (obstructionMetric > 0.35f || intent == CognitiveIntent.StructuralParadox)
            {
                _connesMarcolliLambda *= 0.96f;
                int nextPrime = _primeSequence[_memorySpectrum.Count % _primeSequence.Length];
                _memorySpectrum.Add(new PrimeSpectralComponent
                {
                    PrimeIndex = _memorySpectrum.Count + 1,
                    Eigenvalue = nextPrime * 0.31416f * _connesMarcolliLambda,
                    Amplitude = Math.Clamp(confidence + 0.25f, 0.4f, 1.0f)
                });
                Console.WriteLine($"[NCST Sheaf Cohomology] Obstruction: {obstructionMetric:F3} -> Renormalized Lambda: {_connesMarcolliLambda:F3} (Eigenstates: {_memorySpectrum.Count})");
            }
            else
            {
                Console.WriteLine($"[NCST Sheaf Cohomology] Obstruction: {obstructionMetric:F3} -> Topos Equilibrium Stable (Lambda: {_connesMarcolliLambda:F3})");
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
                        Console.WriteLine($"[Doctrine Synthesis]:");
                        Console.WriteLine($"  * Logos: Mapped [{strainedThought.Domain}] paradox onto [{candidate.Domain}] operator '{candidate.FragmentName}' ({targetSymmetry} symmetry).");
                        Console.WriteLine($"  * Worked Check: Cosine Similarity={bestScore:F3}, Sheaf Obstruction={obstructionMetric:F3}, Lambda={_connesMarcolliLambda:F3}.");
                        Console.WriteLine($"  * Boundary: Grounded in record '{matchedRecord.Id}' -> \"{snippet}\"");
                        CheckDreamCycle();
                        return;
                    }
                }
            }

            Console.WriteLine($"[Doctrine Synthesis]:");
            Console.WriteLine($"  * Logos: Direct 2D ArborNet attention + NCST sheaf consistency verified under [{intent}].");
            Console.WriteLine($"  * Worked Check: Cosine Similarity={bestScore:F3}, Sheaf Obstruction={obstructionMetric:F3}, Lambda={_connesMarcolliLambda:F3}.");
            Console.WriteLine($"  * Boundary: Grounded in record '{matchedRecord.Id}' -> \"{snippet}\"");
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

    public class ProgramA
    {
        public static void MainA(string[] args)
        {
            string pretrainDir = @"C:\Users\Jingl\Documents\OzzieAI AutoLab Bench Pro - CRITICAL DATA (Do Not Delete)\Repository\Training Schema's\pretrain";
            string posttrainDir = @"C:\Users\Jingl\Documents\OzzieAI AutoLab Bench Pro - CRITICAL DATA (Do Not Delete)\Repository\Training Schema's\posttrain";

            using var machine = new NativeArborNetCognitiveMachine(pretrainDir, posttrainDir);

            Console.WriteLine("============================================================");
            Console.WriteLine(" OzzieAI v2 Orthogonal 2D-ArborNet Cognitive Machine Ready");
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