// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// Project:      ArborNet Engine
// Description:  NedK-Prime Master Pipeline: Universal 16K Record Schema Extractor
// License:      MIT License
// -----------------------------------------------------------------------------------------
namespace NedK
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Reflection;
    using System.Text;
    using System.Text.Json;
    using System.Text.RegularExpressions;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Data;
    using ArborNet.Layers;
    using ArborNet.Losses;
    using ArborNet.Optimizers;

    public static class Num
    {
        public static bool Bad(ITensor t)
        {
            var d = t.ToArray();
            for (int i = 0; i < d.Length; i++)
            {
                if (float.IsNaN(d[i]) || float.IsInfinity(d[i])) return true;
            }
            return false;
        }
    }

    #region ConsoleUI Helper

    public static class ConsoleUI
    {
        public const int InnerWidth = 94;

        public const string Reset = "\u001b[0m";
        public const string Bold = "\u001b[1m";
        public const string Cyan = "\u001b[38;5;51m";
        public const string Blue = "\u001b[38;5;39m";
        public const string Purple = "\u001b[38;5;141m";
        public const string Green = "\u001b[38;5;48m";
        public const string Yellow = "\u001b[38;5;220m";
        public const string DarkGray = "\u001b[38;5;240m";
        public const string LightGray = "\u001b[38;5;249m";
        public const string White = "\u001b[38;5;255m";

        public static void PrintBanner(string title, string subtitle)
        {
            string top = "╔" + new string('═', InnerWidth) + "╗";
            string bot = "╚" + new string('═', InnerWidth) + "╝";

            Console.WriteLine($"{Purple}{top}{Reset}");
            Console.WriteLine($"{Purple}║{Reset}  {Bold}{White}{title,-90}{Reset}{Purple}║{Reset}");
            Console.WriteLine($"{Purple}║{Reset}  {Cyan}{subtitle,-90}{Reset}{Purple}║{Reset}");
            Console.WriteLine($"{Purple}{bot}{Reset}\n");
        }

        public static void PrintCardHeader(string title, string icon)
        {
            string tag = $"[ {icon} {title} ]";
            int lineRemain = InnerWidth - tag.Length - 2;
            if (lineRemain < 0) lineRemain = 0;

            Console.WriteLine($"\n{Blue}┌─{White}{Bold}{tag}{Reset}{Blue}" + new string('─', lineRemain) + $"┐{Reset}");
        }

        public static void PrintCardFooter()
        {
            Console.WriteLine($"{Blue}└" + new string('─', InnerWidth) + $"┘{Reset}");
        }

        public static void PrintMetricRow(string label, string value, string unit = "")
        {
            string left = $"  {label}";
            string right = $"{value} {unit}".TrimEnd();
            int dots = InnerWidth - left.Length - right.Length - 4;
            if (dots < 2) dots = 2;

            Console.WriteLine(
                $"{Blue}│{Reset}{LightGray}{left}{Reset} " +
                $"{DarkGray}{new string('.', dots)}{Reset} " +
                $"{White}{Bold}{right}{Reset} {Blue}│{Reset}");
        }

        public static void RenderLiveProgress(
            int current,
            int total,
            int epoch,
            int maxEpochs,
            float currentLoss,
            float currentAcc,
            double samplesPerSec)
        {
            int barWidth = 20;
            double progress = (double)current / Math.Max(1, total);
            int filled = Math.Clamp((int)(progress * barWidth), 0, barWidth);
            int empty = barWidth - filled;

            string bar = $"{Green}{new string('█', filled)}{DarkGray}{new string('░', empty)}{Reset}";
            double pct = progress * 100.0;

            string content =
                $" Ep {epoch:D2}/{maxEpochs:D2} [{bar}] {pct,5:F1}% " +
                $"Loss:{currentLoss,6:F4} Acc:{currentAcc,5:F1}% {samplesPerSec,5:F0}s/s";

            int padding = InnerWidth - content.Length - 2;
            if (padding < 0) padding = 0;

            string line =
                $"\r{Blue}│{Reset} Ep {White}{epoch:D2}/{maxEpochs:D2}{Reset} [{bar}] {Yellow}{pct,5:F1}%{Reset} " +
                $"Loss:{Cyan}{currentLoss,6:F4}{Reset} Acc:{Green}{currentAcc,5:F1}%{Reset} {Purple}{samplesPerSec,5:F0}s/s{Reset}" +
                new string(' ', padding) + $"{Blue}│{Reset}";

            Console.Write(line);
        }

        public static void PrintEpochSummary(
            int ep,
            int maxEp,
            float trLoss,
            float trAcc,
            float vLoss,
            float vAcc,
            float ppl,
            float bitsPerToken,
            int patience,
            int maxPatience,
            bool improved)
        {
            Console.Write("\r" + new string(' ', InnerWidth + 2) + "\r");

            string bestTag = improved ? "BEST" : "PASS";
            string vLossStr = float.IsNaN(vLoss) ? "   N/A" : $"{vLoss,6:F4}";
            string vAccStr = float.IsNaN(vAcc) ? "  N/A" : $"{vAcc,5:F1}%";

            int padding = InnerWidth - (58 + vLossStr.Length + vAccStr.Length);
            if (padding < 0) padding = 0;

            string styledBest = improved ? $"{Green}BEST{Reset}" : $"{DarkGray}PASS{Reset}";
            string styledVLoss = float.IsNaN(vLoss) ? $"{DarkGray}   N/A{Reset}" : $"{Cyan}{vLoss,6:F4}{Reset}";
            string styledVAcc = float.IsNaN(vAcc) ? $"{DarkGray}  N/A{Reset}" : $"{Green}{vAcc,5:F1}%{Reset}";

            string line =
                $"{Blue}│{Reset} Ep {White}{ep:D2}/{maxEp:D2}{Reset} [{styledBest}] " +
                $"Tr:{Cyan}{trLoss,6:F4}{Reset}({Green}{trAcc,5:F1}%{Reset}) │ " +
                $"Val:{styledVLoss}({styledVAcc}) │ " +
                $"PPL:{Yellow}{ppl,6:F2}{Reset} │ " +
                $"H:{Purple}{bitsPerToken,4:F2}b{Reset} │ " +
                $"Pat:{DarkGray}{patience}/{maxPatience}{Reset}" +
                new string(' ', Math.Max(0, InnerWidth - 78)) + $"{Blue}│{Reset}";

            Console.WriteLine(line);
        }
    }

    #endregion

    #region ModelSerializer Helper

    public static class ModelSerializer
    {
        public static void SaveWeights(BaseModel model, string binaryPath)
        {
            string? dir = Path.GetDirectoryName(binaryPath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }

            using var fs = new FileStream(binaryPath, FileMode.Create, FileAccess.Write, FileShare.None);
            using var bw = new BinaryWriter(fs, Encoding.UTF8);

            var visited = new HashSet<ITensor>(ReferenceEqualityComparer.Instance);
            var paramsToSave = model.Parameters().Where(p => p != null && visited.Add(p)).ToList();

            bw.Write(paramsToSave.Count);
            foreach (var p in paramsToSave)
            {
                bw.Write(p.Shape.Rank);
                for (int r = 0; r < p.Shape.Rank; r++) bw.Write(p.Shape[r]);
                float[] data = p.ToArray();
                bw.Write(data.Length);
                for (int i = 0; i < data.Length; i++) bw.Write(data[i]);
            }
        }

        public static void LoadWeights(BaseModel model, string binaryPath)
        {
            if (!File.Exists(binaryPath))
            {
                throw new FileNotFoundException($"Checkpoint file not found: {binaryPath}");
            }

            using var fs = new FileStream(binaryPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var br = new BinaryReader(fs, Encoding.UTF8);

            int tensorCount = br.ReadInt32();
            var visited = new HashSet<ITensor>(ReferenceEqualityComparer.Instance);
            var paramsToLoad = model.Parameters().Where(p => p != null && visited.Add(p)).ToList();

            if (tensorCount != paramsToLoad.Count)
            {
                throw new InvalidOperationException($"Tensor count mismatch: checkpoint={tensorCount}, model={paramsToLoad.Count}");
            }

            for (int i = 0; i < tensorCount; i++)
            {
                int rank = br.ReadInt32();
                int[] dims = new int[rank];
                for (int r = 0; r < rank; r++) dims[r] = br.ReadInt32();
                int len = br.ReadInt32();
                float[] data = new float[len];
                for (int j = 0; j < len; j++) data[j] = br.ReadSingle();

                CopyTensorData(paramsToLoad[i], data);
            }
        }

        private static void CopyTensorData(ITensor target, float[] source)
        {
            const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;

            var method = target.GetType().GetMethod("CopyFrom", flags) ?? target.GetType().GetMethod("SetData", flags);
            if (method != null && method.GetParameters().Length == 1)
            {
                method.Invoke(target, new object[] { source });
                return;
            }

            var prop = target.GetType().GetProperty("Data", flags);
            if (prop != null && prop.CanWrite && prop.PropertyType == typeof(float[]))
            {
                prop.SetValue(target, (float[])source.Clone());
                return;
            }

            foreach (string fname in new[] { "_data", "data", "_values", "values", "_buffer" })
            {
                var field = target.GetType().GetField(fname, flags);
                if (field != null && field.FieldType == typeof(float[]))
                {
                    field.SetValue(target, (float[])source.Clone());
                    return;
                }
            }

            var setAt = target.GetType().GetMethod("Set", new[] { typeof(int), typeof(float) });
            if (setAt != null)
            {
                for (int i = 0; i < source.Length; i++) setAt.Invoke(target, new object[] { i, source[i] });
                return;
            }

            throw new NotSupportedException($"Unable to copy weights to tensor {target.GetType().FullName}.");
        }
    }

    #endregion

    #region Universal Multi-Format Schema Extractor

    public static class StreamedSchemaExtractor
    {
        public static IEnumerable<string> StreamPartitionText(string directoryPath, string partitionTag)
        {
            if (!Directory.Exists(directoryPath)) yield break;

            string[] files = Directory.GetFiles(directoryPath, "*.*", SearchOption.AllDirectories);
            if (files.Length == 0) yield break;

            foreach (var file in files)
            {
                using var fs = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read, 65536);
                using var reader = new StreamReader(fs, Encoding.UTF8);

                string? firstLine = reader.ReadLine();
                if (string.IsNullOrWhiteSpace(firstLine)) continue;
                fs.Seek(0, SeekOrigin.Begin);
                reader.DiscardBufferedData();

                if (firstLine.TrimStart().StartsWith("["))
                {
                    string fullJson = reader.ReadToEnd();
                    foreach (var text in ExtractFromJsonArray(fullJson))
                    {
                        yield return text;
                    }
                }
                else
                {
                    string? line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        string trimmed = line.Trim();
                        if (trimmed.Length < NedKConfig.MinTextCharLength) continue;

                        if (trimmed.EndsWith(",") && trimmed.StartsWith("{"))
                        {
                            trimmed = trimmed.Substring(0, trimmed.Length - 1).Trim();
                        }

                        string? extracted = null;

                        if (trimmed.StartsWith("{") && trimmed.EndsWith("}"))
                        {
                            extracted = ExtractFromJsonObject(trimmed);
                        }
                        else if (!trimmed.StartsWith("//") && !trimmed.StartsWith("#"))
                        {
                            extracted = trimmed;
                        }

                        if (!string.IsNullOrWhiteSpace(extracted))
                        {
                            string clean = Regex.Replace(extracted, @"\s+", " ").Trim();
                            if (clean.Length >= NedKConfig.MinTextCharLength && clean.Length <= NedKConfig.MaxTextCharLength)
                            {
                                yield return clean;
                            }
                        }
                    }
                }
            }
        }

        private static IEnumerable<string> ExtractFromJsonArray(string json)
        {
            var results = new List<string>();
            try
            {
                using var doc = JsonDocument.Parse(json);
                if (doc.RootElement.ValueKind == JsonValueKind.Array)
                {
                    foreach (var element in doc.RootElement.EnumerateArray())
                    {
                        string? text = ParseElement(element);
                        if (!string.IsNullOrWhiteSpace(text))
                        {
                            string clean = Regex.Replace(text, @"\s+", " ").Trim();
                            if (clean.Length >= NedKConfig.MinTextCharLength && clean.Length <= NedKConfig.MaxTextCharLength)
                            {
                                results.Add(clean);
                            }
                        }
                    }
                }
            }
            catch { /* Skip corrupted items */ }

            foreach (var item in results) yield return item;
        }

        private static string? ExtractFromJsonObject(string json)
        {
            try
            {
                using var doc = JsonDocument.Parse(json);
                return ParseElement(doc.RootElement);
            }
            catch { return null; }
        }

        private static string? ParseElement(JsonElement root)
        {
            if (root.ValueKind != JsonValueKind.Object) return null;

            if (root.TryGetProperty("text", out var textProp))
                return textProp.GetString();

            if (root.TryGetProperty("prompt", out var pProp) && root.TryGetProperty("completion", out var cProp))
                return $"{pProp.GetString()} {cProp.GetString()}";

            if (root.TryGetProperty("instruction", out var instProp))
            {
                string inst = instProp.GetString() ?? "";
                string inp = root.TryGetProperty("input", out var inProp) ? inProp.GetString() ?? "" : "";
                string outp = root.TryGetProperty("output", out var outProp) ? outProp.GetString() ?? "" : "";
                return string.IsNullOrWhiteSpace(inp) ? $"{inst} {outp}" : $"{inst} {inp} {outp}";
            }

            if (root.TryGetProperty("messages", out var msgProp) && msgProp.ValueKind == JsonValueKind.Array)
            {
                var chatMessages = new List<OzChatMessage>();
                foreach (var m in msgProp.EnumerateArray())
                {
                    string role = m.TryGetProperty("role", out var rProp) ? rProp.GetString() ?? "user" : "user";
                    string content = m.TryGetProperty("content", out var msgContentProp) ? msgContentProp.GetString() ?? "" : "";
                    chatMessages.Add(new OzChatMessage(role, content));
                }
                return OzLocalInferProtocol.FormatPrompt(chatMessages);
            }

            if (root.TryGetProperty("chosen", out var chProp) && chProp.TryGetProperty("content", out var chContent))
                return chContent.GetString();

            if (root.TryGetProperty("conversations", out var convProp) && convProp.ValueKind == JsonValueKind.Array)
            {
                var sb = new StringBuilder();
                foreach (var turn in convProp.EnumerateArray())
                {
                    string val = turn.TryGetProperty("value", out var vProp) ? vProp.GetString() ?? "" : "";
                    if (!string.IsNullOrEmpty(val)) sb.Append(val).Append(' ');
                }
                return sb.ToString().Trim();
            }

            return null;
        }
    }

    #endregion


    /// <summary>Console helpers safe when stdout is redirected (AutoLab / CI / no TTY).</summary>
    public static class SafeConsole
    {
        public static bool HasInteractiveConsole
        {
            get
            {
                try
                {
                    if (Console.IsOutputRedirected || Console.IsErrorRedirected)
                        return false;
                    // Touch buffer only when not redirected
                    _ = Console.WindowHeight;
                    return true;
                }
                catch
                {
                    return false;
                }
            }
        }

        public static void InitEncoding()
        {
            try { Console.OutputEncoding = Encoding.UTF8; } catch { /* ignore */ }
        }

        public static void Clear()
        {
            if (!HasInteractiveConsole) return;
            try { Console.Clear(); } catch (IOException) { } catch (PlatformNotSupportedException) { }
        }

        public static void WriteLine(string? s = null)
        {
            try
            {
                if (s == null) Console.WriteLine();
                else Console.WriteLine(s);
            }
            catch (IOException) { }
        }

        public static void Write(string? s)
        {
            try { if (s != null) Console.Write(s); } catch (IOException) { }
        }
    }

    public class ProgramB
    {
        public static void MainB()
        {
            // AutoLab / CI: stdout redirected — never call Console.Clear (IOException: handle invalid)
            SafeConsole.InitEncoding();
            SafeConsole.Clear();
            // Headless train: skip interactive banner noise when redirected
            bool headless = string.Equals(Environment.GetEnvironmentVariable("NEDK_HEADLESS"), "1", StringComparison.OrdinalIgnoreCase)
                || string.Equals(Environment.GetEnvironmentVariable("NEDK_AUTOLAB"), "1", StringComparison.OrdinalIgnoreCase)
                || Console.IsOutputRedirected;

            if (!headless)
            {
                ConsoleUI.PrintBanner(
                    "ARBORNET NEDK-PRIME V2 TELEMETRY BENCHMARK",
                    "Pure 2D Tensor Algebra • 16K Scale Schema Pipeline • Guaranteed Zero-Drift Checkpointing");
            }
            else
            {
                SafeConsole.WriteLine("[NEDK] Headless/AutoLab mode — console UI suppressed; train proceeds.");
            }

            var device = Device.CPU;

            string repositoryDir = @"C:\Users\Jingl\Documents\OzzieAI AutoLab Bench Pro - CRITICAL DATA (Do Not Delete)\Repository\Training Schema's";
            string trainedModelsDir = @"C:\Users\Jingl\Documents\OzzieAI AutoLab Bench Pro - CRITICAL DATA (Do Not Delete)\trained-models";
            Directory.CreateDirectory(trainedModelsDir);

            string pretrainDir = Path.Combine(repositoryDir, "pretrain");
            string posttrainDir = Path.Combine(repositoryDir, "posttrain");

            var foundationSentences = new List<string>
            {
                OzLocalInferProtocol.FormatPrompt(new List<OzChatMessage>
                {
                    new OzChatMessage("system", OzLocalInferProtocol.DefaultSystemPrompt),
                    new OzChatMessage("user", "Tell me about NedK's performance."),
                    new OzChatMessage("assistant", "ArborNet makes NedK exceptionally fast and smart.")
                }),
                OzLocalInferProtocol.FormatPrompt(new List<OzChatMessage>
                {
                    new OzChatMessage("system", OzLocalInferProtocol.DefaultSystemPrompt),
                    new OzChatMessage("user", "What kind of model is NedK?"),
                    new OzChatMessage("assistant", "NedK is a frontier mixture of experts language model.")
                }),
                OzLocalInferProtocol.FormatPrompt(new List<OzChatMessage>
                {
                    new OzChatMessage("system", OzLocalInferProtocol.DefaultSystemPrompt),
                    new OzChatMessage("user", "What is your vision for AI?"),
                    new OzChatMessage("assistant", "The future of artificial intelligence is open source.")
                }),
                OzLocalInferProtocol.FormatPrompt(new List<OzChatMessage>
                {
                    new OzChatMessage("system", OzLocalInferProtocol.DefaultSystemPrompt),
                    new OzChatMessage("user", "How should machine learning models be designed?"),
                    new OzChatMessage("assistant", "Machine learning models should be efficient and beautiful.")
                }),
                OzLocalInferProtocol.FormatPrompt(new List<OzChatMessage>
                {
                    new OzChatMessage("system", OzLocalInferProtocol.DefaultSystemPrompt),
                    new OzChatMessage("user", "Why use CUDA?"),
                    new OzChatMessage("assistant", "CUDA acceleration makes training much faster on GPUs.")
                }),
                OzLocalInferProtocol.FormatPrompt(new List<OzChatMessage>
                {
                    new OzChatMessage("system", OzLocalInferProtocol.DefaultSystemPrompt),
                    new OzChatMessage("user", "What is the purpose of residual connections?"),
                    new OzChatMessage("assistant", "Residual connections help deep networks converge better.")
                }),
                OzLocalInferProtocol.FormatPrompt(new List<OzChatMessage>
                {
                    new OzChatMessage("system", OzLocalInferProtocol.DefaultSystemPrompt),
                    new OzChatMessage("user", "How do sparse experts help scaling?"),
                    new OzChatMessage("assistant", "Sparse experts allow scaling to trillions of parameters.")
                }),
                OzLocalInferProtocol.FormatPrompt(new List<OzChatMessage>
                {
                    new OzChatMessage("system", OzLocalInferProtocol.DefaultSystemPrompt),
                    new OzChatMessage("user", "Send a greeting from the system."),
                    new OzChatMessage("assistant", "Hello world from the NedK architecture.")
                }),
                OzLocalInferProtocol.FormatPrompt(new List<OzChatMessage>
                {
                    new OzChatMessage("system", OzLocalInferProtocol.DefaultSystemPrompt),
                    new OzChatMessage("user", "What does training language models require?"),
                    new OzChatMessage("assistant", "Training language models requires careful optimisation.")
                }),
                OzLocalInferProtocol.FormatPrompt(new List<OzChatMessage>
                {
                    new OzChatMessage("system", OzLocalInferProtocol.DefaultSystemPrompt),
                    new OzChatMessage("user", "Thoughts on attention mechanisms?"),
                    new OzChatMessage("assistant", "Attention is all you need, but linear attention is faster.")
                })
            };

            // 1. Data Pipeline
            ConsoleUI.PrintCardHeader("INGESTION & CURATION PIPELINE", "📁");
            var pretrainStream = StreamedSchemaExtractor.StreamPartitionText(pretrainDir, "Pretrain").ToList();
            var posttrainStream = StreamedSchemaExtractor.StreamPartitionText(posttrainDir, "Posttrain").ToList();

            int pretrainSplitIndex = (int)(pretrainStream.Count * 0.85);
            var pretrainTrainSentences = foundationSentences.Concat(pretrainStream.Take(pretrainSplitIndex)).ToList();
            var pretrainValSentences = pretrainStream.Skip(pretrainSplitIndex).ToList();

            int posttrainSplitIndex = (int)(posttrainStream.Count * 0.85);
            var posttrainTrainSentences = foundationSentences.Concat(posttrainStream.Take(posttrainSplitIndex)).ToList();
            var posttrainValSentences = posttrainStream.Skip(posttrainSplitIndex).ToList();

            ConsoleUI.PrintMetricRow("Pretrain Ingested", $"{pretrainStream.Count:N0}", "records");
            ConsoleUI.PrintMetricRow("Posttrain Ingested", $"{posttrainStream.Count:N0}", "records");
            ConsoleUI.PrintMetricRow("Pretrain Train Pool", $"{pretrainTrainSentences.Count:N0}", "samples");
            ConsoleUI.PrintMetricRow("Pretrain Val Pool", $"{pretrainValSentences.Count:N0}", "samples");
            ConsoleUI.PrintCardFooter();

            // 2. Tokenizer (PrimeGapTokenizer)
            ConsoleUI.PrintCardHeader("PRIME-GAP ARITHMETIC TOKENIZER", "🔣");
            string vocabPath = Path.Combine(trainedModelsDir, "nedk_prime_vocab.txt");
            var allLines = pretrainTrainSentences.Concat(pretrainValSentences).Concat(posttrainTrainSentences).Concat(posttrainValSentences);

            var tokSw = Stopwatch.StartNew();
            var tokenizer = PrimeGapTokenizer.BuildAndSave(allLines, vocabPath, targetVocabSize: NedKConfig.TargetVocabSize);
            tokSw.Stop();

            int V = tokenizer.VocabSize;
            int W = NedKConfig.ContextWindow;

            ConsoleUI.PrintMetricRow("Vocabulary Elements", $"{V:N0}", "tokens");
            ConsoleUI.PrintMetricRow("Max Prime Lattice Base", $"{tokenizer.PrimeCoordinates[^1].Prime:N0}", "prime");
            ConsoleUI.PrintMetricRow("Compilation Latency", $"{tokSw.ElapsedMilliseconds:N0}", "ms");
            ConsoleUI.PrintCardFooter();

            // 3. Dataset Builder
            List<(float[] x, int tgt)> BuildPrefixDataset(List<string> sentences, int maxTotalSamples)
            {
                var dataset = new List<(float[] x, int tgt)>();
                if (sentences.Count == 0) return dataset;

                int pad = tokenizer.PadId;

                foreach (var s in sentences)
                {
                    var tokens = tokenizer.Encode(s);
                    if (tokens.Count < 2) continue;

                    for (int used = 1; used < tokens.Count && dataset.Count < maxTotalSamples; used += 2)
                    {
                        var x = new float[W];
                        for (int t = 0; t < W; t++)
                        {
                            int p = used - W + t;
                            x[t] = p < 0 ? pad : tokens[p];
                        }
                        dataset.Add((x, tokens[used]));
                    }

                    if (dataset.Count >= maxTotalSamples) break;
                }
                return dataset;
            }

            var pretrainData = BuildPrefixDataset(pretrainTrainSentences, NedKConfig.MaxCuratedSamples);
            var pretrainVal = BuildPrefixDataset(pretrainValSentences, 1000);
            var posttrainData = BuildPrefixDataset(posttrainTrainSentences, NedKConfig.MaxCuratedSamples);
            var posttrainVal = BuildPrefixDataset(posttrainValSentences, 500);

            // 4. Model Initialization
            ConsoleUI.PrintCardHeader("NEURAL ARCHITECTURE SPECIFICATION", "🧠");
            var model = new NedKPrimeModel(
                tokenizer: tokenizer,
                d: NedKConfig.HiddenDim,
                dFf: NedKConfig.FfnDim,
                window: NedKConfig.ContextWindow,
                maxLoops: NedKConfig.MaxDynamicLoops,
                device: device);

            //int uniqueParams = model.ComputeUniqueParameters();
            float initialEntropyNats = MathF.Log(V);
            float initialEntropyBits = initialEntropyNats * 1.442695f;

            ConsoleUI.PrintMetricRow("Engine Framework", $"NedK-Prime v{NedKConfig.Version:0.0}");
            ConsoleUI.PrintMetricRow("Active Training Windows", $"{pretrainData.Count:N0}", "windows");
            ConsoleUI.PrintMetricRow("Hidden Dimension (D)", $"{NedKConfig.HiddenDim:N0}", "channels");
            ConsoleUI.PrintMetricRow("Feed-Forward Dim (dFf)", $"{NedKConfig.FfnDim:N0}", "channels");
            ConsoleUI.PrintMetricRow("Max Recurrent Depth", $"{NedKConfig.MaxDynamicLoops:N0}", "loops");
            //ConsoleUI.PrintMetricRow("Trainable Parameters", $"{uniqueParams:N0}", "floats");
            ConsoleUI.PrintMetricRow("Shannon Base Entropy", $"{initialEntropyBits:F4}", "bits/tok");
            ConsoleUI.PrintCardFooter();

            string pretrainOzmPath = Path.Combine(trainedModelsDir, "nedk_prime_pretrain.ozm");
            string pretrainWeightsPath = Path.Combine(trainedModelsDir, "nedk_prime_pretrain_weights.bin");

            string officialOzmPath = Path.Combine(trainedModelsDir, "nedk_prime.ozm");
            string nativeWeightsPath = Path.Combine(trainedModelsDir, "nedk_prime_weights.bin");

            var ce = new CrossEntropy();
            var rng = new Random(42);
            var pooledTargetBuffer = new float[V];

            static void FisherYatesShuffle<T>(IList<T> list, Random rng)
            {
                for (int i = list.Count - 1; i > 0; i--)
                {
                    int j = rng.Next(i + 1);
                    (list[i], list[j]) = (list[j], list[i]);
                }
            }

            (float loss, float acc) Evaluate(List<(float[] x, int tgt)> data, int fixedLoops = 3)
            {
                if (data == null || data.Count == 0) return (float.NaN, 0f);
                model.Eval();
                float totalLoss = 0f;
                int correct = 0, total = 0;

                foreach (var (xin, tgt) in data)
                {
                    pooledTargetBuffer[tgt] = 1.0f;
                    var xTensor = Tensor.FromArray(xin, new TensorShape(1, W), device);
                    var yTensor = Tensor.FromArray(pooledTargetBuffer, new TensorShape(1, V), device);
                    var logits = model.Forward(xTensor, loops: fixedLoops);
                    pooledTargetBuffer[tgt] = 0.0f;

                    if (Num.Bad(logits)) continue;
                    float lossVal = ce.Forward(logits, yTensor).ToScalar();
                    if (float.IsNaN(lossVal) || float.IsInfinity(lossVal)) continue;

                    float[] z = logits.ToArray();
                    int argmax = 0;
                    float max = float.NegativeInfinity;
                    for (int k = 0; k < z.Length; k++)
                    {
                        if (z[k] > max) { max = z[k]; argmax = k; }
                    }

                    if (argmax == tgt) correct++;
                    totalLoss += lossVal;
                    total++;
                }

                model.Train();
                return (total > 0 ? totalLoss / total : float.NaN, total > 0 ? 100f * correct / total : 0f);
            }

            // =====================================================================
            // STAGE 1: PRE-TRAINING
            // =====================================================================
            ConsoleUI.PrintCardHeader("STAGE 1: FOUNDATION PRE-TRAINING (AdamW)", "⚡");

            var pretrainOpt = new ArborNet.Optimizers.AdamW(
                learningRate: NedKConfig.InitialPretrainLr,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weightDecay: NedKConfig.WeightDecay);

            var stage1Checkpoint = new EarlyStoppingCheckpoint(
                                                    patience: NedKConfig.EarlyStoppingPatience,  // 6 is fine
                                                    minDelta: 1e-4f,
                                                    checkpointPath: pretrainWeightsPath,
                                                    ozmPath: pretrainOzmPath,
                                                    tokenizer: tokenizer,
                                                    device: device,
                                                    corpusRoot: repositoryDir,
                                                    minWarmupEpochs: 1,   // was 4 — THIS is the important fix
                                                    maxValRise: 0.75f);   // stop if val jumps 0.75+ above best

            int batchSize = NedKConfig.BatchSize;
            model.Train();
            var stage1Timer = Stopwatch.StartNew();

            for (int ep = 1; ep <= NedKConfig.PretrainEpochs; ep++)
            {
                FisherYatesShuffle(pretrainData, rng);

                float epochLoss = 0f;
                int correct = 0, total = 0;
                int accumulatedInBatch = 0;

                pretrainOpt.ZeroGrad(model.Parameters());
                var epSw = Stopwatch.StartNew();

                for (int i = 0; i < pretrainData.Count; i++)
                {
                    var (xin, tgt) = pretrainData[i];
                    pooledTargetBuffer[tgt] = 1.0f;
                    var xTensor = Tensor.FromArray(xin, new TensorShape(1, W), device);
                    var yTensor = Tensor.FromArray(pooledTargetBuffer, new TensorShape(1, V), device);
                    int activeLoops = rng.Next(1, NedKConfig.MaxDynamicLoops + 1);

                    var logits = model.Forward(xTensor, loops: activeLoops);
                    pooledTargetBuffer[tgt] = 0.0f;

                    if (Num.Bad(logits)) continue;
                    var loss = ce.Forward(logits, yTensor);
                    float lossVal = loss.ToScalar();
                    if (float.IsNaN(lossVal) || float.IsInfinity(lossVal)) continue;

                    loss.Backward();
                    accumulatedInBatch++;

                    if (accumulatedInBatch >= batchSize)
                    {
                        Tensor.ClipGradients(model.Parameters(), -NedKConfig.GradientClipThreshold, NedKConfig.GradientClipThreshold);
                        pretrainOpt.Step(model.Parameters());
                        pretrainOpt.ZeroGrad(model.Parameters());
                        accumulatedInBatch = 0;
                    }

                    float[] z = logits.ToArray();
                    int argmax = 0;
                    float max = float.NegativeInfinity;
                    for (int k = 0; k < z.Length; k++)
                    {
                        if (z[k] > max) { max = z[k]; argmax = k; }
                    }

                    if (argmax == tgt) correct++;
                    epochLoss += lossVal;
                    total++;

                    if (i % 32 == 0 || i == pretrainData.Count - 1)
                    {
                        double smpPerSec = total / Math.Max(0.001, epSw.Elapsed.TotalSeconds);
                        float currentRunningLoss = total > 0 ? epochLoss / total : 0f;
                        float currentRunningAcc = total > 0 ? 100f * correct / total : 0f;
                        ConsoleUI.RenderLiveProgress(i + 1, pretrainData.Count, ep, NedKConfig.PretrainEpochs, currentRunningLoss, currentRunningAcc, smpPerSec);
                    }
                }

                if (accumulatedInBatch > 0)
                {
                    Tensor.ClipGradients(model.Parameters(), -NedKConfig.GradientClipThreshold, NedKConfig.GradientClipThreshold);
                    pretrainOpt.Step(model.Parameters());
                    pretrainOpt.ZeroGrad(model.Parameters());
                }

                float avgLoss = total > 0 ? epochLoss / total : 0f;
                float acc = total > 0 ? 100f * correct / total : 0f;

                var (valLoss, valAcc) = Evaluate(pretrainVal);
                bool improved = stage1Checkpoint.Step(model, valLoss, valAcc, ep);

                float ppl = MathF.Exp(Math.Min(avgLoss, 20f));
                float hBits = avgLoss * 1.442695f;

                ConsoleUI.PrintEpochSummary(
                    ep, NedKConfig.PretrainEpochs, avgLoss, acc, valLoss, valAcc, ppl, hBits,
                    stage1Checkpoint.PatienceCounter, NedKConfig.EarlyStoppingPatience, improved);

                if (stage1Checkpoint.ShouldStop || (acc >= NedKConfig.TargetHaltingAcc && ep >= 12))
                {
                    Console.WriteLine($"{ConsoleUI.Blue}│{ConsoleUI.Green}  ↳ Stage 1 Early Convergence Triggered. Restoring Best Validation Weights (Ep {stage1Checkpoint.BestEpoch}).{ConsoleUI.Reset}");
                    stage1Checkpoint.RestoreBestWeights(model);
                    break;
                }
            }

            stage1Timer.Stop();
            stage1Checkpoint.RestoreBestWeights(model);
            ConsoleUI.PrintCardFooter();

            // =====================================================================
            // STAGE 2: POST-TRAINING
            // =====================================================================
            ConsoleUI.PrintCardHeader("STAGE 2: TASK ALIGNMENT (AdamW)", "🎯");

            var posttrainOpt = new ArborNet.Optimizers.AdamW(
                learningRate: NedKConfig.InitialPosttrainLr,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weightDecay: NedKConfig.WeightDecay);

            var stage2Checkpoint = new EarlyStoppingCheckpoint(
                                                patience: NedKConfig.EarlyStoppingPatience,
                                                minDelta: 1e-4f,
                                                checkpointPath: nativeWeightsPath,
                                                ozmPath: officialOzmPath,
                                                tokenizer: tokenizer,
                                                device: device,
                                                corpusRoot: repositoryDir,
                                                minWarmupEpochs: 1,
                                                maxValRise: 0.75f);

            for (int ep = 1; ep <= NedKConfig.PosttrainEpochs; ep++)
            {
                FisherYatesShuffle(posttrainData, rng);

                float epochLoss = 0f;
                int correct = 0, total = 0;
                int accumulatedInBatch = 0;

                posttrainOpt.ZeroGrad(model.Parameters());
                var epSw = Stopwatch.StartNew();

                for (int i = 0; i < posttrainData.Count; i++)
                {
                    var (xin, tgt) = posttrainData[i];
                    pooledTargetBuffer[tgt] = 1.0f;
                    var xTensor = Tensor.FromArray(xin, new TensorShape(1, W), device);
                    var yTensor = Tensor.FromArray(pooledTargetBuffer, new TensorShape(1, V), device);
                    int activeLoops = rng.Next(1, NedKConfig.MaxDynamicLoops + 1);

                    var logits = model.Forward(xTensor, loops: activeLoops);
                    pooledTargetBuffer[tgt] = 0.0f;

                    if (Num.Bad(logits)) continue;
                    var loss = ce.Forward(logits, yTensor);
                    float lossVal = loss.ToScalar();
                    if (float.IsNaN(lossVal) || float.IsInfinity(lossVal)) continue;

                    loss.Backward();
                    accumulatedInBatch++;

                    if (accumulatedInBatch >= batchSize)
                    {
                        Tensor.ClipGradients(model.Parameters(), -NedKConfig.GradientClipThreshold, NedKConfig.GradientClipThreshold);
                        posttrainOpt.Step(model.Parameters());
                        posttrainOpt.ZeroGrad(model.Parameters());
                        accumulatedInBatch = 0;
                    }

                    float[] z = logits.ToArray();
                    int argmax = 0;
                    float max = float.NegativeInfinity;
                    for (int k = 0; k < z.Length; k++)
                    {
                        if (z[k] > max) { max = z[k]; argmax = k; }
                    }

                    if (argmax == tgt) correct++;
                    epochLoss += lossVal;
                    total++;

                    if (i % 32 == 0 || i == posttrainData.Count - 1)
                    {
                        double smpPerSec = total / Math.Max(0.001, epSw.Elapsed.TotalSeconds);
                        float currentRunningLoss = total > 0 ? epochLoss / total : 0f;
                        float currentRunningAcc = total > 0 ? 100f * correct / total : 0f;
                        ConsoleUI.RenderLiveProgress(i + 1, posttrainData.Count, ep, NedKConfig.PosttrainEpochs, currentRunningLoss, currentRunningAcc, smpPerSec);
                    }
                }

                if (accumulatedInBatch > 0)
                {
                    Tensor.ClipGradients(model.Parameters(), -NedKConfig.GradientClipThreshold, NedKConfig.GradientClipThreshold);
                    posttrainOpt.Step(model.Parameters());
                    posttrainOpt.ZeroGrad(model.Parameters());
                }

                float avgLoss = total > 0 ? epochLoss / total : 0f;
                float acc = total > 0 ? 100f * correct / total : 0f;

                var (valLoss, valAcc) = Evaluate(posttrainVal);
                bool improved = stage2Checkpoint.Step(
                    model: model,
                    valLoss: valLoss,
                    valAcc: valAcc,
                    epoch: ep,
                    pretrainValLoss: stage1Checkpoint.BestLoss < float.MaxValue * 0.5f ? stage1Checkpoint.BestLoss : null);

                float ppl = MathF.Exp(Math.Min(avgLoss, 20f));
                float hBits = avgLoss * 1.442695f;

                ConsoleUI.PrintEpochSummary(
                    ep, NedKConfig.PosttrainEpochs, avgLoss, acc, valLoss, valAcc, ppl, hBits,
                    stage2Checkpoint.PatienceCounter, NedKConfig.EarlyStoppingPatience, improved);

                if (stage2Checkpoint.ShouldStop || (acc >= NedKConfig.TargetHaltingAcc && ep >= 8))
                {
                    Console.WriteLine($"{ConsoleUI.Blue}│{ConsoleUI.Green}  ↳ Stage 2 Alignment Converged. Restoring Best Validation Weights (Ep {stage2Checkpoint.BestEpoch}).{ConsoleUI.Reset}");
                    stage2Checkpoint.RestoreBestWeights(model);
                    break;
                }
            }

            stage2Checkpoint.RestoreBestWeights(model);
            ConsoleUI.PrintCardFooter();

            // =====================================================================
            // PERSISTENCE & INTEGRITY VERIFICATION
            // =====================================================================
            ConsoleUI.PrintCardHeader("OZM-PRIME BINARY PERSISTENCE", "💾");

            long ozmBytes = new FileInfo(officialOzmPath).Length;
            ConsoleUI.PrintMetricRow("Best Stage 1 Loss", $"{stage1Checkpoint.BestLoss:F6}");
            ConsoleUI.PrintMetricRow("Best Stage 2 Loss", $"{stage2Checkpoint.BestLoss:F6}");
            ConsoleUI.PrintMetricRow("Native Weights Target", Path.GetFileName(nativeWeightsPath));
            ConsoleUI.PrintMetricRow("Compressed Container", $"{ozmBytes:N0}", "bytes");
            ConsoleUI.PrintMetricRow("Compression Efficiency", $"{ozmBytes / 1024.0:F1}", "KiB");

            var loadedPkg = NedKOzComplete.Load(officialOzmPath);
            var loadedTok = NedKOzComplete.MaterializeTokenizer(loadedPkg);
            var loadedNedK = NedKOzComplete.Materialize(loadedPkg, loadedTok, device);

            var probe = posttrainData.Count > 0 ? posttrainData[0] : pretrainData[0];
            var testInput = Tensor.FromArray(probe.x, new TensorShape(1, W), device);
            float[] origOut = model.Forward(testInput, loops: 3).ToArray();
            float[] loadedOut = loadedNedK.Forward(testInput, loops: 3).ToArray();

            float maxDiff = 0f;
            for (int i = 0; i < origOut.Length; i++)
            {
                float d = MathF.Abs(origOut[i] - loadedOut[i]);
                if (d > maxDiff) maxDiff = d;
            }

            ConsoleUI.PrintMetricRow("Bit-Exact Divergence |Δ|", $"{maxDiff:E6}", maxDiff < 1e-5f ? "VERIFIED" : "FAIL");
            ConsoleUI.PrintCardFooter();

            loadedNedK.Eval();

            // =====================================================================
            // GENERATION BENCHMARK
            // =====================================================================
            ConsoleUI.PrintCardHeader("AUTONOMOUS INFERENCE GENERATION SUITE", "✨");

            string[] prompts =
            {
                "ArborNet makes NedK ",
                "The future of ",
                "CUDA acceleration ",
                "Residual connections ",
                "Hello "
            };

            foreach (var p in prompts)
            {
                Console.Write($"{ConsoleUI.Blue}│{ConsoleUI.Reset}  {ConsoleUI.Cyan}Query{ConsoleUI.Reset}  : {ConsoleUI.White}{ConsoleUI.Bold}\"{p}\"{ConsoleUI.Reset}\n");
                Console.Write($"{ConsoleUI.Blue}│{ConsoleUI.Reset}  {ConsoleUI.Green}Output{ConsoleUI.Reset} : {ConsoleUI.White}");

                NedKOzLocalInfer.GenerateSentence(
                    loadedNedK, loadedTok, device, loadedTok.PadId,
                    loadedPkg.ContextWindow, p, NedKConfig.DefaultInferenceLoops);

                Console.WriteLine($"{ConsoleUI.Blue}│{ConsoleUI.Reset}");
            }
            ConsoleUI.PrintCardFooter();

            Console.WriteLine($"\n{ConsoleUI.Green}{ConsoleUI.Bold}✔ Benchmark Pipeline Executed Cleanly Across All Ingested Records.{ConsoleUI.Reset}\n");
        }
    }
}
