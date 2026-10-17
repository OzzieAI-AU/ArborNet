#nullable enable

namespace NedK
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.IO;
    using System.IO.Compression;
    using System.Linq;
    using System.Reflection;
    using System.Text;
    using System.Text.Json;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Data;

    public sealed class NedKCompletePackage
    {
        public const string Magic = "OZMP";
        public const string FormatId = "ozzieai.nedk-prime/2d";
        public const string Architecture = "NedKPrime2D";

        public Dictionary<string, string> Meta { get; set; } = new(StringComparer.OrdinalIgnoreCase);
        public List<string> Tokens { get; set; } = new();
        public int PadId { get; set; }
        public int UnkId { get; set; } = 1;

        public int VocabSize { get; set; }
        public int HiddenDim { get; set; }
        public int FfnDim { get; set; }
        public int ContextWindow { get; set; }
        public int MaxLoops { get; set; }
        public int DefaultLoops { get; set; } = 4;

        public List<float[]> OrderedTensors { get; set; } = new();
        public List<int[]> OrderedShapes { get; set; } = new();

        public double CalculatePerplexity(float crossEntropyLoss)
        {
            if (float.IsNaN(crossEntropyLoss) || float.IsInfinity(crossEntropyLoss))
                return double.PositiveInfinity;
            return Math.Exp(Math.Clamp((double)crossEntropyLoss, 0.0, 100.0));
        }

        public string GetMetadataSummary()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Package: {Architecture} ({FormatId})");
            sb.AppendLine($"Vocab: {VocabSize} | HiddenDim: {HiddenDim} | FfnDim: {FfnDim} | Window: {ContextWindow} | Loops: {DefaultLoops}/{MaxLoops}");
            if (Meta.TryGetValue("pretrainBestVal", out var pre)) sb.AppendLine($"Pretrain Best Val: {pre}");
            if (Meta.TryGetValue("posttrainBestVal", out var post)) sb.AppendLine($"Posttrain Best Val: {post}");
            if (Meta.TryGetValue("valPerplexity", out var ppl)) sb.AppendLine($"Held-Out Perplexity: {ppl}");
            if (Meta.TryGetValue("dataType", out var dt)) sb.AppendLine($"Target Modality: {dt}");
            return sb.ToString();
        }
    }

    public static class NedKOzComplete
    {
        public static NedKCompletePackage FromModel(
            NedKPrimeModel model,
            PrimeGapTokenizer tokenizer,
            int padId,
            int unkId,
            float? pretrainValLoss = null,
            float? posttrainValLoss = null,
            string corpusRoot = "",
            int defaultLoops = 4)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            if (tokenizer == null) throw new ArgumentNullException(nameof(tokenizer));

            var tokens = new List<string>(tokenizer.VocabSize);
            for (int i = 0; i < tokenizer.VocabSize; i++)
            {
                tokens.Add(tokenizer.Decode(new List<int> { i }));
            }

            var pkg = new NedKCompletePackage
            {
                VocabSize = model.VocabSize,
                HiddenDim = model.HiddenDim,
                FfnDim = NedKConfig.FfnDim,
                ContextWindow = model.Window,
                MaxLoops = model.MaxLoops,
                DefaultLoops = Math.Clamp(defaultLoops, 1, model.MaxLoops),
                Tokens = tokens,
                PadId = padId,
                UnkId = unkId
            };

            foreach (var p in DistinctParams(model))
            {
                pkg.OrderedTensors.Add(p.ToArray());
                int[] shape = new int[p.Shape.Rank];
                for (int r = 0; r < p.Shape.Rank; r++) shape[r] = p.Shape[r];
                pkg.OrderedShapes.Add(shape);
            }

            pkg.Meta["format"] = NedKCompletePackage.FormatId;
            pkg.Meta["magic"] = NedKCompletePackage.Magic;
            pkg.Meta["architecture"] = NedKCompletePackage.Architecture;
            pkg.Meta["family"] = NedKConfig.ModelFamily;
            pkg.Meta["dataType"] = "language_lm";
            pkg.Meta["domain"] = "code";
            pkg.Meta["tokenizer"] = "PrimeGapTokenizer";
            pkg.Meta["inferProtocol"] = OzLocalInferProtocol.ProtocolId;
            pkg.Meta["vocabSize"] = pkg.VocabSize.ToString(CultureInfo.InvariantCulture);
            pkg.Meta["hiddenDim"] = pkg.HiddenDim.ToString(CultureInfo.InvariantCulture);
            pkg.Meta["ffnDim"] = pkg.FfnDim.ToString(CultureInfo.InvariantCulture);
            pkg.Meta["contextWindow"] = pkg.ContextWindow.ToString(CultureInfo.InvariantCulture);
            pkg.Meta["maxLoops"] = pkg.MaxLoops.ToString(CultureInfo.InvariantCulture);
            pkg.Meta["defaultLoops"] = pkg.DefaultLoops.ToString(CultureInfo.InvariantCulture);
            //pkg.Meta["uniqueParams"] = model.ComputeUniqueParameters().ToString(CultureInfo.InvariantCulture);
            pkg.Meta["engineVersion"] = NedKConfig.Version.ToString("0.0", CultureInfo.InvariantCulture);

            if (!string.IsNullOrWhiteSpace(corpusRoot))
            {
                pkg.Meta["corpusRoot"] = corpusRoot;
            }

            if (pretrainValLoss.HasValue)
            {
                pkg.Meta["pretrainBestVal"] = pretrainValLoss.Value.ToString("G9", CultureInfo.InvariantCulture);
                pkg.Meta["pretrainPpl"] = pkg.CalculatePerplexity(pretrainValLoss.Value).ToString("F4", CultureInfo.InvariantCulture);
            }

            if (posttrainValLoss.HasValue)
            {
                pkg.Meta["posttrainBestVal"] = posttrainValLoss.Value.ToString("G9", CultureInfo.InvariantCulture);
                pkg.Meta["valPerplexity"] = pkg.CalculatePerplexity(posttrainValLoss.Value).ToString("F4", CultureInfo.InvariantCulture);
            }

            return pkg;
        }

        public static void Save(string path, NedKCompletePackage pkg, bool writeVocabSidecar = true)
        {
            if (pkg == null) throw new ArgumentNullException(nameof(pkg));
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Save path cannot be null or empty.", nameof(path));

            string? dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

            byte[] payload = BuildUncompressedPayload(pkg);

            using (var fs = File.Create(path))
            using (var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true))
            {
                bw.Write(Encoding.ASCII.GetBytes(NedKCompletePackage.Magic));
                bw.Write((long)payload.Length);
            }

            using (var fs = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.None))
            using (var gzip = new GZipStream(fs, CompressionLevel.Optimal, leaveOpen: true))
            {
                gzip.Write(payload, 0, payload.Length);
            }

            if (writeVocabSidecar)
            {
                string sidecar = Path.ChangeExtension(path, ".vocab.tsv");
                using var sw = new StreamWriter(sidecar, false, Encoding.UTF8);
                for (int i = 0; i < pkg.Tokens.Count; i++)
                {
                    string t = (pkg.Tokens[i] ?? "").Replace('\t', ' ').Replace('\n', ' ').Replace('\r', ' ');
                    sw.WriteLine($"{i}\t{t}");
                }
            }
        }

        public static NedKCompletePackage Load(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException("Package not found", path);

            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: true);

            string magic = Encoding.ASCII.GetString(br.ReadBytes(4));
            if (magic != NedKCompletePackage.Magic)
                throw new InvalidDataException($"Invalid magic identifier '{magic}'. Expected '{NedKCompletePackage.Magic}'.");

            long uncompressed = br.ReadInt64();
            if (uncompressed < 0 || uncompressed > 2L * 1024 * 1024 * 1024)
                throw new InvalidDataException($"Uncompressed length {uncompressed} out of valid bounds.");

            byte[] payload = new byte[uncompressed];
            using (var gzip = new GZipStream(fs, CompressionMode.Decompress, leaveOpen: true))
            {
                int off = 0;
                while (off < payload.Length)
                {
                    int n = gzip.Read(payload, off, payload.Length - off);
                    if (n <= 0) break;
                    off += n;
                }

                if (off != payload.Length)
                {
                    throw new EndOfStreamException($"Expected {payload.Length} bytes decompressed, only read {off}.");
                }
            }

            return ParseUncompressedPayload(payload);
        }

        public static NedKPrimeModel Materialize(NedKCompletePackage pkg, PrimeGapTokenizer tokenizer, Device? device = null)
        {
            if (pkg == null) throw new ArgumentNullException(nameof(pkg));
            if (tokenizer == null) throw new ArgumentNullException(nameof(tokenizer));

            device ??= Device.CPU;
            var model = new NedKPrimeModel(
                tokenizer: tokenizer,
                d: pkg.HiddenDim,
                dFf: pkg.FfnDim,
                window: pkg.ContextWindow,
                maxLoops: pkg.MaxLoops,
                device: device);

            var live = DistinctParams(model).ToList();
            if (live.Count != pkg.OrderedTensors.Count)
            {
                throw new InvalidOperationException(
                    $"Checkpoint parameter count mismatch: Model expects {live.Count} tensors, package contains {pkg.OrderedTensors.Count}.");
            }

            for (int i = 0; i < live.Count; i++)
            {
                int expectedLen = live[i].Shape.TotalElements;
                if (expectedLen <= 0)
                {
                    // Fallback: product of dims when TotalElements unavailable/zero
                    expectedLen = 1;
                    for (int r = 0; r < live[i].Shape.Rank; r++)
                        expectedLen *= live[i].Shape[r];
                }
                if (pkg.OrderedTensors[i].Length != expectedLen)
                {
                    throw new InvalidOperationException(
                        $"Tensor {i} size mismatch: Expected {expectedLen} elements (rank={live[i].Shape.Rank}), got {pkg.OrderedTensors[i].Length}.");
                }

                CopyTensorData(live[i], pkg.OrderedTensors[i]);
            }

            return model;
        }

        public static PrimeGapTokenizer MaterializeTokenizer(NedKCompletePackage pkg)
        {
            if (pkg == null) throw new ArgumentNullException(nameof(pkg));

            var dict = new Dictionary<string, int>(pkg.Tokens.Count, StringComparer.Ordinal);
            for (int i = 0; i < pkg.Tokens.Count; i++)
            {
                string t = pkg.Tokens[i] ?? "<unk>";
                if (!dict.ContainsKey(t)) dict[t] = i;
            }
            return new PrimeGapTokenizer(dict, padId: pkg.PadId, unkId: pkg.UnkId);
        }

        private static byte[] BuildUncompressedPayload(NedKCompletePackage pkg)
        {
            using var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                bw.Write(JsonSerializer.Serialize(pkg.Meta));
                bw.Write(pkg.VocabSize);
                bw.Write(pkg.HiddenDim);
                bw.Write(pkg.FfnDim);
                bw.Write(pkg.ContextWindow);
                bw.Write(pkg.MaxLoops);
                bw.Write(pkg.DefaultLoops);
                bw.Write(pkg.PadId);
                bw.Write(pkg.UnkId);

                bw.Write(pkg.Tokens.Count);
                foreach (var t in pkg.Tokens) bw.Write(t ?? string.Empty);

                bw.Write(pkg.OrderedTensors.Count);
                for (int i = 0; i < pkg.OrderedTensors.Count; i++)
                {
                    int[] shape = pkg.OrderedShapes[i];
                    bw.Write(shape.Length);
                    for (int s = 0; s < shape.Length; s++) bw.Write(shape[s]);

                    float[] tensorData = pkg.OrderedTensors[i];
                    bw.Write(tensorData.Length);
                    byte[] bytes = new byte[tensorData.Length * 4];
                    Buffer.BlockCopy(tensorData, 0, bytes, 0, bytes.Length);
                    bw.Write(bytes);
                }
            }
            return ms.ToArray();
        }

        private static NedKCompletePackage ParseUncompressedPayload(byte[] payload)
        {
            using var ms = new MemoryStream(payload, writable: false);
            using var br = new BinaryReader(ms, Encoding.UTF8);

            string metaJson = br.ReadString();
            var meta = JsonSerializer.Deserialize<Dictionary<string, string>>(metaJson)
                       ?? new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

            var pkg = new NedKCompletePackage
            {
                Meta = meta,
                VocabSize = br.ReadInt32(),
                HiddenDim = br.ReadInt32(),
                FfnDim = br.ReadInt32(),
                ContextWindow = br.ReadInt32(),
                MaxLoops = br.ReadInt32(),
                DefaultLoops = br.ReadInt32(),
                PadId = br.ReadInt32(),
                UnkId = br.ReadInt32()
            };

            int tokCount = br.ReadInt32();
            pkg.Tokens = new List<string>(tokCount);
            for (int i = 0; i < tokCount; i++) pkg.Tokens.Add(br.ReadString());

            int tensorCount = br.ReadInt32();
            pkg.OrderedTensors = new List<float[]>(tensorCount);
            pkg.OrderedShapes = new List<int[]>(tensorCount);

            for (int i = 0; i < tensorCount; i++)
            {
                int rank = br.ReadInt32();
                int[] shape = new int[rank];
                for (int r = 0; r < rank; r++) shape[r] = br.ReadInt32();

                int n = br.ReadInt32();
                byte[] bytes = br.ReadBytes(n * 4);
                float[] data = new float[n];
                Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);

                pkg.OrderedTensors.Add(data);
                pkg.OrderedShapes.Add(shape);
            }

            return pkg;
        }

        private static void CopyTensorData(ITensor target, float[] source)
        {
            const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;

            var copyMethod = target.GetType().GetMethod("CopyFrom", flags, null, new[] { typeof(float[]) }, null)
                          ?? target.GetType().GetMethod("SetData", flags, null, new[] { typeof(float[]) }, null);
            if (copyMethod != null)
            {
                copyMethod.Invoke(target, new object[] { source });
                return;
            }

            var prop = target.GetType().GetProperty("Data", flags);
            if (prop != null && prop.CanWrite && prop.PropertyType == typeof(float[]))
            {
                prop.SetValue(target, (float[])source.Clone());
                return;
            }

            foreach (string fname in new[] { "_data", "data", "_values", "values", "_buffer", "Buffer" })
            {
                var field = target.GetType().GetField(fname, flags);
                if (field != null && field.FieldType == typeof(float[]))
                {
                    field.SetValue(target, (float[])source.Clone());
                    return;
                }
            }

            var setAt = target.GetType().GetMethod("Set", new[] { typeof(int), typeof(float) })
                     ?? target.GetType().GetMethod("SetFlat", new[] { typeof(int), typeof(float) });
            if (setAt != null)
            {
                for (int i = 0; i < source.Length; i++) setAt.Invoke(target, new object[] { i, source[i] });
                return;
            }

            var indexer = target.GetType().GetProperty("Item", new[] { typeof(int) });
            if (indexer != null && indexer.CanWrite)
            {
                for (int i = 0; i < source.Length; i++) indexer.SetValue(target, source[i], new object[] { i });
                return;
            }

            throw new NotSupportedException($"Unable to copy weights into target tensor of type {target.GetType().FullName}.");
        }

        private static IEnumerable<ITensor> DistinctParams(BaseModel model)
        {
            var seen = new HashSet<ITensor>(ReferenceEqualityComparer.Instance);
            foreach (var p in model.Parameters())
            {
                if (p != null && seen.Add(p))
                {
                    yield return p;
                }
            }
        }
    }
}