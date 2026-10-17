// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

#nullable enable

namespace ArborNet.Data
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.IO;
    using System.IO.Compression;
    using System.Linq;
    using System.Text;
    using System.Text.Json;
    using System.Text.Json.Serialization;

    /// <summary>
    /// Portable OzzieAI <c>.ozm</c> codec: read, write, inspect, convert, and compute
    /// information-theoretic Knowledge Density (<c>Φ_KD</c>) statistics for neural models.
    /// Supports the legacy OZM1 (character-level) format and the modern OZM2 (subword token + JSON meta) standard.
    /// </summary>
    /// <remarks>
    /// Exports to:
    /// <list type="bullet">
    ///   <item><description>Standard AutoLab <c>.ozm</c> binary packages with <c>.vocab.tsv</c> sidecars.</description></item>
    ///   <item><description>Formatted JSON configuration and weight payloads.</description></item>
    ///   <item><description>Raw little-endian binary float packs (<c>.ozraw</c>).</description></item>
    ///   <item><description>Standard NumPy archives (<c>.npz</c> ZIP containers with <c>.npy</c> matrices).</description></item>
    ///   <item><description>HuggingFace-compliant little-endian <c>SafeTensors</c> packages.</description></item>
    ///   <item><description>Direct protobuf-encoded ONNX runtime models (MatMul + Scale graph).</description></item>
    /// </list>
    /// </remarks>
    public sealed class OzModelCodec
    {
        /// <summary>The 4-byte ASCII magic identifier for legacy OZM1 models.</summary>
        public const string MagicOzm1 = "OZM1";

        /// <summary>The 4-byte ASCII magic identifier for modern OZM2 models.</summary>
        public const string MagicOzm2 = "OZM2";

        /// <summary>Complete NedK graph package (gzip payload; also projects to OZM2 chassis on load).</summary>
        public const string MagicOzm3 = "OZM3";

        /// <summary>The canonical format identifier URI for AutoLab OZM2.</summary>
        public const string FormatId = "ozzieai.ozm/2";

        /// <summary>Gets the encapsulated in-memory neural model package.</summary>
        public OzModelPackage Package { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="OzModelCodec"/> class with an empty default model structure.
        /// </summary>
        public OzModelCodec()
        {
            Package = OzModelPackage.CreateEmpty(vocab: 256, dim: 64, r: 4);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="OzModelCodec"/> class wrapping an existing package.
        /// </summary>
        /// <param name="package">The concrete <see cref="OzModelPackage"/> to manage.</param>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="package"/> is null.</exception>
        public OzModelCodec(OzModelPackage package)
        {
            Package = package ?? throw new ArgumentNullException(nameof(package));
        }

        #region Loading and Saving

        /// <summary>
        /// Loads an <c>.ozm</c> model from disk, automatically resolving vocabulary sidecars (<c>.vocab.tsv</c>) if present.
        /// </summary>
        /// <param name="path">The local filesystem path to the model file.</param>
        /// <returns>An initialized <see cref="OzModelCodec"/> instance containing the loaded package.</returns>
        /// <exception cref="FileNotFoundException">Thrown when the target file does not exist.</exception>
        public static OzModelCodec Load(string path)
        {
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            {
                throw new FileNotFoundException("Model file not found.", path);
            }

            var pkg = OzModelFileIO.Load(path);

            // Sidecar paths: AutoLab uses "model.vocab.tsv"; NedK tests often write "model.ozm.vocab.tsv"
            foreach (var sidecar in new[]
                     {
                         Path.ChangeExtension(path, ".vocab.tsv"),
                         path + ".vocab.tsv",
                         Path.Combine(Path.GetDirectoryName(path) ?? ".", Path.GetFileNameWithoutExtension(path) + ".vocab.tsv")
                     })
            {
                if (!File.Exists(sidecar)) continue;
                try
                {
                    var sideTokens = OzModelFileIO.LoadVocabTsv(sidecar);
                    if (sideTokens.Count >= 32)
                    {
                        pkg.Tokens = sideTokens;
                        break;
                    }
                }
                catch { /* keep embedded table */ }
            }

            // Tolerate non-D LayerScale (e.g. projected NedK FinalNorm) by pad/truncate
            pkg.NormalizeLayerScale();

            return new OzModelCodec(pkg);
        }

        /// <summary>
        /// Saves the current model package to disk following the <c>ozzieai.ozm/2</c> specification.
        /// </summary>
        /// <param name="path">The destination filesystem path.</param>
        /// <param name="writeVocabSidecar">Whether to write a companion <c>{path}.vocab.tsv</c> file.</param>
        /// <exception cref="ArgumentException">Thrown if <paramref name="path"/> is invalid.</exception>
        public void Save(string path, bool writeVocabSidecar = true)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                throw new ArgumentException("Path required.", nameof(path));
            }

            string fullPath = Path.GetFullPath(path);
            Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);

            OzModelFileIO.Save(fullPath, Package);

            if (writeVocabSidecar)
            {
                string sidecarPath = Path.ChangeExtension(fullPath, ".vocab.tsv");
                OzModelFileIO.SaveVocabTsv(sidecarPath, Package.Tokens);
            }
        }

        /// <summary>
        /// Reloads the underlying model package from the specified path.
        /// </summary>
        /// <param name="path">The path from which to reload.</param>
        public void Reload(string path) => Package = Load(path).Package;

        #endregion

        #region Statistical Analysis and Telemetry

        /// <summary>
        /// Computes comprehensive information-theoretic, spectral, and health statistics for the model.
        /// </summary>
        /// <returns>An immutable <see cref="OzModelStats"/> snapshot.</returns>
        public OzModelStats ComputeStats() => OzModelStats.From(Package);

        /// <summary>
        /// Generates an ASCII diagnostic report detailing model geometry, parameter efficiency,
        /// stability bounds, and proxy Knowledge Density metrics.
        /// </summary>
        /// <param name="includeWeightHistograms">Whether to render the logarithmic weight distribution bars.</param>
        /// <returns>A formatted multi-line string report.</returns>
        public string FormatStatsReport(bool includeWeightHistograms = true)
        {
            var s = ComputeStats();
            var sb = new StringBuilder();
            sb.AppendLine("══════════════════════════════════════════════════════════");
            sb.AppendLine("  OzzieAI · .ozm Model Statistics");
            sb.AppendLine("══════════════════════════════════════════════════════════");
            sb.AppendLine($"  Path            : {s.SourcePath ?? "(in-memory)"}");
            sb.AppendLine($"  Format          : {s.Format}");
            sb.AppendLine($"  Architecture    : {s.Architecture}");
            sb.AppendLine($"  Family          : {s.Family}");
            sb.AppendLine($"  Created (UTC)   : {s.CreatedUtc}");
            sb.AppendLine();
            sb.AppendLine("── Geometry ──");
            sb.AppendLine($"  Vocab size (V)  : {s.VocabSize:N0}");
            sb.AppendLine($"  Hidden dim (D)  : {s.HiddenDim}");
            sb.AppendLine($"  Loops R         : {s.LoopsR}");
            sb.AppendLine($"  Unique params   : {s.UniqueParams:N0}");
            sb.AppendLine($"  Weight bytes    : {s.WeightBytes:N0}  (~{s.WeightBytes / (1024.0 * 1024):F2} MiB float32)");
            sb.AppendLine();
            sb.AppendLine("── Training Metadata ──");
            sb.AppendLine($"  Pretrain best val : {Fmt(s.PretrainBestVal)}");
            sb.AppendLine($"  Posttrain best val: {Fmt(s.PosttrainBestVal)}");
            sb.AppendLine($"  Corpus root       : {s.CorpusRoot ?? "—"}");
            sb.AppendLine();
            sb.AppendLine("── Weight Health ──");
            sb.AppendLine($"  Finite weights   : {(s.AllFinite ? "yes" : "NO — NaN/Inf detected")}");
            sb.AppendLine($"  Embed  L2 / max|w| / mean|w| / zero% : {s.Embed.L2:F4} / {s.Embed.MaxAbs:F4} / {s.Embed.MeanAbs:F6} / {s.Embed.ZeroFraction * 100:F2}%");
            sb.AppendLine($"  Core   L2 / max|w| / mean|w| / zero% : {s.Core.L2:F4} / {s.Core.MaxAbs:F4} / {s.Core.MeanAbs:F6} / {s.Core.ZeroFraction * 100:F2}%");
            sb.AppendLine($"  Head   L2 / max|w| / mean|w| / zero% : {s.Head.L2:F4} / {s.Head.MaxAbs:F4} / {s.Head.MeanAbs:F6} / {s.Head.ZeroFraction * 100:F2}%");
            sb.AppendLine($"  Scale  L2 / max|w| / mean|w|         : {s.LayerScale.L2:F4} / {s.LayerScale.MaxAbs:F4} / {s.LayerScale.MeanAbs:F6}");
            sb.AppendLine();
            sb.AppendLine("── Knowledge Density (Proxies 0–100) ──");
            sb.AppendLine($"  KD overall          : {s.KnowledgeDensityOverall:F1} / 100");
            sb.AppendLine($"    · param efficiency : {s.KdParamEfficiency:F1}   (structure per parameter)");
            sb.AppendLine($"    · multi-scale R    : {s.KdMultiScale:F1}   (loop depth utility)");
            sb.AppendLine($"    · weight utilization: {s.KdWeightUtilization:F1}   (non-degenerate mass)");
            sb.AppendLine($"    · train signal     : {s.KdTrainSignal:F1}   (val loss if known)");
            sb.AppendLine($"    · numeric stability: {s.KdStability:F1}   (finite / scale bounds)");
            sb.AppendLine($"  Bits / param (proxy) : {s.BitsPerParamProxy:F3}");
            sb.AppendLine($"  Compression vs dense : {s.CompressionVsDense:F2}×  (unique vs V·D·R naive)");
            sb.AppendLine();
            sb.AppendLine("── Derived Metrics ──");
            sb.AppendLine($"  Head≈Embed tied?     : {(s.HeadTiedToEmbed ? "likely (corr high)" : "distinct")}");
            sb.AppendLine($"  Core spectral proxy  : Frobenius={s.CoreFrobenius:F4}, diag-mean={s.CoreDiagMean:F6}");
            sb.AppendLine($"  Effective rank proxy : {s.EffectiveRankProxy:F2} / {s.HiddenDim}");
            sb.AppendLine($"  Perplexity (if CE)   : pre={FmtPpl(s.PretrainBestVal)}  post={FmtPpl(s.PosttrainBestVal)}");

            if (includeWeightHistograms)
            {
                sb.AppendLine();
                sb.AppendLine("── |w| Histogram (Embed+Core+Head, log bins) ──");
                sb.AppendLine(s.AbsWeightHistogramText);
            }

            sb.AppendLine("══════════════════════════════════════════════════════════");
            return sb.ToString();

            static string Fmt(double? v) => v.HasValue ? v.Value.ToString("F6", CultureInfo.InvariantCulture) : "—";
            static string FmtPpl(double? ce) =>
                ce.HasValue ? Math.Exp(Math.Max(0, ce.Value)).ToString("F2", CultureInfo.InvariantCulture) : "—";
        }

        #endregion

        #region Universal Exporters

        /// <summary>
        /// Converts and writes the model into every supported format within a target directory.
        /// </summary>
        /// <param name="outputDirectory">The target directory path.</param>
        /// <param name="baseName">The base file stem for output files.</param>
        /// <returns>An <see cref="OzExportManifest"/> containing the written paths and statistics.</returns>
        public OzExportManifest ConvertToAll(string outputDirectory, string baseName = "model")
        {
            if (string.IsNullOrWhiteSpace(outputDirectory))
            {
                throw new ArgumentException("Output directory required.", nameof(outputDirectory));
            }

            Directory.CreateDirectory(outputDirectory);
            var man = new OzExportManifest { OutputDirectory = outputDirectory, BaseName = baseName };
            string stem = Path.Combine(outputDirectory, baseName);

            Save(stem + ".ozm", writeVocabSidecar: true);
            man.Files["ozm"] = stem + ".ozm";
            man.Files["vocab_tsv"] = stem + ".vocab.tsv";

            ExportJson(stem + ".json");
            man.Files["json"] = stem + ".json";

            ExportRawPack(stem + ".ozraw");
            man.Files["raw"] = stem + ".ozraw";

            ExportNumpyZip(stem + ".npz");
            man.Files["numpy_zip"] = stem + ".npz";

            ExportSafeTensors(stem + ".safetensors");
            man.Files["safetensors"] = stem + ".safetensors";

            ExportOnnx(stem + ".onnx");
            man.Files["onnx"] = stem + ".onnx";

            File.WriteAllText(stem + ".stats.txt", FormatStatsReport());
            man.Files["stats"] = stem + ".stats.txt";

            man.Stats = ComputeStats();
            return man;
        }

        /// <summary>Serializes the model package into an indented UTF-8 JSON file.</summary>
        public void ExportJson(string path)
        {
            var dto = new OzJsonDto
            {
                Format = FormatId,
                Meta = Package.Meta,
                VocabSize = Package.VocabSize,
                HiddenDim = Package.HiddenDim,
                LoopsR = Package.LoopsR,
                Tokens = Package.Tokens.ToList(),
                Embed = Package.Embed.ToList(),
                Core = Package.Core.ToList(),
                Head = Package.Head.ToList(),
                LayerScale = Package.LayerScale.ToList()
            };

            string json = JsonSerializer.Serialize(dto, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(path, json, Encoding.UTF8);
        }

        /// <summary>Deserializes a model package from a JSON file.</summary>
        public static OzModelCodec ImportJson(string path)
        {
            var dto = JsonSerializer.Deserialize<OzJsonDto>(File.ReadAllText(path))
                      ?? throw new InvalidDataException("Invalid JSON model.");

            var pkg = new OzModelPackage
            {
                Meta = dto.Meta ?? new Dictionary<string, string>(),
                VocabSize = dto.VocabSize,
                HiddenDim = dto.HiddenDim,
                LoopsR = dto.LoopsR,
                Tokens = dto.Tokens?.ToList() ?? new List<string>(),
                Embed = dto.Embed?.ToArray() ?? Array.Empty<float>(),
                Core = dto.Core?.ToArray() ?? Array.Empty<float>(),
                Head = dto.Head?.ToArray() ?? Array.Empty<float>(),
                LayerScale = dto.LayerScale?.ToArray() ?? Array.Empty<float>(),
                SourcePath = path
            };

            pkg.EnsureShapes();
            return new OzModelCodec(pkg);
        }

        /// <summary>Exports tensors as a contiguous little-endian raw binary packet.</summary>
        public void ExportRawPack(string path)
        {
            using var fs = File.Create(path);
            using var bw = new BinaryWriter(fs, Encoding.UTF8);
            bw.Write(Encoding.ASCII.GetBytes("OZRW"));
            bw.Write(Package.VocabSize);
            bw.Write(Package.HiddenDim);
            bw.Write(Package.LoopsR);
            WriteFloats(bw, Package.Embed);
            WriteFloats(bw, Package.Core);
            WriteFloats(bw, Package.Head);
            WriteFloats(bw, Package.LayerScale);
        }

        /// <summary>Exports tensors to a compressed NumPy <c>.npz</c> ZIP archive.</summary>
        public void ExportNumpyZip(string path)
        {
            if (File.Exists(path)) File.Delete(path);

            using var zip = ZipFile.Open(path, ZipArchiveMode.Create);

            void AddNpy(string name, float[] data, int[] shape)
            {
                var entry = zip.CreateEntry(name, CompressionLevel.Optimal);
                using var s = entry.Open();
                WriteNpy(s, data, shape);
            }

            AddNpy("embed.npy", Package.Embed, new[] { Package.VocabSize, Package.HiddenDim });
            AddNpy("core.npy", Package.Core, new[] { Package.HiddenDim, Package.HiddenDim });
            AddNpy("head.npy", Package.Head, new[] { Package.HiddenDim, Package.VocabSize });
            AddNpy("layer_scale.npy", Package.LayerScale, new[] { Package.HiddenDim });

            var metaEntry = zip.CreateEntry("meta.json");
            using (var ms = metaEntry.Open())
            using (var sw = new StreamWriter(ms, Encoding.UTF8))
            {
                sw.Write(JsonSerializer.Serialize(Package.Meta));
            }
        }

        /// <summary>
        /// Exports tensors into HuggingFace-compatible <c>SafeTensors</c> format (8-byte aligned JSON header + float32 buffers).
        /// </summary>
        public void ExportSafeTensors(string path)
        {
            var tensors = new (string Name, float[] Data, int[] Shape)[]
            {
                ("embed", Package.Embed, new[] { Package.VocabSize, Package.HiddenDim }),
                ("core", Package.Core, new[] { Package.HiddenDim, Package.HiddenDim }),
                ("head", Package.Head, new[] { Package.HiddenDim, Package.VocabSize }),
                ("layer_scale", Package.LayerScale, new[] { Package.HiddenDim })
            };

            long offset = 0;
            var header = new Dictionary<string, object>();

            foreach (var (name, data, shape) in tensors)
            {
                long nbytes = data.LongLength * 4L;
                header[name] = new Dictionary<string, object>
                {
                    ["dtype"] = "F32",
                    ["shape"] = shape,
                    ["data_offsets"] = new[] { offset, offset + nbytes }
                };
                offset += nbytes;
            }

            header["__metadata__"] = Package.Meta.ToDictionary(kv => kv.Key, kv => (object)kv.Value);

            string headerJson = JsonSerializer.Serialize(header);
            byte[] headerBytes = Encoding.UTF8.GetBytes(headerJson);

            int pad = (8 - (headerBytes.Length % 8)) % 8;
            if (pad > 0)
            {
                var padded = new byte[headerBytes.Length + pad];
                Buffer.BlockCopy(headerBytes, 0, padded, 0, headerBytes.Length);
                for (int i = headerBytes.Length; i < padded.Length; i++) padded[i] = (byte)' ';
                headerBytes = padded;
            }

            using var fs = File.Create(path);
            using var bw = new BinaryWriter(fs);
            bw.Write((ulong)headerBytes.Length);
            bw.Write(headerBytes);

            foreach (var (_, data, _) in tensors)
            {
                var bytes = new byte[data.Length * 4];
                Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
                bw.Write(bytes);
            }
        }

        /// <summary>
        /// Writes an ONNX runtime model (opset 13) containing initializers for Embed/Core/Head/Scale
        /// and dynamic MatMul execution nodes.
        /// </summary>
        public void ExportOnnx(string path)
        {
            OzOnnxMinimalWriter.Write(
                path,
                Package.Embed, Package.VocabSize, Package.HiddenDim,
                Package.Core, Package.HiddenDim, Package.HiddenDim,
                Package.Head, Package.HiddenDim, Package.VocabSize,
                Package.LayerScale,
                Package.LoopsR,
                Package.Meta);
        }

        #endregion

        #region Internal Helper Methods

        private static void WriteFloats(BinaryWriter bw, float[] a)
        {
            bw.Write(a.Length);
            var bytes = new byte[a.Length * 4];
            Buffer.BlockCopy(a, 0, bytes, 0, bytes.Length);
            bw.Write(bytes);
        }

        private static void WriteNpy(Stream stream, float[] data, int[] shape)
        {
            var descr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                        string.Join(", ", shape) + (shape.Length == 1 ? "," : "") + "), }";
            byte[] header = Encoding.ASCII.GetBytes(descr);

            int pad = 16 - ((10 + header.Length) % 16);
            if (pad == 0) pad = 16;

            var headerPadded = new byte[header.Length + pad];
            Buffer.BlockCopy(header, 0, headerPadded, 0, header.Length);
            for (int i = header.Length; i < headerPadded.Length - 1; i++) headerPadded[i] = (byte)' ';
            headerPadded[^1] = (byte)'\n';

            using var bw = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);
            bw.Write((byte)0x93);
            bw.Write(Encoding.ASCII.GetBytes("NUMPY"));
            bw.Write((byte)1);
            bw.Write((byte)0);
            bw.Write((ushort)headerPadded.Length);
            bw.Write(headerPadded);

            var bytes = new byte[data.Length * 4];
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            bw.Write(bytes);
        }

        #endregion
    }

    #region Supporting Models and Data Transfer Objects

    /// <summary>
    /// Represents the complete memory package corresponding to the <c>ozzieai.ozm/2</c> binary specification.
    /// </summary>
    public sealed class OzModelPackage
    {
        /// <summary>Gets or sets the arbitrary string metadata dictionary.</summary>
        public Dictionary<string, string> Meta { get; set; } = new(StringComparer.OrdinalIgnoreCase);

        /// <summary>Gets or sets the vocabulary dimension size (V).</summary>
        public int VocabSize { get; set; }

        /// <summary>Gets or sets the hidden model dimension (D).</summary>
        public int HiddenDim { get; set; }

        /// <summary>Gets or sets the recurrence loops count (R).</summary>
        public int LoopsR { get; set; } = 1;

        /// <summary>Gets or sets the sequential list of subword tokens.</summary>
        public List<string> Tokens { get; set; } = new();

        /// <summary>Gets or sets the contiguous float buffer for the token embedding matrix [V x D].</summary>
        public float[] Embed { get; set; } = Array.Empty<float>();

        /// <summary>Gets or sets the contiguous float buffer for the shared recurrent core matrix [D x D].</summary>
        public float[] Core { get; set; } = Array.Empty<float>();

        /// <summary>Gets or sets the contiguous float buffer for the language modeling head matrix [D x V].</summary>
        public float[] Head { get; set; } = Array.Empty<float>();

        /// <summary>Gets or sets the per-dimension residual scale vector [D].</summary>
        public float[] LayerScale { get; set; } = Array.Empty<float>();

        /// <summary>Gets or sets the source path from which the model was loaded.</summary>
        public string? SourcePath { get; set; }

        /// <summary>
        /// Creates a newly zero-initialized model package with default scaling.
        /// </summary>
        public static OzModelPackage CreateEmpty(int vocab, int dim, int r)
        {
            var p = new OzModelPackage
            {
                VocabSize = vocab,
                HiddenDim = dim,
                LoopsR = Math.Clamp(r, 1, 64),
                Embed = new float[vocab * dim],
                Core = new float[dim * dim],
                Head = new float[dim * vocab],
                LayerScale = new float[dim]
            };

            for (int i = 0; i < dim; i++) p.LayerScale[i] = 0.1f;
            p.Meta["format"] = OzModelCodec.FormatId;
            return p;
        }

        /// <summary>
        /// Validates that all weight array dimensions align with the metadata geometry.
        /// </summary>
        /// <exception cref="InvalidDataException">Thrown if any array length does not match expected dimensions.</exception>
        public void EnsureShapes()
        {
            NormalizeLayerScale();
            if (Embed.Length != VocabSize * HiddenDim)
                throw new InvalidDataException($"Embed length {Embed.Length} != V*D ({VocabSize}*{HiddenDim}).");
            if (Core.Length != HiddenDim * HiddenDim)
                throw new InvalidDataException($"Core length {Core.Length} != D*D.");
            if (Head.Length != HiddenDim * VocabSize)
                throw new InvalidDataException($"Head length {Head.Length} != D*V ({HiddenDim}*{VocabSize}).");
            if (LayerScale.Length != HiddenDim)
                throw new InvalidDataException($"LayerScale length {LayerScale.Length} != D.");
        }

        /// <summary>
        /// Pad or truncate <see cref="LayerScale"/> to <see cref="HiddenDim"/> so projected
        /// exports (e.g. NedK FinalNorm) still load cleanly.
        /// </summary>
        public void NormalizeLayerScale()
        {
            if (HiddenDim <= 0) return;
            if (LayerScale == null || LayerScale.Length == 0)
            {
                LayerScale = new float[HiddenDim];
                for (int i = 0; i < HiddenDim; i++) LayerScale[i] = 0.1f;
                return;
            }
            if (LayerScale.Length == HiddenDim) return;
            var scaled = new float[HiddenDim];
            int n = Math.Min(LayerScale.Length, HiddenDim);
            Array.Copy(LayerScale, scaled, n);
            for (int i = n; i < HiddenDim; i++) scaled[i] = 0.1f;
            LayerScale = scaled;
        }

        /// <summary>Flattens all weight matrices into a single contiguous array.</summary>
        public float[] FlattenWeights()
        {
            var all = new float[Embed.Length + Core.Length + Head.Length + LayerScale.Length];
            Buffer.BlockCopy(Embed, 0, all, 0, Embed.Length * 4);
            Buffer.BlockCopy(Core, 0, all, Embed.Length * 4, Core.Length * 4);
            Buffer.BlockCopy(Head, 0, all, (Embed.Length + Core.Length) * 4, Head.Length * 4);
            Buffer.BlockCopy(LayerScale, 0, all, (Embed.Length + Core.Length + Head.Length) * 4, LayerScale.Length * 4);
            return all;
        }

        /// <summary>Restores all model weight matrices from a contiguous flat buffer.</summary>
        public void RestoreWeights(float[] all)
        {
            if (all.Length < Embed.Length + Core.Length + Head.Length + LayerScale.Length)
                throw new ArgumentException("Weight buffer too small.");

            Buffer.BlockCopy(all, 0, Embed, 0, Embed.Length * 4);
            Buffer.BlockCopy(all, Embed.Length * 4, Core, 0, Core.Length * 4);
            Buffer.BlockCopy(all, (Embed.Length + Core.Length) * 4, Head, 0, Head.Length * 4);
            Buffer.BlockCopy(all, (Embed.Length + Core.Length + Head.Length) * 4, LayerScale, 0, LayerScale.Length * 4);
        }

        /// <summary>Gets the total count of learnable float32 parameters.</summary>
        public long UniqueParamCount => Embed.LongLength + Core.LongLength + Head.LongLength + LayerScale.LongLength;
    }

    /// <summary>Captures summary statistical properties of a single tensor.</summary>
    public sealed class TensorStats
    {
        public double L2 { get; init; }
        public double MaxAbs { get; init; }
        public double MeanAbs { get; init; }
        public double ZeroFraction { get; init; }
        public bool AllFinite { get; init; }

        public static TensorStats From(float[] a)
        {
            if (a == null || a.Length == 0)
            {
                return new TensorStats { AllFinite = true };
            }

            double l2 = 0, maxAbs = 0, sumAbs = 0;
            int zeros = 0;
            bool finite = true;

            for (int i = 0; i < a.Length; i++)
            {
                float v = a[i];
                if (!float.IsFinite(v)) { finite = false; continue; }
                double av = Math.Abs(v);
                l2 += v * (double)v;
                if (av > maxAbs) maxAbs = av;
                sumAbs += av;
                if (av < 1e-12) zeros++;
            }

            return new TensorStats
            {
                L2 = Math.Sqrt(l2),
                MaxAbs = maxAbs,
                MeanAbs = sumAbs / a.Length,
                ZeroFraction = (double)zeros / a.Length,
                AllFinite = finite
            };
        }
    }

    /// <summary>Encapsulates metrics and proxy Knowledge Density analytics for a model.</summary>
    public sealed class OzModelStats
    {
        public string? SourcePath { get; init; }
        public string Format { get; init; } = "";
        public string Architecture { get; init; } = "";
        public string Family { get; init; } = "";
        public string CreatedUtc { get; init; } = "";
        public string? CorpusRoot { get; init; }
        public int VocabSize { get; init; }
        public int HiddenDim { get; init; }
        public int LoopsR { get; init; }
        public long UniqueParams { get; init; }
        public long WeightBytes { get; init; }
        public double? PretrainBestVal { get; init; }
        public double? PosttrainBestVal { get; init; }
        public bool AllFinite { get; init; }
        public TensorStats Embed { get; init; } = new();
        public TensorStats Core { get; init; } = new();
        public TensorStats Head { get; init; } = new();
        public TensorStats LayerScale { get; init; } = new();
        public double KnowledgeDensityOverall { get; init; }
        public double KdParamEfficiency { get; init; }
        public double KdMultiScale { get; init; }
        public double KdWeightUtilization { get; init; }
        public double KdTrainSignal { get; init; }
        public double KdStability { get; init; }
        public double BitsPerParamProxy { get; init; }
        public double CompressionVsDense { get; init; }
        public bool HeadTiedToEmbed { get; init; }
        public double CoreFrobenius { get; init; }
        public double CoreDiagMean { get; init; }
        public double EffectiveRankProxy { get; init; }
        public string AbsWeightHistogramText { get; init; } = "";

        public static OzModelStats From(OzModelPackage p)
        {
            p.EnsureShapes();
            var embed = TensorStats.From(p.Embed);
            var core = TensorStats.From(p.Core);
            var head = TensorStats.From(p.Head);
            var scale = TensorStats.From(p.LayerScale);
            bool finite = embed.AllFinite && core.AllFinite && head.AllFinite && scale.AllFinite;

            double? pre = ParseD(p.Meta, "pretrainBestVal");
            double? post = ParseD(p.Meta, "posttrainBestVal");

            long unique = p.UniqueParamCount;
            long naive = ((long)p.VocabSize * p.HiddenDim * Math.Max(1, p.LoopsR)) + ((long)p.HiddenDim * p.HiddenDim);
            double compression = naive > 0 ? (double)naive / Math.Max(1, unique) : 1.0;
            double kdEff = Clamp01(Math.Log(1 + compression) / Math.Log(1 + 8)) * 100.0;

            double kdScale = Clamp01((p.LoopsR - 1) / 15.0) * 100.0;
            double zero = (embed.ZeroFraction + core.ZeroFraction + head.ZeroFraction) / 3.0;
            double kdUtil = Clamp01(1.0 - zero) * 100.0;

            double kdTrain = 40;
            if (post.HasValue || pre.HasValue)
            {
                var ce = post ?? pre!.Value;
                kdTrain = Clamp01((12.0 - ce) / 10.0) * 100.0;
            }

            double kdStab = finite ? 100.0 : 0.0;
            if (scale.MaxAbs > 50 || core.MaxAbs > 100) kdStab *= 0.5;

            double kdOverall = (0.30 * kdEff) + (0.15 * kdScale) + (0.20 * kdUtil) + (0.25 * kdTrain) + (0.10 * kdStab);
            bool tied = ApproximateHeadTied(p);

            double fro = core.L2;
            double diag = 0;
            for (int i = 0; i < p.HiddenDim; i++)
            {
                diag += p.Core[i * p.HiddenDim + i];
            }
            diag /= Math.Max(1, p.HiddenDim);

            double effRank = EstimateEffectiveRank(p.Core, p.HiddenDim);
            double bits = finite ? (8.0 * (1.0 - zero) * (kdTrain / 100.0)) : 0;

            return new OzModelStats
            {
                SourcePath = p.SourcePath,
                Format = Get(p.Meta, "format", OzModelCodec.FormatId),
                Architecture = Get(p.Meta, "architecture", "?"),
                Family = Get(p.Meta, "family", "?"),
                CreatedUtc = Get(p.Meta, "createdUtc", "—"),
                CorpusRoot = Get(p.Meta, "corpusRoot", null),
                VocabSize = p.VocabSize,
                HiddenDim = p.HiddenDim,
                LoopsR = p.LoopsR,
                UniqueParams = unique,
                WeightBytes = unique * 4,
                PretrainBestVal = pre,
                PosttrainBestVal = post,
                AllFinite = finite,
                Embed = embed,
                Core = core,
                Head = head,
                LayerScale = scale,
                KnowledgeDensityOverall = kdOverall,
                KdParamEfficiency = kdEff,
                KdMultiScale = kdScale,
                KdWeightUtilization = kdUtil,
                KdTrainSignal = kdTrain,
                KdStability = kdStab,
                BitsPerParamProxy = bits,
                CompressionVsDense = compression,
                HeadTiedToEmbed = tied,
                CoreFrobenius = fro,
                CoreDiagMean = diag,
                EffectiveRankProxy = effRank,
                AbsWeightHistogramText = BuildHistogram(p.FlattenWeights())
            };
        }

        private static string BuildHistogram(float[] w)
        {
            var edges = new[] { 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0 };
            var counts = new int[edges.Length + 1];

            foreach (var x in w)
            {
                if (!float.IsFinite(x)) continue;
                double a = Math.Abs(x);
                int b = edges.Length;
                for (int i = 0; i < edges.Length; i++)
                {
                    if (a < edges[i]) { b = i; break; }
                }
                counts[b]++;
            }

            int max = Math.Max(1, counts.Max());
            var sb = new StringBuilder();

            for (int i = 0; i < counts.Length; i++)
            {
                string label = i == 0 ? $"< {edges[0]:G}" :
                    i == counts.Length - 1 ? $">= {edges[^1]:G}" :
                    $"[{edges[i - 1]:G}, {edges[i]:G})";
                int barLen = (int)(40.0 * counts[i] / max);
                sb.AppendLine($"  {label,-22} {counts[i],10:N0}  {new string('█', barLen)}");
            }

            return sb.ToString().TrimEnd();
        }

        private static bool ApproximateHeadTied(OzModelPackage p)
        {
            int vMax = Math.Min(p.VocabSize, 64);
            int dMax = Math.Min(p.HiddenDim, 32);
            double dot = 0, n1 = 0, n2 = 0;

            for (int v = 0; v < vMax; v++)
            {
                for (int d = 0; d < dMax; d++)
                {
                    float e = p.Embed[v * p.HiddenDim + d];
                    float h = p.Head[d * p.VocabSize + v];
                    dot += e * h; n1 += e * e; n2 += h * h;
                }
            }

            if (n1 < 1e-12 || n2 < 1e-12) return false;
            return (dot / Math.Sqrt(n1 * n2)) > 0.95;
        }

        private static double EstimateEffectiveRank(float[] core, int d)
        {
            double sum = 0, max = 0;
            for (int i = 0; i < d; i++)
            {
                double r = 0;
                for (int j = 0; j < d; j++)
                {
                    float v = core[i * d + j];
                    r += v * (double)v;
                }
                sum += r;
                if (r > max) max = r;
            }

            if (max < 1e-12) return 0;
            return Math.Min(d, sum / max);
        }

        private static double Clamp01(double x) => x < 0 ? 0 : x > 1 ? 1 : x;

        private static string Get(Dictionary<string, string> m, string k, string? def)
            => m.TryGetValue(k, out var v) && !string.IsNullOrWhiteSpace(v) ? v : def ?? "";

        private static double? ParseD(Dictionary<string, string> m, string k)
        {
            if (!m.TryGetValue(k, out var v) || string.IsNullOrWhiteSpace(v)) return null;
            if (double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var d)) return d;
            return null;
        }
    }

    /// <summary>Describes the set of files produced by <see cref="OzModelCodec.ConvertToAll"/>.</summary>
    public sealed class OzExportManifest
    {
        public string OutputDirectory { get; set; } = "";
        public string BaseName { get; set; } = "model";
        public Dictionary<string, string> Files { get; set; } = new();
        public OzModelStats? Stats { get; set; }
    }

    /// <summary>Data transfer object for full model serialization to JSON.</summary>
    public sealed class OzJsonDto
    {
        public string? Format { get; set; }
        public Dictionary<string, string>? Meta { get; set; }
        public int VocabSize { get; set; }
        public int HiddenDim { get; set; }
        public int LoopsR { get; set; }
        public List<string>? Tokens { get; set; }
        public List<float>? Embed { get; set; }
        public List<float>? Core { get; set; }
        public List<float>? Head { get; set; }
        public List<float>? LayerScale { get; set; }
    }

    #endregion

    #region Binary I/O Engine

    /// <summary>
    /// Low-level binary reader and writer strictly enforcing the <c>ozzieai.ozm/2</c> layout.
    /// </summary>
    public static class OzModelFileIO
    {
        /// <summary>
        /// Saves an <see cref="OzModelPackage"/> to a file path following the OZM2 standard.
        /// </summary>
        public static void Save(string path, OzModelPackage model)
        {
            model.EnsureShapes();
            model.Meta["format"] = OzModelCodec.FormatId;
            model.Meta["vocabSize"] = model.VocabSize.ToString(CultureInfo.InvariantCulture);
            model.Meta["hiddenDim"] = model.HiddenDim.ToString(CultureInfo.InvariantCulture);
            model.Meta["loopsR"] = model.LoopsR.ToString(CultureInfo.InvariantCulture);
            model.Meta["uniqueParams"] = model.UniqueParamCount.ToString(CultureInfo.InvariantCulture);

            if (!model.Meta.ContainsKey("createdUtc"))
            {
                model.Meta["createdUtc"] = DateTime.UtcNow.ToString("o");
            }

            var inv = new string[model.VocabSize];
            for (int i = 0; i < inv.Length; i++)
            {
                inv[i] = i < model.Tokens.Count && model.Tokens[i] != null ? model.Tokens[i] : "<unk>";
            }

            using var fs = File.Create(path);
            using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

            string metaJson = JsonSerializer.Serialize(model.Meta);
            byte[] metaBytes = Encoding.UTF8.GetBytes(metaJson);

            // 1. Magic: 4-byte ASCII "OZM2"
            bw.Write(Encoding.ASCII.GetBytes(OzModelCodec.MagicOzm2));

            // 2. Metadata JSON byte length + payload
            bw.Write(metaBytes.Length);
            bw.Write(metaBytes);

            // 3. Vocab Table: count + length-prefixed UTF-8 strings
            bw.Write(inv.Length);
            foreach (var s in inv)
            {
                byte[] b = Encoding.UTF8.GetBytes(s ?? "");
                bw.Write(b.Length);
                bw.Write(b);
            }

            // 4. Float Blobs
            WriteFloats(bw, model.Embed);
            WriteFloats(bw, model.Core);
            WriteFloats(bw, model.Head);
            WriteFloats(bw, model.LayerScale);
        }

        /// <summary>
        /// Reads and validates an <c>.ozm</c> file into an <see cref="OzModelPackage"/>.
        /// Supports both OZM2 and legacy OZM1.
        /// </summary>
        public static OzModelPackage Load(string path)
        {
            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: false);

            string magic = Encoding.ASCII.GetString(br.ReadBytes(4));
            if (magic == OzModelCodec.MagicOzm3)
            {
                // OZM3: int64 uncompressed + gzip body → project to OZM2 chassis for Micro/AutoLab
                long unc = br.ReadInt64();
                if (unc <= 0 || unc > 512L * 1024 * 1024)
                    throw new InvalidDataException("Invalid OZM3 uncompressed size.");
                var payload = new byte[unc];
                using (var gzip = new System.IO.Compression.GZipStream(fs, System.IO.Compression.CompressionMode.Decompress, leaveOpen: true))
                {
                    int off = 0;
                    while (off < payload.Length)
                    {
                        int n = gzip.Read(payload, off, payload.Length - off);
                        if (n <= 0) throw new EndOfStreamException("Truncated OZM3 gzip payload.");
                        off += n;
                    }
                }
                var pkg3 = ProjectOzm3PayloadToPackage(payload, path);
                pkg3.SourcePath = path;
                pkg3.EnsureShapes();
                return pkg3;
            }

            if (magic != OzModelCodec.MagicOzm1 && magic != OzModelCodec.MagicOzm2)
            {
                throw new InvalidDataException($"Not a valid OzzieAI model file (magic='{magic}').");
            }

            int metaLen = br.ReadInt32();
            if (metaLen < 0 || metaLen > 50_000_000)
            {
                throw new InvalidDataException("Invalid meta length.");
            }

            byte[] metaBytes = br.ReadBytes(metaLen);
            string metaJson = Encoding.UTF8.GetString(metaBytes);
            var metaFlat = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

            try
            {
                using var doc = JsonDocument.Parse(metaJson);
                foreach (var prop in doc.RootElement.EnumerateObject())
                {
                    metaFlat[prop.Name] = prop.Value.ValueKind switch
                    {
                        JsonValueKind.String => prop.Value.GetString() ?? "",
                        JsonValueKind.Number => prop.Value.ToString(),
                        JsonValueKind.True => "true",
                        JsonValueKind.False => "false",
                        _ => prop.Value.ToString()
                    };
                }
            }
            catch (Exception ex)
            {
                throw new InvalidDataException("Corrupt OZM metadata JSON payload.", ex);
            }

            int vocabSize = GetInt(metaFlat, "vocabSize", 0);
            int dim = GetInt(metaFlat, "hiddenDim", 0);
            int r = GetInt(metaFlat, "loopsR", 1);

            var tokens = new List<string>();

            if (magic == OzModelCodec.MagicOzm2)
            {
                int invLen = br.ReadInt32();
                for (int i = 0; i < invLen; i++)
                {
                    int n = br.ReadInt32();
                    if (n < 0 || n > 10_000_000) throw new InvalidDataException("Bad token length in table.");
                    tokens.Add(Encoding.UTF8.GetString(br.ReadBytes(n)));
                }
                if (vocabSize <= 0) vocabSize = invLen;
            }
            else // Legacy OZM1 character mapping
            {
                int invLen = br.ReadInt32();
                for (int i = 0; i < invLen; i++)
                {
                    int ch = br.ReadInt32();
                    tokens.Add(char.ConvertFromUtf32(Math.Clamp(ch, 0, 0x10FFFF)));
                }
                if (vocabSize <= 0) vocabSize = invLen;
            }

            if (dim <= 0) throw new InvalidDataException("hiddenDim is missing or invalid in metadata.");
            if (vocabSize <= 0) throw new InvalidDataException("vocabSize is missing or invalid in metadata.");

            float[] embed = ReadFloats(br);
            float[] core = ReadFloats(br);
            float[] head = ReadFloats(br);
            float[] scale = ReadFloats(br);

            if (embed.Length > 0 && dim > 0 && embed.Length % dim == 0)
            {
                vocabSize = embed.Length / dim;
            }

            var pkg = new OzModelPackage
            {
                Meta = metaFlat,
                VocabSize = vocabSize,
                HiddenDim = dim,
                LoopsR = Math.Clamp(r, 1, 64),
                Tokens = tokens,
                Embed = embed,
                Core = core,
                Head = head,
                LayerScale = scale,
                SourcePath = path
            };

            pkg.EnsureShapes();
            return pkg;
        }

        /// <summary>Writes a TSV file containing token index-to-string mappings.</summary>
        public static void SaveVocabTsv(string path, IReadOnlyList<string> tokens)
        {
            using var sw = new StreamWriter(path, false, Encoding.UTF8);
            for (int i = 0; i < tokens.Count; i++)
            {
                string sanitized = (tokens[i] ?? "").Replace('\t', ' ').Replace('\n', ' ').Replace('\r', ' ');
                sw.WriteLine($"{i.ToString(CultureInfo.InvariantCulture)}\t{sanitized}");
            }
        }

        /// <summary>Loads vocabulary mappings from a TSV sidecar file.</summary>
        public static List<string> LoadVocabTsv(string path)
        {
            var list = new List<string>();
            foreach (string line in File.ReadLines(path, Encoding.UTF8))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                string[] parts = line.Split('\t');
                if (parts.Length >= 2 && int.TryParse(parts[0], out int id))
                {
                    while (list.Count <= id) list.Add("<unk>");
                    list[id] = parts[1];
                }
                else
                {
                    list.Add(parts[0]);
                }
            }
            return list;
        }


        /// <summary>
        /// Parse NedK OZM3 uncompressed payload and project onto Embed/Core/Head/Scale chassis.
        /// Full ordered tensors are stored in Meta keys ozm3.param.{i}.len for diagnostics;
        /// AutoLab MicroRecurrentLm uses the projected chassis.
        /// </summary>
        private static OzModelPackage ProjectOzm3PayloadToPackage(byte[] payload, string path)
        {
            using var ms = new MemoryStream(payload, writable: false);
            using var br = new BinaryReader(ms, Encoding.UTF8);

            int metaLen = br.ReadInt32();
            if (metaLen < 0 || metaLen > 50_000_000) throw new InvalidDataException("Bad OZM3 meta length.");
            string metaJson = Encoding.UTF8.GetString(br.ReadBytes(metaLen));
            var metaFlat = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            try
            {
                using var doc = JsonDocument.Parse(metaJson);
                foreach (var prop in doc.RootElement.EnumerateObject())
                {
                    metaFlat[prop.Name] = prop.Value.ValueKind switch
                    {
                        JsonValueKind.String => prop.Value.GetString() ?? "",
                        JsonValueKind.Number => prop.Value.ToString(),
                        JsonValueKind.True => "true",
                        JsonValueKind.False => "false",
                        _ => prop.Value.ToString()
                    };
                }
            }
            catch (Exception ex)
            {
                throw new InvalidDataException("Corrupt OZM3 metadata JSON.", ex);
            }

            int vocabSize = GetInt(metaFlat, "vocabSize", 0);
            int dim = GetInt(metaFlat, "hiddenDim", 0);
            int ffn = GetInt(metaFlat, "ffnDim", dim);
            int window = GetInt(metaFlat, "contextWindow", 128);
            int maxLoops = GetInt(metaFlat, "maxLoops", 4);
            int loopsR = GetInt(metaFlat, "defaultLoops", Math.Min(3, maxLoops));
            if (loopsR < 1) loopsR = GetInt(metaFlat, "loopsR", 3);

            // geometry block written by NedKOzComplete
            // VocabSize, HiddenDim, FfnDim, ContextWindow, CompressionDim, MaxLoops, DefaultLoops, PadId, UnkId
            if (ms.Position + 9 * 4 <= ms.Length)
            {
                int v2 = br.ReadInt32();
                int d2 = br.ReadInt32();
                int f2 = br.ReadInt32();
                int win = br.ReadInt32();
                int c2 = br.ReadInt32();
                int rmax = br.ReadInt32();
                int defL = br.ReadInt32();
                int padId = br.ReadInt32();
                int unkId = br.ReadInt32();
                if (v2 > 0) vocabSize = v2;
                if (d2 > 0) dim = d2;
                if (f2 > 0) ffn = f2;
                if (win > 0) window = win;
                if (defL > 0) loopsR = defL;
                metaFlat["padId"] = padId.ToString(CultureInfo.InvariantCulture);
                metaFlat["unkId"] = unkId.ToString(CultureInfo.InvariantCulture);
                metaFlat["compressionDim"] = c2.ToString(CultureInfo.InvariantCulture);
                metaFlat["maxLoops"] = rmax.ToString(CultureInfo.InvariantCulture);
            }

            int tokCount = br.ReadInt32();
            var tokens = new List<string>(Math.Max(0, tokCount));
            for (int i = 0; i < tokCount; i++)
            {
                int n = br.ReadInt32();
                if (n < 0 || n > 10_000_000) throw new InvalidDataException("Bad OZM3 token length.");
                tokens.Add(Encoding.UTF8.GetString(br.ReadBytes(n)));
            }
            if (vocabSize <= 0) vocabSize = tokens.Count;

            int tensorCount = br.ReadInt32();
            var tensors = new List<(int[] shape, float[] data)>(tensorCount);
            for (int i = 0; i < tensorCount; i++)
            {
                int rank = br.ReadInt32();
                int[] shape = new int[rank];
                for (int r = 0; r < rank; r++) shape[r] = br.ReadInt32();
                int n = br.ReadInt32();
                byte[] bytes = br.ReadBytes(checked(n * 4));
                if (bytes.Length != n * 4) throw new EndOfStreamException("Truncated OZM3 tensor.");
                float[] data = new float[n];
                Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
                tensors.Add((shape, data));
            }

            if (dim <= 0 || vocabSize <= 0)
                throw new InvalidDataException("OZM3 missing hiddenDim/vocabSize.");

            // Project by shape onto chassis (ModelSerializer order from NedK)
            float[]? embed = null, core = null, head = null, scale = null;
            float[]? w1 = null, w2 = null;

            foreach (var (shape, data) in tensors)
            {
                int len = data.Length;
                if (embed == null && len == vocabSize * dim) { embed = data; continue; }
                if (head == null && len == dim * vocabSize && !ReferenceEquals(data, embed)) { head = data; continue; }
                if (len == dim * dim && core == null) { core = data; continue; }
                if (len == dim && scale == null) { scale = data; continue; } // first D-vector often rec.norm; prefer last later
                if (w1 == null && len == dim * ffn) { w1 = data; continue; }
                if (w2 == null && len == ffn * dim) { w2 = data; continue; }
            }

            // Prefer last length-D tensor as final LayerScale if multiple
            for (int i = tensors.Count - 1; i >= 0; i--)
            {
                if (tensors[i].data.Length == dim)
                {
                    scale = tensors[i].data;
                    break;
                }
            }

            // Contract W1·W2 → Core if Core missing
            if (core == null && w1 != null && w2 != null && ffn > 0)
            {
                core = new float[dim * dim];
                for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                {
                    float sum = 0f;
                    for (int k = 0; k < ffn; k++)
                        sum += w1[i * ffn + k] * w2[k * dim + j];
                    core[i * dim + j] = sum;
                }
            }

            if (embed == null) embed = new float[vocabSize * dim];
            if (core == null)
            {
                core = new float[dim * dim];
                for (int i = 0; i < dim; i++) core[i * dim + i] = 1f;
            }
            if (head == null)
            {
                // weight-tie transpose of embed
                head = new float[dim * vocabSize];
                for (int v = 0; v < vocabSize; v++)
                for (int d = 0; d < dim; d++)
                    head[d * vocabSize + v] = embed[v * dim + d];
            }
            if (scale == null)
            {
                scale = new float[dim];
                Array.Fill(scale, 1f);
            }

            metaFlat["format"] = OzModelCodec.FormatId;
            metaFlat["ozm3"] = "true";
            metaFlat["ozm3.paramCount"] = tensorCount.ToString(CultureInfo.InvariantCulture);
            metaFlat["architecture"] = metaFlat.GetValueOrDefault("architecture", "NedK");
            metaFlat["contextWindow"] = window.ToString(CultureInfo.InvariantCulture);
            metaFlat["loopsR"] = loopsR.ToString(CultureInfo.InvariantCulture);
            metaFlat["vocabSize"] = vocabSize.ToString(CultureInfo.InvariantCulture);
            metaFlat["hiddenDim"] = dim.ToString(CultureInfo.InvariantCulture);
            metaFlat["note"] = "OZM3 projected to OZM2 chassis for AutoLab MicroRecurrentLm";

            return new OzModelPackage
            {
                Meta = metaFlat,
                VocabSize = vocabSize,
                HiddenDim = dim,
                LoopsR = Math.Clamp(loopsR, 1, 64),
                Tokens = tokens,
                Embed = embed,
                Core = core,
                Head = head,
                LayerScale = scale,
                SourcePath = path
            };
        }


        private static int GetInt(Dictionary<string, string> m, string k, int def)
        {
            if (!m.TryGetValue(k, out var v)) return def;
            return int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) ? n : def;
        }

        private static void WriteFloats(BinaryWriter bw, float[] a)
        {
            bw.Write(a.Length);
            var bytes = new byte[a.Length * 4];
            Buffer.BlockCopy(a, 0, bytes, 0, bytes.Length);
            bw.Write(bytes);
        }

        private static float[] ReadFloats(BinaryReader br)
        {
            int n = br.ReadInt32();
            if (n < 0 || n > 500_000_000) throw new InvalidDataException("Invalid float blob element count.");
            byte[] bytes = br.ReadBytes(n * 4);
            if (bytes.Length != n * 4) throw new EndOfStreamException("Truncated float32 weight blob.");
            var a = new float[n];
            Buffer.BlockCopy(bytes, 0, a, 0, bytes.Length);
            return a;
        }
    }

    #endregion

    #region Minimal Protobuf Wire ONNX Writer

    /// <summary>
    /// Pure C# minimal ONNX binary wire encoder without external dependencies.
    /// Serializes tensors as initializers and assembles a dual-head MatMul computation graph.
    /// </summary>
    internal static class OzOnnxMinimalWriter
    {
        public static void Write(
            string path,
            float[] embed, int v, int d,
            float[] core, int coreRows, int coreCols,
            float[] head, int headRows, int headCols,
            float[] layerScale,
            int loopsR,
            Dictionary<string, string> meta)
        {
            var tensors = new List<OnnxTensor>
            {
                OnnxTensor.Float("Embed", new[] { v, d }, embed),
                OnnxTensor.Float("Core", new[] { coreRows, coreCols }, core),
                OnnxTensor.Float("Head", new[] { headRows, headCols }, head),
                OnnxTensor.Float("LayerScale", new[] { d }, layerScale)
            };

            var nodes = new List<OnnxNode>
            {
                new("MatMul", new[] { "hidden", "Core" }, new[] { "core_mm" }, "core_matmul"),
                new("Mul", new[] { "core_mm", "LayerScale" }, new[] { "core_out" }, "core_scale"),
                new("MatMul", new[] { "hidden", "Head" }, new[] { "logits" }, "head_matmul")
            };

            byte[] model = OnnxModel.Build(
                irVersion: 7,
                opset: 13,
                producer: "ArborNet.Data.OzModelCodec",
                doc: $"OzzieAI NedK Engine; R={loopsR}; Architecture={meta.GetValueOrDefault("architecture", "NedK")}",
                inputs: new[]
                {
                    OnnxValueInfo.Float("hidden", new[] { 1L, d })
                },
                outputs: new[]
                {
                    OnnxValueInfo.Float("core_out", new[] { 1L, d }),
                    OnnxValueInfo.Float("logits", new[] { 1L, v })
                },
                initializers: tensors,
                nodes: nodes);

            File.WriteAllBytes(path, model);
        }

        private sealed class OnnxTensor
        {
            public string Name = "";
            public long[] Shape = Array.Empty<long>();
            public float[] Data = Array.Empty<float>();

            public static OnnxTensor Float(string name, int[] shape, float[] data) => new()
            {
                Name = name,
                Shape = shape.Select(x => (long)x).ToArray(),
                Data = data
            };
        }

        private sealed class OnnxNode
        {
            public string Op;
            public string[] Inputs;
            public string[] Outputs;
            public string Name;

            public OnnxNode(string op, string[] inputs, string[] outputs, string name)
            {
                Op = op; Inputs = inputs; Outputs = outputs; Name = name;
            }
        }

        private sealed class OnnxValueInfo
        {
            public string Name = "";
            public long[] Shape = Array.Empty<long>();
            public static OnnxValueInfo Float(string name, long[] shape) => new() { Name = name, Shape = shape };
        }

        private static class OnnxModel
        {
            public static byte[] Build(
                int irVersion, int opset, string producer, string doc,
                OnnxValueInfo[] inputs, OnnxValueInfo[] outputs,
                List<OnnxTensor> initializers, List<OnnxNode> nodes)
            {
                using var ms = new MemoryStream();
                WriteVarintField(ms, 1, irVersion);
                WriteStringField(ms, 2, producer);
                WriteStringField(ms, 3, "1.0");
                WriteStringField(ms, 4, doc);

                byte[] graph = BuildGraph("main", inputs, outputs, initializers, nodes);
                WriteBytesField(ms, 7, graph);

                using (var op = new MemoryStream())
                {
                    WriteStringField(op, 1, "");
                    WriteVarintField(op, 2, opset);
                    WriteBytesField(ms, 8, op.ToArray());
                }

                return ms.ToArray();
            }

            private static byte[] BuildGraph(
                string name, OnnxValueInfo[] inputs, OnnxValueInfo[] outputs,
                List<OnnxTensor> inits, List<OnnxNode> nodes)
            {
                using var ms = new MemoryStream();
                foreach (var n in nodes) WriteBytesField(ms, 1, BuildNode(n));
                WriteStringField(ms, 2, name);
                foreach (var t in inits) WriteBytesField(ms, 5, BuildTensor(t));
                foreach (var i in inputs) WriteBytesField(ms, 11, BuildValueInfo(i));
                foreach (var o in outputs) WriteBytesField(ms, 12, BuildValueInfo(o));
                return ms.ToArray();
            }

            private static byte[] BuildNode(OnnxNode n)
            {
                using var ms = new MemoryStream();
                foreach (var i in n.Inputs) WriteStringField(ms, 1, i);
                foreach (var o in n.Outputs) WriteStringField(ms, 2, o);
                WriteStringField(ms, 3, n.Name);
                WriteStringField(ms, 4, n.Op);
                return ms.ToArray();
            }

            private static byte[] BuildValueInfo(OnnxValueInfo v)
            {
                using var ms = new MemoryStream();
                WriteStringField(ms, 1, v.Name);

                using (var typeMs = new MemoryStream())
                using (var tensorType = new MemoryStream())
                {
                    WriteVarintField(tensorType, 1, 1); // FLOAT
                    using (var shape = new MemoryStream())
                    {
                        foreach (var dim in v.Shape)
                        {
                            using var dimMsg = new MemoryStream();
                            WriteVarintField(dimMsg, 1, dim);
                            WriteBytesField(shape, 1, dimMsg.ToArray());
                        }
                        WriteBytesField(tensorType, 2, shape.ToArray());
                    }
                    WriteBytesField(typeMs, 1, tensorType.ToArray());
                    WriteBytesField(ms, 2, typeMs.ToArray());
                }

                return ms.ToArray();
            }

            private static byte[] BuildTensor(OnnxTensor t)
            {
                using var ms = new MemoryStream();
                foreach (var d in t.Shape) WriteVarintField(ms, 1, d);
                WriteVarintField(ms, 2, 1); // FLOAT

                var bytes = new byte[t.Data.Length * 4];
                Buffer.BlockCopy(t.Data, 0, bytes, 0, bytes.Length);
                WriteBytesField(ms, 8, bytes);
                WriteStringField(ms, 9, t.Name);
                return ms.ToArray();
            }

            private static void WriteVarintField(Stream s, int field, long value)
            {
                WriteKey(s, field, 0);
                WriteVarint(s, unchecked((ulong)value));
            }

            private static void WriteStringField(Stream s, int field, string value)
            {
                byte[] b = Encoding.UTF8.GetBytes(value ?? "");
                WriteBytesField(s, field, b);
            }

            private static void WriteBytesField(Stream s, int field, byte[] data)
            {
                WriteKey(s, field, 2);
                WriteVarint(s, (ulong)data.Length);
                s.Write(data, 0, data.Length);
            }

            private static void WriteKey(Stream s, int field, int wire)
                => WriteVarint(s, ((ulong)field << 3) | (uint)wire);

            private static void WriteVarint(Stream s, ulong v)
            {
                while (v >= 0x80)
                {
                    s.WriteByte((byte)(v | 0x80));
                    v >>= 7;
                }
                s.WriteByte((byte)v);
            }
        }
    }

    #endregion
}