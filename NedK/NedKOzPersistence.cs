namespace NedK
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.IO;
    using System.Linq;
    using ArborNet.Core.Models;
    using ArborNet.Data;

    /// <summary>
    /// High-performance persistence, spectral validation, and weight-tying projection layer
    /// for NedKV2 / NedKPrime recurrent-depth micro-LM architectures.
    /// Manages serialization, spectral radius control, tensor health diagnostics, and OZM package conformance.
    /// </summary>
    public static class NedKOzPersistence
    {
        public const string CurrentArchitecture = "NedKV2-RecurrentDepth";
        public const string LMTaskFamily = "language_lm";
        public const string DefaultTokenizerFamily = "InstantCorpusTokenizer";

        /// <summary>
        /// Converts active model buffers into an OZM-compatible package with validated shapes,
        /// spectral statistics, weight tying enforcement, and training telemetry.
        /// </summary>
        public static OzModelPackage ToPackage(
            BaseModel modelBase,
            int vocabSize,
            int hiddenDim,
            int loopsR,
            IReadOnlyList<string> tokensOrderedById,
            float[] embed,
            float[] coreDxD,
            float[] head,
            float[] layerScale,
            float? pretrainValLoss,
            float? posttrainValLoss,
            string corpusRoot)
        {
            if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize), "VocabSize must be positive.");
            if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim), "HiddenDim must be positive.");

            int clampedLoops = Math.Clamp(loopsR, 1, 64);
            int expectedEmbedSize = vocabSize * hiddenDim;

            // Enforce tied-weight head projection:
            // When head is omitted or zero-length, reuse or mirror the embedding buffer
            // to preserve the parameter budget for recurrent sequence transformations.
            float[] resolvedEmbed = embed ?? Array.Empty<float>();
            float[] resolvedHead = head ?? Array.Empty<float>();
            if (resolvedHead.Length == 0 && resolvedEmbed.Length == expectedEmbedSize)
            {
                resolvedHead = new float[expectedEmbedSize];
                Array.Copy(resolvedEmbed, resolvedHead, expectedEmbedSize);
            }

            // Default LayerScale vector to 1.0f identity scaling if not populated
            float[] resolvedLayerScale = layerScale ?? Array.Empty<float>();
            if (resolvedLayerScale.Length == 0 && hiddenDim > 0)
            {
                resolvedLayerScale = new float[hiddenDim];
                Array.Fill(resolvedLayerScale, 1.0f);
            }

            float[] resolvedCore = coreDxD ?? Array.Empty<float>();

            var pkg = new OzModelPackage
            {
                VocabSize = vocabSize,
                HiddenDim = hiddenDim,
                LoopsR = clampedLoops,
                Tokens = tokensOrderedById != null ? new List<string>(tokensOrderedById) : new List<string>(vocabSize),
                Embed = resolvedEmbed,
                Core = resolvedCore,
                Head = resolvedHead,
                LayerScale = resolvedLayerScale,
            };

            // Structural & task metadata
            pkg.Meta["format"] = OzModelCodec.FormatId;
            pkg.Meta["architecture"] = CurrentArchitecture;
            pkg.Meta["family"] = LMTaskFamily;
            pkg.Meta["tokenizer"] = DefaultTokenizerFamily;
            pkg.Meta["inferProtocol"] = OzLocalInferProtocol.ProtocolId;
            pkg.Meta["contextWindow"] = NedKConfig.ContextWindow.ToString(CultureInfo.InvariantCulture);
            pkg.Meta["vocabSize"] = vocabSize.ToString(CultureInfo.InvariantCulture);
            pkg.Meta["hiddenDim"] = hiddenDim.ToString(CultureInfo.InvariantCulture);
            pkg.Meta["loopsR"] = clampedLoops.ToString(CultureInfo.InvariantCulture);
            pkg.Meta["createdUtc"] = DateTime.UtcNow.ToString("o");
            pkg.Meta["corpusRoot"] = corpusRoot ?? string.Empty;
            pkg.Meta["note"] = "NedKV2 Recurrent-Depth SwiGLU / Tied-Head Causal OZM Package";
            pkg.Meta["weightTying"] = "true";
            pkg.Meta["causalSupervision"] = "true";

            // Numerical health & validation
            ValidateTensors(pkg);

            // Compute diagnostic summary statistics
            float embedL2 = ComputeL2Norm(pkg.Embed);
            float embedStd = ComputeStdDev(pkg.Embed);
            float coreL2 = ComputeL2Norm(pkg.Core);
            float headL2 = ComputeL2Norm(pkg.Head);
            float scaleMean = ComputeMean(pkg.LayerScale);
            float spectralRadius = EstimateSpectralRadius(pkg.Core, hiddenDim);

            pkg.Meta["embedL2Norm"] = embedL2.ToString("G6", CultureInfo.InvariantCulture);
            pkg.Meta["embedStdDev"] = embedStd.ToString("G6", CultureInfo.InvariantCulture);
            pkg.Meta["coreL2Norm"] = coreL2.ToString("G6", CultureInfo.InvariantCulture);
            pkg.Meta["headL2Norm"] = headL2.ToString("G6", CultureInfo.InvariantCulture);
            pkg.Meta["layerScaleMean"] = scaleMean.ToString("G6", CultureInfo.InvariantCulture);
            pkg.Meta["coreSpectralRadius"] = spectralRadius.ToString("G6", CultureInfo.InvariantCulture);

            if (pretrainValLoss.HasValue && !float.IsNaN(pretrainValLoss.Value))
            {
                pkg.Meta["pretrainBestVal"] = pretrainValLoss.Value.ToString("G9", CultureInfo.InvariantCulture);
            }
            if (posttrainValLoss.HasValue && !float.IsNaN(posttrainValLoss.Value))
            {
                pkg.Meta["posttrainBestVal"] = posttrainValLoss.Value.ToString("G9", CultureInfo.InvariantCulture);
            }

            // Unique non-tied active parameters
            long uniqueParams = (long)pkg.Embed.Length + pkg.Core.Length + pkg.LayerScale.Length;
            pkg.Meta["uniqueParams"] = uniqueParams.ToString(CultureInfo.InvariantCulture);
            pkg.Meta["totalAllocatedParams"] = ((long)pkg.Embed.Length + pkg.Core.Length + pkg.Head.Length + pkg.LayerScale.Length).ToString(CultureInfo.InvariantCulture);

            pkg.EnsureShapes();
            return pkg;
        }

        /// <summary>
        /// Saves the model package to disk with atomic write guarantees and optional sidecar vocabulary.
        /// </summary>
        public static void SaveOzm(string path, OzModelPackage package, bool writeVocabSidecar = true)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentNullException(nameof(path));
            if (package == null) throw new ArgumentNullException(nameof(package));

            package.EnsureShapes();
            ValidateTensors(package);

            string? directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            string tempPath = path + ".tmp_" + Guid.NewGuid().ToString("N");
            try
            {
                var codec = new OzModelCodec(package);
                codec.Save(tempPath, writeVocabSidecar: writeVocabSidecar);

                if (File.Exists(path))
                {
                    File.Replace(tempPath, path, null);
                }
                else
                {
                    File.Move(tempPath, path);
                }
            }
            catch
            {
                if (File.Exists(tempPath))
                {
                    try { File.Delete(tempPath); } catch { }
                }
                throw;
            }
        }

        /// <summary>
        /// Loads an existing OZM archive from disk with integrity verification.
        /// </summary>
        public static OzModelCodec LoadOzm(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Target OZM model file not found.", path);
            }
            var codec = OzModelCodec.Load(path);
            ValidateTensors(codec.Package);
            return codec;
        }

        /// <summary>
        /// Cache-optimized contractive projection of dual-matrix representations (e.g. SwiGLU down-projection or linear state-space)
        /// into an equivalent d x d transition core matrix. Uses an i-k-j loop order for maximum spatial locality.
        /// </summary>
        public static float[] ContractCore(float[] w1, float[] w2, int d, int dFf)
        {
            if (w1 == null) throw new ArgumentNullException(nameof(w1));
            if (w2 == null) throw new ArgumentNullException(nameof(w2));
            if (w1.Length != d * dFf) throw new ArgumentException($"w1 dimension mismatch: expected {d * dFf}, got {w1.Length}.");
            if (w2.Length != dFf * d) throw new ArgumentException($"w2 dimension mismatch: expected {dFf * d}, got {w2.Length}.");

            var core = new float[d * d];

            for (int i = 0; i < d; i++)
            {
                int iOff1 = i * dFf;
                int iOffCore = i * d;

                for (int k = 0; k < dFf; k++)
                {
                    float val1 = w1[iOff1 + k];
                    if (val1 == 0f) continue;

                    int kOff2 = k * d;
                    for (int j = 0; j < d; j++)
                    {
                        core[iOffCore + j] += val1 * w2[kOff2 + j];
                    }
                }
            }

            return core;
        }

        /// <summary>
        /// Scales the recurrent core transition matrix to guarantee contractive spectral radius (Lipschitz constant &lt;= targetBound)
        /// ensuring stable Krasnosel'skiĭ-Mann recurrent depth convergence without gradient explosion.
        /// Uses power iteration over A^T A to bound the operator 2-norm / maximum singular value.
        /// </summary>
        public static void BoundSpectralRadius(float[] core, int d, float targetBound = 0.98f)
        {
            if (core == null || core.Length != d * d || d <= 0) return;

            float maxSingularValue = EstimateOperatorNorm(core, d, iterations: 16);
            if (maxSingularValue > targetBound && maxSingularValue > 1e-7f)
            {
                float scale = targetBound / maxSingularValue;
                for (int idx = 0; idx < core.Length; idx++)
                {
                    core[idx] *= scale;
                }
            }
        }

        /// <summary>
        /// Estimates the spectral radius (dominant absolute eigenvalue) of the d x d core matrix via power iteration.
        /// </summary>
        public static float EstimateSpectralRadius(float[] core, int d, int iterations = 16)
        {
            if (core == null || core.Length != d * d || d <= 0) return 0f;

            float[] v = new float[d];
            float[] vNext = new float[d];
            float invSqrtD = 1.0f / (float)Math.Sqrt(d);
            for (int i = 0; i < d; i++) v[i] = invSqrtD;

            float lambda = 0f;
            for (int iter = 0; iter < iterations; iter++)
            {
                Array.Clear(vNext, 0, d);
                for (int i = 0; i < d; i++)
                {
                    int rowOff = i * d;
                    float sum = 0f;
                    for (int j = 0; j < d; j++)
                    {
                        sum += core[rowOff + j] * v[j];
                    }
                    vNext[i] = sum;
                }

                // Rayleigh quotient estimate: v^T (A v) / (v^T v)
                float rayleighNum = 0f;
                float normSq = 0f;
                for (int i = 0; i < d; i++)
                {
                    rayleighNum += v[i] * vNext[i];
                    normSq += vNext[i] * vNext[i];
                }

                float norm = (float)Math.Sqrt(normSq);
                if (norm < 1e-8f) return 0f;

                lambda = Math.Abs(rayleighNum);
                float invNorm = 1.0f / norm;
                for (int i = 0; i < d; i++) v[i] = vNext[i] * invNorm;
            }

            return lambda;
        }

        /// <summary>
        /// Estimates the matrix 2-norm (maximum singular value) via power iteration on A^T * A.
        /// </summary>
        public static float EstimateOperatorNorm(float[] core, int d, int iterations = 16)
        {
            if (core == null || core.Length != d * d || d <= 0) return 0f;

            float[] v = new float[d];
            float[] Av = new float[d];
            float[] AtAv = new float[d];
            float invSqrtD = 1.0f / (float)Math.Sqrt(d);
            for (int i = 0; i < d; i++) v[i] = invSqrtD;

            float sigmaSq = 0f;
            for (int iter = 0; iter < iterations; iter++)
            {
                // Av = A * v
                for (int i = 0; i < d; i++)
                {
                    int rowOff = i * d;
                    float sum = 0f;
                    for (int j = 0; j < d; j++)
                    {
                        sum += core[rowOff + j] * v[j];
                    }
                    Av[i] = sum;
                }

                // AtAv = A^T * Av
                Array.Clear(AtAv, 0, d);
                for (int i = 0; i < d; i++)
                {
                    int rowOff = i * d;
                    float avi = Av[i];
                    for (int j = 0; j < d; j++)
                    {
                        AtAv[j] += core[rowOff + j] * avi;
                    }
                }

                float normSq = 0f;
                for (int i = 0; i < d; i++) normSq += AtAv[i] * AtAv[i];
                float norm = (float)Math.Sqrt(normSq);

                if (norm < 1e-8f) return 0f;
                sigmaSq = norm;

                float invNorm = 1.0f / norm;
                for (int i = 0; i < d; i++) v[i] = AtAv[i] * invNorm;
            }

            return (float)Math.Sqrt(sigmaSq);
        }

        /// <summary>
        /// Synchronizes tied output head projection with token embeddings when weight tying is active.
        /// </summary>
        public static void SyncTiedWeights(OzModelPackage package)
        {
            if (package == null) return;
            if (package.Embed != null && package.Head != null && package.Head.Length == package.Embed.Length)
            {
                Array.Copy(package.Embed, package.Head, package.Embed.Length);
            }
        }

        /// <summary>
        /// Inspects numerical sanity of tensor buffers to prevent propagating NaNs, infinities, or destructive overflows.
        /// </summary>
        public static void ValidateTensors(OzModelPackage package)
        {
            if (package == null) throw new ArgumentNullException(nameof(package));
            CheckBufferSanity(package.Embed, "Embed");
            CheckBufferSanity(package.Core, "Core");
            CheckBufferSanity(package.Head, "Head");
            CheckBufferSanity(package.LayerScale, "LayerScale");
        }

        private static void CheckBufferSanity(float[] buffer, string name)
        {
            if (buffer == null || buffer.Length == 0) return;

            for (int i = 0; i < buffer.Length; i++)
            {
                float v = buffer[i];
                if (float.IsNaN(v))
                {
                    throw new InvalidOperationException($"Tensor sanity check failed: NaN detected in buffer '{name}' at index {i}.");
                }
                if (float.IsInfinity(v))
                {
                    throw new InvalidOperationException($"Tensor sanity check failed: Infinity detected in buffer '{name}' at index {i}.");
                }
                if (Math.Abs(v) > 1e7f)
                {
                    throw new InvalidOperationException($"Tensor sanity check failed: Extreme magnitude value {v} in buffer '{name}' at index {i}.");
                }
            }
        }

        private static float ComputeL2Norm(float[] buffer)
        {
            if (buffer == null || buffer.Length == 0) return 0f;
            double sumSq = 0.0;
            for (int i = 0; i < buffer.Length; i++)
            {
                double v = buffer[i];
                sumSq += v * v;
            }
            return (float)Math.Sqrt(sumSq);
        }

        private static float ComputeMean(float[] buffer)
        {
            if (buffer == null || buffer.Length == 0) return 0f;
            double sum = 0.0;
            for (int i = 0; i < buffer.Length; i++)
            {
                sum += buffer[i];
            }
            return (float)(sum / buffer.Length);
        }

        private static float ComputeStdDev(float[] buffer)
        {
            if (buffer == null || buffer.Length < 2) return 0f;
            double mean = ComputeMean(buffer);
            double sumDiffSq = 0.0;
            for (int i = 0; i < buffer.Length; i++)
            {
                double diff = buffer[i] - mean;
                sumDiffSq += diff * diff;
            }
            return (float)Math.Sqrt(sumDiffSq / (buffer.Length - 1));
        }
    }
}