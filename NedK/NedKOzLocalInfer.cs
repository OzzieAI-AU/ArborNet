#nullable enable

namespace NedK
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Linq;
    using System.Reflection;
    using System.Text;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Data;

    public sealed class OzChatMessage
    {
        public string Role { get; set; } = "user";
        public string Content { get; set; } = "";

        public OzChatMessage() { }

        public OzChatMessage(string role, string content)
        {
            Role = role;
            Content = content ?? string.Empty;
        }
    }

    public sealed class OzLocalInferRequest
    {
        public List<OzChatMessage> Messages { get; set; } = new();
        public int MaxNewTokens { get; set; } = 64;
        public int MinNewTokens { get; set; } = 1;
        public float Temperature { get; set; } = 0.2f;
        public float TopP { get; set; } = 0.90f;
        public int TopK { get; set; } = 40;
        public float MinP { get; set; } = 0.05f;
        public float RepetitionPenalty { get; set; } = 1.15f;
        public float PresencePenalty { get; set; } = 0.0f;
        public float FrequencyPenalty { get; set; } = 0.0f;
        public int Loops { get; set; } = 4;
        public int Window { get; set; } = NedKConfig.ContextWindow;
        public bool StopOnSentenceEnd { get; set; } = false;
        public bool PreserveIndentation { get; set; } = true;
        public List<string> StopTokens { get; set; } = new()
        {
            "<|endoftext|>",
            "<|pad|>",
            "<|eos|>",
            "</s>",
            "<|im_end|>",
            "<|end|>",
            "User:",
            "System:"
        };
    }

    public sealed class OzLocalInferResponse
    {
        public string Role { get; set; } = "assistant";
        public string Text { get; set; } = "";
        public string ProtocolId { get; set; } = OzLocalInferProtocol.ProtocolId;
        public int TokensGenerated { get; set; }
        public string FinishReason { get; set; } = "stop";
    }

    public static class OzLocalInferProtocol
    {
        public const string ProtocolId = "ozzieai.local-infer/1";
        public const string DefaultSystemPrompt =
            "You are a precise engineering assistant specializing in software, systems, and algorithms. "
            + "Answer clearly and directly. Prefer mechanisms over slogans.";

        public static string FormatPrompt(IReadOnlyList<OzChatMessage> messages)
        {
            var sb = new StringBuilder();
            if (messages == null || messages.Count == 0)
            {
                return "User: \nAssistant:";
            }

            foreach (var m in messages)
            {
                if (m == null || string.IsNullOrWhiteSpace(m.Content)) continue;
                string role = (m.Role ?? "user").Trim().ToLowerInvariant();
                string label = role switch
                {
                    "system" => "System",
                    "assistant" or "model" => "Assistant",
                    _ => "User"
                };
                sb.Append(label).Append(": ").Append(m.Content.Trim()).Append('\n');
            }

            if (!sb.ToString().TrimEnd().EndsWith("Assistant:", StringComparison.Ordinal))
            {
                sb.Append("Assistant:");
            }
            return sb.ToString();
        }

        public static bool ShouldStop(string textWindow, bool isCodeContext = false, IReadOnlyList<string>? stopTokens = null)
        {
            if (string.IsNullOrEmpty(textWindow)) return false;

            if (stopTokens != null)
            {
                for (int i = 0; i < stopTokens.Count; i++)
                {
                    string st = stopTokens[i];
                    if (!string.IsNullOrEmpty(st) && textWindow.EndsWith(st, StringComparison.OrdinalIgnoreCase))
                    {
                        return true;
                    }
                }
            }

            if (textWindow.EndsWith("<|endoftext|>", StringComparison.OrdinalIgnoreCase) ||
                textWindow.EndsWith("<|pad|>", StringComparison.OrdinalIgnoreCase) ||
                textWindow.EndsWith("<|eos|>", StringComparison.OrdinalIgnoreCase) ||
                textWindow.EndsWith("</s>", StringComparison.OrdinalIgnoreCase) ||
                textWindow.EndsWith("<|im_end|>", StringComparison.OrdinalIgnoreCase) ||
                textWindow.EndsWith("<|end|>", StringComparison.OrdinalIgnoreCase) ||
                textWindow.EndsWith("User:", StringComparison.OrdinalIgnoreCase) ||
                textWindow.EndsWith("System:", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            return false;
        }
    }

    public delegate ITensor NedKForwardFn(ITensor input, int loops);

    public static class NedKOzLocalInfer
    {
        private static readonly ConcurrentDictionary<Type, MethodInfo?> EncodeMethodCache = new();
        private static readonly ConcurrentDictionary<Type, MethodInfo?> DecodeMethodCache = new();

        public static OzLocalInferResponse Generate(
                    NedKForwardFn forward,
                    int maxLoops,
                    ITokenizer tokenizer,
                    Device device,
                    OzLocalInferRequest request,
                    int padId = 0)
        {
            if (forward == null) throw new ArgumentNullException(nameof(forward));
            if (tokenizer == null) throw new ArgumentNullException(nameof(tokenizer));
            if (request == null) throw new ArgumentNullException(nameof(request));

            int window = request.Window > 0 ? request.Window : NedKConfig.ContextWindow;
            int loops = Math.Clamp(request.Loops, 1, Math.Max(1, maxLoops));
            int maxTokens = Math.Clamp(request.MaxNewTokens, 1, 1024);
            int minTokens = Math.Clamp(request.MinNewTokens, 0, maxTokens);

            if (tokenizer is ITokenizer ict)
            {
                padId = ict.PadId;
            }

            string flat = (request.Messages != null && request.Messages.Count == 1)
                ? request.Messages[0].Content.Trim()
                : OzLocalInferProtocol.FormatPrompt(request.Messages);

            List<int> tokenContext = EncodeAsList(tokenizer, flat);
            if (tokenContext.Count == 0)
            {
                tokenContext.Add(padId);
            }

            bool isCodeMode = request.PreserveIndentation ||
                              flat.Contains("class ", StringComparison.Ordinal) ||
                              flat.Contains("def ", StringComparison.Ordinal) ||
                              flat.Contains("fn ", StringComparison.Ordinal) ||
                              flat.Contains("public ", StringComparison.Ordinal) ||
                              flat.Contains("private ", StringComparison.Ordinal) ||
                              flat.Contains("return ", StringComparison.Ordinal) ||
                              flat.Contains('{') ||
                              flat.Contains(';');

            var emitted = new List<int>(maxTokens);
            var tokenFrequencies = new Dictionary<int, int>(128);
            foreach (int tok in tokenContext)
            {
                tokenFrequencies[tok] = tokenFrequencies.GetValueOrDefault(tok, 0) + 1;
            }

            var rnd = new Random(42);
            var winBuffer = new float[window];
            string finishReason = "length";
            var textTracker = new StringBuilder();

            (int Index, float Logit)[]? candidateBuffer = null;
            (int Index, double Prob)[]? probBuffer = null;

            for (int step = 0; step < maxTokens; step++)
            {
                int copy = Math.Min(tokenContext.Count, window);
                int src = tokenContext.Count - copy;

                for (int t = 0; t < window - copy; t++)
                {
                    winBuffer[t] = padId;
                }
                for (int t = 0; t < copy; t++)
                {
                    winBuffer[window - copy + t] = tokenContext[src + t];
                }

                float[] rawLogits;
                // Deterministic disposal of intermediate inference tensors on every generated token!
                using (new TensorScope())
                {
                    ITensor inputTensor = Tensor.FromArray(winBuffer, new TensorShape(1, window), device);
                    ITensor logitsTensor = forward(inputTensor, loops);
                    rawLogits = logitsTensor.ToArray();
                }

                if (rawLogits.Length == 0)
                {
                    finishReason = "error";
                    break;
                }

                int vocabSize = rawLogits.Length >= window ? rawLogits.Length / window : rawLogits.Length;
                float[] stepLogits = rawLogits.Length > vocabSize
                    ? ExtractLastStepLogits(rawLogits, window, vocabSize)
                    : (float[])rawLogits.Clone();

                if (candidateBuffer == null || candidateBuffer.Length != vocabSize)
                {
                    candidateBuffer = new (int Index, float Logit)[vocabSize];
                    probBuffer = new (int Index, double Prob)[vocabSize];
                }

                if (request.RepetitionPenalty > 1.0f || request.PresencePenalty > 0.0f || request.FrequencyPenalty > 0.0f)
                {
                    ApplyPenalties(stepLogits, tokenFrequencies, request.RepetitionPenalty, request.PresencePenalty, request.FrequencyPenalty);
                }

                // Always suppress <unk> (token 1) so the sampler picks real vocabulary words
                if (stepLogits.Length > 1)
                {
                    stepLogits[1] = float.NegativeInfinity;
                }
                if (step < minTokens && padId >= 0 && padId < stepLogits.Length)
                {
                    stepLogits[padId] = float.NegativeInfinity;
                }

                int next = request.Temperature <= 0.05f
                    ? ArgMax(stepLogits)
                    : SampleTopKTopPMinP(stepLogits, request.Temperature, request.TopK, request.TopP, request.MinP, candidateBuffer, probBuffer, rnd);

                if (next == padId)
                {
                    finishReason = "eos";
                    break;
                }

                tokenContext.Add(next);
                emitted.Add(next);
                tokenFrequencies[next] = tokenFrequencies.GetValueOrDefault(next, 0) + 1;

                string piece = DecodeOne(tokenizer, next);
                if (!string.IsNullOrEmpty(piece) && piece != "<unk>")
                {
                    if (textTracker.Length > 0
                        && char.IsLetterOrDigit(textTracker[^1])
                        && char.IsLetterOrDigit(piece[0]))
                    {
                        textTracker.Append(' ');
                    }
                    textTracker.Append(piece);
                }

                string trailingWindow = textTracker.Length > 64
                    ? textTracker.ToString(textTracker.Length - 64, 64)
                    : textTracker.ToString();

                if (OzLocalInferProtocol.ShouldStop(trailingWindow, isCodeMode, request.StopTokens))
                {
                    finishReason = "stop";
                    break;
                }

                if (request.StopOnSentenceEnd && !isCodeMode && step >= minTokens && (piece.Contains('.') || piece.Contains('\n')))
                {
                    finishReason = "stop";
                    break;
                }
            }

            var sbDecoded = new StringBuilder();
            for (int i = 0; i < emitted.Count; i++)
            {
                string piece = DecodeOne(tokenizer, emitted[i]);
                if (string.IsNullOrEmpty(piece) || piece == "<unk>") continue;

                if (sbDecoded.Length > 0
                    && char.IsLetterOrDigit(sbDecoded[^1])
                    && char.IsLetterOrDigit(piece[0]))
                {
                    sbDecoded.Append(' ');
                }
                sbDecoded.Append(piece);
            }

            string cleaned = CleanGeneratedText(sbDecoded.ToString(), request.StopTokens);

            return new OzLocalInferResponse
            {
                Role = "assistant",
                Text = cleaned,
                ProtocolId = OzLocalInferProtocol.ProtocolId,
                TokensGenerated = emitted.Count,
                FinishReason = finishReason
            };
        }

        public static void GenerateSentence(
            BaseModel model,
            ITokenizer tokenizer,
            Device device,
            int padId,
            int window,
            string rawPrompt,
            int loops,
            int maxTokens = 32)
        {
            var mi = model.GetType().GetMethod("Forward", new[] { typeof(ITensor), typeof(int) })
                     ?? throw new InvalidOperationException("Model missing Forward(ITensor, int).");
            var maxLoopsProp = model.GetType().GetProperty("MaxLoops");
            int maxLoops = maxLoopsProp != null ? Convert.ToInt32(maxLoopsProp.GetValue(model)) : NedKConfig.MaxDynamicLoops;

            NedKForwardFn fwd = (input, L) => (ITensor)mi.Invoke(model, new object[] { input, L })!;

            var resp = Generate(fwd, maxLoops, tokenizer, device, new OzLocalInferRequest
            {
                Messages =
                {
                    new OzChatMessage("system", OzLocalInferProtocol.DefaultSystemPrompt),
                    new OzChatMessage("user", (rawPrompt ?? string.Empty).TrimEnd())
                },
                MaxNewTokens = maxTokens,
                Temperature = 0f,
                Loops = loops,
                Window = window > 0 ? window : NedKConfig.ContextWindow,
                StopOnSentenceEnd = true
            }, padId);

            Console.Write($"{rawPrompt}[");
            Console.Write(resp.Text);
            Console.WriteLine("]");
        }

        private static float[] ExtractLastStepLogits(float[] logits, int seqLen, int vocabSize)
        {
            var slice = new float[vocabSize];
            int offset = (seqLen - 1) * vocabSize;
            if (offset >= 0 && offset + vocabSize <= logits.Length)
            {
                Array.Copy(logits, offset, slice, 0, vocabSize);
            }
            else
            {
                Array.Copy(logits, Math.Max(0, logits.Length - vocabSize), slice, 0, vocabSize);
            }
            return slice;
        }

        private static void ApplyPenalties(float[] logits, Dictionary<int, int> frequencies, float repPenalty, float presencePenalty, float freqPenalty)
        {
            foreach (var kvp in frequencies)
            {
                int id = kvp.Key;
                int count = kvp.Value;
                if (id >= 0 && id < logits.Length && count > 0)
                {
                    if (repPenalty > 1.0f)
                    {
                        if (logits[id] > 0f)
                            logits[id] /= repPenalty;
                        else
                            logits[id] *= repPenalty;
                    }

                    if (presencePenalty > 0.0f)
                    {
                        logits[id] -= presencePenalty;
                    }

                    if (freqPenalty > 0.0f)
                    {
                        logits[id] -= freqPenalty * count;
                    }
                }
            }
        }

        private static List<int> EncodeAsList(ITokenizer tokenizer, string text)
        {
            var type = tokenizer.GetType();
            var enc = EncodeMethodCache.GetOrAdd(type, t => t.GetMethod("Encode", new[] { typeof(string) }));
            if (enc == null) return new List<int>();

            var result = enc.Invoke(tokenizer, new object[] { text });
            if (result is List<int> list) return list;
            if (result is int[] arr) return arr.ToList();
            if (result is IEnumerable<int> en) return en.ToList();
            return new List<int>();
        }

        private static string DecodeOne(ITokenizer tokenizer, int id)
        {
            var type = tokenizer.GetType();
            var dec = DecodeMethodCache.GetOrAdd(type, t =>
                t.GetMethod("Decode", new[] { typeof(List<int>) })
                ?? t.GetMethod("Decode", new[] { typeof(IList<int>) })
                ?? t.GetMethod("Decode", new[] { typeof(int[]) }));

            if (dec == null) return "<unk>";
            var paramType = dec.GetParameters()[0].ParameterType;
            object arg = paramType == typeof(int[])
                ? new[] { id }
                : (object)new List<int> { id };

            return dec.Invoke(tokenizer, new[] { arg }) as string ?? "<unk>";
        }

        private static string DecodeMany(ITokenizer tokenizer, List<int> ids)
        {
            var type = tokenizer.GetType();
            var dec = DecodeMethodCache.GetOrAdd(type, t =>
                t.GetMethod("Decode", new[] { typeof(List<int>) })
                ?? t.GetMethod("Decode", new[] { typeof(IList<int>) })
                ?? t.GetMethod("Decode", new[] { typeof(int[]) }));

            if (dec == null) return string.Join("", ids);
            var paramType = dec.GetParameters()[0].ParameterType;
            object arg = paramType == typeof(int[])
                ? ids.ToArray()
                : (object)ids;

            return dec.Invoke(tokenizer, new[] { arg }) as string ?? string.Empty;
        }

        private static string CleanGeneratedText(string raw, IReadOnlyList<string>? stopTokens = null)
        {
            if (string.IsNullOrEmpty(raw)) return string.Empty;

            var sb = new StringBuilder(raw);
            sb.Replace("<|endoftext|>", "")
              .Replace("<|pad|>", "")
              .Replace("<|eos|>", "")
              .Replace("</s>", "")
              .Replace("<|im_end|>", "")
              .Replace("<|end|>", "")
              .Replace("<unk>", "");

            string result = sb.ToString();

            if (stopTokens != null)
            {
                foreach (var st in stopTokens)
                {
                    if (string.IsNullOrEmpty(st)) continue;
                    int idx = result.IndexOf(st, StringComparison.OrdinalIgnoreCase);
                    if (idx >= 0)
                    {
                        result = result.Substring(0, idx);
                    }
                }
            }

            return result.Trim();
        }

        private static int ArgMax(float[] z)
        {
            int iMax = 0;
            float best = float.NegativeInfinity;
            for (int i = 0; i < z.Length; i++)
            {
                float val = z[i];
                if (float.IsNaN(val) || float.IsInfinity(val)) continue;
                if (val > best)
                {
                    best = val;
                    iMax = i;
                }
            }
            return iMax;
        }

        private static int SampleTopKTopPMinP(
            float[] z,
            float temperature,
            int topK,
            float topP,
            float minP,
            (int Index, float Logit)[] candidates,
            (int Index, double Prob)[] probs,
            Random rnd)
        {
            float max = float.NegativeInfinity;
            for (int i = 0; i < z.Length; i++)
            {
                float v = z[i];
                if (!float.IsNaN(v) && !float.IsInfinity(v) && v > max)
                {
                    max = v;
                }
            }

            if (float.IsNegativeInfinity(max)) return 0;

            float t = Math.Max(0.05f, temperature);
            int count = z.Length;
            for (int i = 0; i < count; i++)
            {
                candidates[i] = (i, z[i]);
            }

            Array.Sort(candidates, 0, count, Comparer<(int Index, float Logit)>.Create((a, b) => b.Logit.CompareTo(a.Logit)));

            int effectiveK = topK > 0 ? Math.Min(topK, count) : count;

            double sum = 0.0;
            for (int i = 0; i < effectiveK; i++)
            {
                float val = candidates[i].Logit;
                double p = (float.IsNaN(val) || float.IsInfinity(val)) ? 0.0 : Math.Exp((val - max) / t);
                probs[i] = (candidates[i].Index, p);
                sum += p;
            }

            if (sum <= 0.0 || double.IsNaN(sum)) return candidates[0].Index;

            for (int i = 0; i < effectiveK; i++)
            {
                probs[i].Prob /= sum;
            }

            if (minP > 0.0f && effectiveK > 1)
            {
                double pMax = probs[0].Prob;
                double threshold = pMax * minP;
                int validCount = 0;
                double filteredSum = 0.0;

                for (int i = 0; i < effectiveK; i++)
                {
                    if (probs[i].Prob >= threshold)
                    {
                        validCount++;
                        filteredSum += probs[i].Prob;
                    }
                    else
                    {
                        break;
                    }
                }

                if (validCount > 0 && filteredSum > 0.0)
                {
                    effectiveK = validCount;
                    for (int i = 0; i < effectiveK; i++)
                    {
                        probs[i].Prob /= filteredSum;
                    }
                }
            }

            double cumulative = 0.0;
            int cutoff = effectiveK - 1;
            for (int i = 0; i < effectiveK; i++)
            {
                cumulative += probs[i].Prob;
                if (cumulative >= topP)
                {
                    cutoff = i;
                    break;
                }
            }

            double activeSum = 0.0;
            for (int i = 0; i <= cutoff; i++)
            {
                activeSum += probs[i].Prob;
            }

            if (activeSum <= 0.0) return probs[0].Index;

            double u = rnd.NextDouble() * activeSum;
            for (int i = 0; i <= cutoff; i++)
            {
                u -= probs[i].Prob;
                if (u <= 0.0) return probs[i].Index;
            }

            return probs[0].Index;
        }
    }
}