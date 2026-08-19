// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// Project:      ArborNet
// Description:  On-graph 2D Kimi K3 char LM — pad-aware next-char (tape-proven ops)
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace KimiK3Test
{
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;
    using ArborNet.Losses;
    using ArborNet.Models;
    using ArborNet.Optimizers;
    using System;
    using System.Collections.Generic;
    using System.Linq;

    public static class Num
    {
        public static bool Bad(ITensor t)
        {
            var d = t.ToArray();
            return d.Any(v => float.IsNaN(v) || float.IsInfinity(v));
        }
    }

    public sealed class MixBlock : BaseLayer
    {
        private readonly Linear _a, _b;

        public MixBlock(int d, Device device)
        {
            this.device = device;
            _a = new Linear(d, d * 2, device);
            _b = new Linear(d * 2, d, device);
        }

        public override ITensor Forward(ITensor x)
            => x.Add(_b.Forward(_a.Forward(x).Relu()));

        public override IEnumerable<ITensor> Parameters()
        {
            foreach (var p in _a.Parameters()) yield return p;
            foreach (var p in _b.Parameters()) yield return p;
        }
    }

    public sealed class KimiK3 : BaseModel
    {
        private readonly Embedding _tok, _pos;
        private readonly MixBlock _mix1, _mix2;
        private readonly Linear _win1, _win2, _winOut;
        private readonly Linear _lastOut;
        private readonly ITensor _posIds;
        private readonly int _d, _window;

        public int Window => _window;

        public KimiK3(int vocab, int d, int window, int maxLen, Device device)
        {
            _d = d;
            _window = window;
            _tok = new Embedding(vocab, d);
            _pos = new Embedding(maxLen, d);
            _mix1 = new MixBlock(d, device);
            _mix2 = new MixBlock(d, device);

            int flat = window * d;
            _win1 = new Linear(flat, 256, device);
            _win2 = new Linear(256, 128, device);
            _winOut = new Linear(128, vocab, device);
            _lastOut = new Linear(d, vocab, device);

            var ids = Enumerable.Range(0, maxLen).Select(i => (float)i).ToArray();
            _posIds = Tensor.FromArray(ids, new TensorShape(1, maxLen), device);
            Register();
        }

        void Register()
        {
            parameters.Clear();
            parameters.AddRange(_tok.Parameters());
            parameters.AddRange(_pos.Parameters());
            parameters.AddRange(_mix1.Parameters());
            parameters.AddRange(_mix2.Parameters());
            parameters.AddRange(_win1.Parameters());
            parameters.AddRange(_win2.Parameters());
            parameters.AddRange(_winOut.Parameters());
            parameters.AddRange(_lastOut.Parameters());
        }

        public override ITensor Forward(ITensor input)
        {
            int T = input.Shape[1];
            if (T != _window)
                throw new ArgumentException($"Forward needs T={_window}, got {T}");

            var x = _tok.Forward(input.Reshape(T));
            var pos = _pos.Forward(_posIds.Slice((0, 1, 1), (0, T, 1)).Reshape(T));
            x = _mix2.Forward(_mix1.Forward(x.Add(pos)));

            var fromWin = _winOut.Forward(
                _win2.Forward(_win1.Forward(x.Reshape(T * _d).Unsqueeze(0)).Relu()).Relu());
            var fromLast = _lastOut.Forward(x.Slice((T - 1, T, 1), (0, _d, 1)));
            return fromWin.Add(fromLast).Clip(-20f, 20f);
        }
    }

    class Program
    {
        static void Main()
        {
            Console.WriteLine("=== Kimi K3 – next-char window LM (pad-aware) ===\n");
            var device = Device.CPU;

            string[] corpus =
            {
                "ArborNet makes Kimi K3 exceptionally fast and smart.",
                "Kimi K3 is a frontier mixture of experts language model.",
                "The future of artificial intelligence is open source.",
                "Machine learning models should be efficient and beautiful.",
                "CUDA acceleration makes training much faster on GPUs.",
                "Residual connections help deep networks converge better.",
                "Sparse experts allow scaling to trillions of parameters.",
                "Hello world from the Kimi K3 architecture.",
                "Training language models requires careful optimisation.",
                "Attention is all you need, but linear attention is faster."
            };

            string stream = string.Join(" ", corpus) + " ";
            const char PadChar = '\u0001';
            var chars = stream.Distinct().OrderBy(c => c).ToList();
            if (!chars.Contains(PadChar)) chars.Insert(0, PadChar);

            int V = chars.Count;
            var c2i = chars.Select((c, i) => (c, i)).ToDictionary(x => x.c, x => x.i);
            var i2c = chars.Select((c, i) => (i, c)).ToDictionary(x => x.i, x => x.c);
            int padId = c2i[PadChar];
            int spaceId = c2i[' '];
            var ids = stream.Select(c => c2i[c]).ToArray();
            Console.WriteLine($"Vocab: {V}  Stream: {ids.Length}  spaceId={spaceId}  padId={padId}");

            const int d = 40, window = 20, maxLen = 40;
            var model = new KimiK3(V, d, window, maxLen, device);

            var data = new List<(float[] x, float[] y, int tgt)>();

            void AddWindow(int[] src, int start)
            {
                var x = new float[window];
                for (int t = 0; t < window; t++)
                {
                    int p = start + t;
                    x[t] = (p < 0) ? padId : src[p];
                }
                int tgtPos = start + window;
                if (tgtPos < 0 || tgtPos >= src.Length) return;
                int tgt = src[tgtPos];
                var y = new float[V];
                y[tgt] = 1f;
                data.Add((x, y, tgt));
            }

            // full-stream sliding windows
            for (int i = 0; i + window < ids.Length; i++)
                AddWindow(ids, i);

            // sentence prefixes with LEFT PAD — this is what short prompts need
            foreach (var s in corpus)
            {
                var sid = s.Select(c => c2i[c]).Concat(new[] { spaceId }).ToArray();
                for (int used = 1; used <= Math.Min(window, sid.Length - 1); used++)
                    AddWindow(sid, used - window);
            }

            Console.WriteLine($"Windows: {data.Count}\n");

            float lr = 2e-2f;
            var opt = new SGD(learningRate: lr);
            var ce = new CrossEntropy();
            model.Train();

            int epochs = 80;
            for (int ep = 1; ep <= epochs; ep++)
            {
                if (ep == 25) { lr = 8e-3f; opt = new SGD(learningRate: lr); }
                if (ep == 45) { lr = 2e-3f; opt = new SGD(learningRate: lr); }
                if (ep == 60) { lr = 5e-4f; opt = new SGD(learningRate: lr); }

                data = data.OrderBy(_ => Guid.NewGuid()).ToList();
                float sum = 0;
                int n = 0, skip = 0, correct = 0, predSpace = 0;

                foreach (var (xin, yin, tgt) in data)
                {
                    var x = Tensor.FromArray(xin, new TensorShape(1, window), device);
                    var y = Tensor.FromArray(yin, new TensorShape(1, V), device);

                    opt.ZeroGrad(model.Parameters());
                    var logits = model.Forward(x);
                    if (Num.Bad(logits)) { skip++; continue; }

                    var loss = ce.Forward(logits, y);
                    float lv = loss.ToScalar();
                    if (float.IsNaN(lv) || float.IsInfinity(lv)) { skip++; continue; }

                    loss.Backward();
                    Tensor.ClipGradients(model.Parameters(), -1f, 1f);
                    opt.Step(model.Parameters());

                    int pred = ArgMax(logits.ToArray());
                    if (pred == tgt) correct++;
                    if (pred == spaceId) predSpace++;
                    sum += lv;
                    n++;
                }

                float acc = n > 0 ? 100f * correct / n : 0;
                if (ep == 1 || ep % 5 == 0 || ep == epochs || acc >= 99f)
                    Console.WriteLine(
                        $"[SFT] Epoch {ep:D2}/{epochs}  loss={(n > 0 ? sum / n : 0):F4}  " +
                        $"acc={acc:F1}%  predSpace={(n > 0 ? 100f * predSpace / n : 0):F1}%  " +
                        $"ok={n}/{data.Count}  skip={skip}");

                if (n == 0) { Console.WriteLine("NaN wipe — stop."); break; }
                if (acc >= 99.5f && ep >= 20)
                {
                    Console.WriteLine("Memorized — stop.");
                    break;
                }
            }

            model.Eval();
            Console.WriteLine("\n=== Greedy  [generated]  · = space  □ = pad ===");
            Gen(model, c2i, i2c, device, padId, "ArborNet makes Kimi ", 90);
            Gen(model, c2i, i2c, device, padId, "The future of ", 70);
            Gen(model, c2i, i2c, device, padId, "Hello ", 55);
            Gen(model, c2i, i2c, device, padId, "Attention is ", 70);
            Gen(model, c2i, i2c, device, padId, "Kimi K3 is ", 60);
            Gen(model, c2i, i2c, device, padId, "Machine learning ", 65);
        }

        static int ArgMax(float[] z)
        {
            int iMax = 0;
            float best = float.NegativeInfinity;
            for (int i = 0; i < z.Length; i++)
            {
                if (float.IsNaN(z[i]) || float.IsInfinity(z[i])) continue;
                if (z[i] > best) { best = z[i]; iMax = i; }
            }
            return iMax;
        }

        static void Gen(KimiK3 model, Dictionary<char, int> c2i, Dictionary<int, char> i2c,
                        Device device, int padId, string prompt, int nTok)
        {
            int W = model.Window;
            var ctx = prompt.Select(c => c2i.ContainsKey(c) ? c2i[c] : padId).ToList();
            Console.Write(prompt);
            Console.Write("[");
            for (int i = 0; i < nTok; i++)
            {
                var win = new float[W];
                int copy = Math.Min(ctx.Count, W);
                int src = ctx.Count - copy;
                for (int t = 0; t < W - copy; t++) win[t] = padId;
                for (int t = 0; t < copy; t++) win[W - copy + t] = ctx[src + t];

                int next = ArgMax(model.Forward(Tensor.FromArray(win, new TensorShape(1, W), device)).ToArray());
                ctx.Add(next);
                char ch = i2c.ContainsKey(next) ? i2c[next] : '?';
                if (ch == '\u0001') ch = '□';
                Console.Write(ch == ' ' ? '·' : ch);
            }
            Console.WriteLine("]");
        }
    }
}
