namespace ArborNet.Core.Autograd
{


    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Core.Initializers;

    public static class AutogradTapeTest
    {
        public static int Run()
        {
            Console.WriteLine("=== ArborNet autograd tape test ===");
            Console.WriteLine("z[4,8] --MatMul--> w[8,8] --ReLU--> Reshape(32) --Unsqueeze(0) --Reshape(4,8) -- y*y --Mean--> loss");
            Console.WriteLine();

            var z = Tensor.Randn(new TensorShape(4, 8), Device.CPU);
            z.RequiresGrad = true;
            var w = Initializers.XavierUniform(new TensorShape(8, 8), Device.CPU);
            w.RequiresGrad = true;

            var y = z.MatMul(w).Relu();
            y = y.Reshape(32).Unsqueeze(0).Reshape(4, 8);
            var loss = y.Multiply(y).Mean();

            Console.WriteLine($"loss = {Fmt(loss.ToScalar())}   finite={!Bad(loss)}");

            loss.Backward();

            bool pass = true;
            pass &= Report("z.Grad", z.Grad, expectNonZero: true);
            pass &= Report("w.Grad", w.Grad, expectNonZero: true);

            // analytic check: d(mean(y^2))/dy = 2y / N
            if (y.Grad != null && !Bad(y) && !Bad(y.Grad))
            {
                var yv = y.ToArray();
                var gv = y.Grad.ToArray();
                int N = yv.Length;
                double maxErr = 0;
                for (int i = 0; i < N; i++)
                    maxErr = Math.Max(maxErr, Math.Abs(gv[i] - 2.0 * yv[i] / N));
                bool ok = maxErr < 1e-5;
                Console.WriteLine($"y.Grad vs 2y/N          maxAbsErr={maxErr:E3}  {(ok ? "PASS" : "FAIL")}");
                pass &= ok;
            }
            else
            {
                Console.WriteLine("y.Grad vs 2y/N          FAIL (missing / non-finite)");
                pass = false;
            }

            Console.WriteLine();
            Console.WriteLine(pass
                ? "RESULT: PASS  tape is connected through MatMul, ReLU, Reshape, Unsqueeze."
                : "RESULT: FAIL  do not train Kimi — Reshape/Unsqueeze/MatMul backward is still broken.");
            return pass ? 0 : 1;
        }

        static bool Report(string name, ITensor? g, bool expectNonZero)
        {
            if (g == null)
            {
                Console.WriteLine($"{name,-22} NULL                         FAIL");
                return false;
            }

            var a = g.ToArray();
            bool nan = a.Any(float.IsNaN);
            bool inf = a.Any(float.IsInfinity);
            float min = a.Min();
            float max = a.Max();
            float l2 = MathF.Sqrt(a.Sum(v => v * v));
            int nz = a.Count(v => Math.Abs(v) > 1e-12f);
            bool finite = !nan && !inf;
            bool nonzero = nz > 0 && l2 > 1e-12f;
            bool ok = finite && (!expectNonZero || nonzero);

            Console.WriteLine(
                $"{name,-22} shape={g.Shape}  min={Fmt(min)}  max={Fmt(max)}  " +
                $"l2={Fmt(l2)}  nonzero={nz}/{a.Length}  nan={nan} inf={inf}  {(ok ? "PASS" : "FAIL")}");
            return ok;
        }

        static bool Bad(ITensor t)
        {
            var d = t.ToArray();
            return d.Any(v => float.IsNaN(v) || float.IsInfinity(v));
        }

        static string Fmt(float v)
            => float.IsNaN(v) ? "NaN" : float.IsInfinity(v) ? v.ToString() : v.ToString("G6");
    }
}