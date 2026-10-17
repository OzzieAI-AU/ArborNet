using System;

namespace ArborNet.Layers
{
    public sealed class GeometricEmbeddingSlice
    {
        public const double Eps = 1e-5;
        public const double MaxNorm = 1.0 - 1e-5;

        private readonly int _entities;
        private readonly int _relations;
        private readonly int _dim;
        private readonly double[][] _entity;
        private readonly double[][] _relation;
        private readonly Random _rng;

        public GeometricEmbeddingSlice(int entityCount, int dim, int relationCount, int seed = 7)
        {
            if (entityCount < 1) throw new ArgumentOutOfRangeException(nameof(entityCount));
            if (relationCount < 1) throw new ArgumentOutOfRangeException(nameof(relationCount));
            if (dim < 1) throw new ArgumentOutOfRangeException(nameof(dim));
            _entities = entityCount;
            _relations = relationCount;
            _dim = dim;
            _rng = new Random(seed);
            _entity = Alloc(entityCount, dim);
            _relation = Alloc(relationCount, dim);
            Init(_entity, 1e-3);
            Init(_relation, 1e-3);
        }

        public int EntityCount { get { return _entities; } }
        public int RelationCount { get { return _relations; } }
        public int Dim { get { return _dim; } }
        public double Curvature { get { return 1.0; } }

        public void CopyEntity(int id, double[] destination)
        {
            CheckEntity(id);
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            if (destination.Length < _dim) throw new ArgumentException("destination shorter than dim", nameof(destination));
            Array.Copy(_entity[id], 0, destination, 0, _dim);
        }

        public void CopyRelation(int id, double[] destination)
        {
            CheckRelation(id);
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            if (destination.Length < _dim) throw new ArgumentException("destination shorter than dim", nameof(destination));
            Array.Copy(_relation[id], 0, destination, 0, _dim);
        }

        public double EntityRadius(int id)
        {
            CheckEntity(id);
            return Math.Sqrt(Dot(_entity[id], _entity[id]));
        }

        public double ConformalFactor(int id)
        {
            CheckEntity(id);
            return Conformal(_entity[id]);
        }

        public double EntityDistance(int a, int b)
        {
            CheckEntity(a);
            CheckEntity(b);
            return PoincareDistance(_entity[a], _entity[b]);
        }

        public double TranslatedDistance(int head, int relation, int tail)
        {
            CheckEntity(head);
            CheckRelation(relation);
            CheckEntity(tail);
            double[] translated = MobiusAdd(_entity[head], _relation[relation]);
            return PoincareDistance(translated, _entity[tail]);
        }

        public double LinkProbability(int a, int b, double radius = 2.0, double temperature = 1.0)
        {
            if (temperature < Eps) throw new ArgumentOutOfRangeException(nameof(temperature));
            double d = EntityDistance(a, b);
            return Sigmoid((radius - d) / temperature);
        }

        public double TripleProbability(int head, int relation, int tail, double radius = 2.0, double temperature = 1.0)
        {
            if (temperature < Eps) throw new ArgumentOutOfRangeException(nameof(temperature));
            double d = TranslatedDistance(head, relation, tail);
            return Sigmoid((radius - d) / temperature);
        }

        public double MarginLoss(int head, int relation, int tailPos, int tailNeg, double margin)
        {
            double hinge = margin + TranslatedDistance(head, relation, tailPos) - TranslatedDistance(head, relation, tailNeg);
            return hinge > 0.0 ? hinge : 0.0;
        }

        public double MeanTripleNll(int[] heads, int[] relations, int[] tails, double radius = 2.0, double temperature = 1.0)
        {
            int n = CheckBatch(heads, relations, tails);
            if (n == 0) return 0.0;
            if (temperature < Eps) throw new ArgumentOutOfRangeException(nameof(temperature));
            double sum = 0.0;
            for (int i = 0; i < n; i++)
            {
                double p = TripleProbability(heads[i], relations[i], tails[i], radius, temperature);
                if (p < Eps) p = Eps;
                sum += -Math.Log(p);
            }
            return sum / n;
        }

        public void AlignTriple(int head, int relation, int tail, double step)
        {
            CheckEntity(head);
            CheckRelation(relation);
            CheckEntity(tail);
            if (step <= 0.0) return;
            double[] h = _entity[head];
            double[] r = _relation[relation];
            double[] t = _entity[tail];
            double[] targetR = MobiusAdd(Neg(h), t);
            double[] targetH = MobiusAdd(t, Neg(r));
            double[] targetT = MobiusAdd(h, r);
            PullToward(r, targetR, step);
            PullToward(h, targetH, step);
            PullToward(t, targetT, step);
        }

        public void ContrastTriple(int head, int relation, int tailPos, int tailNeg, double step, double margin)
        {
            CheckEntity(tailNeg);
            AlignTriple(head, relation, tailPos, step);
            if (tailNeg == head || tailNeg == tailPos) return;
            double d = TranslatedDistance(head, relation, tailNeg);
            if (d >= margin) return;
            double[] translated = MobiusAdd(_entity[head], _relation[relation]);
            PushFrom(_entity[tailNeg], translated, step);
        }

        public double AlignBatch(int[] heads, int[] relations, int[] tails, double step)
        {
            int n = CheckBatch(heads, relations, tails);
            if (n == 0) return 0.0;
            double sum = 0.0;
            for (int i = 0; i < n; i++)
            {
                sum += TranslatedDistance(heads[i], relations[i], tails[i]);
                AlignTriple(heads[i], relations[i], tails[i], step);
            }
            return sum / n;
        }

        public static double PoincareDistance(double[] u, double[] v)
        {
            if (u == null) throw new ArgumentNullException(nameof(u));
            if (v == null) throw new ArgumentNullException(nameof(v));
            if (u.Length != v.Length) throw new ArgumentException("dimension mismatch");
            double nu2 = Dot(u, u);
            double nv2 = Dot(v, v);
            if (nu2 > MaxNorm * MaxNorm) nu2 = MaxNorm * MaxNorm;
            if (nv2 > MaxNorm * MaxNorm) nv2 = MaxNorm * MaxNorm;
            double diff = 0.0;
            for (int i = 0; i < u.Length; i++)
            {
                double d = u[i] - v[i];
                diff += d * d;
            }
            double alpha = 1.0 - nu2;
            double beta = 1.0 - nv2;
            if (alpha < Eps) alpha = Eps;
            if (beta < Eps) beta = Eps;
            double z = 1.0 + (2.0 * diff) / (alpha * beta);
            if (z < 1.0) z = 1.0;
            if (z > 1e150) z = 1e150;
            return Acosh(z);
        }

        public static bool Smoke()
        {
            GeometricEmbeddingSlice g = new GeometricEmbeddingSlice(8, 4, 2, 3);
            double d0 = g.EntityDistance(0, 1);
            g.AlignTriple(0, 0, 1, 0.5);
            double d1 = g.TranslatedDistance(0, 0, 1);
            double p = g.TripleProbability(0, 0, 1, 2.0, 1.0);
            double nll = g.MeanTripleNll(new int[] { 0, 1 }, new int[] { 0, 1 }, new int[] { 1, 2 }, 2.0, 1.0);
            return d0 >= 0.0 && d1 >= 0.0 && p > 0.0 && p < 1.0 && nll >= 0.0
                && !double.IsNaN(d0) && !double.IsNaN(d1) && !double.IsInfinity(d1) && !double.IsNaN(nll);
        }

        private void PullToward(double[] x, double[] target, double step)
        {
            if (step > 1.0) step = 1.0;
            double[] v = LogMap(x, target);
            for (int i = 0; i < v.Length; i++) v[i] *= step;
            double[] y = ExpMap(x, v);
            Array.Copy(y, x, x.Length);
        }

        private static void PushFrom(double[] x, double[] other, double step)
        {
            if (step <= 0.0) return;
            if (step > 1.0) step = 1.0;
            double[] v = LogMap(x, other);
            double nrm = Norm(v);
            if (nrm < Eps)
            {
                v[0] = step;
            }
            else
            {
                double s = step / nrm;
                for (int i = 0; i < v.Length; i++) v[i] = -v[i] * s;
            }
            double[] y = ExpMap(x, v);
            Array.Copy(y, x, x.Length);
        }

        private static double[] ExpMap(double[] x, double[] v)
        {
            int n = x.Length;
            double nv = Norm(v);
            if (nv < Eps)
            {
                double[] copy = new double[n];
                Array.Copy(x, copy, n);
                return copy;
            }
            double lambda = Conformal(x);
            double scale = Math.Tanh(lambda * nv * 0.5) / nv;
            double[] dir = new double[n];
            for (int i = 0; i < n; i++) dir[i] = v[i] * scale;
            return MobiusAdd(x, dir);
        }

        private static double[] LogMap(double[] x, double[] y)
        {
            int n = x.Length;
            double[] o = new double[n];
            double[] m = MobiusAdd(Neg(x), y);
            double nm = Norm(m);
            if (nm < Eps) return o;
            double lambda = Conformal(x);
            double scale = (2.0 / lambda) * Artanh(nm) / nm;
            for (int i = 0; i < n; i++) o[i] = scale * m[i];
            return o;
        }

        private static double[] MobiusAdd(double[] u, double[] v)
        {
            int n = u.Length;
            double nu2 = Dot(u, u);
            double nv2 = Dot(v, v);
            double uv = Dot(u, v);
            double denom = 1.0 + 2.0 * uv + nu2 * nv2;
            if (denom < Eps) denom = Eps;
            double a = (1.0 + 2.0 * uv + nv2) / denom;
            double b = (1.0 - nu2) / denom;
            double[] o = new double[n];
            for (int i = 0; i < n; i++) o[i] = a * u[i] + b * v[i];
            Project(o);
            return o;
        }

        private static double[] Neg(double[] x)
        {
            double[] y = new double[x.Length];
            for (int i = 0; i < x.Length; i++) y[i] = -x[i];
            return y;
        }

        private static void Project(double[] x)
        {
            double n = Norm(x);
            if (n > MaxNorm)
            {
                double s = MaxNorm / n;
                for (int i = 0; i < x.Length; i++) x[i] *= s;
            }
        }

        private static double Conformal(double[] x)
        {
            double n2 = Dot(x, x);
            if (n2 > MaxNorm * MaxNorm) n2 = MaxNorm * MaxNorm;
            double d = 1.0 - n2;
            if (d < Eps) d = Eps;
            return 2.0 / d;
        }

        private static double Dot(double[] a, double[] b)
        {
            double s = 0.0;
            for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
            return s;
        }

        private static double Norm(double[] x)
        {
            return Math.Sqrt(Dot(x, x));
        }

        private static double Artanh(double x)
        {
            if (x > 1.0 - Eps) x = 1.0 - Eps;
            if (x < -1.0 + Eps) x = -1.0 + Eps;
            return 0.5 * Math.Log((1.0 + x) / (1.0 - x));
        }

        private static double Acosh(double x)
        {
            if (x < 1.0) x = 1.0;
            return Math.Log(x + Math.Sqrt(x * x - 1.0));
        }

        private static double Sigmoid(double x)
        {
            if (x >= 0.0)
            {
                double z = Math.Exp(-x);
                return 1.0 / (1.0 + z);
            }
            double e = Math.Exp(x);
            return e / (1.0 + e);
        }

        private void Init(double[][] table, double scale)
        {
            for (int i = 0; i < table.Length; i++)
            {
                double[] row = table[i];
                for (int j = 0; j < row.Length; j++)
                {
                    double u1 = 1.0 - _rng.NextDouble();
                    double u2 = 1.0 - _rng.NextDouble();
                    double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                    row[j] = z * scale;
                }
                Project(row);
            }
        }

        private static double[][] Alloc(int rows, int cols)
        {
            double[][] t = new double[rows][];
            for (int i = 0; i < rows; i++) t[i] = new double[cols];
            return t;
        }

        private void CheckEntity(int id)
        {
            if ((uint)id >= (uint)_entities) throw new ArgumentOutOfRangeException(nameof(id));
        }

        private void CheckRelation(int id)
        {
            if ((uint)id >= (uint)_relations) throw new ArgumentOutOfRangeException(nameof(id));
        }

        private static int CheckBatch(int[] heads, int[] relations, int[] tails)
        {
            if (heads == null) throw new ArgumentNullException(nameof(heads));
            if (relations == null) throw new ArgumentNullException(nameof(relations));
            if (tails == null) throw new ArgumentNullException(nameof(tails));
            if (relations.Length != heads.Length || tails.Length != heads.Length)
                throw new ArgumentException("batch length mismatch");
            return heads.Length;
        }
    }
}
