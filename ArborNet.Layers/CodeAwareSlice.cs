using System;

namespace ArborNet.Layers
{
    public sealed class CodeAwareSlice
    {
        public const int FeatureCount = 8;

        private readonly float[] _features;

        public int Start { get; }
        public int Length { get; }
        public float Score { get; }

        private CodeAwareSlice(int start, int length, float score, float[] features)
        {
            Start = start;
            Length = length;
            Score = score;
            _features = features;
        }

        public float Feature(int index)
        {
            if ((uint)index >= FeatureCount)
                throw new ArgumentOutOfRangeException(nameof(index));
            return _features[index];
        }

        public void CopyFeatures(float[] destination)
        {
            if (destination == null)
                throw new ArgumentNullException(nameof(destination));
            if (destination.Length < FeatureCount)
                throw new ArgumentException("destination shorter than FeatureCount.", nameof(destination));
            Array.Copy(_features, 0, destination, 0, FeatureCount);
        }

        public string Materialize(string source)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            EnsureSpan(Start, Length, source.Length);
            return source.Substring(Start, Length);
        }

        public int[] MaterializeTokens(int[] tokens)
        {
            if (tokens == null)
                throw new ArgumentNullException(nameof(tokens));
            EnsureSpan(Start, Length, tokens.Length);
            var copy = new int[Length];
            if (Length > 0)
                Array.Copy(tokens, Start, copy, 0, Length);
            return copy;
        }

        public static CodeAwareSlice FromSource(string source, int window, int stride)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            if (window < 1)
                throw new ArgumentOutOfRangeException(nameof(window));
            if (stride < 1)
                stride = 1;
            if (source.Length == 0)
                return Pack(0, 0, Zero());
            int length = source.Length < window ? source.Length : window;
            if (source.Length <= window)
                return ScoreSource(source, 0, length);

            CodeAwareSlice best = null;
            int last = source.Length - window;
            for (int start = 0; start <= last; start += stride)
            {
                CodeAwareSlice cand = ScoreSource(source, start, window);
                if (best == null || cand.Score > best.Score)
                    best = cand;
            }
            return best;
        }

        public static CodeAwareSlice FromTokens(int[] tokens, int window, int stride, int[] boundaryIds, int[] openIds, int[] closeIds)
        {
            if (tokens == null)
                throw new ArgumentNullException(nameof(tokens));
            if (window < 1)
                throw new ArgumentOutOfRangeException(nameof(window));
            if (stride < 1)
                stride = 1;
            if (boundaryIds == null)
                boundaryIds = new int[0];
            if (openIds == null)
                openIds = new int[0];
            if (closeIds == null)
                closeIds = new int[0];
            if (openIds.Length != closeIds.Length)
                throw new ArgumentException("openIds and closeIds length mismatch.");
            if (tokens.Length == 0)
                return Pack(0, 0, Zero());
            int length = tokens.Length < window ? tokens.Length : window;
            if (tokens.Length <= window)
                return ScoreTokens(tokens, 0, length, boundaryIds, openIds, closeIds);

            CodeAwareSlice best = null;
            int last = tokens.Length - window;
            for (int start = 0; start <= last; start += stride)
            {
                CodeAwareSlice cand = ScoreTokens(tokens, start, window, boundaryIds, openIds, closeIds);
                if (best == null || cand.Score > best.Score)
                    best = cand;
            }
            return best;
        }

        public static void FillPositionWeights(string source, int start, int length, float[] destination)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            if (destination == null)
                throw new ArgumentNullException(nameof(destination));
            EnsureSpan(start, length, source.Length);
            if (destination.Length < length)
                throw new ArgumentException("destination shorter than length.", nameof(destination));
            for (int i = 0; i < length; i++)
            {
                int abs = start + i;
                char c = source[abs];
                float w = 1f;
                if (c == '\n' || c == ';' || c == '{' || c == '}' || c == '(' || c == ')')
                    w += 0.35f;
                if (abs == 0 || IsStructuralBoundary(source, abs))
                    w += 0.15f;
                destination[i] = w;
            }
        }

        private static CodeAwareSlice ScoreSource(string source, int start, int length)
        {
            float[] feat = Zero();
            int end = start + length;
            int depth = 0;
            int minDepth = 0;
            int peak = 0;
            int newlines = 0;
            int comments = 0;
            int idents = 0;
            int lines = 0;
            int indentSum = 0;
            int indentSq = 0;
            int i = start;
            while (i < end)
            {
                int indent = 0;
                while (i < end && (source[i] == ' ' || source[i] == '\t'))
                {
                    indent += source[i] == '\t' ? 4 : 1;
                    i++;
                }
                if (i >= end)
                    break;
                if (source[i] == '\n' || source[i] == '\r')
                {
                    if (source[i] == '\r' && i + 1 < end && source[i + 1] == '\n')
                        i++;
                    newlines++;
                    i++;
                    continue;
                }
                lines++;
                indentSum += indent;
                indentSq += indent * indent;
                while (i < end && source[i] != '\n' && source[i] != '\r')
                {
                    char c = source[i];
                    if (c == '(' || c == '[' || c == '{')
                    {
                        depth++;
                        if (depth > peak)
                            peak = depth;
                    }
                    else if (c == ')' || c == ']' || c == '}')
                    {
                        depth--;
                        if (depth < minDepth)
                            minDepth = depth;
                        int ad = depth >= 0 ? depth : -depth;
                        if (ad > peak)
                            peak = ad;
                    }
                    if (IsIdentChar(c))
                        idents++;
                    if (c == '/' && i + 1 < end && (source[i + 1] == '/' || source[i + 1] == '*'))
                        comments++;
                    i++;
                }
            }

            float boundaryScore = (start == 0 || IsStructuralBoundary(source, start)) ? 1f : 0f;
            if (start > 0 && IsIdentChar(source[start]) && IsIdentChar(source[start - 1]))
                boundaryScore -= 0.75f;

            float balance = 1f / (1f + Math.Abs(depth) + (minDepth < 0 ? -minDepth : 0));
            float indentScore = 0.5f;
            if (lines > 1)
            {
                float mean = indentSum / (float)lines;
                float var = indentSq / (float)lines - mean * mean;
                if (var < 0f)
                    var = 0f;
                indentScore = 1f / (1f + var);
            }

            int denom = length < 1 ? 1 : length;
            float commentFrac = comments / (float)denom;
            feat[0] = boundaryScore;
            feat[1] = balance;
            feat[2] = indentScore;
            feat[3] = newlines > 0 ? (newlines > 8 ? 1f : newlines / 8f) : 0.15f;
            feat[4] = 1f / (1f + peak);
            feat[5] = 1f - (commentFrac * 4f > 1f ? 1f : commentFrac * 4f);
            feat[6] = idents / (float)denom;
            feat[7] = length <= 0 ? 0f : 1f;
            return Pack(start, length, feat);
        }

        private static CodeAwareSlice ScoreTokens(int[] tokens, int start, int length, int[] boundaryIds, int[] openIds, int[] closeIds)
        {
            float[] feat = Zero();
            int end = start + length;
            int depth = 0;
            int minDepth = 0;
            int peak = 0;
            int bounds = 0;
            for (int i = start; i < end; i++)
            {
                int t = tokens[i];
                if (ContainsId(boundaryIds, t))
                    bounds++;
                if (ContainsId(openIds, t))
                {
                    depth++;
                    if (depth > peak)
                        peak = depth;
                }
                else if (ContainsId(closeIds, t))
                {
                    depth--;
                    if (depth < minDepth)
                        minDepth = depth;
                    int ad = depth >= 0 ? depth : -depth;
                    if (ad > peak)
                        peak = ad;
                }
            }

            bool onBoundary = start == 0 || ContainsId(boundaryIds, tokens[start]);
            float density = bounds / (float)(length < 1 ? 1 : length);
            feat[0] = onBoundary ? 1f : 0.25f * (bounds > 0 ? 1f : 0f);
            feat[1] = 1f / (1f + Math.Abs(depth) + (minDepth < 0 ? -minDepth : 0));
            feat[2] = density;
            feat[3] = 1f / (1f + peak);
            feat[4] = length <= 0 ? 0f : 1f;
            feat[5] = 0f;
            feat[6] = 0f;
            feat[7] = 0f;
            return Pack(start, length, feat);
        }

        private static CodeAwareSlice Pack(int start, int length, float[] feat)
        {
            float score = 1.6f * feat[0] + 1.2f * feat[1] + 0.6f * feat[2] + 0.4f * feat[3] + 0.5f * feat[4] + 0.2f * feat[5];
            return new CodeAwareSlice(start, length, score, feat);
        }

        private static float[] Zero()
        {
            return new float[FeatureCount];
        }

        private static void EnsureSpan(int start, int length, int total)
        {
            if (start < 0 || length < 0 || start > total || length > total - start)
                throw new ArgumentOutOfRangeException(nameof(start));
        }

        private static bool IsStructuralBoundary(string source, int index)
        {
            if (index <= 0)
                return true;
            char prev = source[index - 1];
            if (prev == '\n' || prev == '\r' || prev == ';' || prev == '{' || prev == '}' || prev == '(' || prev == ')')
                return true;
            int j = index - 1;
            while (j >= 0 && (source[j] == ' ' || source[j] == '\t'))
                j--;
            return j < 0 || source[j] == '\n' || source[j] == '\r';
        }

        private static bool IsIdentChar(char c)
        {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        }

        private static bool ContainsId(int[] ids, int value)
        {
            for (int i = 0; i < ids.Length; i++)
            {
                if (ids[i] == value)
                    return true;
            }
            return false;
        }
    }
}
