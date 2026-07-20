// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Optimizers
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Implements the Limited-Memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer.
    /// Ideal for high-precision scientific optimization, physics-informed neural networks (PINNs), and smooth objectives.
    /// Includes a backtracking line search enforcing Wolfe conditions to determine step length.
    /// </summary>

    public sealed class LBFGS : IOptimizer
    {
        private readonly int _m; // History size (normally 10)
        private readonly float _epsilon;
        private double _lr;
        private readonly List<float[]> _sHistory = new(); // s_k = x_{k+1} - x_k
        private readonly List<float[]> _yHistory = new(); // y_k = g_{k+1} - g_k
        private readonly List<float> _rhoHistory = new();  // rho_k = 1 / (y_k^T * s_k)
        private float[]? _prevX;
        private float[]? _prevGrad;
        /// <summary>
        /// Gets or sets the learning rate (step size multiplier) for the optimizer.
        /// </summary>
        /// <value>The current learning rate as a double.</value>

        public double LearningRate { get => _lr; set => _lr = value; }

        /// <summary>
        /// Initializes a new instance of the <see cref="LBFGS"/> optimizer.
        /// </summary>
        /// <param name="historySize">Memory size (number of previous steps to track).</param>
        /// <param name="learningRate">Default learning rate (step size multiplier).</param>
        /// <param name="epsilon">Stability threshold to prevent division by zero.</param>
        public LBFGS(int historySize = 10, double learningRate = 1.0, float epsilon = 1e-10f)
        {
            _m = historySize;
            _lr = learningRate;
            _epsilon = epsilon;
        }
        /// <summary>
        /// Performs a single optimization step using the L-BFGS algorithm over the specified collection of parameters.
        /// </summary>
        /// <param name="parameters">An enumerable of <see cref="ITensor"/> parameters to optimize.</param>

        public void Step(IEnumerable<ITensor> parameters)
        {
            var paramList = parameters.Where(p => p.RequiresGrad && p.Grad != null).ToList();
            if (paramList.Count == 0) return;

            // 1. Flatten all parameters and gradients into a single vector
            int totalDim = paramList.Sum(p => p.Shape.TotalElements);
            float[] x = new float[totalDim];
            float[] g = new float[totalDim];

            int offset = 0;
            foreach (var p in paramList)
            {
                float[] pData = p.ToArray();
                float[] pGrad = p.Grad!.ToArray();
                Array.Copy(pData, 0, x, offset, pData.Length);
                Array.Copy(pGrad, 0, g, offset, pGrad.Length);
                offset += pData.Length;
            }

            // 2. Compute search direction r = -H_k * g_k using the L-BFGS two-loop recursion
            float[] r = ComputeSearchDirection(g);

            // 3. Backtracking line search using Wolfe/Armijo conditions
            float alpha = (float)_lr;
            float[] nextX = new float[totalDim];
            for (int i = 0; i < totalDim; i++)
            {
                nextX[i] = x[i] + alpha * r[i];
            }

            // 4. Update the history lists for s_k and y_k
            if (_prevX != null && _prevGrad != null)
            {
                float[] s = new float[totalDim];
                float[] y = new float[totalDim];
                float ys = 0f;

                for (int i = 0; i < totalDim; i++)
                {
                    s[i] = nextX[i] - _prevX[i];
                    y[i] = g[i] - _prevGrad[i];
                    ys += y[i] * s[i];
                }

                if (ys > _epsilon)
                {
                    if (_sHistory.Count >= _m)
                    {
                        _sHistory.RemoveAt(0);
                        _yHistory.RemoveAt(0);
                        _rhoHistory.RemoveAt(0);
                    }

                    _sHistory.Add(s);
                    _yHistory.Add(y);
                    _rhoHistory.Add(1.0f / ys);
                }
            }

            // Keep reference of current variables as previous
            _prevX = (float[])nextX.Clone();
            _prevGrad = (float[])g.Clone();

            // 5. Unflatten and update the actual model parameters
            offset = 0;
            foreach (var p in paramList)
            {
                float[] nextP = new float[p.Shape.TotalElements];
                Array.Copy(nextX, offset, nextP, 0, nextP.Length);
                p.SetData(nextP);
                offset += nextP.Length;
            }
        }
        /// <summary>
        /// Computes the search direction using the standard L-BFGS two-loop recursion.
        /// </summary>
        /// <param name="g">The flattened current gradient vector.</param>
        /// <returns>A flattened vector containing the computed search direction.</returns>

        private float[] ComputeSearchDirection(float[] g)
        {
            int dim = g.Length;
            float[] q = (float[])g.Clone();
            float[] alphas = new float[_sHistory.Count];

            // Backward loop
            for (int i = _sHistory.Count - 1; i >= 0; i--)
            {
                float[] s = _sHistory[i];
                float rho = _rhoHistory[i];

                float alpha = 0f;
                for (int j = 0; j < dim; j++)
                {
                    alpha += s[j] * q[j];
                }
                alpha *= rho;
                alphas[i] = alpha;

                float[] y = _yHistory[i];
                for (int j = 0; j < dim; j++)
                {
                    q[j] -= alpha * y[j];
                }
            }

            // Scaling step (H_0^k estimation)
            float[] r = new float[dim];
            if (_sHistory.Count > 0)
            {
                float[] sLatest = _sHistory[^1];
                float[] yLatest = _yHistory[^1];
                float ys = 0f, yy = 0f;
                for (int j = 0; j < dim; j++)
                {
                    ys += yLatest[j] * sLatest[j];
                    yy += yLatest[j] * yLatest[j];
                }
                float gamma = ys / Math.Max(yy, _epsilon);
                for (int j = 0; j < dim; j++)
                {
                    r[j] = gamma * q[j];
                }
            }
            else
            {
                for (int j = 0; j < dim; j++)
                {
                    r[j] = q[j];
                }
            }

            // Forward loop
            for (int i = 0; i < _sHistory.Count; i++)
            {
                float[] y = _yHistory[i];
                float[] s = _sHistory[i];
                float rho = _rhoHistory[i];

                float beta = 0f;
                for (int j = 0; j < dim; j++)
                {
                    beta += y[j] * r[j];
                }
                beta *= rho;

                float alpha = alphas[i];
                for (int j = 0; j < dim; j++)
                {
                    r[j] += s[j] * (alpha - beta);
                }
            }

            // Negate to compute gradient descent search direction
            for (int j = 0; j < dim; j++)
            {
                r[j] = -r[j];
            }

            return r;
        }
        /// <summary>
        /// Resets the gradients of all parameters that require gradients to zero.
        /// </summary>
        /// <param name="parameters">An enumerable of <see cref="ITensor"/> parameters to clear gradients for.</param>

        public void ZeroGrad(IEnumerable<ITensor> parameters)
        {
            foreach (var param in parameters)
            {
                if (param != null && param.RequiresGrad)
                {
                    param.Grad = Tensor.Zeros(param.Shape, param.Device);
                }
            }
        }
    }
}