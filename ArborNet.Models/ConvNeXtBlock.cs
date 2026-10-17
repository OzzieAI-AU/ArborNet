// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Models
{
    #region Using Statements:

    using ArborNet.Activations;
    using ArborNet.Core;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;
    using System;
    using System.Collections.Generic;

    #endregion

    public class ConvNeXtBlock : BaseModel
    {
        private readonly int _dim;
        private readonly Conv2D dwConv;
        private readonly LayerNorm norm;
        private readonly Linear pw1;
        private readonly Linear pw2;

        public ConvNeXtBlock(int dim)
        {
            _dim = dim;
            // 7x7 depthwise convolution (groups = dim)
            dwConv = new Conv2D(dim, dim, kernelSize: 7, stride: 1, padding: 3, useBias: true);
            norm = new LayerNorm(new[] { dim });
            pw1 = new Linear(dim, dim * 4);
            pw2 = new Linear(dim * 4, dim);

            parameters.AddRange(dwConv.Parameters());
            parameters.AddRange(norm.Parameters());
            parameters.AddRange(pw1.Parameters());
            parameters.AddRange(pw2.Parameters());
        }

        public override ITensor Forward(ITensor x)
        {
            if (x.Shape.Rank != 4)
                throw new ArgumentException("ConvNeXtBlock expects 4D NCHW input tensor.", nameof(x));

            var residual = x;
            int b = x.Shape[0];
            int c = x.Shape[1];
            int h = x.Shape[2];
            int w = x.Shape[3];

            // 1. Depthwise Convolution in NCHW
            var outDw = dwConv.Forward(x);

            // 2. Permute NCHW -> NHWC to align channels with LayerNorm and Linear projections
            var nhwc = outDw.Transpose(new[] { 0, 2, 3, 1 });
            var flatTokens = nhwc.Reshape(b * h * w, c);

            // 3. Inverted bottleneck MLP: LN -> PW1 -> GELU -> PW2
            var normed = norm.Forward(flatTokens);
            var mlp1 = pw1.Forward(normed).Gelu();
            var mlp2 = pw2.Forward(mlp1);

            // 4. Permute back from NHWC to NCHW
            var backNhwc = mlp2.Reshape(b, h, w, c);
            var backNchw = backNhwc.Transpose(new[] { 0, 3, 1, 2 });

            // 5. Residual addition
            return backNchw.Add(residual);
        }

        public override IEnumerable<ITensor> Parameters() => parameters;
    }
}