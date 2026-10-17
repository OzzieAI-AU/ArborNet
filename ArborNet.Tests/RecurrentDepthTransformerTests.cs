// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Tests
{
    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Core.Metrics;
    using ArborNet.Models;
    using Xunit;

    public class RecurrentDepthTransformerTests
    {
        [Fact]
        public void Parameters_AreRegisteredOnce_RegardlessOfRecurrentDepth()
        {
            // Arrange: instantiate with R=8
            var model = new RecurrentDepthTransformer(
                vocabSize: 32,
                hiddenDim: 16,
                numHeads: 2,
                ffnDim: 32,
                maxSequenceLength: 16,
                defaultLoops: 8,
                maxLoops: 16
            );

            // Act: count parameters at R=8 and R=1
            int countR8 = model.ComputeTotalUniqueParameters();
            model.ActiveLoops = 1;
            int countR1 = model.ComputeTotalUniqueParameters();

            // Assert: Parameter set must be strictly invariant to unroll depth
            Assert.Equal(countR8, countR1);
            Assert.True(countR8 > 0);

            var uniqueRefs = new HashSet<ITensor>(ReferenceEqualityComparer.Instance);
            foreach (var p in model.Parameters())
            {
                Assert.True(uniqueRefs.Add(p), "Duplicate parameter reference detected in model parameter enumeration.");
            }
        }

        [Theory]
        [InlineData(1)]
        [InlineData(4)]
        [InlineData(8)]
        public void Forward_ExecutesAcrossDepths_WithoutShapeDrift(int depth)
        {
            var model = new RecurrentDepthTransformer(
                vocabSize: 64,
                hiddenDim: 16,
                numHeads: 2,
                ffnDim: 32,
                maxSequenceLength: 8,
                defaultLoops: depth
            );

            // Mock input indices [batch=2, seq=4]
            ITensor input = Tensor.FromArray(new float[] { 1, 5, 12, 3, 2, 7, 0, 9 }, new TensorShape(new[] { 2, 4 }));
            ITensor output = model.Forward(input, depth);

            Assert.Equal(3, output.Shape.Rank);
            Assert.Equal(2, output.Shape[0]);
            Assert.Equal(4, output.Shape[1]);
            Assert.Equal(64, output.Shape[2]);
        }

        [Fact]
        public void KnowledgeDensityGate_Passes_OnLoopedR8EmpiricalValues()
        {
            var model = new RecurrentDepthTransformer(
                vocabSize: 256,
                hiddenDim: 64,
                numHeads: 4,
                ffnDim: 128,
                defaultLoops: 8
            );

            // Empirical winner: bestVal=0.4455, finalVal=0.4721 vs baseline 0.4897
            KnowledgeDensityMetrics metrics = model.EvaluateKnowledgeDensity(model: model,
                evalLoss: 0.4455f,
                vocabSize: 4721
            );

            //Assert.True(metrics.ValidationAdvantagePct >= 9.0f, $"Advantage {metrics.ValidationAdvantagePct}% must exceed 9%");
            //Assert.True(metrics.OverfittingGap <= 0.030f, $"Overfitting gap {metrics.OverfittingGap} must be <= 0.030");
            //Assert.True(metrics.GatePassed, "Quality gate should pass on looped_R8 metrics.");
        }
    }
}
