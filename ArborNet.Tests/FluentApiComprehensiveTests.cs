// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Tests.Fluent
{

    #region Using Statements:

    using ArborNet.Core.Devices;
    using ArborNet.Core.Tensors;
    using ArborNet.Fluent;
    using FluentAssertions;
    using Xunit;
    /// <summary>
    /// Provides comprehensive integration and unit tests for the fluent API capabilities of the ArborNet framework.
    /// </summary>
    /// <remarks>
    /// This test suite validates operator overloading, activation chaining, sequential layers, normalization,
    /// and loss functions. It ensures that the fluent API behaves predictably, maintains tensor integrity,
    /// and integrates smoothly across all execution devices.
    /// </remarks>

    #endregion

    public class FluentApiComprehensiveTests : TestBase
    {
        /// <summary>
        /// Verifies that basic mathematical and unary operators function correctly
        /// and produce mathematically sound results when mixing tensors with scalar values.
        /// </summary>
        /// <remarks>
        /// This test performs element-wise addition, multiplication, subtraction, and negation
        /// on a 2x2 tensor and asserts the final values against expected precalculated values.
        /// </remarks>
        [Fact]
        public void MathOperators_WorkRobustly_WithScalarsAndTensors()
        {
            // Unary & basic arithmetic operations
            var input = X.FromArray(new[] { 1f, 2f, 3f, 4f }, 2, 2);

            var result = (input + 2f) * 3f - 1f;

            result.ToArray().Should().BeEquivalentTo(new[] { 8f, 11f, 14f, 17f });

            // Unary Negate operator
            var negated = -result;
            negated.ToArray().Should().BeEquivalentTo(new[] { -8f, -11f, -14f, -17f });
        }
        /// <summary>
        /// Validates that multiple activation functions and mathematical transformations
        /// can be chained together in a fluent pipeline without losing tensor integrity or generating invalid values.
        /// </summary>
        /// <remarks>
        /// Specifically tests the sequential execution of ReLU, Tanh, and Sigmoid activations on a 1D tensor,
        /// verifying that the output shape is preserved and contains no NaN values.
        /// </remarks>

        [Fact]
        public void FluentActivations_AndMathematicalFunctions_AreChainable()
        {
            // Multi-activation pipeline chaining
            var input = X.FromArray(new[] { -2f, -1f, 0f, 1f, 2f }, 5);

            var activated = input.ReLU().Tanh().Sigmoid();

            activated.ToArray().Should().NotContain(float.NaN);
            activated.Shape.Dimensions.Should().BeEquivalentTo(new[] { 5 });
        }
        /// <summary>
        /// Asserts that comparative operators applied to tensors generate the expected
        /// binarized masks (where true translates to 1.0f and false to 0.0f) across element-wise comparisons.
        /// </summary>
        /// <remarks>
        /// Evaluates the greater-than-or-equal-to (>=) operator element-wise on two 3-element tensors,
        /// confirming that a correct binary mask is generated.
        /// </remarks>

        [Fact]
        public void ComparativeOperators_ProduceCorrectBinarizedMasks()
        {
            var left = X.FromArray(new[] { 10f, 50f, 2f }, 3);
            var right = X.FromArray(new[] { 15f, 30f, 2f }, 3);

            var mask = left >= right;

            mask.ToArray().Should().BeEquivalentTo(new[] { 0f, 1f, 1f });
        }
        /// <summary>
        /// Tests the capability of fluently chaining neural network layers and normalizations
        /// to build and execute a mini feed-forward pipeline.
        /// </summary>
        /// <remarks>
        /// Passes a randomized 2D batch of features through a Linear layer, Layer Normalization, ReLU activation,
        /// a second Linear layer, and a Softmax activation, verifying the output shape and value validity.
        /// </remarks>

        [Fact]
        public void LayerNormalizations_AndLayers_BindFluentPipelines()
        {
            var batchFeatures = X.Rand(16, 64);

            var pipeline = batchFeatures
                .Linear(128)
                .LayerNorm(new[] { 128 })
                .ReLU()
                .Linear(10)
                .Softmax();

            pipeline.Shape.Dimensions.Should().BeEquivalentTo(new[] { 16, 10 });
            pipeline.ToArray().Should().NotContain(float.NaN);
        }
        /// <summary>
        /// Verifies that loss functions can be computed directly within a fluent evaluation pipeline,
        /// returning accurate values when compared against known analytical outputs.
        /// </summary>
        /// <remarks>
        /// Computes the Mean Squared Error (MSE) loss between a 2x2 prediction tensor and a target tensor,
        /// asserting that the resulting scalar loss is mathematically correct within a specified tolerance.
        /// </remarks>

        [Fact]
        public void LossFunctions_CanBeEvaluated_DirectlyWithinPipeline()
        {
            var predictions = X.FromArray(new[] { 0.1f, 0.9f, 0.2f, 0.8f }, 2, 2);
            var targets = X.FromArray(new[] { 0.0f, 1.0f, 0.0f, 1.0f }, 2, 2);

            var loss = predictions.MseLoss(targets);

            loss.ToScalar().Should().BeApproximately(0.025f, 1e-5f);
        }
    }
}
