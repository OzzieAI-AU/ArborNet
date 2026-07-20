// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Layers
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Core.Functional;
    using ArborNet.Core.Layers;
    /// <summary>
    /// Represents a fully connected (dense) linear layer in a neural network.
    /// Applies a linear transformation to the incoming data: <c>y = xW + b</c>.
    /// </summary>
    /// <remarks>
    /// This layer maintains a weight matrix and a bias vector. The weights are initialized 
    /// using Xavier (Glorot) Uniform initialization, and the biases are initialized to zero.
    /// It automatically registers backward computation pathways (gradients) for autograd execution.
    /// </remarks>

    #endregion

    public class Linear : BaseLayer
    {
        private ITensor weight;
        private ITensor bias;
        // FIXED: Removed 'private readonly Device device;' to prevent hiding BaseLayer.device

        public Linear(int inFeatures, int outFeatures, Device? device = null)
        {
            this.device = device ?? Device.CPU;
            weight = Initializers.XavierUniform(new TensorShape(inFeatures, outFeatures), this.device);
            bias = Tensor.Zeros(new TensorShape(outFeatures), this.device);
            weight.RequiresGrad = true;
            bias.RequiresGrad = true;
        }
        /// <summary>
        /// Performs the forward pass of the linear layer by computing the matrix multiplication of the input 
        /// with the weights, then adding the bias.
        /// </summary>
        /// <param name="input">The input tensor of shape <c>(batchSize, inFeatures)</c>.</param>
        /// <returns>The computed output tensor of shape <c>(batchSize, outFeatures)</c>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <see langword="null"/>.</exception>

        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var output = input.MatMul(weight).Add(bias);

            if (input.RequiresGrad || weight.RequiresGrad || bias.RequiresGrad)
            {
                var capturedInput = input;
                var capturedWeight = weight;
                var capturedBias = bias;

                output.GradFn = gradOutput =>
                {
                    ITensor? gradInput = null;
                    ITensor? gradWeight = null;
                    ITensor? gradBias = null;

                    if (capturedWeight.RequiresGrad)
                    {
                        gradWeight = capturedInput.Transpose(new[] { 1, 0 }).MatMul(gradOutput);
                        capturedWeight.AccumulateGrad(gradWeight);
                    }

                    if (capturedBias.RequiresGrad)
                    {
                        gradBias = gradOutput.Sum(0);
                        capturedBias.AccumulateGrad(gradBias);
                    }

                    if (capturedInput.RequiresGrad)
                    {
                        gradInput = gradOutput.MatMul(capturedWeight.Transpose(new[] { 1, 0 }));
                        capturedInput.AccumulateGrad(gradInput);
                    }

                    return gradInput ?? gradOutput;
                };
            }
            return output;
        }
        /// <summary>
        /// Retrieves the learnable parameters of this linear layer.
        /// </summary>
        /// <returns>An enumerable collection containing the weight and bias tensors.</returns>

        public override IEnumerable<ITensor> Parameters()
        {
            yield return weight;
            yield return bias;
        }
    }
}