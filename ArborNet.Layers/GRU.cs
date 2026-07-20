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
    using ArborNet.Activations;
    using ArborNet.Core.Layers;
    /// <summary>
    /// Represents a Gated Recurrent Unit (GRU) recurrent neural network layer.
    /// </summary>
    /// <remarks>
    /// This layer processes sequential data one step at a time. It maintains an internal hidden state 
    /// and computes transitions using gating mechanisms. The update formulas applied at each time step are:
    /// <list type="bullet">
    /// <item>
    /// <description>Reset Gate: <c>r = sigmoid(W_ir * x + b_ir + W_hr * h_prev + b_hr)</c></description>
    /// </item>
    /// <item>
    /// <description>Update Gate: <c>z = sigmoid(W_iz * x + b_iz + W_hz * h_prev + b_hz)</c></description>
    /// </item>
    /// <item>
    /// <description>Candidate Hidden State: <c>n = tanh(W_in * x + b_in + r * (W_hn * h_prev + b_hn))</c></description>
    /// </item>
    /// <item>
    /// <description>Output Hidden State: <c>h_new = z * h_prev + (1 - z) * n</c></description>
    /// </item>
    /// </list>
    /// This implementation dynamically adapts to the batch size of the input tensor.
    /// </remarks>

    #endregion

    public class GRU : BaseLayer
    {
        /// <summary>
        /// The dimensionality of the input features for each time step.
        /// </summary>
        private readonly int inputSize;
        /// <summary>
        /// The dimensionality of the hidden state vector.
        /// </summary>
        private readonly int hiddenSize;

        // Weights
        /// <summary>
        /// Input-to-reset gate weight matrix. Shape: (hiddenSize, inputSize).
        /// </summary>
        private readonly ITensor W_ir; // Input to reset gate (hiddenSize, inputSize)
        /// <summary>
        /// Input-to-update gate weight matrix. Shape: (hiddenSize, inputSize).
        /// </summary>
        private readonly ITensor W_iz; // Input to update gate (hiddenSize, inputSize)
        /// <summary>
        /// Input-to-candidate weight matrix. Shape: (hiddenSize, inputSize).
        /// </summary>
        private readonly ITensor W_in; // Input to candidate (hiddenSize, inputSize)
        /// <summary>
        /// Hidden-to-reset gate weight matrix. Shape: (hiddenSize, hiddenSize).
        /// </summary>
        private readonly ITensor W_hr; // Hidden to reset gate (hiddenSize, hiddenSize)
        /// <summary>
        /// Hidden-to-update gate weight matrix. Shape: (hiddenSize, hiddenSize).
        /// </summary>
        private readonly ITensor W_hz; // Hidden to update gate (hiddenSize, hiddenSize)
        /// <summary>
        /// Hidden-to-candidate weight matrix. Shape: (hiddenSize, hiddenSize).
        /// </summary>
        private readonly ITensor W_hn; // Hidden to candidate (hiddenSize, hiddenSize)

        // Biases
        /// <summary>
        /// Bias vector for the input projection of the reset gate. Shape: (hiddenSize,).
        /// </summary>
        private readonly ITensor b_ir; // Reset gate bias (hiddenSize,)
        /// <summary>
        /// Bias vector for the input projection of the update gate. Shape: (hiddenSize,).
        /// </summary>
        private readonly ITensor b_iz; // Update gate bias (hiddenSize,)
        /// <summary>
        /// Bias vector for the input projection of the candidate. Shape: (hiddenSize,).
        /// </summary>
        private readonly ITensor b_in; // Candidate bias (hiddenSize,)
        /// <summary>
        /// Bias vector for the hidden projection of the reset gate. Shape: (hiddenSize,).
        /// </summary>
        private readonly ITensor b_hr; // Hidden reset bias (hiddenSize,)
        /// <summary>
        /// Bias vector for the hidden projection of the update gate. Shape: (hiddenSize,).
        /// </summary>
        private readonly ITensor b_hz; // Hidden update bias (hiddenSize,)
        /// <summary>
        /// Bias vector for the hidden projection of the candidate. Shape: (hiddenSize,).
        /// </summary>
        private readonly ITensor b_hn; // Hidden candidate bias (hiddenSize,)

        /// <summary>
        /// Current hidden state maintained across time steps. Shape: (hiddenSize,).
        /// </summary>
        private ITensor hidden; // Current hidden state (hiddenSize,)

        /// <summary>
        /// Initializes a new instance of the GRU layer.
        /// </summary>
        /// <param name="inputSize">The size of the input features.</param>
        /// <param name="hiddenSize">The size of the hidden state.</param>
        public GRU(int inputSize, int hiddenSize)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;

            // Initialize weights using Xavier uniform initialization
            W_ir = Initializers.XavierUniform(new TensorShape(hiddenSize, inputSize));
            W_iz = Initializers.XavierUniform(new TensorShape(hiddenSize, inputSize));
            W_in = Initializers.XavierUniform(new TensorShape(hiddenSize, inputSize));
            W_hr = Initializers.XavierUniform(new TensorShape(hiddenSize, hiddenSize));
            W_hz = Initializers.XavierUniform(new TensorShape(hiddenSize, hiddenSize));
            W_hn = Initializers.XavierUniform(new TensorShape(hiddenSize, hiddenSize));

            // Initialize biases to zeros
            b_ir = Tensor.Zeros(new TensorShape(hiddenSize));
            b_iz = Tensor.Zeros(new TensorShape(hiddenSize));
            b_in = Tensor.Zeros(new TensorShape(hiddenSize));
            b_hr = Tensor.Zeros(new TensorShape(hiddenSize));
            b_hz = Tensor.Zeros(new TensorShape(hiddenSize));
            b_hn = Tensor.Zeros(new TensorShape(hiddenSize));

            // Initialize hidden state to zeros
            hidden = Tensor.Zeros(new TensorShape(hiddenSize));
        }
        /// <summary>
        /// Computes the forward pass of the GRU layer for a single time step.
        /// </summary>
        /// <param name="input">
        /// The input tensor for the current time step. 
        /// Shape can be <c>(inputSize,)</c> for a single instance or <c>(batchSize, inputSize)</c> for a batch.
        /// </param>
        /// <returns>
        /// The updated hidden state tensor of shape <c>(batchSize, hiddenSize)</c>.
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the input feature dimension does not match the configured <see cref="inputSize"/>.</exception>


        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input);

            int batch = input.Shape.Rank == 2 ? input.Shape[0] : 1;
            int inputFeatures = input.Shape[^1];

            if (inputFeatures != inputSize)
                throw new ArgumentException($"Input feature size mismatch. Expected {inputSize}, got {inputFeatures}");

            ITensor x = input.Reshape(batch, inputSize);

            // Dynamically scale hidden states to match batch dimension
            ITensor h = (hidden.Shape.Rank == 2 && hidden.Shape[0] == batch)
                ? hidden
                : Tensor.Zeros(new TensorShape(batch, hiddenSize), input.Device);

            var r = W_ir.MatMul(x.Transpose(new[] { 1, 0 }))
                .Add(b_ir.Reshape(hiddenSize, 1))
                .Add(W_hr.MatMul(h.Transpose(new[] { 1, 0 })))
                .Add(b_hr.Reshape(hiddenSize, 1));

            r = r.Sigmoid().Transpose(new[] { 1, 0 });

            var z = W_iz.MatMul(x.Transpose(new[] { 1, 0 }))
                .Add(b_iz.Reshape(hiddenSize, 1))
                .Add(W_hz.MatMul(h.Transpose(new[] { 1, 0 })))
                .Add(b_hz.Reshape(hiddenSize, 1));

            z = z.Sigmoid().Transpose(new[] { 1, 0 });

            var n_temp = W_hn.MatMul(h.Transpose(new[] { 1, 0 })).Add(b_hn.Reshape(hiddenSize, 1)).Transpose(new[] { 1, 0 });
            n_temp = r.Multiply(n_temp);

            var n = W_in.MatMul(x.Transpose(new[] { 1, 0 }))
                .Add(b_in.Reshape(hiddenSize, 1))
                .Add(n_temp.Transpose(new[] { 1, 0 }));

            n = n.Tanh().Transpose(new[] { 1, 0 });

            var one = Tensor.Ones(z.Shape, input.Device);
            hidden = z.Multiply(h).Add(one.Subtract(z).Multiply(n));

            return hidden;
        }
        /// <summary>
        /// Retrieves all learnable parameter tensors (weights and biases) of this GRU layer.
        /// </summary>
        /// <returns>An enumerable collection of parameter tensors in the sequence of weights followed by biases.</returns>


        public override IEnumerable<ITensor> Parameters()
        {
            yield return W_ir;
            yield return W_iz;
            yield return W_in;
            yield return W_hr;
            yield return W_hz;
            yield return W_hn;
            yield return b_ir;
            yield return b_iz;
            yield return b_in;
            yield return b_hr;
            yield return b_hz;
            yield return b_hn;
        }
        /// <summary>
        /// Resets the internal hidden state of the GRU to a zero-filled vector of shape <c>(hiddenSize,)</c>.
        /// </summary>

        public void ResetHidden()
        {
            hidden = Tensor.Zeros(new TensorShape(hiddenSize));
        }
    }
}