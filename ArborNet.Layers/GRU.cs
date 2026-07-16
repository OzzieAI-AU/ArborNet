using System;
using System.Collections.Generic;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Functional;
using ArborNet.Activations;
using ArborNet.Core.Layers;

namespace ArborNet.Layers
{
    /// <summary>
    /// Implements a Gated Recurrent Unit (GRU) layer for sequential data processing.
    /// This is a simplified single-step GRU assuming batch size of 1.
    /// </summary>
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
        /// Performs the forward pass of the GRU layer.
        /// Assumes input is a tensor of shape (inputSize,) representing a single time step.
        /// Updates the internal hidden state and returns the new hidden state.
        /// </summary>
        /// <param name="input">The input tensor of shape (inputSize,).</param>
        /// <returns>The output hidden state tensor of shape (hiddenSize,).</returns>
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
        /// Gets the parameters of the GRU layer (weights and biases).
        /// </summary>
        /// <returns>An enumerable collection of the layer's parameters.</returns>
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
        /// Resets the hidden state to zeros.
        /// </summary>
        public void ResetHidden()
        {
            hidden = Tensor.Zeros(new TensorShape(hiddenSize));
        }
    }
}