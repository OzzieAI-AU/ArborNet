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
    using ArborNet.Activations;
    using ArborNet.Core.Functional;
    using ArborNet.Core.Layers;
    /// <summary>
    /// Represents a Long Short-Term Memory (LSTM) recurrent neural network layer.
    /// This layer maintains and updates a cell state and hidden state over sequential inputs,
    /// utilizing gating mechanisms to control information retention and flow.
    /// </summary>

    #endregion

    public class LSTM : BaseLayer
    {
        private readonly int _inputSize;
        private readonly int _hiddenSize;
        private readonly Device _device;

        private ITensor _Wf, _Wi, _Wo, _Wc;
        private ITensor _Uf, _Ui, _Uo, _Uc;
        private ITensor _bf, _bi, _bo, _bc;

        private ITensor _hidden;
        private ITensor _cell;

        public LSTM(int inputSize, int hiddenSize, Device device = null)
        {
            if (inputSize <= 0) throw new ArgumentException("inputSize must be > 0");
            if (hiddenSize <= 0) throw new ArgumentException("hiddenSize must be > 0");

            _inputSize = inputSize;
            _hiddenSize = hiddenSize;
            _device = device ?? Device.CPU;

            _Wf = Initializers.XavierUniform(new TensorShape(hiddenSize, inputSize), _device);
            _Wi = Initializers.XavierUniform(new TensorShape(hiddenSize, inputSize), _device);
            _Wo = Initializers.XavierUniform(new TensorShape(hiddenSize, inputSize), _device);
            _Wc = Initializers.XavierUniform(new TensorShape(hiddenSize, inputSize), _device);

            _Uf = Initializers.XavierUniform(new TensorShape(hiddenSize, hiddenSize), _device);
            _Ui = Initializers.XavierUniform(new TensorShape(hiddenSize, hiddenSize), _device);
            _Uo = Initializers.XavierUniform(new TensorShape(hiddenSize, hiddenSize), _device);
            _Uc = Initializers.XavierUniform(new TensorShape(hiddenSize, hiddenSize), _device);

            _bf = Tensor.Zeros(new TensorShape(hiddenSize), _device);
            _bi = Tensor.Zeros(new TensorShape(hiddenSize), _device);
            _bo = Tensor.Zeros(new TensorShape(hiddenSize), _device);
            _bc = Tensor.Zeros(new TensorShape(hiddenSize), _device);

            _hidden = Tensor.Zeros(new TensorShape(hiddenSize), _device);
            _cell = Tensor.Zeros(new TensorShape(hiddenSize), _device);

            foreach (var t in new[] { _Wf, _Wi, _Wo, _Wc, _Uf, _Ui, _Uo, _Uc, _bf, _bi, _bo, _bc })
                t.RequiresGrad = true;
        }
        /// <summary>
        /// Performs the forward pass of the LSTM layer over the provided input sequence.
        /// </summary>
        /// <param name="input">The input sequence tensor. Supports shape (SeqLen, InputSize) or (Batch, SeqLen, InputSize).</param>
        /// <returns>The updated hidden state tensor after processing the entire sequence, with shape (Batch, HiddenSize).</returns>

        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input);

            int batch = input.Shape.Rank == 3 ? input.Shape[0] : 1;
            int seqLen = input.Shape.Rank == 3 ? input.Shape[1] : input.Shape[0];

            ITensor h = (_hidden.Shape.Rank == 2 && _hidden.Shape[0] == batch)
                ? _hidden
                : Tensor.Zeros(new TensorShape(batch, _hiddenSize), _device);

            ITensor c = (_cell.Shape.Rank == 2 && _cell.Shape[0] == batch)
                ? _cell
                : Tensor.Zeros(new TensorShape(batch, _hiddenSize), _device);

            for (int t = 0; t < seqLen; t++)
            {
                ITensor x = (input.Shape.Rank == 3)
                    ? input.Slice(new (int, int, int)[] { (0, batch, 1), (t, t + 1, 1), (0, _inputSize, 1) }).Reshape(batch, _inputSize)
                    : input.Slice(new (int, int, int)[] { (t, t + 1, 1), (0, _inputSize, 1) }).Reshape(batch, _inputSize);

                var ft = new Sigmoid().Forward(x.MatMul(_Wf).Add(h.MatMul(_Uf)).Add(_bf));
                var i_t = new Sigmoid().Forward(x.MatMul(_Wi).Add(h.MatMul(_Ui)).Add(_bi));
                var ot = new Sigmoid().Forward(x.MatMul(_Wo).Add(h.MatMul(_Uo)).Add(_bo));
                var ct = new Tanh().Forward(x.MatMul(_Wc).Add(h.MatMul(_Uc)).Add(_bc));

                c = ft.Multiply(c).Add(i_t.Multiply(ct));
                h = ot.Multiply(new Tanh().Forward(c));
            }

            _hidden = h;
            _cell = c;

            return _hidden;
        }
        /// <summary>
        /// Enumerates all learnable parameters (weights and biases) of this LSTM layer.
        /// </summary>
        /// <returns>An enumerable collection of parameter tensors.</returns>

        public override IEnumerable<ITensor> Parameters()
        {
            yield return _Wf; yield return _Wi; yield return _Wo; yield return _Wc;
            yield return _Uf; yield return _Ui; yield return _Uo; yield return _Uc;
            yield return _bf; yield return _bi; yield return _bo; yield return _bc;
        }
        /// <summary>
        /// Resets the internal hidden and cell states of the LSTM layer to zero.
        /// </summary>

        public void ResetHidden()
        {
            _hidden = Tensor.Zeros(new TensorShape(_hiddenSize), _device);
            _cell = Tensor.Zeros(new TensorShape(_hiddenSize), _device);
        }
    }
}
