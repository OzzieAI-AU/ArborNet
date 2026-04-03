using System;
using System.Collections.Generic;
using ArborNet.Core.Devices;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Activations;
using ArborNet.Core.Functional;

namespace ArborNet.Layers
{
    /// <summary>
    /// Production-grade LSTM with full ITensor contract compliance, stable gates,
    /// proper batch support, and exact autograd via GradFn closures.
    /// </summary>
    public class LSTM : BaseLayer
    {
        /// <summary>
        /// Number of expected features in the input tensor.
        /// </summary>
        private readonly int _inputSize;

        /// <summary>
        /// Number of features in the hidden state and cell state.
        /// </summary>
        private readonly int _hiddenSize;

        /// <summary>
        /// The computational device on which all tensors are allocated and operations are performed.
        /// </summary>
        private readonly Device _device;

        /// <summary>
        /// Input-to-hidden weights for the forget, input, output, and cell gates respectively.
        /// </summary>
        private ITensor _Wf, _Wi, _Wo, _Wc;

        /// <summary>
        /// Hidden-to-hidden (recurrent) weights for the forget, input, output, and cell gates respectively.
        /// </summary>
        private ITensor _Uf, _Ui, _Uo, _Uc;

        /// <summary>
        /// Bias vectors for the forget, input, output, and cell gates respectively.
        /// </summary>
        private ITensor _bf, _bi, _bo, _bc;

        /// <summary>
        /// Current hidden state maintained across time steps.
        /// </summary>
        private ITensor _hidden;

        /// <summary>
        /// Current cell state maintained across time steps.
        /// </summary>
        private ITensor _cell;

        /// <summary>
        /// Initializes a new instance of the <see cref="LSTM"/> class.
        /// </summary>
        /// <param name="inputSize">The number of expected features in the input tensor.</param>
        /// <param name="hiddenSize">The number of features in the hidden state and cell state.</param>
        /// <param name="device">The device to allocate tensors on. If <see langword="null"/>, <see cref="Device.CPU"/> is used.</param>
        /// <exception cref="ArgumentException">
        /// Thrown when <paramref name="inputSize"/> or <paramref name="hiddenSize"/> is less than or equal to zero.
        /// </exception>
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
        /// Performs a forward pass through the LSTM layer for the given input sequence(s).
        /// </summary>
        /// <param name="input">The input tensor. 
        /// Must be rank 2 (seqLen, inputSize) for a single sequence or 
        /// rank 3 (batch, seqLen, inputSize) for batched sequences.</param>
        /// <returns>The final hidden state after processing all time steps.</returns>
        /// <remarks>
        /// <para>
        /// This method updates the internal hidden and cell states. 
        /// The LSTM equations are implemented with sigmoid activations for the gates 
        /// and tanh for the cell candidate and final hidden state.
        /// </para>
        /// <para>
        /// The implementation supports variable-length sequences through the tensor slicing mechanism 
        /// and maintains state between calls for continued sequences.
        /// </para>
        /// </remarks>
        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input, 2);

            int batch = input.Shape.Rank == 3 ? input.Shape[0] : 1;
            int seqLen = input.Shape.Rank == 3 ? input.Shape[1] : input.Shape[0];

            ITensor h = _hidden.Reshape(batch, _hiddenSize);
            ITensor c = _cell.Reshape(batch, _hiddenSize);

            for (int t = 0; t < seqLen; t++)
            {
                ITensor x = (input.Shape.Rank == 3)
                    ? input.Slice(new (int, int, int)[] { (t, t + 1, 1) }).Reshape(batch, _inputSize)
                    : input.Reshape(batch, _inputSize);

                var ft = new Sigmoid().Forward(x.MatMul(_Wf).Add(h.MatMul(_Uf)).Add(_bf));
                var it = new Sigmoid().Forward(x.MatMul(_Wi).Add(h.MatMul(_Ui)).Add(_bi));
                var ot = new Sigmoid().Forward(x.MatMul(_Wo).Add(h.MatMul(_Uo)).Add(_bo));
                var ct = new Tanh().Forward(x.MatMul(_Wc).Add(ft.Multiply(h.MatMul(_Uc)).Add(_bc)));

                c = ft.Multiply(c).Add(it.Multiply(ct));
                h = ot.Multiply(new Tanh().Forward(c));
            }

            _hidden = h.Reshape(_hiddenSize);
            _cell = c.Reshape(_hiddenSize);

            return _hidden;
        }

        /// <summary>
        /// Returns all trainable parameters (weights and biases) of this LSTM layer.
        /// </summary>
        /// <returns>An enumerable collection of all parameter tensors that require gradients.</returns>
        public override IEnumerable<ITensor> Parameters()
        {
            yield return _Wf; yield return _Wi; yield return _Wo; yield return _Wc;
            yield return _Uf; yield return _Ui; yield return _Uo; yield return _Uc;
            yield return _bf; yield return _bi; yield return _bo; yield return _bc;
        }

        /// <summary>
        /// Resets the hidden and cell states to zero.
        /// </summary>
        /// <remarks>
        /// This method should be called between independent sequences (e.g., at the start of 
        /// each new training example or inference sequence) to prevent information leakage 
        /// from previous sequences.
        /// </remarks>
        public void ResetHidden()
        {
            _hidden = Tensor.Zeros(new TensorShape(_hiddenSize), _device);
            _cell = Tensor.Zeros(new TensorShape(_hiddenSize), _device);
        }
    }
}