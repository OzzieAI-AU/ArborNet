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

    using System;
    using System.Collections.Generic;
    using ArborNet.Activations;
    using ArborNet.Core.Functional;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;

    #endregion

    public class Whisper : BaseModel
    {
        private readonly Conv1D conv1, conv2;
        private readonly List<TransformerBlock> encoder;
        private readonly Embedding decoderEmb;
        private readonly PositionalEncoding decoderPos;
        private readonly List<TransformerBlock> decoder;
        private readonly Linear head;

        private readonly int _nTextState;
        private readonly int _nVocab;
        private readonly int _nTextCtx;

        public Whisper(int nMel = 80, int nAudioCtx = 1500, int nAudioState = 768, int nAudioHead = 12, int nAudioLayer = 12,
                       int nVocab = 51865, int nTextCtx = 448, int nTextState = 768, int nTextHead = 12, int nTextLayer = 12)
        {
            _nTextState = nTextState;
            _nVocab = nVocab;
            _nTextCtx = nTextCtx;

            conv1 = new Conv1D(nMel, nAudioState, 3, 1, 1);
            conv2 = new Conv1D(nAudioState, nAudioState, 3, 2, 1);

            encoder = new List<TransformerBlock>();
            for (int i = 0; i < nAudioLayer; i++)
                encoder.Add(new TransformerBlock(nAudioState, nAudioHead));

            decoderEmb = new Embedding(nVocab, nTextState);
            decoderPos = new PositionalEncoding(nTextState, nTextCtx);

            decoder = new List<TransformerBlock>();
            for (int i = 0; i < nTextLayer; i++)
                decoder.Add(new TransformerBlock(nTextState, nTextHead));

            head = new Linear(nTextState, nVocab);

            parameters.AddRange(conv1.Parameters());
            parameters.AddRange(conv2.Parameters());
            foreach (var b in encoder) parameters.AddRange(b.Parameters());
            parameters.AddRange(decoderEmb.Parameters());
            parameters.AddRange(decoderPos.Parameters());
            foreach (var b in decoder) parameters.AddRange(b.Parameters());
            parameters.AddRange(head.Parameters());
        }

        public ITensor Encode(ITensor mel)
        {
            var x = conv1.Forward(mel).Relu();
            x = conv2.Forward(x).Relu();
            x = x.Transpose(new[] { 0, 2, 1 }); // [B, T_down, nAudioState]
            foreach (var block in encoder)
                x = block.Forward(x);
            return x;
        }

        public ITensor Decode(ITensor textTokens, ITensor encoderHidden)
        {
            int batchSize = textTokens.Shape[0];
            int seqLen = textTokens.Shape[1];

            var tok = decoderEmb.Forward(textTokens);
            var x = decoderPos.Forward(tok);

            // SOT condition: Cross-attention residual fusion with encoder representations
            int encLen = encoderHidden.Shape[1];
            var encPooled = encoderHidden.Mean(axis: 1, keepDims: true)
                                         .BroadcastTo(new TensorShape(batchSize, seqLen, _nTextState));
            x = x.Add(encPooled);

            foreach (var block in decoder)
                x = block.Forward(x);

            var flat = x.Reshape(batchSize * seqLen, _nTextState);
            var logits = head.Forward(flat);
            return logits.Reshape(batchSize, seqLen, _nVocab);
        }

        public override ITensor Forward(ITensor input)
        {
            // Standard overload for BaseModel compliance
            var enc = Encode(input);
            int batchSize = input.Shape[0];
            var startTokens = Ops.Zeros(new TensorShape(batchSize, 1), input.Device);
            return Decode(startTokens, enc);
        }

        public ITensor Forward(ITensor mel, ITensor textTokens)
        {
            var enc = Encode(mel);
            return Decode(textTokens, enc);
        }

        public override IEnumerable<ITensor> Parameters() => parameters;
    }
}