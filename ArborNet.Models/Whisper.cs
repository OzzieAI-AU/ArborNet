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

    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Layers;
    using System.Collections.Generic;
    /// <summary>
    /// Represents the Whisper model architecture designed for automatic speech recognition (ASR) and speech translation.
    /// This implementation combines a convolutional neural network (CNN) frontend for feature extraction from audio spectrograms 
    /// with a transformer-based encoder-decoder architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The architecture consists of:
    /// <list type="bullet">
    /// <item><description>A convolutional frontend (two 1D convolutional layers with ReLU activations) to subsample the input audio representation.</description></item>
    /// <item><description>A Transformer encoder that processes the downsampled audio features.</description></item>
    /// <item><description>A Transformer decoder that autoregressively generates text tokens guided by the encoder's hidden states.</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// Note: The current <see cref="Forward(ITensor)"/> implementation focuses on the audio encoder forward pass. 
    /// The decoder blocks, embeddings, positional encodings, and output head are fully initialized and registered to ensure parameter tracking 
    /// and compatibility with complete sequence-to-sequence training and inference workflows.
    /// </para>
    /// </remarks>

    #endregion

    public class Whisper : BaseModel
    {
        /// <summary>
        /// Convolutional layers responsible for initial feature extraction from the mel spectrogram input.
        /// </summary>
        private readonly Conv1D conv1, conv2;

        /// <summary>
        /// The stack of transformer blocks that form the audio encoder.
        /// </summary>
        private readonly List<TransformerBlock> encoder;

        /// <summary>
        /// Token embedding layer for the text decoder.
        /// </summary>
        private readonly Embedding decoderEmb;

        /// <summary>
        /// Positional encoding component for decoder input tokens.
        /// </summary>
        private readonly PositionalEncoding decoderPos;

        /// <summary>
        /// The stack of transformer blocks that form the text decoder.
        /// </summary>
        private readonly List<TransformerBlock> decoder;

        /// <summary>
        /// Final linear projection layer mapping decoder hidden states to vocabulary logits.
        /// </summary>
        private readonly Linear head;
        /// <summary>
        /// Retrieves all trainable parameters (weights, biases, etc.) associated with this model,
        /// including those of the convolutional frontend, the encoder stack, the decoder stack, embeddings, and the projection head.
        /// </summary>
        /// <returns>An <see cref="IEnumerable{ITensor}"/> containing all registered parameters of the model.</returns>

        public override IEnumerable<ITensor> Parameters() => parameters;

        /// <summary>
        /// Initializes a new instance of the <see cref="Whisper"/> class.
        /// </summary>
        /// <param name="nMel">Number of mel frequency bins in the input spectrogram. Default is 80.</param>
        /// <param name="nAudioCtx">Maximum audio context length (number of frames). Default is 1500.</param>
        /// <param name="nAudioState">Hidden state dimension for the audio encoder and convolution layers. Default is 768.</param>
        /// <param name="nAudioHead">Number of attention heads in each audio encoder transformer block. Default is 12.</param>
        /// <param name="nAudioLayer">Number of transformer blocks in the audio encoder. Default is 12.</param>
        /// <param name="nVocab">Size of the output vocabulary. Default is 51865.</param>
        /// <param name="nTextCtx">Maximum text context length (number of tokens). Default is 448.</param>
        /// <param name="nTextState">Hidden state dimension for the text decoder. Default is 768.</param>
        /// <param name="nTextHead">Number of attention heads in each text decoder transformer block. Default is 12.</param>
        /// <param name="nTextLayer">Number of transformer blocks in the text decoder. Default is 12.</param>
        public Whisper(int nMel = 80, int nAudioCtx = 1500, int nAudioState = 768, int nAudioHead = 12, int nAudioLayer = 12,
                       int nVocab = 51865, int nTextCtx = 448, int nTextState = 768, int nTextHead = 12, int nTextLayer = 12)
        {
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
        /// <summary>
        /// Executes the forward pass of the audio encoder pipeline.
        /// </summary>
        /// <param name="input">
        /// An <see cref="ITensor"/> representing the input Mel-spectrogram. 
        /// Expected tensor shape is <c>(batch_size, nMel, time_steps)</c>.
        /// </param>
        /// <returns>
        /// An <see cref="ITensor"/> representing the encoded audio features. 
        /// The tensor is transposed to shape <c>(batch_size, downsampled_time_steps, nAudioState)</c> 
        /// ready for consumption by the decoder or downstream tasks.
        /// </returns>

        public override ITensor Forward(ITensor input)
        {
            var x = conv1.Forward(input).Relu();
            x = conv2.Forward(x).Relu();
            x = x.Transpose(new[] { 0, 2, 1 });
            foreach (var b in encoder)
                x = b.Forward(x);
            return x;
        }
    }
}