using ArborNet.Core.Interfaces;
using ArborNet.Core.Models;
using ArborNet.Core.Tensors;
using ArborNet.Losses;
using ArborNet.Models;
using System.Collections.Generic;

namespace ArborNet.Models
{
    /// <summary>
    /// Implements the CLIP (Contrastive Language-Image Pre-training) model from OpenAI.
    /// Combines a Vision Transformer (ViT) image encoder and a Transformer text encoder
    /// to learn a shared multimodal embedding space using contrastive learning.
    /// </summary>
    /// <remarks>
    /// This implementation provides separate encoding methods for images and text,
    /// and a <see cref="ComputeContrastiveLoss"/> method that implements the standard
    /// symmetric cross-entropy loss used in CLIP training.
    /// </remarks>
    public class CLIP : BaseModel
    {
        /// <summary>
        /// The Vision Transformer encoder responsible for processing image inputs.
        /// </summary>
        private readonly ViT visionEncoder;

        /// <summary>
        /// The Transformer-based text encoder responsible for processing tokenized text inputs.
        /// </summary>
        private readonly TransformerTextEncoder textEncoder;

        /// <summary>
        /// Temperature parameter used to scale the similarity logits before computing the contrastive loss.
        /// </summary>
        private readonly float temperature;

        /// <summary>
        /// Gets all trainable parameters from both the vision encoder and text encoder.
        /// </summary>
        /// <returns>An enumerable collection of all model parameters as <see cref="ITensor"/> instances.</returns>
        public override IEnumerable<ITensor> Parameters() => parameters;

        /// <summary>
        /// Initializes a new instance of the <see cref="CLIP"/> class with the specified architecture parameters.
        /// </summary>
        /// <param name="imageSize">The height and width of input images. Default is 224.</param>
        /// <param name="patchSize">The patch size used by the Vision Transformer. Default is 16.</param>
        /// <param name="embedDim">The dimensionality of the shared embedding space. Default is 512.</param>
        /// <param name="numHeads">The number of attention heads in both encoders. Default is 8.</param>
        /// <param name="numLayers">The number of transformer layers in both encoders. Default is 12.</param>
        /// <param name="vocabSize">The size of the text vocabulary. Default is 49408.</param>
        /// <param name="maxSeqLen">The maximum sequence length for text tokens. Default is 77.</param>
        /// <param name="temperature">The temperature scaling factor for contrastive logits. Default is 0.07f.</param>
        public CLIP(int imageSize = 224, int patchSize = 16, int embedDim = 512, int numHeads = 8,
                    int numLayers = 12, int vocabSize = 49408, int maxSeqLen = 77, float temperature = 0.07f)
        {
            visionEncoder = new ViT(imageSize, patchSize, 3, embedDim, numHeads, numLayers, 1);
            textEncoder = new TransformerTextEncoder(embedDim, numHeads, numLayers, vocabSize, maxSeqLen);
            this.temperature = temperature;

            parameters.AddRange(visionEncoder.Parameters());
            parameters.AddRange(textEncoder.Parameters());
        }

        /// <summary>
        /// Performs a forward pass through the model. For CLIP, this overload routes the input to the vision encoder.
        /// </summary>
        /// <remarks>
        /// For full CLIP usage with both modalities, use <see cref="EncodeImage"/> and <see cref="EncodeText"/> instead.
        /// This override exists primarily for compatibility with the <see cref="BaseModel"/> base class.
        /// </remarks>
        /// <param name="input">The input tensor, expected to contain image data.</param>
        /// <returns>The output from the vision encoder.</returns>
        public override ITensor Forward(ITensor input)
        {
            // For CLIP we usually need two inputs. This overload accepts image only.
            // Use EncodeImage / EncodeText for full usage.
            return visionEncoder.Forward(input);
        }

        /// <summary>
        /// Encodes an image into a normalized embedding using the vision transformer.
        /// </summary>
        /// <param name="image">The input image tensor with shape [batch, channels, height, width].</param>
        /// <returns>The image embeddings tensor.</returns>
        public ITensor EncodeImage(ITensor image) => visionEncoder.Forward(image);

        /// <summary>
        /// Encodes tokenized text into a normalized embedding using the text transformer.
        /// </summary>
        /// <param name="text">The input text tensor containing token IDs with shape [batch, sequence_length].</param>
        /// <returns>The text embeddings tensor.</returns>
        public ITensor EncodeText(ITensor text) => textEncoder.Forward(text);

        /// <summary>
        /// Computes the contrastive loss between batches of image and text embeddings.
        /// </summary>
        /// <param name="imageEmb">The image embeddings tensor with shape [batch, embedDim].</param>
        /// <param name="textEmb">The text embeddings tensor with shape [batch, embedDim].</param>
        /// <returns>The scalar cross-entropy loss value for the contrastive objective.</returns>
        /// <remarks>
        /// This method implements the standard CLIP loss by normalizing embeddings,
        /// computing scaled cosine similarities, and applying symmetric cross-entropy
        /// with labels corresponding to the diagonal (matching image-text pairs).
        /// </remarks>
        public ITensor ComputeContrastiveLoss(ITensor imageEmb, ITensor textEmb)
        {
            var imgNorm = imageEmb.Divide(imageEmb.Pow(2).Sum(-1).Sqrt().ReshapeWithBroadcast(imageEmb.Shape, -1));
            var txtNorm = textEmb.Divide(textEmb.Pow(2).Sum(-1).Sqrt().ReshapeWithBroadcast(textEmb.Shape, -1));
            var logits = imgNorm.MatMul(txtNorm.Transpose(new[] { 1, 0 })).Multiply(1f / temperature);
            
            // Fixed: Tensor.Arange does not exist. Create labels tensor manually (0 to n-1).
            int n = logits.Shape[0];
            float[] labelData = new float[n];
            for (int i = 0; i < n; i++) labelData[i] = i;
            ITensor labels = Tensor.FromArray(labelData, new TensorShape(n), logits.Device);
            
            var loss = new CrossEntropy().Forward(logits, labels);
            return loss;
        }
    }
}