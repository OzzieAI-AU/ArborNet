using System;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Devices;
using ArborNet.Core.Functional;

namespace ArborNet.Data
{
    /// <summary>
    /// Provides common image transformation operations for tensors.
    /// All methods assume 4D image tensors with shape [batch, channels, height, width].
    /// </summary>
    public enum InterpolationMode
    {
        /// <summary>
        /// Nearest neighbor interpolation. Fast but may produce pixelated results.
        /// </summary>
        Nearest,

        /// <summary>
        /// Bilinear interpolation. Uses weighted average of four nearest neighbors for smoother results.
        /// </summary>
        Bilinear
    }

    /// <summary>
    /// Provides static utility methods for common geometric transformations on image tensors.
    /// </summary>
    /// <remarks>
    /// All operations work with 4D tensors of shape [batch, channels, height, width].
    /// Methods are functional and do not modify the input tensor, returning new tensor instances instead.
    /// </remarks>
    public static class Transforms
    {
        /// <summary>
        /// Resizes a 4D tensor [batch, channels, height, width] using nearest-neighbor or bilinear interpolation.
        /// </summary>
        /// <param name="input">The input 4D tensor to be resized.</param>
        /// <param name="newHeight">The target height of the output tensor.</param>
        /// <param name="newWidth">The target width of the output tensor.</param>
        /// <param name="mode">The interpolation mode to use during resizing.</param>
        /// <returns>A new <see cref="ITensor"/> containing the resized data with shape [batch, channels, newHeight, newWidth].</returns>
        /// <exception cref="ArgumentException">Thrown if the input tensor is not 4-dimensional or if newHeight/newWidth are not positive.</exception>
        public static ITensor Resize(ITensor input, int newHeight, int newWidth,
            InterpolationMode mode = InterpolationMode.Bilinear)
        {
            if (input.Shape.Rank != 4)
                throw new ArgumentException("Input must be a 4D tensor [batch, channels, height, width].");

            int batch = input.Shape[0];
            int channels = input.Shape[1];
            int oldHeight = input.Shape[2];
            int oldWidth = input.Shape[3];

            if (newHeight <= 0 || newWidth <= 0)
                throw new ArgumentException("newHeight and newWidth must be positive.");

            var outputShape = new TensorShape(batch, channels, newHeight, newWidth);
            var inputData = input.ToArray();
            var outputData = new float[outputShape.TotalElements];

            float scaleH = (float)oldHeight / newHeight;
            float scaleW = (float)oldWidth / newWidth;

            int inStrideB = channels * oldHeight * oldWidth;
            int inStrideC = oldHeight * oldWidth;
            int inStrideH = oldWidth;

            int outStrideB = channels * newHeight * newWidth;
            int outStrideC = newHeight * newWidth;
            int outStrideH = newWidth;

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < newHeight; h++)
                    {
                        for (int w = 0; w < newWidth; w++)
                        {
                            float val;

                            if (mode == InterpolationMode.Nearest)
                            {
                                int oh = Math.Min((int)(h * scaleH), oldHeight - 1);
                                int ow = Math.Min((int)(w * scaleW), oldWidth - 1);
                                int inIdx = b * inStrideB + c * inStrideC + oh * inStrideH + ow;
                                val = inputData[inIdx];
                            }
                            else // Bilinear
                            {
                                float fh = h * scaleH;
                                float fw = w * scaleW;
                                int oh0 = (int)fh;
                                int ow0 = (int)fw;
                                int oh1 = Math.Min(oh0 + 1, oldHeight - 1);
                                int ow1 = Math.Min(ow0 + 1, oldWidth - 1);

                                float dh = fh - oh0;
                                float dw = fw - ow0;

                                int idx00 = b * inStrideB + c * inStrideC + oh0 * inStrideH + ow0;
                                int idx01 = b * inStrideB + c * inStrideC + oh0 * inStrideH + ow1;
                                int idx10 = b * inStrideB + c * inStrideC + oh1 * inStrideH + ow0;
                                int idx11 = b * inStrideB + c * inStrideC + oh1 * inStrideH + ow1;

                                val = inputData[idx00] * (1 - dh) * (1 - dw) +
                                      inputData[idx01] * (1 - dh) * dw +
                                      inputData[idx10] * dh * (1 - dw) +
                                      inputData[idx11] * dh * dw;
                            }

                            int outIdx = b * outStrideB + c * outStrideC + h * outStrideH + w;
                            outputData[outIdx] = val;
                        }
                    }
                }
            }

            return Ops.FromArray(outputData, outputShape, input.Device);
        }

        /// <summary>
        /// Flips the tensor horizontally (left-right).
        /// </summary>
        /// <param name="input">The input 4D image tensor.</param>
        /// <returns>A new <see cref="ITensor"/> with the same shape as the input but with horizontal flipping applied to each image.</returns>
        /// <remarks>This operation is performed per batch and channel independently.</remarks>
        public static ITensor FlipHorizontal(ITensor input)
        {
            if (input.Shape.Rank != 4)
                throw new ArgumentException("Input must be 4D [batch, channels, height, width].");

            int B = input.Shape[0], C = input.Shape[1], H = input.Shape[2], W = input.Shape[3];
            var shape = new TensorShape(B, C, H, W);
            var data = input.ToArray();
            var result = new float[data.Length];

            int strideB = C * H * W;
            int strideC = H * W;
            int strideH = W;

            for (int b = 0; b < B; b++)
                for (int c = 0; c < C; c++)
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++)
                        {
                            int srcIdx = b * strideB + c * strideC + h * strideH + w;
                            int dstIdx = b * strideB + c * strideC + h * strideH + (W - 1 - w);
                            result[dstIdx] = data[srcIdx];
                        }

            return Ops.FromArray(result, shape, input.Device);
        }

        /// <summary>
        /// Flips the tensor vertically (up-down).
        /// </summary>
        /// <param name="input">The input 4D image tensor.</param>
        /// <returns>A new <see cref="ITensor"/> with the same shape as the input but with vertical flipping applied to each image.</returns>
        /// <remarks>This operation is performed per batch and channel independently.</remarks>
        public static ITensor FlipVertical(ITensor input)
        {
            if (input.Shape.Rank != 4)
                throw new ArgumentException("Input must be 4D [batch, channels, height, width].");

            int B = input.Shape[0], C = input.Shape[1], H = input.Shape[2], W = input.Shape[3];
            var shape = new TensorShape(B, C, H, W);
            var data = input.ToArray();
            var result = new float[data.Length];

            int strideB = C * H * W;
            int strideC = H * W;
            int strideH = W;

            for (int b = 0; b < B; b++)
                for (int c = 0; c < C; c++)
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++)
                        {
                            int srcIdx = b * strideB + c * strideC + h * strideH + w;
                            int dstIdx = b * strideB + c * strideC + (H - 1 - h) * strideH + w;
                            result[dstIdx] = data[srcIdx];
                        }

            return Ops.FromArray(result, shape, input.Device);
        }

        /// <summary>
        /// Rotates the tensor 90 degrees (clockwise or counterclockwise).
        /// Output shape becomes [batch, channels, oldWidth, oldHeight].
        /// </summary>
        /// <param name="input">The input 4D image tensor.</param>
        /// <param name="clockwise">If <c>true</c>, rotates the image 90 degrees clockwise; otherwise rotates counterclockwise.</param>
        /// <returns>A new <see cref="ITensor"/> containing the rotated images. Height and width dimensions are swapped.</returns>
        /// <remarks>The rotation is applied to each image in the batch independently.</remarks>
        public static ITensor Rotate90(ITensor input, bool clockwise = true)
        {
            if (input.Shape.Rank != 4)
                throw new ArgumentException("Input must be 4D [batch, channels, height, width].");

            int B = input.Shape[0], C = input.Shape[1], H = input.Shape[2], W = input.Shape[3];
            var outputShape = new TensorShape(B, C, W, H);
            var data = input.ToArray();
            var result = new float[outputShape.TotalElements];

            int inStrideB = C * H * W;
            int inStrideC = H * W;
            int inStrideH = W;

            int outStrideB = C * W * H;
            int outStrideC = W * H;
            int outStrideH = H; // stride for the new height dimension (old width)

            for (int b = 0; b < B; b++)
                for (int c = 0; c < C; c++)
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++)
                        {
                            int srcIdx = b * inStrideB + c * inStrideC + h * inStrideH + w;

                            int outH, outW;
                            if (clockwise)
                            {
                                outH = w;
                                outW = H - 1 - h;
                            }
                            else
                            {
                                outH = W - 1 - w;
                                outW = h;
                            }

                            int dstIdx = b * outStrideB + c * outStrideC + outH * outStrideH + outW;
                            result[dstIdx] = data[srcIdx];
                        }

            return Ops.FromArray(result, outputShape, input.Device);
        }

        /// <summary>
        /// Applies random augmentations (horizontal flip, vertical flip, and 0-3 rotations of 90°).
        /// </summary>
        /// <param name="input">The input 4D tensor to augment.</param>
        /// <param name="random">An optional <see cref="Random"/> instance to use for randomness. 
        /// If <c>null</c>, a new <see cref="Random"/> instance will be created.</param>
        /// <returns>A new <see cref="ITensor"/> with randomly applied augmentations.</returns>
        /// <remarks>
        /// The augmentations are applied sequentially in the following order:
        /// <list type="bullet">
        ///   <item>50% chance of horizontal flip</item>
        ///   <item>50% chance of vertical flip</item>
        ///   <item>0-3 clockwise 90-degree rotations (chosen randomly)</item>
        /// </list>
        /// </remarks>
        public static ITensor Augment(ITensor input, Random? random = null)
        {
            random ??= new Random();

            ITensor result = input;

            if (random.Next(2) == 1)
                result = FlipHorizontal(result);

            if (random.Next(2) == 1)
                result = FlipVertical(result);

            int rotations = random.Next(4);
            for (int i = 0; i < rotations; i++)
                result = Rotate90(result, clockwise: true);

            return result;
        }
    }
}