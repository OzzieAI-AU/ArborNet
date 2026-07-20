// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Data
{

    #region Using Statements:

    using System;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Functional;

    #endregion

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
    /// Provides high-performance static utility methods for spatial, geometric, and data augmentation transformations 
    /// on multi-dimensional image tensors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// All operations in this class expect 4-dimensional tensors conforming to the NCHW channel layout:
    /// <list type="bullet">
    /// <item><description><b>N (Batch):</b> Represents the batch size.</description></item>
    /// <item><description><b>C (Channels):</b> Represents color channels (e.g., RGB, Grayscale, or feature maps).</description></item>
    /// <item><description><b>H (Height):</b> Represents the vertical spatial dimension.</description></item>
    /// <item><description><b>W (Width):</b> Represents the horizontal spatial dimension.</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The methods are pure functions and do not mutate the input tensor. Instead, they allocate and return 
    /// a new <see cref="ITensor"/> containing the transformed data on the same execution device.
    /// </para>
    /// </remarks>

    public static class Transforms
    {
        /// <summary>
        /// Resizes a 4D image tensor of shape <c>[batch, channels, height, width]</c> to a new spatial resolution.
        /// </summary>
        /// <param name="input">The source 4D tensor to be resized.</param>
        /// <param name="newHeight">The target spatial height of the output tensor. Must be greater than zero.</param>
        /// <param name="newWidth">The target spatial width of the output tensor. Must be greater than zero.</param>
        /// <param name="mode">The interpolation method to apply. Defaults to <see cref="InterpolationMode.Bilinear"/>.</param>
        /// <returns>A new <see cref="ITensor"/> containing the resized data with the shape <c>[batch, channels, newHeight, newWidth]</c> on the same device.</returns>
        /// <exception cref="ArgumentException">
        /// Thrown if the <paramref name="input"/> tensor does not have exactly 4 dimensions, 
        /// or if <paramref name="newHeight"/> or <paramref name="newWidth"/> are less than or equal to zero.
        /// </exception>
        /// <remarks>
        /// Supports <see cref="InterpolationMode.Nearest"/> for discrete, fast pixel replication and 
        /// <see cref="InterpolationMode.Bilinear"/> for smooth continuous coordinate mapping using 2D bilinear weights.
        /// </remarks>
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
        /// Flips the spatial width dimension of a 4D image tensor horizontally (left-to-right mirror effect).
        /// </summary>
        /// <param name="input">The source 4-dimensional tensor of shape <c>[batch, channels, height, width]</c> to flip.</param>
        /// <returns>A new <see cref="ITensor"/> containing the horizontally flipped image data, retaining the original shape and execution device.</returns>
        /// <exception cref="ArgumentNullException">Thrown if the <paramref name="input"/> tensor is null.</exception>
        /// <exception cref="ArgumentException">Thrown if the <paramref name="input"/> tensor does not have exactly 4 dimensions.</exception>
        /// <remarks>
        /// This operation maps index <c>x</c> in the width dimension to <c>width - 1 - x</c> for all batches, channels, and heights.
        /// </remarks>

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
        /// Flips the spatial height dimension of a 4D image tensor vertically (upside-down mirror effect).
        /// </summary>
        /// <param name="input">The source 4-dimensional tensor of shape <c>[batch, channels, height, width]</c> to flip.</param>
        /// <returns>A new <see cref="ITensor"/> containing the vertically flipped image data, retaining the original shape and execution device.</returns>
        /// <exception cref="ArgumentNullException">Thrown if the <paramref name="input"/> tensor is null.</exception>
        /// <exception cref="ArgumentException">Thrown if the <paramref name="input"/> tensor does not have exactly 4 dimensions.</exception>
        /// <remarks>
        /// This operation maps index <c>y</c> in the height dimension to <c>height - 1 - y</c> for all batches, channels, and widths.
        /// </remarks>

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
        /// Rotates the spatial dimensions of a 4D image tensor by 90 degrees either clockwise or counter-clockwise.
        /// </summary>
        /// <param name="input">The source 4-dimensional tensor of shape <c>[batch, channels, height, width]</c> to rotate.</param>
        /// <param name="clockwise">If set to <see langword="true"/>, rotates 90 degrees clockwise; if <see langword="false"/>, rotates 90 degrees counter-clockwise. Defaults to <see langword="true"/>.</param>
        /// <returns>A new <see cref="ITensor"/> with spatial dimensions transposed, resulting in an output shape of <c>[batch, channels, width, height]</c>.</returns>
        /// <exception cref="ArgumentNullException">Thrown if the <paramref name="input"/> tensor is null.</exception>
        /// <exception cref="ArgumentException">Thrown if the <paramref name="input"/> tensor does not have exactly 4 dimensions.</exception>
        /// <remarks>
        /// This operation swaps the vertical (height) and horizontal (width) dimensions of the tensor, altering the output shape. 
        /// It operates on each channel and batch slice independently.
        /// </remarks>

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
        /// Applies random spatial data augmentations to an input 4D image tensor, including horizontal and vertical flips, as well as discrete 90-degree rotations.
        /// </summary>
        /// <param name="input">The source 4-dimensional tensor of shape <c>[batch, channels, height, width]</c> to augment.</param>
        /// <param name="random">An optional pseudorandom number generator instance of <see cref="Random"/>. If <see langword="null"/>, a default thread-safe instance is created internally.</param>
        /// <returns>A new <see cref="ITensor"/> representing the augmented image, which may have transposed spatial dimensions if a non-zero rotation was applied.</returns>
        /// <exception cref="ArgumentNullException">Thrown if the <paramref name="input"/> tensor is null.</exception>
        /// <exception cref="ArgumentException">Thrown if the <paramref name="input"/> tensor does not have exactly 4 dimensions (propagated from called methods).</exception>
        /// <remarks>
        /// The pipeline executes the following sequential random transformations:
        /// <list type="number">
        /// <item><description><b>Horizontal Flip:</b> Evaluates a 50% probability to apply <see cref="FlipHorizontal"/>.</description></item>
        /// <item><description><b>Vertical Flip:</b> Evaluates a 50% probability to apply <see cref="FlipVertical"/>.</description></item>
        /// <item><description><b>90-Degree Rotations:</b> Applies clockwise rotations 0, 1, 2, or 3 times based on an even probability distribution using <see cref="Rotate90"/>.</description></item>
        /// </list>
        /// This method is highly suited for training neural network pipelines to enforce spatial invariance.
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