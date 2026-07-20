// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Activations
{

    #region Using Statements:

    using System;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    /// <summary>
    /// Represents the Scaled Exponential Linear Unit (SELU) activation function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SELU is an activation function that induces self-normalizing properties in deep neural networks.
    /// Mathematically, it is defined element-wise as:
    /// <code>
    /// SELU(x) = scale * x, if x &gt; 0
    /// SELU(x) = scale * alpha * (exp(x) - 1), if x &lt;= 0
    /// </code>
    /// </para>
    /// <para>
    /// The constants <see cref="Alpha"/> and <see cref="Scale"/> are meticulously pre-calculated 
    /// to ensure that the mean and variance of the activations are preserved (specifically converging 
    /// to a mean of 0 and a variance of 1) across successive layers of a deep neural network. 
    /// This self-normalizing property holds true assuming the inputs are normalized and the weights 
    /// are initialized appropriately (e.g., using LeCun normal initialization).
    /// </para>
    /// <para>
    /// For more details, refer to the seminal paper: 
    /// "Self-Normalizing Neural Networks" by Günter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter.
    /// </para>
    /// </remarks>
    /// <seealso cref="BaseActivation"/>
    /// <seealso cref="ITensor"/>

    #endregion

    public class SELU : BaseActivation
    {
        /// <summary>
        /// The predefined scale parameter (\alpha) applied to negative values to control the saturation level.
        /// </summary>
        /// <value>
        /// Approximately <c>1.6732632423543772848170429916717</c>.
        /// </value>
        private const float Alpha = 1.6732632423543772848170429916717f;

        /// <summary>
        /// The predefined overall scaling factor (\lambda) applied to the activation output to maintain variance.
        /// </summary>
        /// <value>
        /// Approximately <c>1.0507009873554804934193349852946</c>.
        /// </value>
        private const float Scale = 1.0507009873554804934193349852946f;
        /// <summary>
        /// Performs the forward pass calculation of the SELU activation function.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> to be activated.</param>
        /// <returns>An <see cref="ITensor"/> containing the activated output values.</returns>
        /// <exception cref="ArgumentNullException">Thrown when the <paramref name="input"/> tensor is <see langword="null"/>.</exception>
        /// <remarks>
        /// <para>
        /// This method calculates the forward activation of SELU. If the <paramref name="input"/> tensor 
        /// has its <see cref="ITensor.RequiresGrad"/> property set to <see langword="true"/>, a backward 
        /// gradient function (<c>GradFn</c>) is attached to the output tensor to support automatic differentiation.
        /// </para>
        /// <para>
        /// The gradient of the SELU function with respect to the input is derived as:
        /// <code>
        /// dSELU(x)/dx = scale, if x &gt; 0
        /// dSELU(x)/dx = scale * alpha * exp(x), if x &lt;= 0
        /// </code>
        /// which is reformatted in the backward pass closure using the output tensor to optimize computational efficiency.
        /// </para>
        /// </remarks>

        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            var positive = input.GreaterThan(Tensor.Zeros(input.Shape, input.Device));
            var expPart = input.Exp().Subtract(Tensor.Ones(input.Shape, input.Device)).Multiply(Alpha);
            var output = input.Multiply(positive)
                              .Add(expPart.Multiply(positive.LogicalNot()))
                              .Multiply(Scale);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var mask = positive;
                    var seluGrad = output.Divide(Scale).Add(Tensor.FromScalar(Alpha, input.Device)).Multiply(positive.LogicalNot());
                    return gradOutput.Multiply(mask.Add(seluGrad));
                };
            }

            return output;
        }
    }
}