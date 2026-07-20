// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Layers.Normalization
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Functional;
    using ArborNet.Activations;
    using ArborNet.Core.Layers;
    /// <summary>
    /// Abstract base class for all normalization layers in ArborNet.
    /// Provides thread-safe, atomic parameter accumulation and numerical safety guarantees.
    /// </summary>
    /// <remarks>
    /// This class orchestrates the forward and backward passes for normalization operations. 
    /// It manages learnable affine parameters (<see cref="Gamma"/> and <see cref="Beta"/>), 
    /// handles integration with the autograd engine by registering custom <c>GradFn</c> closures, 
    /// and delegates concrete normalization calculations to derived classes.
    /// </remarks>

    #endregion

    public abstract class BaseNormalization : BaseLayer
    {
        /// <summary>
        /// Gets the learnable scaling parameter (weight) for the affine transformation.
        /// </summary>
        /// <value>
        /// An <see cref="ITensor"/> initialized to ones, or <c>null</c> if <see cref="UseAffine"/> is <c>false</c>.
        /// </value>
        protected ITensor Gamma { get; private set; }
        /// <summary>
        /// Gets the learnable shifting parameter (bias) for the affine transformation.
        /// </summary>
        /// <value>
        /// An <see cref="ITensor"/> initialized to zeros, or <c>null</c> if <see cref="UseAffine"/> is <c>false</c>.
        /// </value>
        protected ITensor Beta { get; private set; }
        protected readonly float Eps;
        protected readonly bool UseAffine;

        protected BaseNormalization(int numFeatures, float eps = 1e-5f, bool useAffine = true)
        {
            Eps = eps;
            UseAffine = useAffine;

            if (useAffine)
            {
                Gamma = Tensor.Ones(new TensorShape(numFeatures));
                Beta = Tensor.Zeros(new TensorShape(numFeatures));
                Gamma.RequiresGrad = Beta.RequiresGrad = true;
            }
        }
        /// <summary>
        /// Performs the forward pass of the normalization layer.
        /// </summary>
        /// <param name="input">The input tensor to normalize.</param>
        /// <returns>A normalized and optionally affine-scaled <see cref="ITensor"/>.</returns>
        /// <remarks>
        /// This method validates the input tensor, performs the core normalization step, applies 
        /// the optional affine transformation (<see cref="Gamma"/> and <see cref="Beta"/>), and 
        /// registers an analytical backward gradient closure if the input requires gradient tracking.
        /// </remarks>

        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input);

            var normalized = Normalize(input);

            if (UseAffine)
            {
                normalized = normalized.Multiply(Gamma).Add(Beta);
            }

            if (input.RequiresGrad)
            {
                normalized.GradFn = gradOutput =>
                {
                    var gradInput = ComputeGradInput(input, gradOutput);

                    if (UseAffine)
                    {
                        // Thread-Safe Atomic parameter updates
                        Gamma.AccumulateGrad(gradOutput.Multiply(normalized));
                        Beta.AccumulateGrad(gradOutput);
                    }

                    return gradInput;
                };
            }

            return normalized;
        }
        /// <summary>
        /// When overridden in a derived class, normalizes the input tensor according to the specific normalization algorithm.
        /// </summary>
        /// <param name="input">The input tensor to normalize.</param>
        /// <returns>The normalized tensor before affine transformations are applied.</returns>

        protected abstract ITensor Normalize(ITensor input);
        /// <summary>
        /// When overridden in a derived class, computes the gradient of the loss with respect to the input tensor.
        /// </summary>
        /// <param name="input">The original input tensor passed to the forward pass.</param>
        /// <param name="gradOutput">The gradient of the loss with respect to the output of this layer.</param>
        /// <returns>The computed gradient tensor with respect to the input.</returns>
        protected abstract ITensor ComputeGradInput(ITensor input, ITensor gradOutput);
        /// <summary>
        /// Returns an enumerable collection of all learnable parameter tensors associated with this normalization layer.
        /// </summary>
        /// <returns>
        /// An <see cref="IEnumerable{ITensor}"/> containing <see cref="Gamma"/> and <see cref="Beta"/> if <see cref="UseAffine"/> is <c>true</c>; 
        /// otherwise, an empty sequence.
        /// </returns>

        public override IEnumerable<ITensor> Parameters()
        {
            if (UseAffine)
            {
                yield return Gamma;
                yield return Beta;
            }
        }
    }
}

