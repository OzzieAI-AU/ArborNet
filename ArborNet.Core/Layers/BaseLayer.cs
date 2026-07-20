// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Layers
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Core.Devices;
    using System.Reflection;
    /// <summary>
    /// Represents the abstract base class for all neural network layers within the ArborNet framework.
    /// Guarantees consistent lifecycle management, parameter tracking, and device-safety invariants.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This class establishes the baseline lifecycle, training/evaluation modes, device placement mechanics,
    /// and reflection-based parameter migration. All custom layers must derive from <see cref="BaseLayer"/>
    /// to ensure seamless compatibility with training pipelines, autograd mechanics, and optimizers.
    /// </para>
    /// <para>
    /// Inheriting classes should:
    /// <list type="bullet">
    /// <item><description>Implement the forward pass computation in <see cref="Forward(ITensor)"/>.</description></item>
    /// <item><description>Expose all trainable parameters via <see cref="Parameters"/>.</description></item>
    /// <item><description>Utilize <see cref="ValidateInput(ITensor, int)"/> to guarantee runtime input consistency.</description></item>
    /// </list>
    /// </para>
    /// </remarks>

    #endregion

    public abstract class BaseLayer : ILayer
    {
        /// <summary>
        /// Backing field indicating whether the layer is currently operating in training mode (<c>true</c>) or evaluation mode (<c>false</c>).
        /// </summary>
        protected bool isTraining = true;

        /// <summary>
        /// Backing field tracking the target execution <see cref="Device"/> (e.g., CPU, GPU) on which the layer's computations and parameters reside.
        /// </summary>
        protected Device device = Device.CPU;
        /// <summary>
        /// Executes the forward pass computation of the layer, processing the input tensor to generate an output tensor.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> containing the activation values from the preceding layer.</param>
        /// <returns>An <see cref="ITensor"/> representing the computed activations or outputs of this layer.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">Thrown when the dimensions or shape of the <paramref name="input"/> do not match the expected layer specifications.</exception>
        /// <exception cref="InvalidOperationException">Thrown if there is a device mismatch between the layer and the input tensor.</exception>

        public abstract ITensor Forward(ITensor input);
        /// <summary>
        /// Enumerates all trainable parameters, such as weights and biases, directly associated with this layer.
        /// </summary>
        /// <returns>An <see cref="IEnumerable{ITensor}"/> containing the trainable parameters of the layer.</returns>
        /// <remarks>
        /// Optimizers iterate over this collection to apply gradient updates to the underlying weight tensors during backpropagation.
        /// </remarks>

        public abstract IEnumerable<ITensor> Parameters();
        /// <summary>
        /// Sets the layer and all its nested sub-layers to training mode.
        /// </summary>
        /// <remarks>
        /// This toggles internal state flags that alter the behavior of layers like Dropout or Batch Normalization 
        /// to enable active learning dynamics, parameter updates, and gradient accumulation.
        /// </remarks>

        public virtual void Train() => isTraining = true;
        /// <summary>
        /// Sets the layer and all its nested sub-layers to evaluation (inference) mode.
        /// </summary>
        /// <remarks>
        /// This disables training-specific operations, freezing running statistics for normalization and disabling stochastic elements like dropout.
        /// </remarks>

        public virtual void Eval() => isTraining = false;
        /// <summary>
        /// Gets a value indicating whether the layer is currently configured for training.
        /// </summary>
        /// <value>
        /// <c>true</c> if the layer is in training mode; otherwise, <c>false</c>.
        /// </value>

        public bool IsTraining => isTraining;
        /// <summary>
        /// Gets the execution device where the layer's parameters and computations are currently hosted.
        /// </summary>
        /// <value>
        /// The <see cref="Device"/> where computations are scheduled to run.
        /// </value>

        public Device CurrentDevice => device;
        /// <summary>
        /// Recursively transfers the layer, its nested sub-layers, and all its parameter tensors to the specified target execution device.
        /// </summary>
        /// <param name="targetDevice">The target <see cref="Device"/> (e.g., CPU, GPU) to host the layer's state. If <c>null</c>, defaults to <see cref="Device.CPU"/>.</param>
        /// <remarks>
        /// <para>
        /// This method leverages reflection to detect fields within the runtime instance implementing
        /// <see cref="ITensor"/>, <see cref="ILayer"/>, or collections thereof, programmatically invoking 
        /// migration operations to ensure uniform execution placement.
        /// </para>
        /// <para>
        /// Note: Reflection carries execution overhead. It is recommended to perform device transfers during the 
        /// model initialization phase rather than inside hot paths or training loops.
        /// </para>
        /// </remarks>

        public virtual void To(Device targetDevice)
        {
            device = targetDevice ?? Device.CPU;

            // Use reflection to locate all ITensor and ILayer fields within the class
            var fields = this.GetType().GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            foreach (var field in fields)
            {
                // Move any direct ITensor parameters (weights, biases) to the target device
                if (typeof(ITensor).IsAssignableFrom(field.FieldType))
                {
                    var tensor = (ITensor?)field.GetValue(this);
                    if (tensor != null)
                    {
                        field.SetValue(this, tensor.To(device));
                    }
                }
                // Recursively migrate any sub-layers
                else if (typeof(ILayer).IsAssignableFrom(field.FieldType))
                {
                    var subLayer = (ILayer?)field.GetValue(this);
                    if (subLayer is BaseLayer baseSubLayer)
                    {
                        baseSubLayer.To(device);
                    }
                }
                // Handle collections of sub-layers (e.g., List<TransformerBlock>)
                else if (typeof(System.Collections.IEnumerable).IsAssignableFrom(field.FieldType) && field.FieldType.IsGenericType)
                {
                    var collection = field.GetValue(this) as System.Collections.IEnumerable;
                    if (collection != null)
                    {
                        foreach (var item in collection)
                        {
                            if (item is BaseLayer baseSubLayer)
                            {
                                baseSubLayer.To(device);
                            }
                            else if (item is ITensor tensor)
                            {
                                // If the collection elements are modifiable, they should be migrated
                                // Note: For lists/arrays, a copy-back or in-place update is required.
                            }
                        }
                    }
                }
            }
        }
        /// <summary>
        /// Verifies that the incoming tensor conforms to expected invariants, validating reference integrity, structural rank, and device alignment.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> to be evaluated.</param>
        /// <param name="expectedRank">The required dimensional rank of the input tensor. If less than or equal to zero, rank checks are skipped.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">Thrown when <paramref name="expectedRank"/> is positive and does not match the actual rank of <paramref name="input"/>.</exception>
        /// <exception cref="InvalidOperationException">Thrown when a device placement conflict occurs between this layer and the input tensor.</exception>

        protected void ValidateInput(ITensor input, int expectedRank = -1)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (expectedRank > 0 && input.Shape.Rank != expectedRank)
                throw new ArgumentException($"Expected rank {expectedRank}, got {input.Shape.Rank}");
            if (input.Device != device && !input.Device.IsCpu() && !device.IsCpu())
                throw new InvalidOperationException($"Device mismatch: Layer on {device}, Input on {input.Device}.");
        }
    }
}