using System;
using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Devices;

namespace ArborNet.Layers
{
    /// <summary>
    /// World-class abstract base for all layers in ArborNet.
    /// Guarantees consistent lifecycle, parameter management, and device safety.
    /// Fully compliant with ILayer and supports autograd + training modes.
    /// </summary>
    /// <remarks>
    /// This abstract base class provides standardized behavior for all layers including
    /// training/evaluation mode management, device placement, input validation, and parameter
    /// exposure. All concrete layers in the ArborNet framework must inherit from this class
    /// to ensure consistent behavior with optimizers, trainers, and the autograd system.
    /// </remarks>
    public abstract class BaseLayer : ILayer
    {
        /// <summary>Current training mode state.</summary>
        protected bool isTraining = true;

        /// <summary>Device this layer operates on (defaults to CPU).</summary>
        protected Device device = Device.CPU;

        /// <summary>
        /// Performs the forward pass of the layer.
        /// Must be implemented by derived classes.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor after applying the layer.</returns>
        /// <remarks>
        /// Implementations must respect the current <see cref="IsTraining"/> state and 
        /// ensure all operations are performed on the layer's assigned <see cref="Device"/>.
        /// </remarks>
        public abstract ITensor Forward(ITensor input);

        /// <summary>
        /// Returns all learnable parameters of this layer.
        /// Used by optimizers and trainers.
        /// </summary>
        /// <returns>Enumerable of trainable tensors.</returns>
        /// <remarks>
        /// Only tensors that require gradient updates should be returned. The collection 
        /// is used by optimizers during the parameter update step.
        /// </remarks>
        public abstract IEnumerable<ITensor> Parameters();

        /// <summary>
        /// Sets the layer (and any sub-layers) to training mode.
        /// Enables dropout, batch-norm updates, etc.
        /// </summary>
        /// <remarks>
        /// Override this method in composite layers (e.g. Sequential, Residual) to 
        /// propagate training mode to all child layers.
        /// </remarks>
        public virtual void Train()
        {
            isTraining = true;
        }

        /// <summary>
        /// Sets the layer (and any sub-layers) to evaluation mode.
        /// Disables stochastic operations for inference.
        /// </summary>
        /// <remarks>
        /// Override this method in composite layers to propagate evaluation mode 
        /// to all child layers. This typically disables dropout and enables running 
        /// statistics in batch normalization.
        /// </remarks>
        public virtual void Eval()
        {
            isTraining = false;
        }

        /// <summary>
        /// Returns whether the layer is currently in training mode.
        /// </summary>
        /// <value>
        /// <c>true</c> if the layer is in training mode; otherwise, <c>false</c>.
        /// </value>
        public bool IsTraining => isTraining;

        /// <summary>
        /// Sets the device for this layer and all parameters.
        /// </summary>
        /// <param name="targetDevice">Target device (CPU/CUDA).</param>
        /// <remarks>
        /// Derived classes should override this method to move all learnable parameters 
        /// and buffers to the target device.
        /// </remarks>
        public virtual void To(Device targetDevice)
        {
            device = targetDevice ?? Device.CPU;
        }

        /// <summary>
        /// Protected helper to validate input tensor shape and device.
        /// </summary>
        /// <param name="input">Input tensor to be validated.</param>
        /// <param name="expectedRank">Expected tensor rank. Use -1 to skip rank validation.</param>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="input"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown if the tensor rank does not match the expected rank.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the input device is incompatible with the layer's device.</exception>
        protected void ValidateInput(ITensor input, int expectedRank = -1)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (expectedRank > 0 && input.Shape.Rank != expectedRank)
                throw new ArgumentException($"Expected rank {expectedRank}, got {input.Shape.Rank}");
            if (input.Device != device && !input.Device.IsCpu() && !device.IsCpu())
                throw new InvalidOperationException("Device mismatch between layer and input.");
        }
    }
}