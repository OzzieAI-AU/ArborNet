using ArborNet.Core.Interfaces;
using System.Collections.Generic;


namespace ArborNet.Core.Models
{
    /// <summary>
    /// Base class for neural network models in ArborNet.
    /// Implements IModel and provides common functionality for managing parameters and training/evaluation modes.
    /// Subclasses should implement the Forward method and manage their specific layers and parameters.
    /// </summary>
    /// <remarks>
    /// This abstract base class implements the <see cref="IModel"/> interface and provides 
    /// foundational functionality for all neural network models in the ArborNet framework.
    /// It manages the training/evaluation state and a centralized parameter collection.
    /// All concrete models should inherit from this class and implement the <see cref="Forward(ITensor)"/> method.
    /// </remarks>
    public abstract class BaseModel : IModel
    {
        /// <summary>
        /// Indicates whether the model is in training mode.
        /// </summary>
        /// <remarks>
        /// This flag controls the behavior of layers that have different execution modes
        /// (e.g. Dropout, BatchNorm). Protected to allow derived classes to read the current state.
        /// </remarks>
        protected bool isTraining = true;

        /// <summary>
        /// Collection of model parameters (e.g., weights and biases).
        /// Subclasses should populate this with their parameters.
        /// </summary>
        /// <remarks>
        /// This list is used by optimizers and training infrastructure to access all learnable tensors.
        /// Subclasses are responsible for adding their parameter tensors during initialization.
        /// </remarks>
        protected List<ITensor> parameters = new List<ITensor>();

        /// <summary>
        /// Performs the forward pass of the model on the given input tensor.
        /// Must be implemented by subclasses.
        /// </summary>
        /// <param name="input">The input tensor to the model.</param>
        /// <returns>The output tensor after applying the model's transformations.</returns>
        /// <remarks>
        /// This is the core method that defines the model's computation graph.
        /// All layers, operations, and data flow should be orchestrated within the implementation.
        /// </remarks>
        public abstract ITensor Forward(ITensor input);

        /// <summary>
        /// Gets the parameters of the model (e.g., weights and biases).
        /// Returns a copy of the internal parameters list.
        /// </summary>
        /// <returns>An enumerable collection of the model's parameters.</returns>
        /// <remarks>
        /// Returns a shallow copy of the internal list to prevent external code from modifying 
        /// the model's parameter collection directly. This method is virtual to allow 
        /// subclasses to augment the returned parameters if needed.
        /// </remarks>
        public virtual IEnumerable<ITensor> Parameters()
        {
            return new List<ITensor>(parameters);
        }

        /// <summary>
        /// Sets the model to training mode, enabling operations like dropout and batch normalization updates.
        /// </summary>
        /// <remarks>
        /// This method should be called before training loops to ensure layers that behave 
        /// differently during training (Dropout, BatchNorm, etc.) operate in training mode.
        /// </remarks>
        public virtual void Train()
        {
            isTraining = true;
        }

        /// <summary>
        /// Sets the model to evaluation mode, disabling operations like dropout for inference.
        /// </summary>
        /// <remarks>
        /// This method should be called before inference or validation to ensure deterministic 
        /// behavior and disable training-specific features such as dropout.
        /// </remarks>
        public virtual void Eval()
        {
            isTraining = false;
        }
    }
}