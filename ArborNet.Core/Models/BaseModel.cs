// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Models
{

    #region Using Statements:

    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using System.Collections;
    using System.Reflection;
    /// <summary>
    /// Serves as the abstract base class for all neural network models within the ArborNet framework.
    /// Implements the <see cref="IModel"/> interface and provides foundational capabilities for 
    /// tracking parameters, managing execution modes (training vs. evaluation), and migrating 
    /// parameters across execution devices (CPU/GPU).
    /// </summary>
    /// <remarks>
    /// This class simplifies the creation of custom neural networks by providing automated 
    /// mechanisms for parameter collection and execution device state management. Concrete subclasses 
    /// must define the execution graph by implementing the <see cref="Forward(ITensor)"/> method.
    /// </remarks>

    #endregion


    public abstract class BaseModel : IModel
    {
        /// <summary>
        /// Indicates whether the model is currently in training mode (true) or evaluation mode (false).
        /// </summary>
        /// <value>
        /// <c>true</c> if the model is operating in training mode; otherwise, <c>false</c> for evaluation and inference.
        /// </value>
        protected bool isTraining = true;

        /// <summary>
        /// A list storing the parameters (tensors) of the model.
        /// </summary>
        /// <remarks>
        /// This acts as a centralized cache of all trainable parameters belonging to this model 
        /// and its nested layers or sub-models, typically populated during initialization or device migration.
        /// </remarks>
        protected List<ITensor> parameters = new List<ITensor>();

        /// <summary>
        /// Represents the current execution device (e.g., CPU, GPU) on which the model's parameters and operations reside.
        /// </summary>
        /// <value>
        /// The current execution environment target, defaulted to <see cref="Device.CPU"/>.
        /// </value>
        protected Device currentDevice = Device.CPU;
        /// <summary>
        /// Performs the forward propagation pass of the neural network model.
        /// </summary>
        /// <param name="input">The input <see cref="ITensor"/> containing the features to be processed.</param>
        /// <returns>An <see cref="ITensor"/> representing the output predictions or transformations of the network.</returns>
        /// <remarks>
        /// This method must be implemented by concrete classes to specify the sequence of layer operations and mathematical 
        /// computations that define the model's architecture.
        /// </remarks>

        public abstract ITensor Forward(ITensor input);
        /// <summary>
        /// Retrieves an enumerable collection of all trainable tensors (parameters) managed by this model.
        /// </summary>
        /// <returns>An <see cref="IEnumerable{ITensor}"/> containing the current model parameters.</returns>
        /// <remarks>
        /// This method returns a shallow copy of the underlying parameter list to prevent direct external 
        /// modification of the model's internal structural cache.
        /// </remarks>

        public virtual IEnumerable<ITensor> Parameters()
        {
            return new List<ITensor>(parameters);
        }
        /// <summary>
        /// Transitions the model into training mode.
        /// </summary>
        /// <remarks>
        /// Setting the model to training mode updates the internal <see cref="isTraining"/> state to <c>true</c>. 
        /// This activates training-specific behaviors in layers such as dropout layers and batch normalization.
        /// </remarks>

        public virtual void Train()
        {
            isTraining = true;
        }
        /// <summary>
        /// Transitions the model into evaluation (inference) mode.
        /// </summary>
        /// <remarks>
        /// Setting the model to evaluation mode updates the internal <see cref="isTraining"/> state to <c>false</c>.
        /// This disables training-only operations like dropout and switches layers like batch normalization to 
        /// use their running/historical statistics instead of batch-specific statistics.
        /// </remarks>

        public virtual void Eval()
        {
            isTraining = false;
        }
        /// <summary>
        /// Migrates the model, its constituent layers, sub-models, and parameters to the specified execution device.
        /// </summary>
        /// <param name="device">The target <see cref="Device"/> (e.g., CPU or CUDA GPU) to migrate the model parameters to.</param>
        /// <exception cref="ArgumentNullException">Thrown when the provided <paramref name="device"/> is <c>null</c>.</exception>
        /// <remarks>
        /// This method utilizes reflection to recursively scan all instance fields of the model. It automatically 
        /// detects <see cref="BaseLayer"/> structures, <see cref="IModel"/> instances, individual <see cref="ITensor"/> objects, 
        /// and collections containing these types, migrating each to the new execution target. After resolving 
        /// device transitions, it rebuilds the internal parameter cache.
        /// </remarks>

        public virtual void To(Device device)
        {
            if (device == null) throw new ArgumentNullException(nameof(device));
            currentDevice = device;

            var flags = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
            var fields = this.GetType().GetFields(flags);

            foreach (var field in fields)
            {
                var value = field.GetValue(this);
                if (value == null) continue;

                // 1. Move direct Layer fields
                if (value is BaseLayer baseLayer)
                {
                    baseLayer.To(device);
                }
                else if (value is IModel subModel)
                {
                    subModel.To(device);
                }
                // 2. Move direct Tensor fields
                else if (value is ITensor tensor)
                {
                    field.SetValue(this, tensor.To(device));
                }
                // 3. Move Collections of Tensors or Layers
                else if (value is IList list)
                {
                    for (int i = 0; i < list.Count; i++)
                    {
                        var element = list[i];
                        if (element is BaseLayer bl)
                        {
                            bl.To(device);
                        }
                        else if (element is IModel sm)
                        {
                            sm.To(device);
                        }
                        else if (element is ITensor t)
                        {
                            list[i] = t.To(device);
                        }
                    }
                }
            }

            // Re-populate the flat parameter cache with migrated parameters
            parameters.Clear();
            CollectParameters(this);
        }
        /// <summary>
        /// Recursively scans and collects trainable tensors from the specified object, appending them to the local parameter cache.
        /// </summary>
        /// <param name="obj">The object to scan for parameters, layers, sub-models, or collections of parameters.</param>
        /// <exception cref="System.NullReferenceException">Thrown if the provided <paramref name="obj"/> is null.</exception>
        /// <remarks>
        /// This helper method walks the fields of the provided object via reflection. Any discovered parameters 
        /// that have <see cref="ITensor.RequiresGrad"/> set to <c>true</c> are cached. It avoids adding duplicate 
        /// parameter references to the flat parameter store.
        /// </remarks>

        private void CollectParameters(object obj)
        {
            var flags = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
            var fields = obj.GetType().GetFields(flags);

            foreach (var field in fields)
            {
                var value = field.GetValue(obj);
                if (value == null) continue;

                if (value is ILayer layer)
                {
                    parameters.AddRange(layer.Parameters());
                }
                else if (value is IModel model && model != this)
                {
                    parameters.AddRange(model.Parameters());
                }
                else if (value is ITensor tensor && !parameters.Contains(tensor))
                {
                    if (tensor.RequiresGrad)
                        parameters.Add(tensor);
                }
                else if (value is IList list)
                {
                    foreach (var item in list)
                    {
                        if (item is ILayer l) parameters.AddRange(l.Parameters());
                        else if (item is IModel m) parameters.AddRange(m.Parameters());
                        else if (item is ITensor t && !parameters.Contains(t))
                        {
                            if (t.RequiresGrad) parameters.Add(t);
                        }
                    }
                }
            }
        }
    }
}