namespace ArborNet.Core.Models
{
    
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using System.Collections;
    using System.Reflection;


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

        protected bool isTraining = true;
        protected List<ITensor> parameters = new List<ITensor>();
        protected Device currentDevice = Device.CPU;

        public abstract ITensor Forward(ITensor input);

        public virtual IEnumerable<ITensor> Parameters()
        {
            return new List<ITensor>(parameters);
        }

        public virtual void Train()
        {
            isTraining = true;
        }

        public virtual void Eval()
        {
            isTraining = false;
        }

        /// <summary>
        /// Production-grade recursive device migration for entire model graphs.
        /// Locates and updates all sub-layers, tensors, lists, and dictionaries.
        /// </summary>
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