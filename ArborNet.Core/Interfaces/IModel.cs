using System;
using System.Collections.Generic;
using ArborNet.Core.Devices;

namespace ArborNet.Core.Interfaces
{
    /// <summary>
    /// Interface for neural network models in ArborNet.
    /// </summary>
    public interface IModel
    {
        ITensor Forward(ITensor input);
        IEnumerable<ITensor> Parameters();
        void Train();
        void Eval();

        /// <summary>
        /// Recursively migrates the model, its layers, and all underlying parameters to the target device.
        /// </summary>
        void To(Device device);
    }
}