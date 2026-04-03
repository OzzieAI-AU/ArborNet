using ArborNet.Core.Interfaces;
using System.Collections.Generic;

namespace ArborNet.Optimizers
{
    /// <summary>
    /// Provides static utility methods for neural network optimizers within the ArborNet framework.
    /// </summary>
    public static class Optimizers
    {
        /// <summary>
        /// Zeros the gradients associated with the specified tensor parameters.
        /// </summary>
        /// <param name="parameters">An enumerable collection of <see cref="ITensor"/> parameters whose gradients should be zeroed.</param>
        /// <remarks>
        /// This method iterates over the provided parameters and sets their <c>Grad</c> property to <c>null</c> if it is not already null.
        /// This serves as a placeholder implementation. In a production environment, it should create and assign
        /// a zero-filled tensor matching the original gradient's shape and device.
        /// Refer to the inline code comments within the method for additional implementation details.
        /// </remarks>
        public static void ZeroGrad(IEnumerable<ITensor> parameters)
        {
            foreach (var param in parameters)
            {
                if (param.Grad != null)
                {
                    // To zero the gradient, create a zero tensor of the same shape and device.
                    // Assuming the tensor implementation has a way to create zeros, but since it's interface,
                    // we use a stub: set to null for now (in real implementation, replace with proper zero tensor creation).
                    param.Grad = null;
                }
            }
        }
    }
}