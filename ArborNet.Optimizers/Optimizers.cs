// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Optimizers
{

    #region Using Statements:

    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using System.Collections.Generic;
    /// <summary>
    /// Provides static utility methods for neural network optimizers within the ArborNet framework.
    /// </summary>
    /// <remarks>
    /// This class serves as a centralized suite of helper operations commonly used during the 
    /// optimization and backpropagation phases of training deep learning models.
    /// </remarks>

    #endregion

    public static class Optimizers
    {
        /// <summary>
        /// Resets the gradients of all target parameters to a zero-filled tensor matching their shape and device.
        /// </summary>
        /// <param name="parameters">An enumerable collection of <see cref="ITensor"/> parameters whose gradients should be cleared.</param>
        /// <remarks>
        /// This method iterates through the collection of parameters. For each non-null parameter where 
        /// <see cref="ITensor.RequiresGrad"/> is <see langword="true"/>, the gradient (<see cref="ITensor.Grad"/>) 
        /// is overwritten with a new zero tensor allocated on the same execution device and with the identical shape.
        /// </remarks>
        /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="parameters"/> is <see langword="null"/>.</exception>
        public static void ZeroGrad(IEnumerable<ITensor> parameters)
        {
            foreach (var param in parameters)
            {
                if (param != null && param.RequiresGrad)
                {
                    param.Grad = Tensor.Zeros(param.Shape, param.Device);
                }
            }
        }
    }
}
