using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace ArborNet.Core.Holonomic
{

    /// <summary>
    /// Provides non-linear activation functions for Complex numbers.
    /// </summary>
    public static class ComplexActivations
    {
        /// <summary>
        /// Complex Hyperbolic Tangent. Bounds both the real and imaginary parts, 
        /// serving as an effective non-linearity for wave-interference neural networks.
        /// </summary>
        public static Complex Tanh(Complex z)
        {
            return Complex.Tanh(z);
        }
    }
}