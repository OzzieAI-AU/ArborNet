// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Holonomic
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.Numerics;
    using System.Text;
    /// <summary>
    /// Provides a collection of non-linear activation functions specifically designed for <see cref="Complex"/> numbers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These functions are intended for use in complex-valued neural networks (CVNNs) and holonomic systems 
    /// where maintaining phase, amplitude, and wave-interference properties is essential.
    /// </para>
    /// <para>
    /// Unlike real-valued activation functions, complex-valued activation functions must balance the mapping 
    /// of both the real and imaginary components, often satisfying the Cauchy-Riemann equations to preserve 
    /// holomorphicity, or intentionally violating them (such as in split-complex activations) to achieve specific 
    /// optimization properties.
    /// </para>
    /// </remarks>
    /// <seealso cref="System.Numerics.Complex"/>

    #endregion

    public static class ComplexActivations
    {
        /// <summary>
        /// Computes the complex hyperbolic tangent of the specified complex number.
        /// </summary>
        /// <param name="z">The input <see cref="Complex"/> number representing the pre-activation value (net input to the neuron).</param>
        /// <returns>
        /// A <see cref="Complex"/> number representing the hyperbolic tangent of <paramref name="z"/>.
        /// </returns>
        /// <remarks>
        /// <para>
        /// The complex hyperbolic tangent bounds both the real and imaginary parts of the input,
        /// serving as an effective non-linearity for wave-interference neural networks. 
        /// It is defined mathematically as:
        /// </para>
        /// <para>
        /// <c>tanh(z) = sinh(z) / cosh(z) = (e^z - e^-z) / (e^z + e^-z)</c>
        /// </para>
        /// <para>
        /// This function is fully holomorphic (complex-differentiable) over the entire complex plane, 
        /// except at the poles where <c>cosh(z) = 0</c> (specifically, at <c>z = i * pi * (k + 0.5)</c> for any integer <c>k</c>).
        /// </para>
        /// <para>
        /// Behavior with special values:
        /// <list type="bullet">
        /// <item>
        /// <description>If <paramref name="z"/> is <see cref="Complex.Zero"/>, the result is <see cref="Complex.Zero"/>.</description>
        /// </item>
        /// <item>
        /// <description>If either the real or imaginary component of <paramref name="z"/> is <see cref="double.NaN"/>, the resulting <see cref="Complex"/> value will contain <see cref="double.NaN"/> components.</description>
        /// </item>
        /// </list>
        /// </para>
        /// </remarks>
        /// <example>
        /// The following example demonstrates how to apply the <see cref="Tanh"/> activation to a complex number:
        /// <code>
        /// Complex input = new Complex(1.0, 0.5);
        /// Complex output = ComplexActivations.Tanh(input);
        /// Console.WriteLine($"Tanh({input}) = {output}");
        /// </code>
        /// </example>
        /// <seealso cref="Complex.Tanh(Complex)"/>
        public static Complex Tanh(Complex z)
        {
            return Complex.Tanh(z);
        }
    }
}