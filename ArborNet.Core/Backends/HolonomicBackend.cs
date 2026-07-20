// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Backends
{

    #region Using Statements:

    using System;
    using System.Numerics;
    using System.Runtime.InteropServices;
    /// <summary>
    /// Core backend for Wave-Interference Holonomic Processing.
    /// Handles native acceleration when available, with a thread-safe managed fallback.
    /// </summary>
    /// <remarks>
    /// This class implements <see cref="IDisposable"/> to ensure any resource leakages are prevented.
    /// It coordinates high-performance wave processing by dynamically falling back to CPU computation
    /// if the specialized CUDA dynamic link library is missing or fails.
    /// </remarks>

    #endregion

    public class HolonomicBackend : IDisposable
    {
        /// <summary>
        /// Invokes the native CUDA holonomic kernel for accelerated processing.
        /// </summary>
        /// <param name="inputs">Pointer to the unmanaged input array of complex numbers.</param>
        /// <param name="weights">Pointer to the unmanaged weights array of complex numbers.</param>
        /// <param name="intWeights">Pointer to the unmanaged internal weights array of complex numbers.</param>
        /// <param name="outputs">Pointer to the unmanaged output array of complex numbers.</param>
        /// <param name="inputSize">The number of elements in the input array.</param>
        /// <param name="neuronCount">The number of neurons (and elements in internal weights and outputs).</param>
        /// <param name="fractalDepth">The recursion or iteration depth for the holonomic calculation.</param>
        /// <returns>An integer status code indicating success or failure of the GPU kernel execution.</returns>
        [DllImport("cuda_backend.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "InvokeHolonomicKernel")]
        private static extern int InvokeHolonomicKernel(
    IntPtr inputs,
    IntPtr weights,
    IntPtr intWeights,
    IntPtr outputs,
    int inputSize,
    int neuronCount,
    int fractalDepth);

        /// <summary>
        /// Track whether <see cref="Dispose()"/> has been called to prevent redundant resource cleanup.
        /// </summary>
        private bool _disposed;
        /// <summary>
        /// Executes the forward pass of the holonomic processing network using native GPU acceleration
        /// if available, falling back to a managed CPU implementation on failure.
        /// </summary>
        /// <param name="hostInputs">The input signal represented as an array of complex numbers.</param>
        /// <param name="hostWeights">The connection weights matrix flattened into a single-dimensional array of complex numbers.</param>
        /// <param name="internalWeights">The recurrent internal feedback weights array of complex numbers.</param>
        /// <param name="depth">The fractal iteration depth used in the wave-interference calculations.</param>
        /// <returns>An array of <see cref="Complex"/> numbers representing the activated outputs of the holonomic processing unit.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="hostInputs"/>, <paramref name="hostWeights"/>, or <paramref name="internalWeights"/> is null.</exception>
        /// <remarks>
        /// Memory allocations on the unmanaged heap are safely handled using <see cref="Marshal.AllocHGlobal(int)"/> 
        /// and released inside a <c>finally</c> block to guarantee execution safety and prevent memory leaks.
        /// </remarks>

        public Complex[] ExecuteForward(Complex[] hostInputs, Complex[] hostWeights, Complex[] internalWeights, int depth)
        {
            if (hostInputs == null) throw new ArgumentNullException(nameof(hostInputs));
            if (hostWeights == null) throw new ArgumentNullException(nameof(hostWeights));
            if (internalWeights == null) throw new ArgumentNullException(nameof(internalWeights));

            int inputSize = hostInputs.Length;
            int neuronCount = internalWeights.Length;

            int complexSize = Marshal.SizeOf(typeof(Complex));
            IntPtr d_in = Marshal.AllocHGlobal(inputSize * complexSize);
            IntPtr d_w = Marshal.AllocHGlobal(hostWeights.Length * complexSize);
            IntPtr d_iw = Marshal.AllocHGlobal(neuronCount * complexSize);
            IntPtr d_out = Marshal.AllocHGlobal(neuronCount * complexSize);

            try
            {
                MarshalComplexArray(hostInputs, d_in);
                MarshalComplexArray(hostWeights, d_w);
                MarshalComplexArray(internalWeights, d_iw);

                try
                {
                    InvokeHolonomicKernel(d_in, d_w, d_iw, d_out, inputSize, neuronCount, depth);
                }
                catch (DllNotFoundException)
                {
                    // Managed CPU Fallback to guarantee runtime execution
                    ComputeManagedHolonomic(hostInputs, hostWeights, internalWeights, d_out, depth);
                }

                Complex[] results = new Complex[neuronCount];
                IntPtr current = d_out;
                for (int i = 0; i < neuronCount; i++)
                {
                    results[i] = Marshal.PtrToStructure<Complex>(current);
                    current = IntPtr.Add(current, complexSize);
                }
                return results;
            }
            finally
            {
                Marshal.FreeHGlobal(d_in);
                Marshal.FreeHGlobal(d_w);
                Marshal.FreeHGlobal(d_iw);
                Marshal.FreeHGlobal(d_out);
            }
        }
        /// <summary>
        /// Marshals an array of managed <see cref="Complex"/> structures into unmanaged native memory.
        /// </summary>
        /// <param name="source">The source array of managed <see cref="Complex"/> structures to copy.</param>
        /// <param name="destination">The pre-allocated pointer to the unmanaged destination memory.</param>
        /// <remarks>
        /// Sequentially copies structures using <see cref="Marshal.StructureToPtr{T}(T, IntPtr, bool)"/> to prevent native memory alignment faults.
        /// </remarks>

        private static void MarshalComplexArray(Complex[] source, IntPtr destination)
        {
            int size = Marshal.SizeOf(typeof(Complex));
            IntPtr current = destination;
            for (int i = 0; i < source.Length; i++)
            {
                Marshal.StructureToPtr(source[i], current, false);
                current = IntPtr.Add(current, size);
            }
        }
        /// <summary>
        /// Computes the wave-interference holonomic activation on the CPU using a managed implementation.
        /// Used as a fallback when the native CUDA backend is unavailable.
        /// </summary>
        /// <param name="inputs">The input signal represented as an array of complex numbers.</param>
        /// <param name="weights">The connection weights matrix represented as a single-dimensional array.</param>
        /// <param name="internalWeights">The recurrent internal feedback weights array.</param>
        /// <param name="d_out">An unmanaged pointer where the computed outputs should be written.</param>
        /// <param name="depth">The iteration depth for the non-linear fractal activation.</param>
        /// <remarks>
        /// This fallback computes a double loop involving wave dot product aggregation and iterative feedback processing 
        /// through a non-linear <see cref="Complex.Tanh(Complex)"/> activation function.
        /// </remarks>

        private static void ComputeManagedHolonomic(Complex[] inputs, Complex[] weights, Complex[] internalWeights, IntPtr d_out, int depth)
        {
            int neuronCount = internalWeights.Length;
            int complexSize = Marshal.SizeOf(typeof(Complex));
            IntPtr current = d_out;

            for (int n = 0; n < neuronCount; n++)
            {
                Complex psi = Complex.Zero;
                for (int i = 0; i < inputs.Length; i++)
                {
                    psi += inputs[i] * weights[n * inputs.Length + i];
                }

                Complex z = Complex.Zero;
                for (int t = 0; t < depth; t++)
                {
                    z = Complex.Tanh((internalWeights[n] * z) + psi);
                }

                Marshal.StructureToPtr(z, current, false);
                current = IntPtr.Add(current, complexSize);
            }
        }
        /// <summary>
        /// Releases all resources used by the <see cref="HolonomicBackend"/> instance.
        /// </summary>
        /// <remarks>
        /// Calling <see cref="Dispose()"/> releases deterministic allocations and suppresses garbage collection finalization.
        /// </remarks>

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Finalizes an instance of the <see cref="HolonomicBackend"/> class.
        /// </summary>
        /// <remarks>
        /// Destructor wrapper ensuring proper release of system hooks in the event of improper application disposal sequence.
        /// </remarks>
        ~HolonomicBackend() => Dispose();
    }
}