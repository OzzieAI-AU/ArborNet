using System;
using System.Numerics;
using System.Runtime.InteropServices;

namespace ArborNet.Core.Backends
{
    /// <summary>
    /// Core backend for Wave-Interference Holonomic Processing.
    /// Handles native acceleration when available, with a thread-safe managed fallback.
    /// </summary>
    public class HolonomicBackend : IDisposable
    {
        [DllImport("cuda_backend.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "InvokeHolonomicKernel")]
        private static extern int InvokeHolonomicKernel(
            IntPtr inputs,
            IntPtr weights,
            IntPtr intWeights,
            IntPtr outputs,
            int inputSize,
            int neuronCount,
            int fractalDepth);

        private bool _disposed;

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

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~HolonomicBackend() => Dispose();
    }
}