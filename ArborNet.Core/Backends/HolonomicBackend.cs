// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// Project: ArborNet
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Backends
{
    using System;
    using System.Numerics;
    using System.Runtime.InteropServices;
    using ArborNet.Core.Native.PInvoke;

    public sealed class HolonomicBackend : IDisposable
    {
        private bool _disposed;

        public Complex[] ExecuteForward(Complex[] hostInputs, Complex[] hostWeights, Complex[] internalWeights, int depth)
        {
            if (hostInputs == null) throw new ArgumentNullException(nameof(hostInputs));
            if (hostWeights == null) throw new ArgumentNullException(nameof(hostWeights));
            if (internalWeights == null) throw new ArgumentNullException(nameof(internalWeights));
            if (depth < 0) throw new ArgumentOutOfRangeException(nameof(depth));

            int inputSize = hostInputs.Length;
            int neuronCount = internalWeights.Length;
            if (neuronCount == 0) return Array.Empty<Complex>();
            if (hostWeights.Length != neuronCount * inputSize)
                throw new ArgumentException("hostWeights length must equal neuronCount * inputSize.");

            if (!CUDA.IsAvailable())
                return ComputeManagedHolonomic(hostInputs, hostWeights, internalWeights, depth);

            int complexSize = Marshal.SizeOf<Complex>();
            ulong inBytes = (ulong)inputSize * (ulong)complexSize;
            ulong wBytes = (ulong)hostWeights.Length * (ulong)complexSize;
            ulong iwBytes = (ulong)neuronCount * (ulong)complexSize;

            IntPtr dIn = IntPtr.Zero, dW = IntPtr.Zero, dIw = IntPtr.Zero, dOut = IntPtr.Zero;
            GCHandle hIn = default, hW = default, hIw = default, hOut = default;

            try
            {
                CUDA.CudaMalloc(out dIn, inBytes);
                CUDA.CudaMalloc(out dW, wBytes);
                CUDA.CudaMalloc(out dIw, iwBytes);
                CUDA.CudaMalloc(out dOut, iwBytes);

                hIn = GCHandle.Alloc(hostInputs, GCHandleType.Pinned);
                hW = GCHandle.Alloc(hostWeights, GCHandleType.Pinned);
                hIw = GCHandle.Alloc(internalWeights, GCHandleType.Pinned);

                CUDA.CudaMemcpy(dIn, hIn.AddrOfPinnedObject(), inBytes, CUDA.cudaMemcpyKind.cudaMemcpyHostToDevice);
                CUDA.CudaMemcpy(dW, hW.AddrOfPinnedObject(), wBytes, CUDA.cudaMemcpyKind.cudaMemcpyHostToDevice);
                CUDA.CudaMemcpy(dIw, hIw.AddrOfPinnedObject(), iwBytes, CUDA.cudaMemcpyKind.cudaMemcpyHostToDevice);

                CUDA.InvokeHolonomicKernel(dIn, dW, dIw, dOut, inputSize, neuronCount, depth);
                CUDA.Synchronize();

                Complex[] results = new Complex[neuronCount];
                hOut = GCHandle.Alloc(results, GCHandleType.Pinned);
                CUDA.CudaMemcpy(hOut.AddrOfPinnedObject(), dOut, iwBytes, CUDA.cudaMemcpyKind.cudaMemcpyDeviceToHost);
                return results;
            }
            catch (DllNotFoundException)
            {
                return ComputeManagedHolonomic(hostInputs, hostWeights, internalWeights, depth);
            }
            catch (EntryPointNotFoundException)
            {
                return ComputeManagedHolonomic(hostInputs, hostWeights, internalWeights, depth);
            }
            finally
            {
                if (hIn.IsAllocated) hIn.Free();
                if (hW.IsAllocated) hW.Free();
                if (hIw.IsAllocated) hIw.Free();
                if (hOut.IsAllocated) hOut.Free();

                // Free whatever succeeded. cudaMalloc on failure leaves a null pointer.
                if (dOut != IntPtr.Zero) { try { CUDA.CudaFree(dOut); } catch { /* context dying */ } }
                if (dIw != IntPtr.Zero) { try { CUDA.CudaFree(dIw); } catch { } }
                if (dW != IntPtr.Zero) { try { CUDA.CudaFree(dW); } catch { } }
                if (dIn != IntPtr.Zero) { try { CUDA.CudaFree(dIn); } catch { } }
            }
        }

        private static Complex[] ComputeManagedHolonomic(Complex[] inputs, Complex[] weights, Complex[] internalWeights, int depth)
        {
            int neuronCount = internalWeights.Length;
            var results = new Complex[neuronCount];
            for (int n = 0; n < neuronCount; n++)
            {
                Complex psi = Complex.Zero;
                int row = n * inputs.Length;
                for (int i = 0; i < inputs.Length; i++)
                    psi += inputs[i] * weights[row + i];

                Complex z = Complex.Zero;
                for (int t = 0; t < depth; t++)
                    z = Complex.Tanh((internalWeights[n] * z) + psi);
                results[n] = z;
            }
            return results;
        }

        public void Dispose()
        {
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
}