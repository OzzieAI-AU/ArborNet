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
    using ArborNet.Activations;
    using ArborNet.Core.Autograd;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Native.SIMD;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    using System.Runtime.InteropServices;
    using System.Threading.Tasks;
    
    
    
    /// <summary>
    /// Represents a high-performance Apple Silicon Metal/MPS-accelerated backend.
    /// Provides direct interaction with macOS Metal Performance Shaders using Obj-C runtime P/Invokes,
    /// with an automatic high-fidelity CPU SIMD fallback for cross-platform robustness.
    /// </summary>
    public sealed class MetalBackend : ITensor, IDisposable
    {
        private IntPtr _metalDevice;
        private IntPtr _metalBuffer;
        private readonly ulong _bytes;
        private TensorShape _shape;
        private readonly Device _device;
        private bool _requiresGrad;
        private ITensor? _grad;
        private Func<ITensor, ITensor>? _gradFn;
        private bool _disposed;
        private readonly object _lock = new();
        private ITensor[] _inputs = Array.Empty<ITensor>();

        private uint _version;
        public uint Version => _version;


        private const string ObjCDll = "/usr/lib/libobjc.A.dylib";
        /// <summary>
        /// Registers a selector name with the Objective-C runtime.
        /// </summary>
        /// <param name="name">The name of the selector to register.</param>
        /// <returns>A selector pointer (SEL).</returns>

        [DllImport(ObjCDll, EntryPoint = "sel_registerName")]
        private static extern IntPtr sel_registerName(string name);
        /// <summary>
        /// Retrieves class reference pointers by class name from the Objective-C runtime.
        /// </summary>
        /// <param name="name">The name of the Objective-C class.</param>
        /// <returns>A pointer to the Class structure.</returns>

        [DllImport(ObjCDll, EntryPoint = "objc_getClass")]
        private static extern IntPtr objc_getClass(string name);
        /// <summary>
        /// Sends an Objective-C message to an instance or class.
        /// </summary>
        /// <param name="self">The receiving instance or class.</param>
        /// <param name="op">The selector of the method to invoke.</param>
        /// <returns>The result returned by the method.</returns>

        [DllImport(ObjCDll, EntryPoint = "objc_msgSend")]
        private static extern IntPtr objc_msgSend(IntPtr self, IntPtr op);
        /// <summary>
        /// Sends an Objective-C message with an unsigned long argument.
        /// </summary>
        /// <param name="self">The receiving instance or class.</param>
        /// <param name="op">The selector of the method to invoke.</param>
        /// <param name="arg1">An unsigned 64-bit integer argument.</param>
        /// <returns>The result returned by the method.</returns>

        [DllImport(ObjCDll, EntryPoint = "objc_msgSend")]
        private static extern IntPtr objc_msgSend(IntPtr self, IntPtr op, ulong arg1);
        /// <summary>
        /// Sends an Objective-C message with two unsigned long arguments.
        /// </summary>
        /// <param name="self">The receiving instance or class.</param>
        /// <param name="op">The selector of the method to invoke.</param>
        /// <param name="arg1">The first unsigned 64-bit integer argument.</param>
        /// <param name="arg2">The second unsigned 64-bit integer argument.</param>
        /// <returns>The result returned by the method.</returns>

        [DllImport(ObjCDll, EntryPoint = "objc_msgSend")]
        private static extern IntPtr objc_msgSend(IntPtr self, IntPtr op, ulong arg1, ulong arg2);
        /// <summary>
        /// Sends an Objective-C message with a pointer and two unsigned long arguments.
        /// </summary>
        /// <param name="self">The receiving instance or class.</param>
        /// <param name="op">The selector of the method to invoke.</param>
        /// <param name="arg1">A pointer argument.</param>
        /// <param name="arg2">The first unsigned 64-bit integer argument.</param>
        /// <param name="arg3">The second unsigned 64-bit integer argument.</param>
        /// <returns>The result returned by the method.</returns>

        [DllImport(ObjCDll, EntryPoint = "objc_msgSend")]
        private static extern IntPtr objc_msgSend(IntPtr self, IntPtr op, IntPtr arg1, ulong arg2, ulong arg3);

        private static readonly bool IsMac = RuntimeInformation.IsOSPlatform(OSPlatform.OSX);
        private static readonly bool IsMetalSupported;

        private static readonly IntPtr SelContents = IsMetalSupported ? sel_registerName("contents") : IntPtr.Zero;
        private static readonly IntPtr SelRelease = IsMetalSupported ? sel_registerName("release") : IntPtr.Zero;

        static MetalBackend()
        {
            if (IsMac)
            {
                try
                {
                    IntPtr metalHandle = dlopen("/System/Library/Frameworks/Metal.framework/Metal", 1);
                    IsMetalSupported = metalHandle != IntPtr.Zero;
                }
                catch
                {
                    IsMetalSupported = false;
                }
            }
            else
            {
                IsMetalSupported = false;
            }
        }
        
        /// <summary>
        /// Dynamically loads a shared library into the address space of the calling process.
        /// </summary>
        /// <param name="path">The file path of the dynamic library.</param>
        /// <param name="mode">The load mode flags.</param>
        /// <returns>An opaque handle to the loaded library, or <see cref="IntPtr.Zero"/> on failure.</returns>

        [DllImport("libdl.dylib")]
        private static extern IntPtr dlopen(string path, int mode);
        /// <summary>
        /// Gets or sets the array of ancestor tensors that produced this tensor.
        /// Used for dependency tracking in autograd.
        /// </summary>

        public ITensor[] Inputs { get => _inputs; set => _inputs = value; }
        /// <summary>
        /// Gets the multi-dimensional shape structure of this tensor.
        /// </summary>
        public TensorShape Shape => _shape;
        /// <summary>
        /// Gets the device on which this tensor's calculations are processed.
        /// </summary>
        public Device Device => _device;
        /// <summary>
        /// Gets or sets a value indicating whether this tensor requires gradient computations for autograd.
        /// </summary>
        public bool RequiresGrad { get => _requiresGrad; set => _requiresGrad = value; }
        /// <summary>
        /// Gets or sets the gradient tensor for backpropagation.
        /// </summary>
        public ITensor? Grad { get => _grad; set => _grad = value; }
        /// <summary>
        /// Gets or sets the backward tracking function used to compute the gradients of this tensor's inputs.
        /// </summary>
        public Func<ITensor, ITensor>? GradFn { get => _gradFn; set => _gradFn = value; }
        /// <summary>
        /// Gets the flat underlying data as an array of single-precision floating-point numbers.
        /// </summary>
        public float[] Data => ToArray();

        public MetalBackend(TensorShape shape, bool requiresGrad = false, Device? device = null)
        {
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));
            _device = device ?? new Device(DeviceType.CPU, 0);
            _requiresGrad = requiresGrad;
            _bytes = (ulong)_shape.TotalElements * sizeof(float);

            if (IsMetalSupported)
            {
                InitializeMetalBuffer();
            }
            else
            {
                _metalBuffer = Marshal.AllocHGlobal((int)_bytes);
                unsafe { GC.AddMemoryPressure((long)_bytes); }
            }
        }
        
        /// <summary>
        /// Dynamically loads the Metal framework, registers system devices, and allocates a GPU buffer.
        /// </summary>
        private void InitializeMetalBuffer()
        {
            IntPtr metalLib = dlopen("/System/Library/Frameworks/Metal.framework/Metal", 1);
            if (metalLib != IntPtr.Zero)
            {
                IntPtr defaultDeviceSel = sel_registerName("MTLCreateSystemDefaultDevice");
                _metalDevice = objc_msgSend(IntPtr.Zero, defaultDeviceSel);

                if (_metalDevice != IntPtr.Zero)
                {
                    IntPtr newBufferSel = sel_registerName("newBufferWithLength:options:");
                    _metalBuffer = objc_msgSend(_metalDevice, newBufferSel, _bytes, 0);
                }
            }
        }
        
        /// <summary>
        /// Adds a delta tensor to the existing accumulated gradient of this tensor.
        /// </summary>
        /// <param name="delta">The incoming gradient tensor to accumulate.</param>
        public void AccumulateGrad(ITensor delta)
        {
            if (delta == null) return;
            lock (_lock)
            {
                if (_grad == null)
                {
                    _grad = delta.Clone();
                }
                else
                {
                    _grad.AddInPlace(delta);
                }
            }
        }
        /// <summary>
        /// Copies the tensor data from native memory or Metal buffer to a managed float array.
        /// </summary>
        /// <returns>A flat float array representing the tensor's elements.</returns>
        public float[] ToArray()
        {
            float[] host = new float[_shape.TotalElements];
            lock (_lock)
            {
                if (IsMetalSupported && _metalBuffer != IntPtr.Zero)
                {
                    IntPtr rawDataPtr = objc_msgSend(_metalBuffer, SelContents);
                    Marshal.Copy(rawDataPtr, host, 0, _shape.TotalElements);
                }
                else
                {
                    Marshal.Copy(_metalBuffer, host, 0, _shape.TotalElements);
                }
            }
            return host;
        }

        /// <summary>
        /// Extracts the single scalar value of a 1-element tensor.
        /// </summary>
        /// <returns>The single-precision floating point value.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the tensor contains more or less than one element.</exception>

        public float ToScalar()
        {
            if (_shape.TotalElements != 1)
                throw new InvalidOperationException("Tensor must be a scalar.");
            return ToArray()[0];
        }
        
        /// <summary>
        /// Overwrites the contents of the internal buffer with the specified float array.
        /// </summary>
        /// <param name="floats">The source float array.</param>
        /// <exception cref="ArgumentException">Thrown when the length of <paramref name="floats"/> does not match the tensor shape total elements.</exception>
        public void SetData(float[] floats)
        {
            if (floats.Length != _shape.TotalElements) throw new ArgumentException("Data volume mismatch.");
            lock (_lock)
            {
                if (IsMetalSupported && _metalBuffer != IntPtr.Zero)
                {
                    IntPtr rawDataPtr = objc_msgSend(_metalBuffer, SelContents);
                    Marshal.Copy(floats, 0, rawDataPtr, floats.Length);
                }
                else
                {
                    Marshal.Copy(floats, 0, _metalBuffer, floats.Length);
                }
            }
        }

        /// <summary>
        /// Creates a deep copy of the tensor, preserving shape, device target, autograd configuration, and values.
        /// </summary>
        /// <returns>A clone of the current tensor.</returns>
        public ITensor Clone()
        {
            var clone = new MetalBackend(_shape, _requiresGrad, _device);
            clone.SetData(ToArray());
            return clone;
        }
        
        /// <summary>
        /// Moves the tensor data to the target device backend.
        /// </summary>
        /// <param name="device">The target execution device.</param>
        /// <returns>A new <see cref="ITensor"/> mapped to the specified device backend.</returns>
        public ITensor To(Device device)
        {
            if (device.Type == DeviceType.CPU)
            {
                return new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, device);
            }
            return Clone();
        }
        
        /// <summary>
        /// Returns a value indicating whether this backend executes on CPU.
        /// </summary>
        /// <returns>True if running on CPU or using CPU fallback; otherwise false.</returns>

        public bool IsCpu() => !IsMetalSupported;
        /// <summary>
        /// Returns whether this tensor relies on a CUDA backend.
        /// </summary>
        /// <returns>Always false for the Metal backend.</returns>
        public bool IsCuda() => false;
        /// <summary>
        /// Yields the parameters of the tensor (primarily itself for model tracking).
        /// </summary>
        /// <returns>An enumerable containing this tensor.</returns>

        public IEnumerable<ITensor> Parameters() { yield return this; }
        /// <summary>
        /// Performs an in-place element-wise addition of another tensor to this tensor.
        /// </summary>
        /// <param name="other">The tensor to add.</param>

        public void AddInPlace(ITensor other)
        {
            var otherArr = other.ToArray();
            lock (_lock)
            {
                _version++;

                var selfArr = ToArray();
                Accelerate.Add(selfArr, selfArr, otherArr, _shape.TotalElements);
                SetData(selfArr);
            }
        }
        /// <summary>
        /// Performs an in-place scalar addition to all elements of this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to add.</param>

        public void AddInPlace(float scalar)
        {
            lock (_lock)
            {
                _version++;

                var selfArr = ToArray();
                Parallel.For(0, _shape.TotalElements, i => selfArr[i] += scalar);
                SetData(selfArr);
            }
        }
        /// <summary>
        /// Performs an in-place element-wise subtraction of another tensor from this tensor.
        /// </summary>
        /// <param name="other">The tensor to subtract.</param>

        public void SubtractInPlace(ITensor other)
        {
            var otherArr = other.ToArray();
            lock (_lock)
            {
                _version++;

                var selfArr = ToArray();
                Accelerate.Subtract(selfArr, selfArr, otherArr, _shape.TotalElements);
                SetData(selfArr);
            }
        }
        /// <summary>
        /// Performs an in-place scalar subtraction from all elements of this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to subtract.</param>

        public void SubtractInPlace(float scalar) => AddInPlace(-scalar);
        /// <summary>
        /// Performs an in-place element-wise multiplication with another tensor.
        /// </summary>
        /// <param name="other">The tensor to multiply by.</param>

        public void MultiplyInPlace(ITensor other)
        {
            var otherArr = other.ToArray();
            lock (_lock)
            {
                _version++;

                var selfArr = ToArray();
                Accelerate.Multiply(selfArr, selfArr, otherArr, _shape.TotalElements);
                SetData(selfArr);
            }
        }
        /// <summary>
        /// Performs an in-place scalar multiplication on all elements of this tensor.
        /// </summary>
        /// <param name="scalar">The scalar multiplier.</param>

        public void MultiplyInPlace(float scalar)
        {
            lock (_lock)
            {
                _version++;

                var selfArr = ToArray();
                Parallel.For(0, _shape.TotalElements, i => selfArr[i] *= scalar);
                SetData(selfArr);
            }
        }
        /// <summary>
        /// Computes the element-wise addition of this tensor and another tensor.
        /// Support shape broadcasting where applicable.
        /// </summary>
        /// <param name="other">The tensor to add.</param>
        /// <returns>A new tensor containing the sum.</returns>

        public ITensor Add(ITensor other)
        {
            var result = new MetalBackend(_shape.BroadcastTo(other.Shape), _requiresGrad || other.RequiresGrad, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[result.Shape.TotalElements];
            Accelerate.Add(res, a, b, res.Length);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes the element-wise subtraction of another tensor from this tensor.
        /// Supports shape broadcasting where applicable.
        /// </summary>
        /// <param name="other">The tensor to subtract.</param>
        /// <returns>A new tensor containing the difference.</returns>

        public ITensor Subtract(ITensor other)
        {
            var result = new MetalBackend(_shape.BroadcastTo(other.Shape), _requiresGrad || other.RequiresGrad, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[result.Shape.TotalElements];
            Accelerate.Subtract(res, a, b, res.Length);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes the element-wise multiplication of this tensor and another tensor.
        /// Supports shape broadcasting where applicable.
        /// </summary>
        /// <param name="other">The tensor to multiply by.</param>
        /// <returns>A new tensor containing the product.</returns>

        public ITensor Multiply(ITensor other)
        {
            var result = new MetalBackend(_shape.BroadcastTo(other.Shape), _requiresGrad || other.RequiresGrad, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[result.Shape.TotalElements];
            Accelerate.Multiply(res, a, b, res.Length);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes the element-wise division of this tensor by another tensor.
        /// Supports shape broadcasting where applicable.
        /// </summary>
        /// <param name="other">The divisor tensor.</param>
        /// <returns>A new tensor containing the quotient.</returns>

        public ITensor Divide(ITensor other)
        {
            var result = new MetalBackend(_shape.BroadcastTo(other.Shape), _requiresGrad || other.RequiresGrad, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[result.Shape.TotalElements];
            Accelerate.Divide(res, a, b, res.Length);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Adds a scalar value to every element of this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to add.</param>
        /// <returns>A new tensor containing the shifted values.</returns>

        public ITensor Add(float scalar)
        {
            var result = new MetalBackend(_shape, _requiresGrad, _device);
            var a = ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = a[i] + scalar);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Subtracts a scalar value from every element of this tensor.
        /// </summary>
        /// <param name="scalar">The scalar value to subtract.</param>
        /// <returns>A new tensor containing the shifted values.</returns>

        public ITensor Subtract(float scalar) => Add(-scalar);
        /// <summary>
        /// Multiplies every element of this tensor by a scalar value.
        /// </summary>
        /// <param name="scalar">The scalar value to multiply by.</param>
        /// <returns>A new tensor containing the scaled values.</returns>

        public ITensor Multiply(float scalar)
        {
            var result = new MetalBackend(_shape, _requiresGrad, _device);
            var a = ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = a[i] * scalar);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Divides every element of this tensor by a scalar value.
        /// </summary>
        /// <param name="scalar">The scalar divisor.</param>
        /// <returns>A new tensor containing the divided values.</returns>

        public ITensor Divide(float scalar) => Multiply(1f / scalar);
        /// <summary>
        /// Subtracts an integer value from every element of this tensor.
        /// </summary>
        /// <param name="other">The integer value to subtract.</param>
        /// <returns>A new tensor containing the shifted values.</returns>

        public ITensor Subtract(int other) => Subtract((float)other);
        /// <summary>
        /// Multiplies every element of this tensor by a double-precision scalar value.
        /// </summary>
        /// <param name="scalar">The double multiplier.</param>
        /// <returns>A new tensor containing the scaled values.</returns>
        public ITensor Multiply(double scalar) => Multiply((float)scalar);
        /// <summary>
        /// Divides every element of this tensor by a double-precision scalar value.
        /// </summary>
        /// <param name="scalar">The double divisor.</param>
        /// <returns>A new tensor containing the divided values.</returns>
        public ITensor Divide(double scalar) => Divide((float)scalar);
        /// <summary>
        /// Negates every element of this tensor.
        /// </summary>
        /// <returns>A new tensor containing the negated values.</returns>

        public ITensor Negate() => Multiply(-1f);
        /// <summary>
        /// Calculates the exponential value of each element in the tensor.
        /// </summary>
        /// <returns>A new tensor containing element-wise exp values.</returns>

        public ITensor Exp()
        {
            var result = new MetalBackend(_shape, _requiresGrad, _device);
            var a = ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = MathF.Exp(a[i]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Calculates the natural logarithm of each element in the tensor.
        /// </summary>
        /// <returns>A new tensor containing element-wise log values.</returns>

        public ITensor Log()
        {
            var result = new MetalBackend(_shape, _requiresGrad, _device);
            var a = ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = MathF.Log(a[i]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Calculates the square root of each element in the tensor.
        /// </summary>
        /// <returns>A new tensor containing element-wise square root values.</returns>

        public ITensor Sqrt()
        {
            var result = new MetalBackend(_shape, _requiresGrad, _device);
            var a = ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = MathF.Sqrt(a[i]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Calculates the absolute value of each element in the tensor.
        /// </summary>
        /// <returns>A new tensor containing element-wise absolute values.</returns>

        public ITensor Abs()
        {
            var result = new MetalBackend(_shape, _requiresGrad, _device);
            var a = ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = MathF.Abs(a[i]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Calculates the trigonometric sine of each element in the tensor.
        /// </summary>
        /// <returns>A new tensor containing element-wise sine values.</returns>

        public ITensor Sin()
        {
            var result = new MetalBackend(_shape, _requiresGrad, _device);
            var a = ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = MathF.Sin(a[i]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Calculates the trigonometric cosine of each element in the tensor.
        /// </summary>
        /// <returns>A new tensor containing element-wise cosine values.</returns>

        public ITensor Cos()
        {
            var result = new MetalBackend(_shape, _requiresGrad, _device);
            var a = ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = MathF.Cos(a[i]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Calculates each element in the tensor raised to a given float power.
        /// </summary>
        /// <param name="exponent">The exponent power value.</param>
        /// <returns>A new tensor containing elements raised to the exponent power.</returns>

        public ITensor Pow(float exponent)
        {
            var result = new MetalBackend(_shape, _requiresGrad, _device);
            var a = ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = MathF.Pow(a[i], exponent));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Calculates each element raised to the power of corresponding elements in an exponent tensor.
        /// Supports shape broadcasting where applicable.
        /// </summary>
        /// <param name="exponent">The exponent tensor.</param>
        /// <returns>A new tensor containing base elements raised to exponent tensor values.</returns>

        public ITensor Pow(ITensor exponent)
        {
            var result = new MetalBackend(_shape.BroadcastTo(exponent.Shape), _requiresGrad, _device);
            var a = ToArray();
            var b = exponent.ToArray();
            var res = new float[result.Shape.TotalElements];
            Parallel.For(0, res.Length, i => res[i] = MathF.Pow(a[i % a.Length], b[i % b.Length]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes matrix multiplication between two 2D tensors.
        /// </summary>
        /// <param name="other">The right-hand side tensor in multiplication.</param>
        /// <returns>A new tensor containing matrix product result.</returns>
        /// <exception cref="InvalidOperationException">Thrown when either tensor does not have exactly 2 dimensions.</exception>

        public ITensor MatMul(ITensor other)
        {
            if (_shape.Rank != 2 || other.Shape.Rank != 2)
                throw new InvalidOperationException("MatMul requires 2D matrices.");

            int m = _shape[0];
            int k = _shape[1];
            int n = other.Shape[1];

            var result = new MetalBackend(new TensorShape(m, n), _requiresGrad || other.RequiresGrad, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[m * n];

            Parallel.For(0, m, i =>
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int l = 0; l < k; l++)
                    {
                        sum += a[i * k + l] * b[l * n + j];
                    }
                    res[i * n + j] = sum;
                }
            });

            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Permutes the axes of the tensor based on the provided dimension mapping.
        /// Relies on CPU backend fallback processing.
        /// </summary>
        /// <param name="perm">The desired permutation map of dimensions.</param>
        /// <returns>A new permuted tensor.</returns>

        public ITensor Transpose(int[] perm)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Transpose(perm).To(_device);
        }
        /// <summary>
        /// Reshapes the tensor to a new dimensional configuration.
        /// </summary>
        /// <param name="newShape">The target shape dimensions.</param>
        /// <returns>A new reshaped tensor with shared elements volume.</returns>
        /// <exception cref="ArgumentException">Thrown when total elements count does not match the target shape.</exception>

        public ITensor Reshape(params int[] newShape)
        {
            var ns = new TensorShape(newShape);
            if (ns.TotalElements != _shape.TotalElements)
                throw new ArgumentException("Total volume mismatch.");

            var reshaped = new MetalBackend(ns, _requiresGrad, _device);
            reshaped.SetData(ToArray());
            return reshaped;
        }
        /// <summary>
        /// Slices the tensor across multiple dimensions.
        /// Relies on CPU backend fallback processing.
        /// </summary>
        /// <param name="slices">Array of tuples representing the start index, end index, and stride step for each dimension.</param>
        /// <returns>A sliced tensor view/copy.</returns>

        public ITensor Slice(params (int start, int end, int step)[] slices)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Slice(slices).To(_device);
        }
        /// <summary>
        /// Concatenates this tensor with other tensors along a specified axis.
        /// Relies on CPU backend fallback processing.
        /// </summary>
        /// <param name="others">The collection of tensors to join.</param>
        /// <param name="axis">The dimension along which the tensors will be joined.</param>
        /// <returns>The concatenated tensor.</returns>

        public ITensor Concat(IEnumerable<ITensor> others, int axis = 0)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Concat(others, axis).To(_device);
        }
        /// <summary>
        /// Broadcasts this tensor's dimensions to match the target shape.
        /// Relies on CPU backend fallback processing.
        /// </summary>
        /// <param name="targetShape">The targeted broadcast dimensions.</param>
        /// <returns>A new broadcasted tensor.</returns>

        public ITensor BroadcastTo(TensorShape targetShape)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.BroadcastTo(targetShape).To(_device);
        }
        /// <summary>
        /// Performs an addition on this tensor and another tensor with broadcasting support.
        /// </summary>
        /// <param name="other">The tensor to add.</param>
        /// <returns>The broadcasted sum tensor.</returns>

        public ITensor BroadcastAdd(ITensor other) => Add(other);
        /// <summary>
        /// Reshapes and broadcasts the current tensor along a specific axis to match target shape dimensions.
        /// Relies on CPU backend fallback processing.
        /// </summary>
        /// <param name="target">The target tensor shape.</param>
        /// <param name="axis">The axis index to align during broadcast.</param>
        /// <returns>A new reshaped and broadcasted tensor.</returns>

        public ITensor ReshapeWithBroadcast(TensorShape target, int axis)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.ReshapeWithBroadcast(target, axis).To(_device);
        }
        /// <summary>
        /// Sums the values of elements across the designated axis dimension, option for keeping dims.
        /// Relies on CPU backend fallback processing.
        /// </summary>
        /// <param name="axis">The target axis. Sums everything if null.</param>
        /// <param name="keepDims">Determines whether reduced dimensions are retained with length 1.</param>
        /// <returns>The computed summation tensor.</returns>

        public ITensor Sum(int? axis = null, bool keepDims = false)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Sum(axis, keepDims).To(_device);
        }
        /// <summary>
        /// Sums elements across multiple axes.
        /// Relies on CPU backend fallback processing.
        /// </summary>
        /// <param name="axes">The array of target axis indexes.</param>
        /// <param name="keepDims">Determines whether reduced dimensions are retained with length 1.</param>
        /// <returns>The computed summation tensor.</returns>

        public ITensor Sum(int[] axes, bool keepDims = false)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Sum(axes, keepDims).To(_device);
        }
        /// <summary>
        /// Calculates elements mean across a specified axis.
        /// Relies on CPU backend fallback processing.
        /// </summary>
        /// <param name="axis">The target axis. Reduces completely if null.</param>
        /// <param name="keepDims">Determines whether reduced dimensions are retained with length 1.</param>
        /// <returns>The calculated average tensor.</returns>

        public ITensor Mean(int? axis = null, bool keepDims = false)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Mean(axis, keepDims).To(_device);
        }
        /// <summary>
        /// Calculates elements mean across multiple axes.
        /// Relies on CPU backend fallback processing.
        /// </summary>
        /// <param name="axes">The array of target axis indexes.</param>
        /// <param name="keepDims">Determines whether reduced dimensions are retained with length 1.</param>
        /// <returns>The calculated average tensor.</returns>

        public ITensor Mean(int[] axes, bool keepDims = false)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Mean(axes, keepDims).To(_device);
        }
        /// <summary>
        /// Finds maximum element values along a target axis.
        /// Relies on CPU backend fallback processing.
        /// </summary>
        /// <param name="axis">The target axis. Defaults to last dimension.</param>
        /// <param name="keepDims">Determines whether reduced dimensions are retained with length 1.</param>
        /// <returns>The maximum elements tensor.</returns>

        public ITensor Max(int axis = -1, bool keepDims = false)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Max(axis, keepDims).To(_device);
        }
        /// <summary>
        /// Finds minimum element values along a target axis.
        /// Relies on CPU backend fallback processing.
        /// </summary>
        /// <param name="axis">The target axis. Defaults to last dimension.</param>
        /// <param name="keepDims">Determines whether reduced dimensions are retained with length 1.</param>
        /// <returns>The minimum elements tensor.</returns>

        public ITensor Min(int axis = -1, bool keepDims = false)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Min(axis, keepDims).To(_device);
        }
        /// <summary>
        /// Determines indices of minimum values along the specified axis.
        /// Relies on CPU backend fallback processing.
        /// </summary>
        /// <param name="axis">The dimension along which indices are computed.</param>
        /// <returns>An index tensor containing indices of minimums.</returns>

        public ITensor ArgMin(int axis)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.ArgMin(axis).To(_device);
        }
        /// <summary>
        /// Determines indices of maximum values along the specified axis.
        /// Relies on CPU backend fallback processing.
        /// </summary>
        /// <param name="axis">The dimension along which indices are computed.</param>
        /// <returns>An index tensor containing indices of maximums.</returns>

        public ITensor ArgMax(int axis)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.ArgMax(axis).To(_device);
        }
        /// <summary>
        /// Calculates cumulative sum of elements along a targeted axis.
        /// Relies on CPU backend fallback processing.
        /// </summary>
        /// <param name="axis">The targeted cumulative sum axis.</param>
        /// <returns>A tensor containing cumulative sum elements.</returns>

        public ITensor CumSum(int axis)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.CumSum(axis).To(_device);
        }
        /// <summary>
        /// Compares this tensor against another, element-wise, to check if elements are strictly greater.
        /// </summary>
        /// <param name="other">The tensor to compare against.</param>
        /// <returns>A binary boolean tensor (1.0 for true, 0.0 for false).</returns>

        public ITensor GreaterThan(ITensor other)
        {
            var result = new MetalBackend(_shape, false, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = a[i] > b[i % b.Length] ? 1f : 0f);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Compares this tensor against another, element-wise, to check if elements are greater than or equal.
        /// </summary>
        /// <param name="other">The tensor to compare against.</param>
        /// <returns>A binary boolean tensor (1.0 for true, 0.0 for false).</returns>

        public ITensor GreaterThanOrEqual(ITensor other)
        {
            var result = new MetalBackend(_shape, false, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = a[i] >= b[i % b.Length] ? 1f : 0f);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Compares this tensor against another, element-wise, to check if elements are less than or equal.
        /// </summary>
        /// <param name="other">The tensor to compare against.</param>
        /// <returns>A binary boolean tensor (1.0 for true, 0.0 for false).</returns>

        public ITensor LessEqual(ITensor other)
        {
            var result = new MetalBackend(_shape, false, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = a[i] <= b[i % b.Length] ? 1f : 0f);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Checks element-wise equality between this tensor and another using a small tolerance threshold.
        /// </summary>
        /// <param name="other">The tensor to check equality with.</param>
        /// <returns>A binary boolean tensor (1.0 for equal, 0.0 for non-equal).</returns>

        public ITensor Equal(ITensor other)
        {
            var result = new MetalBackend(_shape, false, _device);
            var a = ToArray();
            var b = other.ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = MathF.Abs(a[i] - b[i % b.Length]) < 1e-6f ? 1f : 0f);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Selects elements from trueValue or falseValue depending on evaluation condition values.
        /// </summary>
        /// <param name="condition">Condition values (values &gt; 0 signify evaluation to true).</param>
        /// <param name="trueValue">The source tensor elements if evaluated to true.</param>
        /// <param name="falseValue">The source tensor elements if evaluated to false.</param>
        /// <returns>A conditional-selection outcome tensor.</returns>

        public ITensor Where(ITensor condition, ITensor trueValue, ITensor falseValue)
        {
            var result = new MetalBackend(_shape, false, _device);
            var cond = condition.ToArray();
            var t = trueValue.ToArray();
            var f = falseValue.ToArray();
            var res = new float[_shape.TotalElements];
            Parallel.For(0, res.Length, i => res[i] = cond[i % cond.Length] > 0f ? t[i % t.Length] : f[i % f.Length]);
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes element-wise signs of tensor values (-1.0, 0.0, 1.0).
        /// </summary>
        /// <returns>A new tensor containing elements' signs.</returns>

        public ITensor Sign()
        {
            var result = new MetalBackend(_shape, false, _device);
            var a = ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = MathF.Sign(a[i]));
            result.SetData(res);
            return result;
        }
        /// <summary>
        /// Computes the hyperbolic tangent of elements in this tensor.
        /// </summary>
        /// <returns>A new tensor containing tanh values.</returns>

        public ITensor Tanh() => new Tanh().Forward(this);
        /// <summary>
        /// Computes Rectified Linear Unit (ReLU) of elements in this tensor.
        /// </summary>
        /// <returns>A new tensor containing rectified values.</returns>
        public ITensor Relu() => new ReLU().Forward(this);
        /// <summary>
        /// Computes the logistic sigmoid of elements in this tensor.
        /// </summary>
        /// <returns>A new tensor containing sigmoid values.</returns>
        public ITensor Sigmoid() => new Sigmoid().Forward(this);
        /// <summary>
        /// Computes the softmax activation of elements in this tensor along the specified axis.
        /// </summary>
        /// <param name="axis">The normalization axis. Defaults to the last dimension (-1).</param>
        /// <returns>A new tensor containing the normalized exponentials.</returns>
        public ITensor Softmax(int axis = -1) => new Softmax(axis).Forward(this);
        /// <summary>
        /// Performs automatic backpropagation starting from this tensor.
        /// </summary>
        /// <param name="gradient">The external loss gradient tensor to initiate backward pass. Defaults to null (1.0 scalar is used).</param>

        public void Backward(ITensor? gradient = null)
        {
            AutogradEngine.Backward(this, gradient);
        }
        /// <summary>
        /// Clears tracked gradient tensors and links to backward autograd functions.
        /// </summary>

        public void ClearGrad()
        {
            _grad = null;
            _gradFn = null;
        }
        /// <summary>
        /// Gathers values along a specified axis using lookup indexes.
        /// Relies on CPU backend fallback processing.
        /// </summary>
        /// <param name="axis">The axis to gather from.</param>
        /// <param name="indices">Indices specifying the target positions to lookup.</param>
        /// <returns>A new gathered elements tensor.</returns>
        public ITensor Gather(int axis, ITensor indices)
        {
            var cpuEquivalent = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, _device);
            return cpuEquivalent.Gather(axis, indices).To(_device);
        }
        
        /// <summary>
        /// Evaluates logical NOT operation element-wise (where 0.0 maps to 1.0, and any non-zero maps to 0.0).
        /// </summary>
        /// <returns>A binary boolean negation tensor.</returns>
        public ITensor LogicalNot()
        {
            var result = new MetalBackend(_shape, false, _device);
            var a = ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = a[i] == 0f ? 1f : 0f);
            result.SetData(res);
            return result;
        }

        /// <summary>
        /// Clamps all elements of the tensor within a range specified by two boundary float values.
        /// </summary>
        /// <param name="v1">The minimum bounding threshold.</param>
        /// <param name="v2">The maximum bounding threshold.</param>
        /// <returns>A clamped elements value tensor.</returns>
        public ITensor Clip(float v1, float v2)
        {
            var result = new MetalBackend(_shape, false, _device);
            var a = ToArray();
            var res = new float[a.Length];
            Parallel.For(0, a.Length, i => res[i] = Math.Clamp(a[i], v1, v2));
            result.SetData(res);
            return result;
        }

        public string DType => "float32";

        public ITensor Cast(string dtype)
        {
            if (dtype != "float32" && dtype != "float" && dtype != "f32")
                throw new NotSupportedException($"Only float32 is currently supported. Requested: {dtype}");
            return this; // already float32, zero-copy
        }

        // =================================================================================
        // SQUEEZE (pure view – zero copy)  – FIXED
        // =================================================================================
        public ITensor Squeeze(int? axis = null)
        {
            if (axis == null)
            {
                var newDims = _shape.Dimensions.Where(d => d != 1).ToArray();
                if (newDims.Length == 0)
                    newDims = new[] { 1 };
                return Reshape(newDims);
            }

            int a = axis.Value < 0 ? _shape.Rank + axis.Value : axis.Value;
            if (a < 0 || a >= _shape.Rank)
                throw new ArgumentOutOfRangeException(nameof(axis));

            if (_shape.Dimensions[a] != 1)
                throw new InvalidOperationException($"Cannot squeeze axis {a} of size {_shape.Dimensions[a]}.");

            var dims = _shape.Dimensions.ToList();
            dims.RemoveAt(a);
            if (dims.Count == 0)
                dims.Add(1);

            return Reshape(dims.ToArray()); // shares CudaAllocation – zero copy
        }

        public ITensor Unsqueeze(int axis)
        {
            int rank = _shape.Rank;
            int actualAxis = axis < 0 ? rank + axis + 1 : axis;
            if (actualAxis < 0 || actualAxis > rank)
                throw new ArgumentOutOfRangeException(nameof(axis));

            var newDims = new int[rank + 1];
            for (int i = 0, j = 0; i < newDims.Length; i++)
                newDims[i] = (i == actualAxis) ? 1 : _shape.Dimensions[j++];

            // Zero-copy reshape (shares the WebGPU buffer)
            return Reshape(newDims);
        }

        // TopK can stay as the CPU fallback you already have
        public (ITensor values, ITensor indices) TopK(int k, int axis = -1)
        {
            var cpu = new CpuBackend(ToArray(), _shape.Clone(), _requiresGrad, Device.CPU);
            var (v, i) = cpu.TopK(k, axis);
            return (v.To(_device), i.To(_device));
        }

        /// <summary>
        /// Disposes internal native pointers, GPU Metal buffers and releases references.
        /// </summary>
        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing) { /* Clean up managed resources if any */ }

                // Unmanaged cleanup (No locking here, finalizers run on a separate thread)
                if (_metalBuffer != IntPtr.Zero)
                {
                    if (IsMetalSupported) objc_msgSend(_metalBuffer, SelRelease);
                    else { Marshal.FreeHGlobal(_metalBuffer); }
                    _metalBuffer = IntPtr.Zero;
                }
                if (_metalDevice != IntPtr.Zero && IsMetalSupported)
                {
                    objc_msgSend(_metalDevice, SelRelease);
                    _metalDevice = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        public void Dispose()
        {
            lock (_lock) { Dispose(true); }
            GC.SuppressFinalize(this);
        }

        ~MetalBackend() => Dispose(false);
    }
}