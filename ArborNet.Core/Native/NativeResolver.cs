// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

using System;
using System.Reflection;
using System.Runtime.InteropServices;

namespace ArborNet.Core.Native
{
    /// <summary>
    /// Provides platform-agnostic native library resolution.
    /// Intercepts P/Invoke calls and resolves library names based on the executing operating system.
    /// </summary>
    /// <remarks>
    /// This static class registers a custom resolver callback that maps standard Windows native dynamic-link library (DLL) names 
    /// (such as "cudart64_12.dll" and "cuda_backend.dll") to their corresponding platform-specific equivalents 
    /// (shared objects on Linux and dynamic libraries on macOS). This enables the library to achieve seamless 
    /// cross-platform compatibility without modification of the underlying <see cref="DllImportAttribute"/> declarations.
    /// </remarks>
    public static class NativeResolver
    {
        private static bool _registered = false;
        private static readonly object _lock = new();
        /// <summary>
        /// Registers the dynamic DLL import resolver for the executing assembly.
        /// </summary>
        /// <remarks>
        /// This method is thread-safe and guarantees that the resolver callback is registered exactly once.
        /// It configures the <see cref="NativeLibrary.SetDllImportResolver"/> for the currently executing assembly, 
        /// mapping Windows target library names to their Unix/macOS equivalent filenames depending on the runtime platform.
        /// </remarks>

        public static void Register()
        {
            lock (_lock)
            {
                if (_registered) return;

                NativeLibrary.SetDllImportResolver(Assembly.GetExecutingAssembly(), (libraryName, assembly, searchPath) =>
                {
                    if (libraryName == "cudart64_12.dll")
                    {
                        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                            return NativeLibrary.Load("libcudart.so", assembly, searchPath);
                        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                            return NativeLibrary.Load("libcudart.dylib", assembly, searchPath);
                        return NativeLibrary.Load("cudart64_12.dll", assembly, searchPath);
                    }
                    if (libraryName == "cuda_backend.dll")
                    {
                        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                            return NativeLibrary.Load("libcuda_backend.so", assembly, searchPath);
                        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                            return NativeLibrary.Load("libcuda_backend.dylib", assembly, searchPath);
                        return NativeLibrary.Load("cuda_backend.dll", assembly, searchPath);
                    }
                    return IntPtr.Zero;
                });

                _registered = true;
            }
        }
    }
}