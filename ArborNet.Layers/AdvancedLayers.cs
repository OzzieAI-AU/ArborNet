using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;

namespace ArborNet.Layers
{
    /// <summary>
    /// Container for advanced layers (LayerNorm, etc.). 
    /// This is no longer a static class inheriting from BaseLayer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This static class acts as an organizational container for advanced neural network 
    /// layer implementations that do not follow the standard <see cref="BaseLayer"/> inheritance pattern.
    /// </para>
    /// <para>
    /// Advanced layers such as Layer Normalization are maintained in their own dedicated files 
    /// for better maintainability. This class can be extended with static factory methods or 
    /// helper utilities for other specialized layers (e.g., GroupNorm, RMSNorm, etc.).
    /// </para>
    /// </remarks>
    public static class AdvancedLayers
    {
        // LayerNorm is in its own file. Add other advanced layers here as needed.
    }
}