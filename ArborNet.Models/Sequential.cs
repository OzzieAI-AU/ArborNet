// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Models
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.Linq;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;
    /// <summary>
    /// Represents a linear stack of layers for building feed-forward neural networks.
    /// </summary>
    /// <remarks>
    /// The <see cref="Sequential"/> model provides a simple and intuitive way to construct 
    /// deep learning architectures by chaining <see cref="ILayer"/> components. Data flows 
    /// sequentially through the added layers during the forward pass.
    /// </remarks>

    #endregion

    public class Sequential : BaseModel
    {
        /// <summary>
        /// The internal ordered collection of layers that comprise the sequential architecture.
        /// </summary>
        private readonly List<ILayer> layers = new();
        // FIXED: Removed 'private bool isTraining = true;' to prevent hiding BaseModel.isTraining

        /// <summary>
        /// Initializes a new, empty instance of the <see cref="Sequential"/> class.
        /// </summary>
        public Sequential() { }

        /// <summary>
        /// Initializes a new instance of the <see cref="Sequential"/> class with a predefined collection of layers.
        /// </summary>
        /// <param name="initialLayers">An enumerable collection of <see cref="ILayer"/> instances to populate the model.</param>
        public Sequential(IEnumerable<ILayer> initialLayers)
        {
            if (initialLayers != null)
                layers.AddRange(initialLayers);
        }
        /// <summary>
        /// Appends a new layer to the end of the sequential model's pipeline.
        /// </summary>
        /// <param name="layer">The <see cref="ILayer"/> to add to the network architecture.</param>
        /// <remarks>
        /// If the <paramref name="layer"/> is <see langword="null"/>, it will not be added to the network.
        /// </remarks>

        public void Add(ILayer layer)
        {
            if (layer != null)
                layers.Add(layer);
        }
        /// <summary>
        /// Executes a forward pass through the network, sequentially propagating the input tensor through each layer.
        /// </summary>
        /// <param name="input">The starting input <see cref="ITensor"/> to feed into the first layer.</param>
        /// <returns>The final computed <see cref="ITensor"/> after passing through all layers in the pipeline.</returns>
        /// <exception cref="ArgumentNullException">Thrown when the provided <paramref name="input"/> is <see langword="null"/>.</exception>

        public override ITensor Forward(ITensor input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));

            ITensor current = input;
            foreach (var layer in layers)
            {
                current = layer.Forward(current);
            }
            return current;
        }
        /// <summary>
        /// Flattens and retrieves all trainable parameters (such as weights and biases) from every layer within the sequential model.
        /// </summary>
        /// <returns>An <see cref="IEnumerable{ITensor}"/> containing all the trainable parameters of the underlying layers.</returns>

        public override IEnumerable<ITensor> Parameters()
        {
            return layers.SelectMany(l => l.Parameters());
        }
        /// <summary>
        /// Transitions the model and all constituent layers into training mode.
        /// </summary>
        /// <remarks>
        /// This ensures stateful layers (e.g., Dropout, Batch Normalization) behave correctly during backpropagation and weight updates.
        /// Layers that inherit from <see cref="BaseLayer"/> will have their training state updated accordingly.
        /// </remarks>

        public override void Train()
        {
            isTraining = true;
            foreach (var layer in layers)
            {
                if (layer is BaseLayer baseLayer)
                    baseLayer.Train();
            }
        }
        /// <summary>
        /// Transitions the model and all constituent layers into evaluation (inference) mode.
        /// </summary>
        /// <remarks>
        /// This ensures deterministic output by disabling training-specific behaviors in stateful layers (e.g., freezing Dropout rates).
        /// Layers that inherit from <see cref="BaseLayer"/> will have their evaluation state updated accordingly.
        /// </remarks>

        public override void Eval()
        {
            isTraining = false;
            foreach (var layer in layers)
            {
                if (layer is BaseLayer baseLayer)
                    baseLayer.Eval();
            }
        }
    }
}