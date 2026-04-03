using System;
using System.Collections.Generic;
using System.Linq;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Models;
using ArborNet.Core.Tensors;
using ArborNet.Layers;

namespace ArborNet.Models
{
    public class Sequential : BaseModel
    {
        private readonly List<ILayer> layers = new();
        private bool isTraining = true;

        public Sequential() { }

        public Sequential(IEnumerable<ILayer> initialLayers)
        {
            if (initialLayers != null)
                layers.AddRange(initialLayers);
        }

        public void Add(ILayer layer)
        {
            if (layer != null)
                layers.Add(layer);
        }

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

        public override IEnumerable<ITensor> Parameters()
        {
            return layers.SelectMany(l => l.Parameters());
        }

        public override void Train()
        {
            isTraining = true;
            foreach (var layer in layers)
            {
                if (layer is BaseLayer baseLayer)
                    baseLayer.Train();
            }
        }

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