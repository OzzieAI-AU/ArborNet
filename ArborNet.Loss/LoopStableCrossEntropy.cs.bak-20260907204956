namespace ArborNet.Loss
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Core.Tensors;
    using ArborNet.Layers;

    public sealed class LoopStableCrossEntropy
    {
        public float LabelSmoothing { get; set; } = 0.0f;
        public float MaxGradNorm { get; set; } = 1.0f;
        public (Tensor loss, Tensor logits) Forward(Tensor logits, Tensor targets)
        {
            var loss = CrossEntropy.Forward(logits, targets, LabelSmoothing);
            return (loss, logits);
        }
        public void ClipParameters(IEnumerable<Parameter> parameters)
        {
            // global L2 clip — critical for R∈{4,8,16}
            ParameterClip.ClipByGlobalNorm(parameters, MaxGradNorm);
        }
    }
}
