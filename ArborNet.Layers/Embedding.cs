// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Layers
{


    #region Using Statements:

    using ArborNet.Core.Functional;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Layers;
    using ArborNet.Core.Native.PInvoke;
    using ArborNet.Core.Tensors;
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;

    using ArborNet.Core.Devices;
    using ArborNet.Core.Backends;


    #endregion

    /// <summary>
    /// Represents a thread-safe categorical embedding table layer that maps integer indices to dense vectors.
    /// </summary>
    public class Embedding : BaseLayer
    {
        private ITensor _weights;
        public int NumEmbeddings { get; }
        public int EmbeddingDim { get; }

        public Embedding(int numEmbeddings, int embeddingDim)
        {
            NumEmbeddings = numEmbeddings;
            EmbeddingDim = embeddingDim;

            this.device = Device.CUDA;
            if (this.device.Type == DeviceType.CUDA && !CUDA.IsAvailable())
            {
                this.device = Device.CPU;
            }

            _weights = Initializers.Normal(new TensorShape(numEmbeddings, embeddingDim), this.device);
            _weights.RequiresGrad = true;
        }

        public override ITensor Forward(ITensor indices)
        {
            ValidateInput(indices);

            var outShapeList = new List<int>(indices.Shape.Dimensions) { EmbeddingDim };
            var outShape = new TensorShape(outShapeList.ToArray());

            if (indices.Device.Type == DeviceType.CUDA && CUDA.IsAvailable())
            {
                var idxRaw = Tensor.Unwrap(indices) as CudaBackend ?? throw new InvalidOperationException("Indices must reside on CUDA GPU.");
                var wRaw = Tensor.Unwrap(_weights) as CudaBackend ?? throw new InvalidOperationException("Weights must reside on CUDA GPU.");

                var result = new Tensor(new CudaBackend(outShape, _weights.RequiresGrad, indices.Device));
                var resRaw = Tensor.Unwrap(result) as CudaBackend ?? throw new InvalidOperationException("Result initialization failed.");

                CUDA.NativeEmbedding(wRaw.DevicePointer, idxRaw.DevicePointer, resRaw.DevicePointer, NumEmbeddings, EmbeddingDim, indices.Shape.TotalElements);

                if (_weights.RequiresGrad)
                {
                    var capturedIndices = idxRaw;
                    var capturedWeights = wRaw;
                    int numWords = NumEmbeddings;
                    int embedDim = EmbeddingDim;
                    int totalIndices = indices.Shape.TotalElements;

                    result.GradFn = gradOut =>
                    {
                        var goRaw = Tensor.Unwrap(gradOut) as CudaBackend ?? throw new InvalidOperationException("Upstream gradient must be on CUDA GPU.");

                        var gradWeights = new Tensor(new CudaBackend(capturedWeights.Shape, false, gradOut.Device));
                        var gwRaw = Tensor.Unwrap(gradWeights) as CudaBackend ?? throw new InvalidOperationException("Gradient allocation failed.");
                        CUDA.CudaMemset(gwRaw.DevicePointer, 0, (ulong)capturedWeights.Shape.TotalElements * sizeof(float));

                        CUDA.NativeEmbeddingGrad(goRaw.DevicePointer, capturedIndices.DevicePointer, gwRaw.DevicePointer, numWords, embedDim, totalIndices);

                        capturedWeights.AccumulateGrad(gradWeights);
                        return Tensor.Zeros(capturedIndices.Shape, capturedIndices.Device);
                    };
                }

                return result;
            }
            else
            {
                // HIGH-PERFORMANCE PARALLEL CPU FALLBACK
                float[] idxData = indices.ToArray();
                float[] wData = _weights.ToArray();
                float[] outData = new float[outShape.TotalElements];

                int totalIndices = indices.Shape.TotalElements;

                Parallel.For(0, totalIndices, i =>
                {
                    int token = (int)idxData[i];
                    if (token >= 0 && token < NumEmbeddings)
                    {
                        Array.Copy(wData, token * EmbeddingDim, outData, i * EmbeddingDim, EmbeddingDim);
                    }
                });

                var result = Tensor.FromArray(outData, outShape, indices.Device);

                if (_weights.RequiresGrad)
                {
                    var capturedIndices = indices;
                    var capturedWeights = _weights;

                    result.GradFn = gradOut =>
                    {
                        float[] goData = gradOut.ToArray();
                        float[] gwData = new float[capturedWeights.Shape.TotalElements];

                        for (int i = 0; i < totalIndices; i++)
                        {
                            int token = (int)idxData[i];
                            if (token >= 0 && token < NumEmbeddings)
                            {
                                for (int d = 0; d < EmbeddingDim; d++)
                                {
                                    gwData[token * EmbeddingDim + d] += goData[i * EmbeddingDim + d];
                                }
                            }
                        }

                        capturedWeights.AccumulateGrad(Tensor.FromArray(gwData, capturedWeights.Shape, gradOut.Device));
                        return Tensor.Zeros(capturedIndices.Shape, capturedIndices.Device);
                    };
                }

                return result;
            }
        }

        public override IEnumerable<ITensor> Parameters()
        {
            yield return _weights;
        }
    }
}