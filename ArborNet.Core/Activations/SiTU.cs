// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Core.Activations
{


    #region Using Statements:

    using ArborNet.Activations;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;

    #endregion

    /// <summary>
    /// Implements the Sigmoid Tanh Unit (SiTU) activation function.
    /// Formula: SiTU(x) = Sigmoid(x) * Tanh(x)
    /// </summary>
    public sealed class SiTU : BaseActivation
    {
        public override ITensor Forward(ITensor input)
        {
            ValidateInput(input);
            var device = input.Device;

            var sigmoid = new Sigmoid().Forward(input);
            var tanh = new Tanh().Forward(input);
            var output = sigmoid.Multiply(tanh);

            if (input.RequiresGrad)
            {
                output.GradFn = gradOutput =>
                {
                    var ones = Tensor.Ones(input.Shape, device);

                    // dSigmoid/dx = Sigmoid(x) * (1 - Sigmoid(x))
                    var dSigmoid = sigmoid.Multiply(ones.Subtract(sigmoid));

                    // dTanh/dx = 1 - Tanh(x)^2
                    var dTanh = ones.Subtract(tanh.Multiply(tanh));

                    // Product Rule: dSiTU = dSigmoid * Tanh + Sigmoid * dTanh
                    var localGrad = dSigmoid.Multiply(tanh).Add(sigmoid.Multiply(dTanh));
                    var gradInput = gradOutput.Multiply(localGrad);

                    input.AccumulateGrad(gradInput);
                    return gradInput;
                };
            }

            return output;
        }
    }
}