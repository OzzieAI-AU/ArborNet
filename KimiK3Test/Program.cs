namespace KimiK3Test
{
    using ArborNet.Core.Devices;
    using ArborNet.Core.Functional;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Serialization;
    using ArborNet.Core.Tensors;
    using ArborNet.Losses;
    using ArborNet.Models;
    using ArborNet.Optimizers;
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Runtime.Intrinsics.X86;

    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=================================================");
            Console.WriteLine("        Kimi K3 Frontier Model Demo App          ");
            Console.WriteLine("=================================================");

            // 1. Choose Execution Device (Gracefully falls back to CPU if CUDA is unavailable)
            Device device = Device.CPU;
            if (device.Type == DeviceType.CUDA && !ArborNet.Core.Native.PInvoke.CUDA.IsAvailable())
            {
                Console.WriteLine("[Warning] CUDA Hardware was selected but is not available. Falling back to CPU.");
                device = Device.CPU;
            }
            Console.WriteLine($"Running on Target Device: {device}");

            // 2. Configure Ultra-lightweight K3 Hyperparameters for local training demonstration
            int vocabSize = 16;       // 16 distinct tokens
            int dModel = 16;          // Hidden embedding dimension
            int nHeads = 2;           // Number of attention heads
            int numLayers = 2;        // Stacked K3 Blocks
            int numExperts = 4;       // 4 total MoE Experts
            int activeExperts = 1;    // Route 1 active expert per token
            int expertCapacity = 8;   // Max capacity per expert block before soft token dropping
            int maxSeqLen = 5;        // Sequence prediction length

            Console.WriteLine("Initializing Kimi K3 Model architecture...");
            var model = new KimiK3(
                vocabSize,
                dModel,
                nHeads,
                numLayers,
                numExperts,
                activeExperts,
                expertCapacity,
                maxSeqLen,
                device
            );

            // Set model to active training mode
            model.Train();

            // 3. Generate Synthetic Sequence Dataset (e.g., Simple Increment Patterns: [0, 1, 2, 3, 4] -> [1, 2, 3, 4, 5])
            int batchSize = 4;
            Console.WriteLine($"Generating synthetic dataset (Batch Size: {batchSize})...");

            float[] inputsData = new float[batchSize * maxSeqLen];
            float[] targetsData = new float[batchSize * maxSeqLen * vocabSize]; // One-hot targets for stable MSE Loss

            for (int b = 0; b < batchSize; b++)
            {
                int patternStart = b % 4; // Varying patterns per batch slice
                for (int t = 0; t < maxSeqLen; t++)
                {
                    int inputTokenId = patternStart + t;
                    int targetTokenId = patternStart + t + 1;

                    inputsData[b * maxSeqLen + t] = inputTokenId;

                    // Map one-hot targets: targetsData[batch, seq, tokenId] = 1.0f
                    int flatTargetIndex = (b * maxSeqLen * vocabSize) + (t * vocabSize) + targetTokenId;
                    targetsData[flatTargetIndex] = 1.0f;
                }
            }

            ITensor inputTensor = Tensor.FromArray(inputsData, new TensorShape(batchSize, maxSeqLen), device);
            ITensor targetsTensor = Tensor.FromArray(targetsData, new TensorShape(batchSize, maxSeqLen, vocabSize), device);

            // 4. Configure Optimizers and Loss Functions
            var optimizer = new Adam(learningRate: 0.01);
            var lossFunction = new MSE();

            Console.WriteLine("\n--- Starting Training Loop (10 Epochs) ---");
            for (int epoch = 1; epoch <= 10; epoch++)
            {
                // Clear any leftover gradient states
                optimizer.ZeroGrad(model.Parameters());

                // Forward execution pass
                ITensor outputLogits = model.Forward(inputTensor);

                // Compute output loss against our targets
                ITensor loss = lossFunction.Forward(outputLogits, targetsTensor);

                // Run reverse-mode automatic differentiation
                loss.Backward();

                // Step optimizer to update weights in-place
                optimizer.Step(model.Parameters());

                Console.WriteLine($"Epoch {epoch:D2}/10 | Training Loss (MSE): {loss.ToScalar():F6}");
            }

            // 5. Test Inference Predictions on New Sequence Context
            model.Eval(); // Switch model to evaluation state (disables dropout, freezes capacities)
            Console.WriteLine("\n--- Testing Model Inference ---");

            // Define custom test input: [1, 2, 3, 4, 5]
            float[] testInputSequence = new float[] { 1f, 2f, 3f, 4f, 5f };
            ITensor testTensor = Tensor.FromArray(testInputSequence, new TensorShape(1, maxSeqLen), device);

            // Execute next token generation
            ITensor predictedTokenTensor = model.GenerateNextToken(testTensor);
            int predictedTokenId = (int)predictedTokenTensor.ToArray()[0];

            Console.WriteLine($"Input Context sequence: [{string.Join(", ", testInputSequence)}]");
            Console.WriteLine($"Predicted Next Token ID : {predictedTokenId}");

            // 6. Save Model Weights utilizing SafetensorsSerializer
            string modelFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "kimi_k3_demo.safetensors");
            Console.WriteLine($"\nSaving model weights to disk: '{modelFilePath}'...");

            SaveModelWeights(model, modelFilePath);

            // 7. Load Model Weights back into a Fresh Architecture Instance
            Console.WriteLine("Instantiating a new, untrained Kimi K3 Model instance...");
            var freshModel = new KimiK3(
                vocabSize,
                dModel,
                nHeads,
                numLayers,
                numExperts,
                activeExperts,
                expertCapacity,
                maxSeqLen,
                device
            );

            Console.WriteLine("Loading saved weights back into fresh model...");
            LoadModelWeights(freshModel, modelFilePath, device);

            // Verify prediction consistency on the reloaded model
            freshModel.Eval();
            ITensor freshPredictedTokenTensor = freshModel.GenerateNextToken(testTensor);
            int reloadedPredictedTokenId = (int)freshPredictedTokenTensor.ToArray()[0];

            Console.WriteLine($"Reloaded Model Prediction Outcome: {reloadedPredictedTokenId}");

            if (predictedTokenId == reloadedPredictedTokenId)
            {
                Console.WriteLine("\nSuccess: Weight Saving, Loading, and Prediction Parity Confirmed!");
            }
            else
            {
                Console.WriteLine("\nError: Reloaded model prediction diverges. Check file serialization context.");
            }
        }

        #region Serialization Helper Utilities

        /// <summary>
        /// Collects model parameters, assigns structured safe keys, and serializes them using the Safetensors protocol.
        /// </summary>
        private static void SaveModelWeights(IModel model, string filePath)
        {
            var dict = new Dictionary<string, ITensor>();
            var modelParams = model.Parameters().ToList();

            for (int i = 0; i < modelParams.Count; i++)
            {
                // Generate deterministic key mapping
                dict[$"kimi_k3_parameter_{i}"] = modelParams[i];
            }

            // Write to file stream
            SafetensorsSerializer.Save(filePath, dict);
        }

        /// <summary>
        /// Reads serialized weights from a Safetensors file and applies them back into the model's parameters in-place.
        /// </summary>
        private static void LoadModelWeights(IModel model, string filePath, Device device)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("Target weight serialization file not found.", filePath);

            var loadedTensors = SafetensorsSerializer.Load(filePath, device);
            var modelParams = model.Parameters().ToList();

            for (int i = 0; i < modelParams.Count; i++)
            {
                string key = $"kimi_k3_parameter_{i}";
                if (loadedTensors.TryGetValue(key, out var loadedTensor))
                {
                    modelParams[i].SetData(loadedTensor.ToArray());
                }
                else
                {
                    throw new InvalidDataException($"Mismatch during weight restoration. Key '{key}' not found in safetensors payload.");
                }
            }
        }

        #endregion
    }
}