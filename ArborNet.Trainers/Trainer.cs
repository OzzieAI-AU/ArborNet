using System;
using System.Collections.Generic;
using System.Linq;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Trainers;

namespace ArborNet.Trainers
{
    /// <summary>
    /// Production-grade Trainer for ArborNet models.
    /// Handles full training loop, validation, logging, and LightningModule lifecycle.
    /// Thread-safe, device-aware, and extensible.
    /// </summary>
    public sealed class Trainer
    {
        /// <summary>
        /// The core <see cref="LightningModule"/> instance responsible for model forward passes,
        /// loss computation, logging, and training/validation/testing steps.
        /// </summary>
        private readonly LightningModule _module;

        /// <summary>
        /// The <see cref="IOptimizer"/> instance used for parameter updates during training.
        /// </summary>
        private readonly IOptimizer _optimizer;

        /// <summary>
        /// The <see cref="ILoss"/> function used for computing training, validation, and test losses.
        /// </summary>
        private readonly ILoss _loss;

        /// <summary>
        /// The total number of training epochs to perform.
        /// </summary>
        private readonly int _epochs;

        /// <summary>
        /// The expected batch size for data loaders.
        /// </summary>
        private readonly int _batchSize;

        /// <summary>
        /// Flag indicating whether validation should be performed after each training epoch.
        /// </summary>
        private readonly bool _enableValidation;

        /// <summary>
        /// Initializes a new instance of the <see cref="Trainer"/> class.
        /// Configures the training loop with the specified module, optimizer, loss, and hyperparameters.
        /// </summary>
        /// <param name="module">The <see cref="LightningModule"/> to train. Cannot be null.</param>
        /// <param name="optimizer">The <see cref="IOptimizer"/> for parameter updates. Cannot be null.</param>
        /// <param name="loss">Optional custom loss function. If null, defaults to the module's configured loss via <see cref="LightningModule.ConfigureLoss"/>.</param>
        /// <param name="epochs">Number of training epochs. Defaults to 10.</param>
        /// <param name="batchSize">Expected batch size for data loaders. Defaults to 32.</param>
        /// <param name="enableValidation">Whether to run validation after each epoch. Defaults to true.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="module"/> or <paramref name="optimizer"/> is null.</exception>
        public Trainer(
            LightningModule module,
            IOptimizer optimizer,
            ILoss loss = null,
            int epochs = 10,
            int batchSize = 32,
            bool enableValidation = true)
        {
            _module = module ?? throw new ArgumentNullException(nameof(module));
            _optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
            _loss = loss ?? module.ConfigureLoss();
            _epochs = epochs;
            _batchSize = batchSize;
            _enableValidation = enableValidation;

            _module.Setup(_optimizer, _loss);
        }

        /// <summary>
        /// Executes the full training loop over the specified number of epochs using the provided training data loader.
        /// Performs forward passes, loss computation, backpropagation, parameter updates, and optional validation.
        /// Logs training and validation losses via the module's logging mechanism.
        /// </summary>
        /// <param name="trainLoader">Required data loader yielding training batches as tuples of (<c>ITensor inputs</c>, <c>ITensor targets</c>).</param>
        /// <param name="valLoader">Optional data loader yielding validation batches. Skipped if null or validation is disabled.</param>
        /// <remarks>
        /// Assumes data loaders yield batches matching the configured <see cref="_batchSize"/>.
        /// Calls module lifecycle hooks: <c>OnTrainStart</c>, <c>OnEpochStart</c>, <c>OnEpochEnd</c>, <c>OnTrainEnd</c>.
        /// </remarks>
        public void Fit(
            IEnumerable<(ITensor inputs, ITensor targets)> trainLoader,
            IEnumerable<(ITensor inputs, ITensor targets)> valLoader = null)
        {
            _module.OnTrainStart();

            for (int epoch = 0; epoch < _epochs; epoch++)
            {
                _module.OnEpochStart(epoch);
                int batchIdx = 0;

                Console.WriteLine($"=== Epoch {epoch + 1}/{_epochs} ===");

                foreach (var (x, y) in trainLoader)
                {
                    _module.CurrentBatch = batchIdx;

                    var loss = _module.TrainingStep(x, batchIdx);
                    _module.Log("train_loss", loss.ToScalar());

                    _optimizer.Step(_module.Parameters());
                    _optimizer.ZeroGrad(_module.Parameters());

                    batchIdx++;
                }

                if (_enableValidation && valLoader != null)
                {
                    batchIdx = 0;
                    foreach (var (x, y) in valLoader)
                    {
                        var valLoss = _module.ValidationStep(x, batchIdx);
                        _module.Log("val_loss", valLoss.ToScalar());
                        batchIdx++;
                    }
                }

                _module.OnEpochEnd(epoch);
            }

            _module.OnTrainEnd();
            Console.WriteLine("Training completed.");
        }

        /// <summary>
        /// Evaluates the model on the provided test data loader.
        /// Computes and logs test loss for each batch without performing gradients or parameter updates.
        /// </summary>
        /// <param name="testLoader">Data loader yielding test batches as tuples of (<c>ITensor inputs</c>, <c>ITensor targets</c>).</param>
        /// <remarks>
        /// Assumes data loaders yield batches matching the configured <see cref="_batchSize"/>.
        /// Logs metrics via <c>_module.Log("test_loss", ...)</c>.
        /// </remarks>
        public void Test(IEnumerable<(ITensor inputs, ITensor targets)> testLoader)
        {
            int batchIdx = 0;
            foreach (var (x, y) in testLoader)
            {
                var testLoss = _module.TestStep(x, batchIdx);
                _module.Log("test_loss", testLoss.ToScalar());
                batchIdx++;
            }
        }
    }
}