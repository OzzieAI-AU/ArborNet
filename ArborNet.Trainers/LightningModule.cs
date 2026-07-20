// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Trainers
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Tensors;
    using ArborNet.Core.Devices;
    using ArborNet.Core.Models;
    /// <summary>
    /// Represents a world-class LightningModule, serving as the recommended base class for all machine learning models in ArborNet.
    /// It provides standardized hooks for training, validation, and testing phases, manages model optimization and loss evaluation,
    /// and integrates fully with the library's autograd and hardware acceleration capabilities.
    /// </summary>

    #endregion

    public abstract class LightningModule : BaseModel
    {
        /// <summary>
        /// Gets the configured optimizer instance utilized for training and parameter optimization.
        /// </summary>
        /// <value>The <see cref="IOptimizer"/> used to update model parameters, or <c>null</c> if not yet initialized.</value>
        protected IOptimizer? Optimizer { get; private set; }
        /// <summary>
        /// Gets the configured loss function utilized for computing training and validation loss.
        /// </summary>
        /// <value>The <see cref="ILoss"/> instance used for error calculation, or <c>null</c> if not yet initialized.</value>

        protected ILoss? LossFn { get; private set; }
        /// <summary>
        /// Gets the zero-based index of the current training epoch.
        /// </summary>
        /// <value>An integer representing the current training epoch.</value>

        protected int CurrentEpoch { get; private set; }
        /// <summary>
        /// Gets or sets the zero-based index of the current batch within the active epoch.
        /// </summary>
        /// <value>An integer representing the active batch index.</value>

        internal int CurrentBatch { get; set; }
        /// <summary>
        /// Configures the optimizer(s) to be utilized during model training.
        /// This abstract method is invoked automatically by the Trainer during the setup phase.
        /// </summary>
        /// <returns>The configured <see cref="IOptimizer"/> instance.</returns>

        public abstract IOptimizer ConfigureOptimizers();
        /// <summary>
        /// Configures and returns the loss function to be utilized during model execution.
        /// This method can be overridden in derived classes to specify a custom loss function.
        /// </summary>
        /// <returns>The <see cref="ILoss"/> instance. Defaults to a new instance of <see cref="Losses.MSE"/>.</returns>

        public virtual ILoss ConfigureLoss() => new Losses.MSE();
        /// <summary>
        /// Executes a single training step using the provided batch and index.
        /// This method must be implemented by derived classes to define the forward pass and loss computation.
        /// </summary>
        /// <param name="batch">The input data batch <see cref="ITensor"/> containing features and target labels.</param>
        /// <param name="batchIdx">The zero-based index of the current batch within the epoch.</param>
        /// <returns>A scalar <see cref="ITensor"/> representing the loss computed for this training batch.</returns>

        public abstract ITensor TrainingStep(ITensor batch, int batchIdx);
        /// <summary>
        /// Executes a single validation step using the provided batch and index.
        /// By default, this method delegates directly to <see cref="TrainingStep(ITensor, int)"/> but can be overridden for custom validation logic.
        /// </summary>
        /// <param name="batch">The validation data batch <see cref="ITensor"/> containing features and target labels.</param>
        /// <param name="batchIdx">The zero-based index of the current validation batch within the validation run.</param>
        /// <returns>A scalar <see cref="ITensor"/> representing validation loss or metrics computed for this batch.</returns>

        public virtual ITensor ValidationStep(ITensor batch, int batchIdx)
    => TrainingStep(batch, batchIdx); // default = same as training
        /// <summary>
        /// Executes a single test step using the provided batch and index.
        /// By default, this method delegates directly to <see cref="TrainingStep(ITensor, int)"/> but can be overridden for custom test evaluation logic.
        /// </summary>
        /// <param name="batch">The test data batch <see cref="ITensor"/> containing features and target labels.</param>
        /// <param name="batchIdx">The zero-based index of the current test batch within the test run.</param>
        /// <returns>A scalar <see cref="ITensor"/> representing test loss or metrics computed for this batch.</returns>

        public virtual ITensor TestStep(ITensor batch, int batchIdx)
    => TrainingStep(batch, batchIdx);
        /// <summary>
        /// A lifecycle hook invoked immediately before the training process begins.
        /// Override this method to perform custom initialization, resource allocation, or state setup.
        /// </summary>

        public virtual void OnTrainStart() { }
        /// <summary>
        /// A lifecycle hook invoked immediately after the training process completes.
        /// Override this method to finalize metrics, save checkpoints, or release resources.
        /// </summary>

        public virtual void OnTrainEnd() { }
        /// <summary>
        /// A lifecycle hook invoked at the beginning of each training epoch.
        /// Updates the current epoch tracker and can be overridden for epoch-specific setup.
        /// </summary>
        /// <param name="epoch">The zero-based index of the epoch that is starting.</param>

        public virtual void OnEpochStart(int epoch) => CurrentEpoch = epoch;
        /// <summary>
        /// A lifecycle hook invoked at the end of each training epoch.
        /// Override this method to perform end-of-epoch operations such as logging summary metrics or performing scheduler steps.
        /// </summary>
        /// <param name="epoch">The zero-based index of the epoch that is ending.</param>

        public virtual void OnEpochEnd(int epoch) { }
        /// <summary>
        /// Initializes the module by assigning the optimizer, establishing the loss function, and executing startup hooks.
        /// This is called internally by the Trainer before beginning training execution.
        /// </summary>
        /// <param name="optimizer">The <see cref="IOptimizer"/> to associate with this module.</param>
        /// <param name="loss">The optional <see cref="ILoss"/> instance; falls back to <see cref="ConfigureLoss"/> if <c>null</c>.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="optimizer"/> is <c>null</c>.</exception>

        internal void Setup(IOptimizer optimizer, ILoss? loss = null)
        {
            Optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
            LossFn = loss ?? ConfigureLoss();
            OnTrainStart();
        }
        /// <summary>
        /// Helper utility to log metrics to the standard console output, prefixed with current epoch and batch markers.
        /// </summary>
        /// <param name="name">The name of the metric to log.</param>
        /// <param name="value">The floating-point value of the metric.</param>

        public void Log(string name, float value)
        {
            Console.WriteLine($"[Epoch {CurrentEpoch} | Batch {CurrentBatch}] {name}: {value:F6}");
        }
    }
}