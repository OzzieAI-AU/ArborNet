using System;
using System.Collections.Generic;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Tensors;
using ArborNet.Core.Devices;
using ArborNet.Core.Models;

namespace ArborNet.Trainers
{
    /// <summary>
    /// World-class LightningModule - the recommended base for all models in ArborNet.
    /// Provides training/validation/test hooks, optimizer configuration, and full autograd support.
    /// </summary>
    public abstract class LightningModule : BaseModel
    {
        /// <summary>
        /// Gets the configured optimizer instance used for training.
        /// </summary>
        protected IOptimizer? Optimizer { get; private set; }

        /// <summary>
        /// Gets the configured loss function used for computing training loss.
        /// </summary>
        protected ILoss? LossFn { get; private set; }

        /// <summary>
        /// Gets the current training epoch index (zero-based).
        /// </summary>
        protected int CurrentEpoch { get; private set; }

        /// <summary>
        /// Gets or sets the current batch index within the epoch.
        /// </summary>
        internal int CurrentBatch { get; set; }

        /// <summary>
        /// Configure optimizer(s). Called automatically by Trainer.
        /// </summary>
        /// <returns>The optimizer instance(s) to use for training.</returns>
        public abstract IOptimizer ConfigureOptimizers();

        /// <summary>
        /// Define loss function. Can be overridden per-module.
        /// </summary>
        /// <returns>The loss function instance to use during training.</returns>
        public virtual ILoss ConfigureLoss() => new Losses.MSE();

        /// <summary>
        /// Training step - returns the loss.
        /// </summary>
        /// <param name="batch">The input batch tensor containing data and labels.</param>
        /// <param name="batchIdx">The zero-based index of the batch within the epoch.</param>
        /// <returns>The scalar loss tensor computed for this batch.</returns>
        public abstract ITensor TrainingStep(ITensor batch, int batchIdx);

        /// <summary>
        /// Validation step - returns the loss (or metrics).
        /// </summary>
        /// <param name="batch">The input batch tensor containing data and labels.</param>
        /// <param name="batchIdx">The zero-based index of the batch within the epoch.</param>
        /// <returns>The scalar loss tensor (or metrics) computed for this batch.</returns>
        public virtual ITensor ValidationStep(ITensor batch, int batchIdx)
            => TrainingStep(batch, batchIdx); // default = same as training

        /// <summary>
        /// Test step.
        /// </summary>
        /// <param name="batch">The input batch tensor containing data and labels.</param>
        /// <param name="batchIdx">The zero-based index of the batch within the epoch.</param>
        /// <returns>The scalar loss tensor (or metrics) computed for this batch.</returns>
        public virtual ITensor TestStep(ITensor batch, int batchIdx)
            => TrainingStep(batch, batchIdx);

        /// <summary>
        /// Called before training begins.
        /// </summary>
        public virtual void OnTrainStart() { }

        /// <summary>
        /// Called after training finishes.
        /// </summary>
        public virtual void OnTrainEnd() { }

        /// <summary>
        /// Called at the start of each epoch.
        /// </summary>
        /// <param name="epoch">The zero-based index of the current epoch.</param>
        public virtual void OnEpochStart(int epoch) => CurrentEpoch = epoch;

        /// <summary>
        /// Called at the end of each epoch.
        /// </summary>
        /// <param name="epoch">The zero-based index of the current epoch.</param>
        public virtual void OnEpochEnd(int epoch) { }

        /// <summary>
        /// Called by the Trainer to set up the module before training.
        /// </summary>
        /// <param name="optimizer">The optimizer instance to configure for training.</param>
        /// <param name="loss">The optional loss function; falls back to <see cref="ConfigureLoss"/> if null.</param>
        internal void Setup(IOptimizer optimizer, ILoss? loss = null)
        {
            Optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
            LossFn = loss ?? ConfigureLoss();
            OnTrainStart();
        }

        /// <summary>
        /// Helper to log metrics to the console.
        /// </summary>
        /// <param name="name">The name of the metric being logged.</param>
        /// <param name="value">The value of the metric.</param>
        public void Log(string name, float value)
        {
            Console.WriteLine($"[Epoch {CurrentEpoch} | Batch {CurrentBatch}] {name}: {value:F6}");
        }
    }
}