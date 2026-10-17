// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// Project:      ArborNet NCST + TIR Enterprise Cognitive Engine
// Description:  NCST Cognitive Engine, spanning far beyond simple text chat into systems engineering, hardware telemetry, cybersecurity, and autonomous machine learning workflows
// License:      MIT License
// -----------------------------------------------------------------------------------------
namespace NedK
{   
    
    public class AppSettingsModel
    {
        public List<ProviderConfig> Providers { get; set; } = new List<ProviderConfig>();
        public string ActiveProviderName { get; set; } = "Google Gemini";
        public string DefaultWorkspaceDirectory { get; set; } = string.Empty;
        public string EmpiricalTargetProject { get; set; } = string.Empty;
        public string EmpiricalReadOnlySource { get; set; } = string.Empty;
        public bool EnableTelemetry { get; set; } = true;

        // Persistent Fields for Studio Tabs
        public string CreatePrompt { get; set; } = "Build a high performance C# mathematical tensor library specializing in vector math manipulation.";
        public string CreateRefineInstruction { get; set; } = "Ensure comprehensive line-by-line documentation, SIMD acceleration, and unit tests.";
        public int CreateMaxPasses { get; set; } = 10;

        public string PerfectPrompt { get; set; } = "Perform a comprehensive structural perfection sweep: enforce full thread-safety, non-blocking async, robust error handling, memory efficiency, and comprehensive line-by-line XML documentation.";
        public string PerfectRefineInstruction { get; set; } = "Prioritize ConcurrentDictionary, lock-free Interlocked, cancellation token propagation, and zero deadlocks.";
        public int PerfectMaxPasses { get; set; } = 10;

        public string ExperimentPrompt { get; set; } = "Iterate using ArborNet and TVLEngine to optimize neural layer weights, gradient flow, and activation functions for high-accuracy pattern recognition.";
        public string ExperimentRefineInstruction { get; set; } = "Incorporate dynamic learning rate schedules, LeakyReLU activations, and TVLEngine memory lookups.";
        public string ExperimentRefLib { get; set; } = "ArborNet";
        public int ExperimentGenerations { get; set; } = 20;
    }
}