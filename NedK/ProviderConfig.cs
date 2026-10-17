// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// Project:      ArborNet NCST + TIR Enterprise Cognitive Engine
// Description:  NCST Cognitive Engine, spanning far beyond simple text chat into systems engineering, hardware telemetry, cybersecurity, and autonomous machine learning workflows
// License:      MIT License
// -----------------------------------------------------------------------------------------
namespace NedK
{
    public class ProviderConfig
    {
        public string ProviderName { get; set; } = string.Empty;
        public string ApiKey { get; set; } = string.Empty;
        public string EndpointUrl { get; set; } = string.Empty;
        public string SelectedModel { get; set; } = string.Empty;
        public List<string> AvailableModels { get; set; } = new List<string>();
        public bool IsEnabled { get; set; }
    }
}