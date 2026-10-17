// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// Project:      ArborNet NCST + TIR Enterprise Cognitive Engine
// Description:  NCST Cognitive Engine, spanning far beyond simple text chat into systems engineering, hardware telemetry, cybersecurity, and autonomous machine learning workflows
// License:      MIT License
// -----------------------------------------------------------------------------------------
namespace NedK
{


using System.Text.Json;


    public static class AppSettingsManager
    {
        private static readonly string SettingsFilePath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "AxiomCODAR",
            "axiom_settings.json");

        public static ProviderConfig GetActiveProvider(AppSettingsModel settings)
        {
            if (settings.Providers == null || settings.Providers.Count == 0)
            {
                return new ProviderConfig { ProviderName = "Ollama", IsEnabled = true, SelectedModel = "openai/gpt-4o" };
            }

            var match = settings.Providers.FirstOrDefault(p => p.ProviderName.Equals(settings.ActiveProviderName, StringComparison.OrdinalIgnoreCase) && p.IsEnabled);
            return match ?? settings.Providers.FirstOrDefault(p => p.IsEnabled) ?? settings.Providers[0];
        }

        public static AppSettingsModel LoadSettings()
        {
            AppSettingsModel settings = new AppSettingsModel();
            try
            {
                if (File.Exists(SettingsFilePath))
                {
                    string json = File.ReadAllText(SettingsFilePath);
                    var loaded = JsonSerializer.Deserialize<AppSettingsModel>(json);
                    if (loaded != null && loaded.Providers != null && loaded.Providers.Count > 0)
                    {
                        settings = loaded;
                    }
                }
            }
            catch (Exception ex)
            {
                //System.Windows.MessageBox.Show($"Error loading settings: {ex.Message}", "Settings Warning", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Warning);
            }

            EnsureDefaultProviders(settings);
            return settings;
        }

        public static void SaveSettings(AppSettingsModel settings)
        {
            try
            {
                string? dir = Path.GetDirectoryName(SettingsFilePath);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }

                string json = JsonSerializer.Serialize(settings, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(SettingsFilePath, json);
            }
            catch (Exception ex)
            {
                //System.Windows.MessageBox.Show($"Error saving settings: {ex.Message}", "Settings Error", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Error);
            }
        }

        private static void EnsureDefaultProviders(AppSettingsModel settings)
        {
            if (settings.Providers == null) settings.Providers = new List<ProviderConfig>();

            MergeProvider(settings, "OpenRouter", "https://openrouter.ai/api/v1", new List<string> { "apodex/apodex-1.1-mini:free", "google/gemini-2.0-flash-001", "anthropic/claude-3.5-sonnet", "openai/gpt-4o", "meta-llama/llama-3.3-70b-instruct" });
            MergeProvider(settings, "Google Gemini", "https://generativelanguage.googleapis.com/v1beta", new List<string> { "gemini-3.8-flash", "gemini-3.7-flash", "gemini-flash-latest", "gemini-pro-latest", "gemini-2.5-pro" });
            MergeProvider(settings, "OpenAI", "https://api.openai.com/v1", new List<string> { "gpt-4o", "gpt-4o-mini", "o1-preview" });
            MergeProvider(settings, "Anthropic", "https://api.anthropic.com/v1", new List<string> { "claude-3-5-sonnet-20241022", "claude-3-opus-20240229" });
            MergeProvider(settings, "Ollama", "http://localhost:11434/api", new List<string> { "gemma4:e2b", "qwen3:8b", "llama3", "mistral" });
        }

        private static void MergeProvider(AppSettingsModel settings, string name, string defaultEndpoint, List<string> defaultModels)
        {
            var existing = settings.Providers.Find(p => p.ProviderName.Equals(name, StringComparison.OrdinalIgnoreCase));
            if (existing == null)
            {
                settings.Providers.Add(new ProviderConfig
                {
                    ProviderName = name,
                    EndpointUrl = defaultEndpoint,
                    AvailableModels = defaultModels,
                    SelectedModel = defaultModels.Count > 0 ? defaultModels[0] : string.Empty,
                    IsEnabled = name == "Google Gemini" || name == "OpenRouter"
                });
            }
            else
            {
                if (string.IsNullOrWhiteSpace(existing.EndpointUrl)) existing.EndpointUrl = defaultEndpoint;
                if (existing.AvailableModels == null || existing.AvailableModels.Count == 0) existing.AvailableModels = defaultModels;
                if (string.IsNullOrWhiteSpace(existing.SelectedModel) && existing.AvailableModels.Count > 0) existing.SelectedModel = existing.AvailableModels[0];

                // Auto-heal known decommissioned/dead models
                if (existing.SelectedModel.Equals("stealth/space-bunny-alpha", StringComparison.OrdinalIgnoreCase))
                {
                    existing.SelectedModel = defaultModels[0];
                }
                if (existing.SelectedModel.Equals("gemini-1.5-flash", StringComparison.OrdinalIgnoreCase) ||
                    existing.SelectedModel.Equals("gemini-2.5-flash", StringComparison.OrdinalIgnoreCase) ||
                    existing.SelectedModel.Equals("gemini-3.1-pro-preview", StringComparison.OrdinalIgnoreCase))
                {
                    existing.SelectedModel = "gemini-3.8-flash";
                }
            }
        }

        public static async Task<List<string>> FetchRemoteModelsAsync(string providerName, string apiKey, string endpoint)
        {
            var models = new List<string>();
            try
            {
                using var client = new HttpClient();
                client.Timeout = TimeSpan.FromSeconds(15);

                if (providerName.Equals("OpenRouter", StringComparison.OrdinalIgnoreCase))
                {
                    client.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", apiKey);
                    var response = await client.GetAsync("https://openrouter.ai/api/v1/models?limit=200");
                    if (response.IsSuccessStatusCode)
                    {
                        string json = await response.Content.ReadAsStringAsync();
                        using var doc = JsonDocument.Parse(json);
                        if (doc.RootElement.TryGetProperty("data", out var dataArr))
                        {
                            foreach (var el in dataArr.EnumerateArray())
                            {
                                if (el.TryGetProperty("id", out var idProp))
                                {
                                    string? id = idProp.GetString();
                                    if (!string.IsNullOrEmpty(id)) models.Add(id);
                                }
                            }
                        }
                    }
                }
                else if (providerName.Equals("Google Gemini", StringComparison.OrdinalIgnoreCase))
                {
                    string cleanEndpoint = endpoint.TrimEnd('/');
                    if (!cleanEndpoint.EndsWith("/v1beta", StringComparison.OrdinalIgnoreCase))
                    {
                        cleanEndpoint = $"{cleanEndpoint}/v1beta";
                    }

                    var response = await client.GetAsync($"{cleanEndpoint}/models?key={apiKey}");
                    if (response.IsSuccessStatusCode)
                    {
                        string json = await response.Content.ReadAsStringAsync();
                        using var doc = JsonDocument.Parse(json);
                        if (doc.RootElement.TryGetProperty("models", out var modelsArr))
                        {
                            foreach (var el in modelsArr.EnumerateArray())
                            {
                                if (el.TryGetProperty("name", out var nameProp))
                                {
                                    string? name = nameProp.GetString();
                                    if (!string.IsNullOrEmpty(name))
                                    {
                                        models.Add(name.Replace("models/", ""));
                                    }
                                }
                            }
                        }
                    }
                }
                else if (providerName.Equals("Ollama", StringComparison.OrdinalIgnoreCase))
                {
                    string baseUri = endpoint.EndsWith("/") ? endpoint.Substring(0, endpoint.Length - 1) : endpoint;
                    var response = await client.GetAsync($"{baseUri}/tags");
                    if (response.IsSuccessStatusCode)
                    {
                        string json = await response.Content.ReadAsStringAsync();
                        using var doc = JsonDocument.Parse(json);
                        if (doc.RootElement.TryGetProperty("models", out var modelsArr))
                        {
                            foreach (var el in modelsArr.EnumerateArray())
                            {
                                if (el.TryGetProperty("name", out var nameProp))
                                {
                                    string? name = nameProp.GetString();
                                    if (!string.IsNullOrEmpty(name)) models.Add(name);
                                }
                            }
                        }
                    }
                }
            }
            catch
            {
                // Fallback on network failure
            }
            return models;
        }
    }
}