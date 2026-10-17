// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// Project:      ArborNet NCST + TIR Enterprise Cognitive Engine
// Description:  NCST Cognitive Engine, spanning far beyond simple text chat into systems engineering, hardware telemetry, cybersecurity, and autonomous machine learning workflows
// License:      MIT License
// -----------------------------------------------------------------------------------------
namespace NedK
{

    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Net.Http;
    using System.Net.Http.Headers;
    using System.Text;
    using System.Text.Json;
    using System.Text.Json.Serialization;
    using System.Threading.Tasks;


    public class MultiProviderClient
    {
        private ProviderConfig _config;
        private readonly HttpClient _http;

        public Action<string, string>? Logger { get; set; }
        public string ActiveModel => string.IsNullOrWhiteSpace(_config?.SelectedModel) ? "gemini-3.8-flash" : _config.SelectedModel;
        public ProviderConfig ActiveConfig => _config;

        public MultiProviderClient(string apiKey)
        {
            _config = new ProviderConfig
            {
                ProviderName = "Google Gemini",
                ApiKey = apiKey,
                EndpointUrl = "https://generativelanguage.googleapis.com/v1beta",
                SelectedModel = "gemini-3.8-flash",
                IsEnabled = true
            };
            _http = new HttpClient { Timeout = TimeSpan.FromMinutes(120) };
        }

        public MultiProviderClient(ProviderConfig? config)
        {
            _config = config ?? new ProviderConfig
            {
                ProviderName = "Google Gemini",
                ApiKey = "",
                EndpointUrl = "https://generativelanguage.googleapis.com/v1beta",
                SelectedModel = "gemini-3.8-flash",
                IsEnabled = true
            };
            // Change line 31 & 45 in MultiProviderClient.cs:
            _http = new HttpClient { Timeout = TimeSpan.FromSeconds(120) };
        }

        public async Task<string> GenerateResponseAsync(List<ChatMessage> history)
        {
            try
            {
                return await ExecuteProviderWithModelFallbackAsync(_config, history);
            }
            catch (Exception ex)
            {
                string shortError = ex.Message.Trim();
                if (shortError.Length > 120) shortError = shortError.Substring(0, 117) + "...";
                string errorBody = $"Active Provider: {_config.ProviderName}\nEndpoint: {_config.EndpointUrl}\nModel: {_config.SelectedModel}\nDiagnostics: {ex.Message}\nTimestamp: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC";
                Logger?.Invoke($"Provider Error [{_config.ProviderName}]: {shortError}\n{errorBody}", "PROVIDER_ERROR");

                var backupProviders = FindBackupProviders(_config.ProviderName);
                if (backupProviders.Count == 0)
                {
                    throw new Exception($"Provider Error [{_config.ProviderName}] ({ex.Message}), and no other enabled backup providers are configured in settings.");
                }

                List<string> errorHistory = new List<string> { $"{_config.ProviderName}: {ex.Message.Trim()}" };

                foreach (var backup in backupProviders)
                {
                    Logger?.Invoke($"[FAILOVER] Switching from '{_config.ProviderName}' to backup '{backup.ProviderName}' ({backup.SelectedModel})...", "WARN");
                    try
                    {
                        string result = await ExecuteProviderWithModelFallbackAsync(backup, history);

                        _config = backup;
                        Logger?.Invoke($"[FAILOVER SUCCESS] Connected to '{backup.ProviderName}' ({backup.SelectedModel}). Continuing task.", "SUCCESS");
                        return result;
                    }
                    catch (Exception bEx)
                    {
                        string bShort = bEx.Message.Trim();
                        if (bShort.Length > 120) bShort = bShort.Substring(0, 117) + "...";
                        string bBody = $"Backup Provider: {backup.ProviderName}\nEndpoint: {backup.EndpointUrl}\nModel: {backup.SelectedModel}\nDiagnostics: {bEx.Message}\nTimestamp: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC";
                        Logger?.Invoke($"Provider Error [{backup.ProviderName}]: {bShort}\n{bBody}", "PROVIDER_ERROR");
                        errorHistory.Add($"{backup.ProviderName}: {bEx.Message.Trim()}");
                    }
                }

                throw new Exception($"All AI neural providers failed with Provider Errors:\n" + string.Join("\n", errorHistory));
            }
        }

        private async Task<string> ExecuteProviderWithModelFallbackAsync(ProviderConfig config, List<ChatMessage> history)
        {
            try
            {
                return await ExecuteProviderRequestAsync(config, history);
            }
            catch (Exception ex) when (IsModelOrEndpointNotFoundError(ex))
            {
                var fallbacks = GetFallbackModelsForProvider(config.ProviderName);
                foreach (var fallbackModel in fallbacks)
                {
                    if (fallbackModel.Equals(config.SelectedModel, StringComparison.OrdinalIgnoreCase)) continue;

                    Logger?.Invoke($"[MODEL FALLBACK] Model '{config.SelectedModel}' unavailable on {config.ProviderName}. Retrying with '{fallbackModel}'...", "WARN");
                    var tempConfig = new ProviderConfig
                    {
                        ProviderName = config.ProviderName,
                        ApiKey = config.ApiKey,
                        EndpointUrl = config.EndpointUrl,
                        SelectedModel = fallbackModel,
                        IsEnabled = config.IsEnabled
                    };

                    try
                    {
                        string result = await ExecuteProviderRequestAsync(tempConfig, history);
                        config.SelectedModel = fallbackModel;
                        Logger?.Invoke($"[MODEL FALLBACK SUCCESS] '{config.ProviderName}' accepted model '{fallbackModel}'.", "SUCCESS");
                        return result;
                    }
                    catch
                    {
                        // Try next fallback model
                    }
                }

                throw;
            }
        }

        private bool IsModelOrEndpointNotFoundError(Exception ex)
        {
            string msg = ex.Message.ToLowerInvariant();
            return msg.Contains("404") ||
                   msg.Contains("not found") ||
                   msg.Contains("no endpoints found") ||
                   msg.Contains("not_found") ||
                   msg.Contains("no longer available") ||
                   msg.Contains("does not exist") ||
                   msg.Contains("invalid model") ||
                   msg.Contains("model_not_found");
        }

        private List<string> GetFallbackModelsForProvider(string providerName)
        {
            string p = (providerName ?? "").ToLowerInvariant();
            if (p.Contains("openrouter"))
            {
                return new List<string>
                {
                    "apodex/apodex-1.1-mini:free",
                    "liquid/lfm-2.5-2.6b:free",
                    "meta-llama/llama-3.3-70b-instruct",
                    "anthropic/claude-3.5-sonnet",
                    "google/gemini-2.0-flash-001",
                    "openai/gpt-4o"
                };
            }
            if (p.Contains("gemini"))
            {
                return new List<string>
                {
                    "gemini-3.8-flash",
                    "gemini-3.7-flash",
                    "gemini-flash-latest",
                    "gemini-pro-latest",
                    "gemini-2.5-pro"
                };
            }
            if (p.Contains("openai"))
            {
                return new List<string> { "gpt-4o", "gpt-4o-mini", "gpt-4-turbo" };
            }
            if (p.Contains("anthropic"))
            {
                return new List<string> { "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307" };
            }
            if (p.Contains("ollama"))
            {
                return new List<string> { "gemma4:e2b", "qwen3:8b", "llama3", "mistral" };
            }
            return new List<string>();
        }

        private List<ProviderConfig> FindBackupProviders(string currentProviderName)
        {
            var backups = new List<ProviderConfig>
            {
                new ProviderConfig
                {
                    ProviderName = "Ollama",
                    EndpointUrl = "http://localhost:11434/api",
                    SelectedModel = "gemma4:e2b",
                    IsEnabled = true
                },
                new ProviderConfig
                {
                    ProviderName = "Google Gemini",
                    ApiKey = Environment.GetEnvironmentVariable("GEMINI_API_KEY") ?? "",
                    EndpointUrl = "https://generativelanguage.googleapis.com/v1beta",
                    SelectedModel = "gemini-3.8-flash",
                    IsEnabled = !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("GEMINI_API_KEY"))
                }
            };

            return backups
                .Where(p => p.IsEnabled && !p.ProviderName.Equals(currentProviderName, StringComparison.OrdinalIgnoreCase))
                .ToList();
        }

        private async Task<string> ExecuteProviderRequestAsync(ProviderConfig config, List<ChatMessage> history)
        {
            string provider = (config.ProviderName ?? "").ToLowerInvariant();
            string endpoint = (config.EndpointUrl ?? "").TrimEnd('/');
            string apiKey = config.ApiKey ?? "";
            string model = string.IsNullOrWhiteSpace(config.SelectedModel) ? "gpt-4o" : config.SelectedModel;

            if (provider.Contains("openrouter"))
            {
                string targetUrl = string.IsNullOrWhiteSpace(endpoint) ? "https://openrouter.ai/api/v1/chat/completions" : $"{endpoint}/chat/completions";
                return await SendOpenAICompatibleRequestAsync(targetUrl, apiKey, model, history, true, config.ProviderName);
            }
            else if (provider.Contains("openai"))
            {
                string targetUrl = string.IsNullOrWhiteSpace(endpoint) ? "https://api.openai.com/v1/chat/completions" : $"{endpoint}/chat/completions";
                return await SendOpenAICompatibleRequestAsync(targetUrl, apiKey, model, history, false, config.ProviderName);
            }
            else if (provider.Contains("ollama"))
            {
                string targetUrl = string.IsNullOrWhiteSpace(endpoint) ? "http://localhost:11434/api/chat" : $"{endpoint}/chat";
                return await SendOllamaRequestAsync(targetUrl, model, history);
            }
            else if (provider.Contains("anthropic"))
            {
                string targetUrl = string.IsNullOrWhiteSpace(endpoint) ? "https://api.anthropic.com/v1/messages" : $"{endpoint}/messages";
                return await SendAnthropicRequestAsync(targetUrl, apiKey, model, history);
            }
            else
            {
                string targetEndpoint = string.IsNullOrWhiteSpace(endpoint) ? "https://generativelanguage.googleapis.com/v1beta" : endpoint;
                return await SendGeminiRequestAsync(targetEndpoint, apiKey, model, history);
            }
        }

        private async Task<string> SendOpenAICompatibleRequestAsync(string url, string apiKey, string model, List<ChatMessage> history, bool isOpenRouter, string providerName)
        {
            var requestMessage = new HttpRequestMessage(HttpMethod.Post, url);
            if (!string.IsNullOrWhiteSpace(apiKey))
            {
                requestMessage.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
            }

            if (isOpenRouter)
            {
                requestMessage.Headers.Add("HTTP-Referer", "https://axiomcodar.local");
                requestMessage.Headers.Add("X-Title", "Axiom CODAR WPF");
            }

            var payload = new
            {
                model = model,
                messages = history.Select(h => new { role = h.Role == "model" ? "assistant" : h.Role, content = h.Content }).ToList()
            };

            requestMessage.Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");
            var response = await _http.SendAsync(requestMessage);
            string resJson = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
                throw new Exception($"Provider Error [{providerName}] ({(int)response.StatusCode} {response.StatusCode}): {resJson}");

            using var doc = JsonDocument.Parse(resJson);

            // SAFE EXTRACTION USING TryGetProperty TO PREVENT KeyNotFoundException
            if (doc.RootElement.TryGetProperty("choices", out var choices) && choices.GetArrayLength() > 0)
            {
                if (choices[0].TryGetProperty("message", out var msgProp) && msgProp.TryGetProperty("content", out var contentProp))
                {
                    return contentProp.GetString() ?? "";
                }
            }

            return resJson; // Fallback if structure differs
        }

        private async Task<string> SendOllamaRequestAsync(string url, string model, List<ChatMessage> history)
        {
            var payload = new
            {
                model = model,
                messages = history.Select(h => new { role = h.Role == "model" ? "assistant" : h.Role, content = h.Content }).ToList(),
                stream = false
            };

            var content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");
            var response = await _http.PostAsync(url, content);
            string resJson = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
                throw new Exception($"Ollama Error ({(int)response.StatusCode} {response.StatusCode}): {resJson}");

            using var doc = JsonDocument.Parse(resJson);
            if (doc.RootElement.TryGetProperty("message", out var msgProp) && msgProp.TryGetProperty("content", out var contentProp))
            {
                return contentProp.GetString() ?? "";
            }
            return resJson;
        }

        private async Task<string> SendAnthropicRequestAsync(string url, string apiKey, string model, List<ChatMessage> history)
        {
            var requestMessage = new HttpRequestMessage(HttpMethod.Post, url);
            requestMessage.Headers.Add("x-api-key", apiKey);
            requestMessage.Headers.Add("anthropic-version", "2023-06-01");

            var payload = new
            {
                model = model,
                max_tokens = 8000,
                messages = history.Select(h => new { role = h.Role == "model" ? "assistant" : h.Role, content = h.Content }).ToList()
            };

            requestMessage.Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");
            var response = await _http.SendAsync(requestMessage);
            string resJson = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
                throw new Exception($"Anthropic Error ({(int)response.StatusCode} {response.StatusCode}): {resJson}");

            using var doc = JsonDocument.Parse(resJson);
            if (doc.RootElement.TryGetProperty("content", out var contentArr) && contentArr.GetArrayLength() > 0)
            {
                if (contentArr[0].TryGetProperty("text", out var textProp))
                {
                    return textProp.GetString() ?? "";
                }
            }
            return resJson;
        }

        private async Task<string> SendGeminiRequestAsync(string endpoint, string apiKey, string model, List<ChatMessage> history)
        {
            string cleanEndpoint = endpoint.TrimEnd('/');
            if (!cleanEndpoint.EndsWith("/v1beta", StringComparison.OrdinalIgnoreCase) && !cleanEndpoint.Contains("/v1beta/"))
            {
                cleanEndpoint = $"{cleanEndpoint}/v1beta";
            }

            string cleanModel = string.IsNullOrWhiteSpace(model) ? "gemini-3.8-flash" : model;
            if (cleanModel.StartsWith("models/")) cleanModel = cleanModel.Substring(7);

            // Automatically upgrade deprecated models that return 404
            if (cleanModel.Equals("gemini-1.5-flash", StringComparison.OrdinalIgnoreCase) ||
                cleanModel.Equals("gemini-2.5-flash", StringComparison.OrdinalIgnoreCase) ||
                cleanModel.Equals("gemini-3.1-pro-preview", StringComparison.OrdinalIgnoreCase))
            {
                cleanModel = "gemini-3.8-flash";
            }

            var url = $"{cleanEndpoint}/models/{cleanModel}:generateContent?key={apiKey}";

            // Extract system messages for top-level system_instruction per Gemini API spec
            var systemMessages = history.Where(h => h.Role.Equals("system", StringComparison.OrdinalIgnoreCase)).ToList();
            var nonSystemMessages = history.Where(h => !h.Role.Equals("system", StringComparison.OrdinalIgnoreCase)).ToList();

            object? systemInstruction = null;
            if (systemMessages.Count > 0)
            {
                var systemParts = systemMessages.SelectMany(m => m.Parts.Select(p => new { text = p.Text ?? "" })).ToList();
                systemInstruction = new { parts = systemParts };
            }

            // In Gemini contents, only 'user' and 'model' are valid roles
            var formattedContents = nonSystemMessages.Select(h => new
            {
                role = h.Role == "model" || h.Role == "assistant" ? "model" : "user",
                parts = h.Parts.Select(p => new { text = p.Text ?? "" }).ToArray()
            }).ToArray();

            if (formattedContents.Length == 0)
            {
                formattedContents = new[]
                {
                    new { role = "user", parts = new[] { new { text = "Please execute the scientific request." } } }
                };
            }

            var payloadObj = systemInstruction != null
                ? (object)new { system_instruction = systemInstruction, contents = formattedContents }
                : (object)new { contents = formattedContents };

            var content = new StringContent(JsonSerializer.Serialize(payloadObj), Encoding.UTF8, "application/json");

            var response = await _http.PostAsync(url, content);
            string resJson = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
                throw new Exception($"Gemini Error ({(int)response.StatusCode} {response.StatusCode}): {resJson}");

            using var doc = JsonDocument.Parse(resJson);

            // SAFE PARSING CHAIN WITH TryGetProperty TO PREVENT KEYNOTFOUNDEXCEPTIONS
            if (doc.RootElement.TryGetProperty("candidates", out var candidates) && candidates.GetArrayLength() > 0)
            {
                var firstCandidate = candidates[0];
                if (firstCandidate.TryGetProperty("content", out var contentProp) &&
                    contentProp.TryGetProperty("parts", out var partsProp) &&
                    partsProp.GetArrayLength() > 0)
                {
                    var firstPart = partsProp[0];
                    if (firstPart.TryGetProperty("text", out var textProp))
                    {
                        return textProp.GetString() ?? "";
                    }
                }
            }

            // Fallback: Return raw JSON string if structure was filtered or modified
            return resJson;
        }
    }

    public class UniversalMultiProviderClient : MultiProviderClient
    {
        public UniversalMultiProviderClient(ProviderConfig config) : base(config) { }
        public UniversalMultiProviderClient(string apiKey) : base(apiKey) { }
    }

    public class ChatMessage
    {
        [JsonPropertyName("role")]
        public string Role { get; set; }

        [JsonPropertyName("parts")]
        public List<MessagePart> Parts { get; set; } = new();

        [JsonIgnore]
        public string Content => Parts.FirstOrDefault(p => p.Text != null)?.Text ?? "[Media]";

        public ChatMessage(string role, string text)
        {
            Role = role;
            if (!string.IsNullOrEmpty(text)) Parts.Add(new MessagePart { Text = text });
        }
    }

    public class MessagePart
    {
        [JsonPropertyName("text")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string? Text { get; set; }

        [JsonPropertyName("inline_data")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public InlineData? InlineData { get; set; }
    }

    public class InlineData
    {
        [JsonPropertyName("mime_type")]
        public string MimeType { get; set; } = "";

        [JsonPropertyName("data")]
        public string Data { get; set; } = "";
    }
}