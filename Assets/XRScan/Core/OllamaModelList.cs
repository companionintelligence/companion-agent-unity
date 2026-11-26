using System;
using System.Collections.Generic;
using UnityEngine;

namespace XRScan.Core
{
    /// <summary>
    /// The models an Ollama server has, read from its <c>GET /api/tags</c> list, for the settings panel's model row.
    /// <para>
    /// The panel used to offer only the names baked into <see cref="CompanionConfig.llmModelChoices"/>, so a headset
    /// pointed at a server that had none of them (on 2026-11-25 no server on the tailnet had llava:7b,
    /// llama3.2-vision:11b or qwen2.5vl:7b) could not select a model that would answer, and changing the list meant a
    /// rebuild. The panel now asks the server and offers what is there.
    /// </para>
    /// Plain parsing and filtering, free of UI and networking, so it is unit-testable; the panel does the request.
    /// </summary>
    public static class OllamaModelList
    {
        public const string TagsPath = "/api/tags";

        [Serializable]
        private class TagsReply
        {
            public TagEntry[] models;
        }

        [Serializable]
        private class TagEntry
        {
            public string name;
            public string model;
        }

        /// <summary>The model list's URL under an LLM base URL, with or without a trailing slash.</summary>
        public static string TagsUrl(string baseUrl) => (baseUrl ?? string.Empty).TrimEnd('/') + TagsPath;

        /// <summary>
        /// The model names in a <c>/api/tags</c> reply, in the server's order, without duplicates. An empty list for a
        /// server with no models; null for a reply that is not a model list at all (an HTML page, a proxy's error).
        /// </summary>
        public static List<string> Parse(string json)
        {
            if (string.IsNullOrWhiteSpace(json) || json.IndexOf("\"models\"", StringComparison.Ordinal) < 0) return null;

            TagsReply reply;
            try
            {
                reply = JsonUtility.FromJson<TagsReply>(json);
            }
            catch (ArgumentException)
            {
                return null;
            }
            if (reply == null || reply.models == null) return null;

            var names = new List<string>();
            foreach (var entry in reply.models)
            {
                if (entry == null) continue;
                string name = !string.IsNullOrWhiteSpace(entry.name) ? entry.name.Trim()
                    : !string.IsNullOrWhiteSpace(entry.model) ? entry.model.Trim() : null;
                if (name != null && !names.Contains(name)) names.Add(name);
            }
            return names;
        }

        /// <summary>
        /// The models a person can talk to: every name except the reply check's model (any tag of it) and embedding
        /// models, which turn text into vectors and cannot answer.
        /// </summary>
        public static List<string> ChatModels(IEnumerable<string> names, string safetyModel)
        {
            var chat = new List<string>();
            if (names == null) return chat;
            string safetyBase = BaseName(safetyModel);
            foreach (var raw in names)
            {
                if (string.IsNullOrWhiteSpace(raw)) continue;
                string name = raw.Trim();
                if (safetyBase.Length > 0 && string.Equals(BaseName(name), safetyBase, StringComparison.OrdinalIgnoreCase)) continue;
                if (name.IndexOf("embed", StringComparison.OrdinalIgnoreCase) >= 0) continue;
                if (!chat.Contains(name)) chat.Add(name);
            }
            return chat;
        }

        /// <summary>The name without its tag: <c>granite3-guardian:2b</c> becomes <c>granite3-guardian</c>.</summary>
        private static string BaseName(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) return string.Empty;
            string trimmed = name.Trim();
            int colon = trimmed.LastIndexOf(':');
            return colon > 0 ? trimmed.Substring(0, colon) : trimmed;
        }
    }
}
