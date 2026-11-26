using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using XRScan.Core;

namespace XRScan.Settings
{
    /// <summary>
    /// Reads the LLM server's model list (<c>GET /api/tags</c>) for the settings panel's model row.
    /// <para>
    /// The URL comes from <see cref="LlmConfig.BaseUrl"/>, which <see cref="CleartextPolicy"/> gates, so the request goes
    /// only where the LLM backend itself may send prompts (TransportGateTests lists this call site). Parsing and the
    /// choice of models are in <see cref="OllamaModelList"/> and <see cref="CompanionSettingsModel"/>.
    /// </para>
    /// </summary>
    public static class ServerModelLister
    {
        /// <summary>
        /// The list's URL for the effective LLM endpoint, or null, with the reason in <paramref name="problem"/>, when the
        /// endpoint is unset or the transport policy refuses it.
        /// </summary>
        public static string ListUrl(LlmConfig llm, out string problem)
        {
            if (llm == null)
            {
                problem = "no LLM settings";
                return null;
            }
            var check = llm.ResolveEndpoint();
            if (!check.IsUsable)
            {
                problem = check.Message;
                return null;
            }
            problem = null;
            return OllamaModelList.TagsUrl(llm.BaseUrl());
        }

        /// <summary>
        /// Asks <paramref name="url"/> for the list and reports the model names, or why there are none, through the callbacks.
        /// Waits at most <paramref name="timeoutSeconds"/>, clamped to 1..15.
        /// </summary>
        public static IEnumerator Fetch(string url, float timeoutSeconds, Action<List<string>> onModels, Action<string> onProblem)
        {
            using (var request = UnityWebRequest.Get(url))
            {
                request.timeout = Mathf.Clamp(Mathf.CeilToInt(timeoutSeconds), 1, 15);
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    onProblem?.Invoke(request.responseCode > 0 ? "HTTP " + request.responseCode : request.error);
                    yield break;
                }

                var names = OllamaModelList.Parse(request.downloadHandler.text);
                if (names != null) onModels?.Invoke(names);
                else onProblem?.Invoke("the server did not answer with an Ollama model list");
            }
        }
    }
}
