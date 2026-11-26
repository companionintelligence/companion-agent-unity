using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using XRScan.Core;
using XRScan.Llm;

namespace XRScan.Tests.EditMode
{
    /// <summary>
    /// Every Ollama request asks for its context window (<c>options.num_ctx</c>). Left to the server's default, 32768 on
    /// beta-red, llava:7b and the reply check's model did not fit side by side in 10 GB and unloaded each other on every
    /// reply; at 4096 both stay loaded.
    /// </summary>
    public class ContextLengthTests
    {
        [Serializable]
        private class Body
        {
            public Options options;
        }

        [Serializable]
        private class Options
        {
            public int num_ctx;
            public int num_predict;
            public float temperature;
        }

        private LlmConfig _config;

        [SetUp]
        public void SetUp()
        {
            ConfigOverrides.EnvironmentReader = _ => null;
            _config = ScriptableObject.CreateInstance<LlmConfig>();
        }

        [TearDown]
        public void TearDown()
        {
            UnityEngine.Object.DestroyImmediate(_config);
            ConfigOverrides.ResetToDefaults();
        }

        private Options LanguageModelOptions() =>
            JsonUtility.FromJson<Body>(OllamaBackend.BuildRequestJson(_config, "what is on the table?", null)).options;

        private static Options ReplyCheckOptions(int contextLength) =>
            JsonUtility.FromJson<Body>(GraniteGuardian.BuildRequestJson(LlmConfig.DefaultSafetyModel, "harm", "q", "r", contextLength)).options;

        [Test]
        public void TheDefault_Is4096_ForTheLanguageModelAndTheReplyCheck()
        {
            Assert.AreEqual(4096, LlmConfig.DefaultContextLength);
            Assert.AreEqual(4096, _config.ContextLength);
            Assert.AreEqual(4096, _config.SafetyContextLength);
        }

        [Test]
        public void EveryLanguageModelRequest_AsksForTheConfiguredWindow()
        {
            Assert.AreEqual(4096, LanguageModelOptions().num_ctx);

            _config.contextLength = 8192;
            var options = LanguageModelOptions();
            Assert.AreEqual(8192, options.num_ctx);
            Assert.AreEqual(_config.maxTokens, options.num_predict, "the other options stay");
        }

        [Test]
        public void EveryReplyCheck_AsksForItsOwnWindow()
        {
            Assert.AreEqual(4096, JsonUtility.FromJson<Body>(
                GraniteGuardian.BuildRequestJson(LlmConfig.DefaultSafetyModel, "harm", "q", "r")).options.num_ctx, "the default");

            _config.safetyContextLength = 2048;
            Assert.AreEqual(2048, ReplyCheckOptions(_config.SafetyContextLength).num_ctx);
            Assert.AreEqual(0f, ReplyCheckOptions(_config.SafetyContextLength).temperature, "the check stays deterministic");
        }

        [Test]
        public void AWindowOutsideWhatOllamaAccepts_IsClamped()
        {
            _config.contextLength = 0;
            Assert.AreEqual(LlmConfig.MinContextLength, LanguageModelOptions().num_ctx);

            _config.contextLength = 10_000_000;
            Assert.AreEqual(LlmConfig.MaxContextLength, LanguageModelOptions().num_ctx);

            Assert.AreEqual(LlmConfig.MinContextLength, ReplyCheckOptions(-1).num_ctx);
        }

        [Test]
        public void TheBackend_SendsTheReplyCheckItsOwnWindow()
        {
            // The check's requests are built inside a coroutine that needs a server, so read the call itself.
            string source = File.ReadAllText(Path.Combine(Application.dataPath, "XRScan", "Llm", "OllamaBackend.cs"));
            StringAssert.Contains("GraniteGuardian.BuildRequestJson(model, risks[i], question, reply, config.SafetyContextLength)", source);
        }
    }
}
