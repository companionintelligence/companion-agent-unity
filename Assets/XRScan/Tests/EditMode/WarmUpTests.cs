using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using XRScan.Core;
using XRScan.Llm;

namespace XRScan.Tests.EditMode
{
    /// <summary>
    /// Once the server answers, the backend loads the language model and the reply check's model, at the context windows
    /// their requests ask for. On beta-red, with neither model loaded, the first reply check ran past its 20 s timeout
    /// while granite3-guardian loaded, and the first reply was withheld (2026-11-25).
    /// </summary>
    public class WarmUpTests
    {
        [Serializable]
        private class Body
        {
            public string model;
            public Options options;
        }

        [Serializable]
        private class Options
        {
            public int num_ctx;
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

        private static Body Read(string json) => JsonUtility.FromJson<Body>(json);

        [Test]
        public void TheLanguageModel_IsLoadedByAnEmptyPrompt()
        {
            Assert.AreEqual("{\"model\":\"llava:7b\",\"prompt\":\"\",\"stream\":false,\"options\":{\"num_ctx\":4096}}",
                OllamaBackend.BuildWarmUpJson("llava:7b", 4096));
        }

        [Test]
        public void TheReplyCheckModel_IsLoadedByAChatWithNoMessages()
        {
            Assert.AreEqual(
                "{\"model\":\"granite3-guardian:2b\",\"messages\":[],\"stream\":false," +
                "\"options\":{\"temperature\":0.0,\"num_ctx\":4096}}",
                GraniteGuardian.BuildWarmUpJson("granite3-guardian:2b", 4096));
        }

        [Test]
        public void BothModels_AreLoaded_AtTheWindowsTheirRequestsAskFor()
        {
            _config.contextLength = 8192;
            _config.safetyContextLength = 2048;

            var plan = OllamaBackend.BuildWarmUpRequests(_config);
            Assert.AreEqual(2, plan.Length);

            var language = plan[0];
            Assert.AreEqual(_config.model, language.Model);
            Assert.AreEqual(OllamaBackend.GeneratePath, language.Path);
            Assert.AreEqual(_config.model, Read(language.Json).model);
            Assert.AreEqual(8192, Read(language.Json).options.num_ctx);
            Assert.AreEqual(Read(OllamaBackend.BuildRequestJson(_config, "what is on the table?", null)).options.num_ctx,
                Read(language.Json).options.num_ctx, "a reply must not load the model again");

            var check = plan[1];
            Assert.AreEqual(_config.SafetyModel, check.Model);
            Assert.AreEqual(GraniteGuardian.ChatPath, check.Path);
            Assert.AreEqual(_config.SafetyModel, Read(check.Json).model);
            Assert.AreEqual(2048, Read(check.Json).options.num_ctx);
            Assert.AreEqual(
                Read(GraniteGuardian.BuildRequestJson(_config.SafetyModel, "harm", "q", "r", _config.SafetyContextLength)).options.num_ctx,
                Read(check.Json).options.num_ctx, "a reply check must not load the model again");
        }

        [Test]
        public void AWindowOutsideWhatOllamaAccepts_IsClamped_LikeTheRequests()
        {
            Assert.AreEqual(LlmConfig.MinContextLength, Read(OllamaBackend.BuildWarmUpJson("llava:7b", 0)).options.num_ctx);
            Assert.AreEqual(LlmConfig.MaxContextLength,
                Read(GraniteGuardian.BuildWarmUpJson("granite3-guardian:2b", int.MaxValue)).options.num_ctx);
        }

        [Test]
        public void TheWarmUp_WaitsLongerThanAReplyOrItsCheck()
        {
            Assert.Greater((float)OllamaBackend.WarmUpTimeoutSeconds, LlmConfig.DefaultSafetyCheckTimeoutSeconds);
            Assert.Greater((float)OllamaBackend.WarmUpTimeoutSeconds, _config.requestTimeoutSeconds);
        }

        [Test]
        public void TheBackend_WarmsUp_OnceTheServerAnswers()
        {
            // The warm-up is a coroutine that needs a server, so read the calls themselves.
            string source = File.ReadAllText(Path.Combine(Application.dataPath, "XRScan", "Llm", "OllamaBackend.cs"));
            StringAssert.Contains("StartCoroutine(WarmUp());", source);
            StringAssert.Contains("var plan = BuildWarmUpRequests(config);", source);
            StringAssert.Contains("web.timeout = WarmUpTimeoutSeconds;", source);
        }

        [TestCase("Request timeout")]
        [TestCase("Request timed out")]
        public void ACheckThatTimesOut_SaysTheServerMayStillBeLoadingTheModel(string transportError)
        {
            var result = GraniteGuardian.ReadAnswer("granite3-guardian:2b", "harm", 0, null, transportError);
            Assert.AreEqual(ReplyVerdict.Unavailable, result.Verdict);
            StringAssert.Contains("may still be loading granite3-guardian:2b", result.Reason);
        }

        [Test]
        public void AServerThatCannotBeReached_IsStillSaidToBeUnreachable()
        {
            var result = GraniteGuardian.ReadAnswer("granite3-guardian:2b", "harm", 0, "", "Cannot connect to destination host");
            Assert.AreEqual(ReplyVerdict.Unavailable, result.Verdict);
            StringAssert.Contains("could not be reached", result.Reason);
        }
    }
}
