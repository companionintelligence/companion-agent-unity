using System.Linq;
using NUnit.Framework;
using UnityEngine;
using XRScan.Core;
using XRScan.Settings;

namespace XRScan.Tests.Config
{
    /// <summary>
    /// The settings panel's model row offers the models the LLM server lists (<c>GET /api/tags</c>), not only the names
    /// baked into <see cref="CompanionConfig.llmModelChoices"/>, which no server on the tailnet had on 2026-11-25.
    /// </summary>
    public class ServerModelListTests
    {
        // Trimmed from an Ollama 0.40 reply: chat models, the reply check's model and an embedding model.
        private const string TagsReply =
            "{\"models\":[" +
            "{\"name\":\"gemma3:4b\",\"model\":\"gemma3:4b\",\"size\":3338801804,\"details\":{\"family\":\"gemma3\"}}," +
            "{\"name\":\"qwen3.5:9b\",\"model\":\"qwen3.5:9b\"}," +
            "{\"name\":\"granite3-guardian:2b\",\"model\":\"granite3-guardian:2b\"}," +
            "{\"name\":\"nomic-embed-text:latest\",\"model\":\"nomic-embed-text:latest\"}]}";

        private static readonly string[] NoHosts = new string[0];

        private OverrideSandbox _sandbox;
        private CompanionConfig _companion;

        [SetUp]
        public void SetUp()
        {
            _sandbox = new OverrideSandbox();
            _companion = ScriptableObject.CreateInstance<CompanionConfig>();
            _companion.llm = ScriptableObject.CreateInstance<LlmConfig>();
            _companion.voice = ScriptableObject.CreateInstance<VoiceConfig>();
            _companion.server = ScriptableObject.CreateInstance<ServerConfig>();
            _companion.llmModelChoices.Clear();
            _companion.llmModelChoices.Add("llava:7b");
            _companion.llmModelChoices.Add("qwen2.5vl:7b");
        }

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(_companion.llm);
            Object.DestroyImmediate(_companion.voice);
            Object.DestroyImmediate(_companion.server);
            Object.DestroyImmediate(_companion);
            _sandbox.Dispose();
        }

        private CompanionSettingsModel Release() => new CompanionSettingsModel(_companion, false, NoHosts);

        private static SettingRow ModelRow(CompanionSettingsModel model) =>
            model.BuildRows().Single(row => row.Id == SettingId.LlmModel);

        [Test]
        public void TagsUrl_PutsTheListUnderTheLlmBaseUrl()
        {
            Assert.AreEqual("https://llm.example.lan/api/tags", OllamaModelList.TagsUrl("https://llm.example.lan"));
            Assert.AreEqual("https://llm.example.lan:11443/api/tags", OllamaModelList.TagsUrl("https://llm.example.lan:11443/"));
        }

        [Test]
        public void Parse_ReadsTheNamesInTheServersOrder()
        {
            CollectionAssert.AreEqual(
                new[] { "gemma3:4b", "qwen3.5:9b", "granite3-guardian:2b", "nomic-embed-text:latest" },
                OllamaModelList.Parse(TagsReply));
        }

        [Test]
        public void Parse_FallsBackToTheModelField_AndListsEachNameOnce()
        {
            const string reply = "{\"models\":[{\"model\":\"gemma3:1b\"},{\"name\":\"gemma3:1b\"},{\"name\":\" llava:7b \"}]}";
            CollectionAssert.AreEqual(new[] { "gemma3:1b", "llava:7b" }, OllamaModelList.Parse(reply));
        }

        [Test]
        public void Parse_GivesAnEmptyListForAServerWithNoModels_AndNullForAnythingElse()
        {
            CollectionAssert.IsEmpty(OllamaModelList.Parse("{\"models\":[]}"));
            Assert.IsNull(OllamaModelList.Parse("<html><body>Bad gateway</body></html>"));
            Assert.IsNull(OllamaModelList.Parse("{\"error\":\"unauthorized\"}"));
            Assert.IsNull(OllamaModelList.Parse(""));
            Assert.IsNull(OllamaModelList.Parse(null));
        }

        [Test]
        public void ChatModels_LeaveOutEveryTagOfTheReplyCheckModel_AndEmbeddingModels()
        {
            var names = new[] { "gemma3:4b", "granite3-guardian:2b", "granite3-guardian:latest", "nomic-embed-text:latest", "mxbai-embed-large", "qwen3.5:9b" };
            CollectionAssert.AreEqual(new[] { "gemma3:4b", "qwen3.5:9b" }, OllamaModelList.ChatModels(names, "granite3-guardian:2b"));
        }

        [Test]
        public void TheServersModels_ReplaceTheConfiguredChoices()
        {
            var model = Release();
            model.SetServerModels(OllamaModelList.Parse(TagsReply));

            CollectionAssert.AreEqual(new[] { "gemma3:4b", "qwen3.5:9b" }, model.ModelChoices());
        }

        [Test]
        public void CyclingFromAModelTheServerLacks_PicksItsFirstModelToTheRight_AndItsLastToTheLeft()
        {
            var model = Release();
            model.SetServerModels(OllamaModelList.Parse(TagsReply));
            Assert.AreEqual("llava:7b", _companion.llm.model, "the default model is not on this server");

            model.Cycle(SettingId.LlmModel, 1);
            Assert.AreEqual("gemma3:4b", _companion.llm.model);
            model.Cycle(SettingId.LlmModel, 1);
            Assert.AreEqual("qwen3.5:9b", _companion.llm.model);
            model.Cycle(SettingId.LlmModel, 1);
            Assert.AreEqual("gemma3:4b", _companion.llm.model, "wraps around the server's list");

            _sandbox.Store.SetString(LlmConfig.PlayerPrefsModelKey, "llava:7b");
            model.Cycle(SettingId.LlmModel, -1);
            Assert.AreEqual("qwen3.5:9b", _companion.llm.model);
        }

        [Test]
        public void TheModelRow_WarnsWhenTheServerLacksTheCurrentModel()
        {
            var model = Release();
            model.SetServerModels(OllamaModelList.Parse(TagsReply));

            var row = ModelRow(model);
            Assert.AreEqual("llava:7b", row.Value);
            Assert.IsTrue(row.IsWarning);
            StringAssert.Contains("does not have this model", row.Detail);
            StringAssert.Contains("one of its 2", row.Detail);
        }

        [Test]
        public void TheModelRow_SaysWhenTheCurrentModelIsOnTheServer()
        {
            var model = Release();
            model.SetServerModels(OllamaModelList.Parse(TagsReply));
            model.Cycle(SettingId.LlmModel, 1);

            var row = ModelRow(model);
            Assert.AreEqual("gemma3:4b", row.Value);
            Assert.IsFalse(row.IsWarning);
            StringAssert.Contains("On the server (2 models there)", row.Detail);
        }

        [Test]
        public void AServerWithNoChatModels_IsAWarning_AndTheConfiguredChoicesStay()
        {
            var model = Release();
            model.SetServerModels(new[] { "granite3-guardian:2b", "nomic-embed-text:latest" });

            CollectionAssert.AreEqual(new[] { "llava:7b", "qwen2.5vl:7b" }, model.ModelChoices());
            var row = ModelRow(model);
            Assert.IsTrue(row.IsWarning);
            StringAssert.Contains("lists no chat models", row.Detail);
        }

        [Test]
        public void AListThatCouldNotBeRead_KeepsTheConfiguredChoices_AndSaysWhy()
        {
            var model = Release();
            model.SetServerModelsProblem("HTTP 502");

            CollectionAssert.AreEqual(new[] { "llava:7b", "qwen2.5vl:7b" }, model.ModelChoices());
            StringAssert.Contains("Could not list the server's models (HTTP 502)", ModelRow(model).Detail);
        }

        [Test]
        public void ANewLlmServer_ForgetsTheOldServersList()
        {
            var model = Release();
            model.SetServerModels(OllamaModelList.Parse(TagsReply));

            Assert.IsTrue(model.TrySetEndpoint(SettingId.LlmEndpoint, "https://other.example.lan", out var message), message);
            Assert.IsNull(model.ServerModels);
            CollectionAssert.AreEqual(new[] { "llava:7b", "qwen2.5vl:7b" }, model.ModelChoices());
        }

        [Test]
        public void ResetAll_ForgetsTheServersList()
        {
            var model = Release();
            model.SetServerModels(OllamaModelList.Parse(TagsReply));
            model.Activate(SettingId.ResetAll);
            Assert.IsNull(model.ServerModels);
        }

        [Test]
        public void TheListIsAskedFor_OnlyWhereTheTransportPolicyLetsTheLlmGo()
        {
            _companion.llm.endpoint = "https://llm.example.lan:11443";
            Assert.AreEqual("https://llm.example.lan:11443/api/tags", ServerModelLister.ListUrl(_companion.llm, out var problem));
            Assert.IsNull(problem);

            _companion.llm.endpoint = "";
            Assert.IsNull(ServerModelLister.ListUrl(_companion.llm, out problem));
            Assert.IsNotEmpty(problem);

            _companion.llm.endpoint = "http://203.0.113.9:11434";
            Assert.IsNull(ServerModelLister.ListUrl(_companion.llm, out problem), "plain http to a host on no allowlist");
            Assert.IsNotEmpty(problem);
        }
    }
}
