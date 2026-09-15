import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.content.adapter import StoryAdapter, validate_draft
from src.content.critic import BatchCritic, prepare_batch, recent_scripts
from tests.test_resilience import DRAFT, FACTS, verdict
from tests import test_resilience


class StoryQualityTests(unittest.IsolatedAsyncioTestCase):
    def adapter(self):
        async def generate(title, text, **kwargs):
            return {**DRAFT, "evidence": [{"claim": "word", "fact_ids": [kwargs["fact_id"]]}]}
        return SimpleNamespace(adapt_story=AsyncMock(side_effect=generate), last_error=None)

    async def test_batch_accept_and_recent_scripts(self):
        adapter = self.adapter()
        critic = SimpleNamespace(review=AsyncMock(return_value={i: verdict(i) for i in range(1, 4)}))
        result = await prepare_batch(adapter, FACTS, ["previous"], critic)
        self.assertEqual([r["status"] for r in result], ["ACCEPTED"] * 3)
        critic.review.assert_awaited_once()
        self.assertEqual(len(critic.review.call_args.args[0]), 3)
        self.assertEqual(adapter.adapt_story.await_count, 3)
        self.assertTrue(all(c.kwargs["recent_scripts"] == ["previous"] for c in adapter.adapt_story.await_args_list))

    async def test_exactly_one_rewrite_and_feedback(self):
        for final in ("accept", "reject", "rewrite"):
            with self.subTest(final=final):
                adapter = self.adapter()
                second = {**DRAFT, "text": "changed " + "word " * 44}
                adapter.adapt_story.side_effect = [DRAFT, second]
                first = {**verdict(1), "verdict": "rewrite", "fixes": ["Remove sentence two"]}
                critic = SimpleNamespace(review=AsyncMock(side_effect=[{1: first}, {1: {**first, "verdict": final}}]))
                result = (await prepare_batch(adapter, FACTS[:1], critic=critic))[0]
                self.assertEqual(adapter.adapt_story.await_count, 2)
                self.assertEqual(critic.review.await_count, 2)
                feedback = adapter.adapt_story.call_args.kwargs["feedback"]
                self.assertEqual(feedback["fixes"], first["fixes"])
                self.assertIn("original_draft", feedback)
                self.assertEqual(result["status"], "ACCEPTED" if final == "accept" else "SKIPPED")
                if final == "accept":
                    self.assertTrue(result["story"]["text"].startswith("changed"))
                else:
                    self.assertNotIn("story", result)

    async def test_skip_and_hard_gate_and_unavailable(self):
        for mode in ("skip", "hard", "low", "unavailable"):
            adapter = self.adapter()
            value = verdict(1)
            if mode == "skip":
                adapter.adapt_story.side_effect = [{"status": "skip", "reason": "weak material"}]
            if mode == "hard":
                value["hard_failures"] = ["invented motive"]
            if mode == "low":
                value["scores"]["factuality"] = 4
            critic = SimpleNamespace(review=AsyncMock(return_value={1: value},
                side_effect=TimeoutError("down") if mode == "unavailable" else None))
            result = (await prepare_batch(adapter, FACTS[:1], critic=critic))[0]
            self.assertNotIn("story", result)
            self.assertIn(result["status"], ("FAILED", "SKIPPED"))
            if mode == "skip":
                critic.review.assert_not_awaited()

    async def test_real_critic_fallback_and_malformed_item_isolation(self):
        with patch("src.content.adapter.AsyncOpenAI"), patch("src.content.llm.asyncio.sleep", new_callable=AsyncMock):
            adapter = StoryAdapter("test")
            response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
                {"items": [verdict(1), {"item": 2}, verdict(3)]})))])
            adapter.client.chat.completions.create = AsyncMock(side_effect=[TimeoutError(), TimeoutError(), response])
            result = await BatchCritic(adapter).review([{"item": i} for i in range(1, 4)], [])
            self.assertIn("invalid", result[2])
            self.assertEqual(result[1]["verdict"], "accept")
            self.assertEqual(adapter.client.chat.completions.create.await_count, 3)
            self.assertEqual(adapter.client.chat.completions.create.call_args.kwargs["model"], adapter.fallback_model)

    async def test_no_schema_regeneration(self):
        with patch("src.content.adapter.AsyncOpenAI"):
            adapter = StoryAdapter("test")
            response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))])
            adapter.client.chat.completions.create = AsyncMock(return_value=response)
            self.assertIsNone(await adapter.adapt_story("title", "facts"))
            adapter.client.chat.completions.create.assert_awaited_once()

    async def test_critic_rejects_missing_duplicate_unknown_ids(self):
        with patch("src.content.adapter.AsyncOpenAI"):
            adapter = StoryAdapter("test")
            for values in ([], [verdict(1), verdict(1)], [verdict(2)]):
                response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                    content=json.dumps({"items": values})))])
                adapter.client.chat.completions.create = AsyncMock(return_value=response)
                with self.assertRaises(ValueError):
                    await BatchCritic(adapter).review([{"item": 1}], [])

    async def test_failed_final_critic_preserves_first_pass_acceptance(self):
        adapter = self.adapter()
        first = {i: verdict(i) for i in range(1, 4)}
        first[2].update(verdict="rewrite", fixes=["Remove second sentence"])
        critic = SimpleNamespace(review=AsyncMock(side_effect=[first, TimeoutError("down")]))
        results = await prepare_batch(adapter, FACTS, critic=critic)
        self.assertEqual([r["status"] for r in results], ["ACCEPTED", "FAILED", "ACCEPTED"])
        self.assertEqual(adapter.adapt_story.await_count, 4)
        self.assertEqual(len(critic.review.call_args.args[0]), 1)

    def test_script_model_overrides_and_defaults(self):
        from src.config import Settings
        import os
        with patch.dict(os.environ, {"PRIMARY_MODEL": "primary", "FALLBACK_MODEL": "fallback",
                                     "SCRIPT_GENERATOR_MODEL": "", "SCRIPT_CRITIC_MODEL": ""}, clear=True):
            self.assertEqual(Settings().script_generator_model, "primary")
            self.assertEqual(Settings().script_critic_model, "primary")
            os.environ.update(SCRIPT_GENERATOR_MODEL="generator", SCRIPT_CRITIC_MODEL="critic")
            with patch("src.content.adapter.AsyncOpenAI"):
                adapter = StoryAdapter("test")
                self.assertEqual(adapter.model, "generator")
                self.assertEqual(BatchCritic(adapter).model, "critic")
                self.assertEqual(adapter.fallback_model, "fallback")

    def test_local_validation(self):
        for change in ({"script": "[PAUSE] " + DRAFT["text"]},
                       {"evidence": [{"claim": "test", "fact_ids": ["F99"]}]},
                       {"evidence": []}, {"text": ""}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_draft({**DRAFT, **change}, {"F1"})

    def test_recent_only_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch_1"
            path.mkdir()
            (path / "stories.json").write_text(json.dumps([
                {"text": "good", "status": "rendered", "critic": verdict(1)},
                {"text": "bad", "status": "failed", "critic": verdict(2)}]))
            self.assertEqual(recent_scripts(Path(tmp)), ["good"])

    async def test_integration_reject_has_no_tts_or_render(self):
        # Existing integration helper exercises real pipeline staging with mocked IO.
        helper = test_resilience.BatchResilienceTests()
        original = __import__("generate").prepare_batch
        async def prepare(adapter, facts, recent):
            critic = SimpleNamespace(review=AsyncMock(return_value={
                1: verdict(1), 2: {**verdict(2), "verdict": "reject"}, 3: verdict(3)}))
            return await original(adapter, facts, recent, critic)
        with tempfile.TemporaryDirectory() as tmp, patch("generate.prepare_batch", side_effect=prepare):
            code, batch, result = await helper.run_generate(Path(tmp))
            self.assertEqual((code, result["status"], result["successful"]), (1, "PARTIAL_SUCCESS", 2))
            self.assertEqual(result["items"][1]["status"], "SKIPPED")
            self.assertEqual(len(list(batch.glob("*.mp4"))), 2)
            self.assertEqual((batch / "posting_plan.txt").read_text(encoding="utf-8").count(".mp4"), 2)

    async def test_integration_rewrite_tts_receives_only_final_script(self):
        import generate
        from src.pipeline import ShortsPipeline
        original = generate.prepare_batch
        original_run = ShortsPipeline.run
        for final in ("accept", "reject", "rewrite", "unavailable", "skip"):
            observed = []
            async def prepare(adapter, facts, recent):
                async def draft(title, text, **kwargs):
                    if final == "skip":
                        return {"status": "skip", "reason": "weak"}
                    return {**DRAFT, "text": ("final " if kwargs["feedback"] else "first ") + "word " * 44,
                            "evidence": [{"claim": "word", "fact_ids": [kwargs["fact_id"]]}]}
                adapter.adapt_story.side_effect = draft
                first = {i: {**verdict(i), "verdict": "rewrite", "fixes": ["Remove second sentence"]}
                         for i in range(1, 4)}
                last = {i: {**first[i], "verdict": final} for i in range(1, 4)}
                critic = SimpleNamespace(review=AsyncMock(side_effect=[first,
                    TimeoutError("down") if final == "unavailable" else last]))
                result = await original(adapter, facts, recent, critic)
                self.assertEqual(adapter.adapt_story.await_count, 3 if final == "skip" else 6)
                self.assertEqual(critic.review.await_count, 0 if final == "skip" else 2)
                return result
            async def run(pipeline, text, output_path):
                observed.append(text)
                return await original_run(pipeline, text, output_path)
            with self.subTest(final=final), tempfile.TemporaryDirectory() as tmp, \
                    patch("generate.prepare_batch", side_effect=prepare), \
                    patch("src.pipeline.ShortsPipeline.run", new=run):
                _, batch, result = await test_resilience.BatchResilienceTests().run_generate(Path(tmp))
                self.assertEqual(len(observed), 3 if final == "accept" else 0)
                self.assertTrue(all(text.startswith("final ") for text in observed))
                self.assertEqual(len(list(batch.glob("*.mp4"))), len(observed))
