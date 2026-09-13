import asyncio
import io
import json
import os
import tempfile
import unittest
from argparse import Namespace
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import aiohttp
import httpx
from edge_tts.exceptions import WebSocketError
from openai import APIConnectionError, APIStatusError

import batch_generate
import fetch_content
import generate
from src.config import Settings, llm_models
from src.content.adapter import StoryAdapter
from src.content.llm import complete_with_fallback
from src.pipeline import ShortsPipeline
from src.runtime import log_failure, log_to, safe_error
from src.tts import TTSError, WordBoundary, generate_tts


STORY = {"title": "A test story", "text": " ".join(["word"] * 45), "tags": ["#test"]}
FACTS = [{"title": f"Fact {i}", "raw_data": "source", "topic": f"topic {i}",
          "category": "systems", "sources": []} for i in range(1, 4)]


class LLMResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_status_and_network_errors_use_fallback(self):
        request = httpx.Request("POST", "https://example.test")
        errors = [APIConnectionError(request=request), TimeoutError("slow"),
                  httpx.ConnectError("DNS unavailable")]
        errors += [APIStatusError("response body", response=httpx.Response(s, request=request), body=None)
                   for s in (429, 500, 502, 503, 599)]
        for error in errors:
            with self.subTest(error=error), patch("src.content.llm.asyncio.sleep", new_callable=AsyncMock) as sleep:
                call = AsyncMock(side_effect=[error, error, "ok"])
                self.assertEqual(await complete_with_fallback(call, "primary", "fallback"), "ok")
                self.assertEqual([c.args[0] for c in call.await_args_list], ["primary", "primary", "fallback"])
                sleep.assert_awaited_once_with(2)

    async def test_nonretryable_errors_and_cancellation_are_propagated(self):
        request = httpx.Request("POST", "https://example.test")
        errors = [ValueError("bad request"), TypeError("bug"), asyncio.CancelledError()]
        errors += [APIStatusError("bad", response=httpx.Response(s, request=request), body=None)
                   for s in (400, 401, 403, 404, 422)]
        for error in errors:
            with self.subTest(error=error), patch("src.content.llm.asyncio.sleep", new_callable=AsyncMock) as sleep:
                call = AsyncMock(side_effect=error)
                with self.assertRaises(type(error)):
                    await complete_with_fallback(call, "primary", "fallback")
                call.assert_awaited_once_with("primary")
                sleep.assert_not_awaited()

    async def test_fallback_error_is_not_replaced_by_primary_error(self):
        call = AsyncMock(side_effect=[TimeoutError(), TimeoutError(), ValueError("fallback bug")])
        with patch("src.content.llm.asyncio.sleep", new_callable=AsyncMock):
            with self.assertRaisesRegex(ValueError, "fallback bug"):
                await complete_with_fallback(call, "primary", "fallback")
        self.assertEqual(call.await_count, 3)

    async def test_adapter_uses_shared_policy_and_disables_sdk_retries(self):
        with patch("src.content.adapter.AsyncOpenAI") as sdk, patch("src.content.llm.asyncio.sleep", new_callable=AsyncMock):
            adapter = StoryAdapter("test")
            response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(STORY)))])
            call = AsyncMock(side_effect=[TimeoutError("network"), TimeoutError("network"), response])
            adapter.client.chat.completions.create = call
            self.assertEqual((await adapter.adapt_story("title", "source"))["text"], STORY["text"])
            self.assertEqual([c.kwargs["model"] for c in call.await_args_list],
                             [adapter.model, adapter.model, adapter.fallback_model])
            self.assertEqual(sdk.call_args.kwargs["max_retries"], 0)

    async def test_adapter_request_value_error_is_not_a_response_repair(self):
        with patch("src.content.adapter.AsyncOpenAI"):
            adapter = StoryAdapter("test")
            adapter.client.chat.completions.create = AsyncMock(side_effect=ValueError("request bug"))
            self.assertIsNone(await adapter.adapt_story("title", "source"))
            adapter.client.chat.completions.create.assert_awaited_once()
            self.assertIsInstance(adapter.last_error, ValueError)

    def test_model_env_is_read_at_construction_and_legacy_is_preserved(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(llm_models(), ("ag/gemini-3.7-flash-medium", "ag/claude-sonnet-4-6"))
            os.environ["ANYMODEL_MODEL"] = "legacy"
            self.assertEqual(llm_models()[0], "legacy")
            os.environ.update(PRIMARY_MODEL="primary", FALLBACK_MODEL="fallback")
            with patch("src.content.adapter.AsyncOpenAI"):
                adapter = StoryAdapter("test")
                self.assertEqual((adapter.model, adapter.fallback_model), ("primary", "fallback"))
                self.assertEqual(StoryAdapter("test", model="explicit").model, "explicit")


class TTSResilienceTests(unittest.IsolatedAsyncioTestCase):
    def fake_communicate(self, outcomes):
        def factory(text, voice, **kwargs):
            outcome = outcomes.pop(0)
            async def stream():
                yield {"type": "audio", "data": b"audio"}
                yield {"type": "WordBoundary", "text": "word", "offset": 0, "duration": 1000000}
                if outcome:
                    raise outcome
            return SimpleNamespace(stream=stream)
        return factory

    async def test_three_primary_failures_then_one_fallback_with_clean_audio_and_logs(self):
        error = WebSocketError("WebSocket connection closed unexpectedly")
        with tempfile.TemporaryDirectory() as tmp, patch("edge_tts.Communicate", side_effect=self.fake_communicate([error] * 3 + [None])) as communicate, patch("src.tts.asyncio.sleep", new_callable=AsyncMock) as sleep, self.assertLogs("shorts", level="WARNING") as logs:
            audio = Path(tmp) / "audio.mp3"
            words = await generate_tts("word", audio, "ru-RU-DmitryNeural", "+10%")
            self.assertEqual(audio.read_bytes(), b"audio")
            self.assertEqual(len(words), 1)
            self.assertEqual([c.args[1] for c in communicate.call_args_list], ["ru-RU-DmitryNeural"] * 3 + ["ru-RU-SvetlanaNeural"])
            self.assertEqual([c.args for c in sleep.await_args_list], [(3.0,), (6.0,)])
            self.assertEqual(list(Path(tmp).iterdir()), [audio])
            self.assertIn("WebSocketError", "\n".join(logs.output))
            self.assertIn("WebSocket connection closed unexpectedly", "\n".join(logs.output))

    async def test_exhaustion_preserves_existing_audio_and_removes_partial(self):
        error = WebSocketError("disconnected")
        with tempfile.TemporaryDirectory() as tmp, patch("edge_tts.Communicate", side_effect=self.fake_communicate([error] * 4)) as communicate, patch("src.tts.asyncio.sleep", new_callable=AsyncMock):
            audio = Path(tmp) / "audio.mp3"
            audio.write_bytes(b"existing")
            with self.assertRaisesRegex(TTSError, "WebSocketError: disconnected"):
                await generate_tts("word", audio, "ru-RU-DmitryNeural", "+10%")
            self.assertEqual(communicate.call_count, 4)
            self.assertEqual(audio.read_bytes(), b"existing")
            self.assertEqual(list(Path(tmp).iterdir()), [audio])

    async def test_invalid_voice_and_local_errors_are_not_retried(self):
        for error in (ValueError("Unsupported voice"), TypeError("bad config"), PermissionError("disk")):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp, patch("edge_tts.Communicate", side_effect=error) as communicate, patch("src.tts.asyncio.sleep", new_callable=AsyncMock) as sleep:
                with self.assertRaises(TTSError):
                    await generate_tts("word", Path(tmp) / "audio.mp3", "ru-RU-DmitryNeural", "+10%")
                communicate.assert_called_once()
                sleep.assert_not_awaited()

    async def test_actual_timeout_is_bounded_and_falls_back(self):
        async def stalled():
            await asyncio.Event().wait()
            yield {}
        with tempfile.TemporaryDirectory() as tmp, patch("edge_tts.Communicate", return_value=SimpleNamespace(stream=stalled)) as communicate, patch("src.tts.asyncio.sleep", new_callable=AsyncMock):
            with self.assertRaisesRegex(TTSError, "TimeoutError"):
                await generate_tts("word", Path(tmp) / "audio.mp3", "ru-RU-DmitryNeural", "+10%", timeout=0.001)
            self.assertEqual(communicate.call_count, 4)
            self.assertFalse(list(Path(tmp).iterdir()))

    async def test_cancellation_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp, patch("edge_tts.Communicate", side_effect=self.fake_communicate([asyncio.CancelledError()])) as communicate:
            with self.assertRaises(asyncio.CancelledError):
                await generate_tts("word", Path(tmp) / "audio.mp3", "ru-RU-DmitryNeural", "+10%")
            communicate.assert_called_once()
            self.assertFalse(list(Path(tmp).iterdir()))

    async def test_http_429_and_5xx_retry_but_400_does_not(self):
        for status in (400, 429, 503):
            error = aiohttp.ClientResponseError(SimpleNamespace(real_url="https://example.test"), (), status=status, message="service failure")
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp, patch("edge_tts.Communicate", side_effect=self.fake_communicate([error, None])) as communicate, patch("src.tts.asyncio.sleep", new_callable=AsyncMock):
                if status == 400:
                    with self.assertRaises(TTSError):
                        await generate_tts("word", Path(tmp) / "audio.mp3", "ru-RU-DmitryNeural", "+10%")
                    self.assertEqual(communicate.call_count, 1)
                else:
                    await generate_tts("word", Path(tmp) / "audio.mp3", "ru-RU-DmitryNeural", "+10%")
                    self.assertEqual(communicate.call_count, 2)

    async def test_custom_voice_never_switches_language(self):
        with tempfile.TemporaryDirectory() as tmp, patch("edge_tts.Communicate", side_effect=self.fake_communicate([WebSocketError("down")] * 3)) as communicate, patch("src.tts.asyncio.sleep", new_callable=AsyncMock):
            with self.assertRaises(TTSError):
                await generate_tts("word", Path(tmp) / "audio.mp3", "en-US-AriaNeural", "+10%")
            self.assertEqual([c.args[1] for c in communicate.call_args_list], ["en-US-AriaNeural"] * 3)


class BatchResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def run_generate(self, root, stage=None, failed=(2,), research_error=None, dry_run=False):
        with ExitStack() as stack:
            stack.enter_context(patch.object(generate, "ROOT", root))
            stack.enter_context(patch("generate.load_dotenv"))
            stack.enter_context(patch.dict(os.environ, {"ANYMODEL_API_KEY": "test"}))
            stack.enter_context(patch("generate.Settings.from_env", return_value=Settings(root=root)))
            researcher = stack.enter_context(patch("generate.FactResearcher")).return_value
            researcher.find_facts = AsyncMock(side_effect=research_error, return_value=FACTS)
            researcher.close = AsyncMock()
            adapter = stack.enter_context(patch("generate.StoryAdapter")).return_value
            adapter.client.close = AsyncMock()
            adapter.last_error = None
            async def adapt(title, text):
                if stage == "adaptation" and int(title.split()[-1]) in failed:
                    raise ValueError("adaptation unavailable")
                return dict(STORY)
            adapter.adapt_story = AsyncMock(side_effect=adapt)
            # Exercise real pipeline staging, commit and cleanup with external work mocked.
            stack.enter_context(patch("src.pipeline.ShortsPipeline.preflight"))
            stack.enter_context(patch("src.pipeline.probe_duration", return_value=24.0))
            stack.enter_context(patch("src.pipeline.ShortsPipeline._select_backgrounds", return_value=([root / "bg.mp4"], [0.0])))
            counter = {"item": 0}
            async def tts(text, audio, voice, rate, **kwargs):
                counter["item"] += 1
                if stage == "tts" and counter["item"] in failed:
                    raise TTSError("WebSocket disconnected")
                audio.write_bytes(b"audio")
                return [WordBoundary("word", 0, 1)]
            stack.enter_context(patch("src.pipeline.generate_tts", side_effect=tts))
            def subtitles(*args):
                if stage == "subtitles" and counter["item"] in failed:
                    raise ValueError("subtitle failure")
            stack.enter_context(patch("src.pipeline.write_ass", side_effect=subtitles))
            def render(bg, audio, ass, partial, *args):
                partial.write_bytes(b"rendered")
                if stage == "render" and counter["item"] in failed:
                    raise RuntimeError("render failed after writing partial")
            stack.enter_context(patch("src.pipeline.render_video", side_effect=render))
            def validate(*args, **kwargs):
                if stage == "validation" and counter["item"] in failed:
                    raise ValueError("invalid output")
                return {"duration": 24}
            stack.enter_context(patch("src.pipeline.validate_output", side_effect=validate))
            stack.enter_context(redirect_stdout(io.StringIO()))
            code = await generate.main_async(generate.parse_args(["--dry-run"] if dry_run else []))
            researcher.close.assert_awaited_once()
            if not research_error:
                adapter.client.close.assert_awaited_once()
            batch = next((root / "output").iterdir())
            return code, batch, json.loads((batch / "batch_result.json").read_text())

    async def test_middle_item_failure_at_each_stage_preserves_first_and_third(self):
        for stage in ("adaptation", "tts", "subtitles", "render", "validation"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                code, batch, result = await self.run_generate(Path(tmp), stage)
                self.assertEqual(code, 1)
                self.assertEqual(result["status"], "PARTIAL_SUCCESS")
                self.assertEqual(result["successful"], 2)
                self.assertEqual(sorted(p.name for p in batch.glob("*.mp4")), ["video_01.mp4", "video_03.mp4"])
                plan = (batch / "posting_plan.txt").read_text(encoding="utf-8")
                self.assertIn("video_01.mp4", plan)
                self.assertIn("video_03.mp4", plan)
                self.assertNotIn("video_02.mp4", plan)
                self.assertIn("03]", plan)
                self.assertEqual(result["failures"][0]["stage"], stage)
                self.assertEqual(result["failures"][0]["item"], 2)
                self.assertIn("test_resilience.py", result["failures"][0]["traceback"])
                self.assertFalse(list(Path(tmp).rglob("*.partial.mp4")))

    async def test_all_items_failed_and_source_failure_report_failed(self):
        for stage in ("adaptation", "tts", "render", "research"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                code, batch, result = await self.run_generate(Path(tmp), stage, (1, 2, 3),
                    RuntimeError("source unavailable") if stage == "research" else None)
                self.assertEqual((code, result["status"], result["successful"]), (1, "FAILED", 0))
                self.assertFalse(list(batch.glob("*.mp4")))
                if (batch / "posting_plan.txt").exists():
                    self.assertNotIn("video_", (batch / "posting_plan.txt").read_text())

    async def test_happy_path_and_dry_run_keep_success_semantics(self):
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run), tempfile.TemporaryDirectory() as tmp:
                code, batch, result = await self.run_generate(Path(tmp), dry_run=dry_run)
                self.assertEqual((code, result["status"], result["successful"]), (0, "SUCCESS", 3))
                self.assertEqual(len(list(batch.glob("*.mp4"))), 0 if dry_run else 3)
                self.assertEqual((batch / "posting_plan.txt").read_text(encoding="utf-8").count(".mp4"), 3)

    async def test_batch_cli_isolates_items_and_reports_all_failed(self):
        for failed in ((2,), (1, 2, 3), ()):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as tmp, patch("batch_generate.ShortsPipeline") as factory, patch("batch_generate.Settings.from_env"), redirect_stdout(io.StringIO()):
                root = Path(tmp)
                source = root / "stories.json"
                source.write_text(json.dumps([STORY] * 3))
                pipeline = factory.return_value
                pipeline.last_stage = "tts"
                async def run(text, output_path):
                    if int(output_path.stem.split("_")[-1]) in failed:
                        raise TTSError("voice unavailable")
                    output_path.write_bytes(b"video")
                pipeline.run = AsyncMock(side_effect=run)
                code = await batch_generate.async_main(Namespace(input=source, output_dir=root / "output", voice="voice", rate="+10%", fps=30, channel_tag="tag"))
                batch = next((root / "output").iterdir())
                result = json.loads((batch / "batch_result.json").read_text())
                self.assertEqual(result["successful"], 3 - len(failed))
                self.assertEqual(code, 1 if failed else 0)
                plan = (batch / "posting_plan.txt").read_text(encoding="utf-8")
                for index in range(1, 4):
                    name = f"video_{index:02d}.mp4"
                    self.assertEqual((batch / name).exists(), index not in failed)
                    self.assertEqual(name in plan, index not in failed)

    async def test_fetch_content_isolates_malformed_items_and_all_failures(self):
        for outcomes in ([STORY, ValueError("bad"), STORY], [None] * 3):
            with tempfile.TemporaryDirectory() as tmp, patch("fetch_content.load_dotenv"), patch.dict(os.environ, {"ANYMODEL_API_KEY": "test"}), patch("fetch_content.StoryAdapter") as factory, redirect_stdout(io.StringIO()):
                root = Path(tmp)
                source = root / "facts.json"
                source.write_text(json.dumps(FACTS))
                adapter = factory.return_value
                adapter.adapt_story = AsyncMock(side_effect=outcomes)
                adapter.client.close = AsyncMock()
                output = root / "stories.json"
                code = await fetch_content.run(Namespace(facts=source, output=output, model="primary"))
                self.assertEqual(code, 1)
                self.assertEqual(len(json.loads(output.read_text())), 2 if outcomes[0] else 0)
                self.assertEqual(adapter.adapt_story.await_count, 3)
                adapter.client.close.assert_awaited_once()


class DiagnosticsTests(unittest.TestCase):
    def test_secrets_and_sdk_payloads_are_excluded_from_error_and_traceback(self):
        request = httpx.Request("POST", "https://example.test")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ANYMODEL_API_KEY": "private-key-value"}):
            path = Path(tmp) / "log.jsonl"
            with log_to(path):
                try:
                    raise RuntimeError("private-key-value Authorization: Bearer credential")
                except RuntimeError as exc:
                    log_failure("tts", exc, item=2)
                error = APIStatusError("sensitive full article", response=httpx.Response(503, request=request), body={"payload": "secret"})
                self.assertIn("503", safe_error(error))
                self.assertNotIn("sensitive full article", safe_error(error))
            text = path.read_text()
            self.assertNotIn("private-key-value", text)
            self.assertNotIn("credential", text)
            data = json.loads(text)
            self.assertEqual((data["stage"], data["item"]), ("tts", 2))
            self.assertIn("test_resilience.py", data["traceback"])


class PipelineIntegrityTests(unittest.IsolatedAsyncioTestCase):
    async def test_duration_rejection_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp, patch("src.pipeline.ShortsPipeline.preflight"), patch("src.pipeline.generate_tts", new_callable=AsyncMock, return_value=[WordBoundary("word", 0, 1)]) as tts, patch("src.pipeline.probe_duration", return_value=27.0), patch("src.pipeline.render_video") as render:
            root = Path(tmp)
            output = root / "video.mp4"
            output.write_bytes(b"previous success")
            pipeline = ShortsPipeline(Settings(root=root, retry_attempts=2, retry_delay=0, tts_timeout=10))
            with self.assertRaisesRegex(TTSError, "duration limit"):
                await pipeline.run("test story", output)
            tts.assert_awaited_once()
            self.assertEqual(tts.call_args.kwargs, {"attempts": 2, "retry_delay": 0, "timeout": 10})
            render.assert_not_called()
            self.assertEqual(output.read_bytes(), b"previous success")
            self.assertIsNone(pipeline.last_report)
            self.assertFalse(list((root / "work").iterdir()))
