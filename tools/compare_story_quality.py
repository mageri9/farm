"""Run from repository root: python -m tools.compare_story_quality. No TTS/render."""
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.content.adapter import StoryAdapter
from src.content.critic import BatchCritic, prepare_batch
from src.runtime import atomic_json, safe_error
from tools.old_adapter_snapshot import StoryAdapter as OldAdapter


async def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    destination = root / "work" / "story_comparison.json"
    facts = []
    seen = set()
    for path in sorted((root / "output").glob("batch_*/facts.json")):
        for fact in json.loads(path.read_text(encoding="utf-8")):
            if fact["topic"] not in seen:
                facts.append(fact)
                seen.add(fact["topic"])
            if len(facts) == 5:
                break
        if len(facts) == 5:
            break
    if len(facts) < 5:
        raise ValueError("Need five existing facts")
    old = OldAdapter(os.environ["ANYMODEL_API_KEY"])
    new = StoryAdapter(os.environ["ANYMODEL_API_KEY"])
    report = {"facts": facts, "old": [], "new": [], "requests": [],
              "models": {"old": old.model, "new": new.model, "critic": BatchCritic(new).model}}
    for label, adapter in (("old", old), ("new", new)):
        original = adapter.client.chat.completions.create
        async def counted(_label=label, _original=original, **kwargs):
            entry = {"arm": _label, "model": kwargs["model"]}
            report["requests"].append(entry)
            try:
                result = await _original(**kwargs)
                entry["usage"] = result.usage.model_dump() if result.usage else None
                return result
            except Exception as exc:
                entry["error"] = safe_error(exc)
                raise
            finally:
                atomic_json(destination, report)
        adapter.client.chat.completions.create = counted
    try:
        for i, fact in enumerate(facts, 1):
            print(f"OLD {i}/5", flush=True)
            script = await old.adapt_story(fact["title"], fact["raw_data"])
            report["old"].append({"item": i, "draft": script,
                                  "error": safe_error(old.last_error) if old.last_error else None})
            atomic_json(destination, report)
        print("NEW batch generation + critic + bounded rewrites", flush=True)
        report["new"] = await prepare_batch(new, facts)
        atomic_json(destination, report)
        items = [{"item": r["item"], "fact_id": f"F{r['item']}", "fact": facts[r["item"] - 1],
                  "caveats": facts[r["item"] - 1].get("caveats", []), "draft": r["draft"]}
                 for r in report["old"] if r["draft"]]
        if items:
            print("OLD batch critic (evaluation only)", flush=True)
            report["old_reviews"] = await BatchCritic(new).review(items, [])
    except Exception as exc:
        report["error"] = safe_error(exc)
    finally:
        atomic_json(destination, report)
        await old.client.close()
        await new.client.close()
    print(destination, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
