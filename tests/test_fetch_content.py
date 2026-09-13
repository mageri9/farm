import unittest
import json

from fetch_content import DEFAULT_FACTS


class FetchContentTests(unittest.TestCase):
    def test_seed_facts_provide_unique_topics_and_source_material(self) -> None:
        facts = json.loads(DEFAULT_FACTS.read_text(encoding="utf-8"))
        self.assertTrue(facts)
        topics = [fact["topic"] for fact in facts]
        self.assertEqual(len(topics), len(set(topics)))
        self.assertTrue(all(fact["title"] and fact["raw_data"] for fact in facts))


if __name__ == "__main__":
    unittest.main()
