import json
import tempfile
import unittest
from pathlib import Path

from interaction_models.train import load_prompt_completion_rows


class TrainDatasetTest(unittest.TestCase):
    def test_loads_prompt_completion_rows(self):
        row = {
            "events": [{"role": "user", "text": "make this warmer", "dt_ms": 0}],
            "target": {"action": "respond", "messages": ["I'll make it warmer."]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            rows = load_prompt_completion_rows(path, max_samples=None)

        self.assertEqual(list(rows[0]), ["prompt", "completion"])
        self.assertTrue(rows[0]["prompt"].endswith("</conversation>\n"))
        self.assertEqual(
            rows[0]["completion"],
            "<act>respond</act>\n<msg>I'll make it warmer.</msg>\n<done/>",
        )


if __name__ == "__main__":
    unittest.main()
