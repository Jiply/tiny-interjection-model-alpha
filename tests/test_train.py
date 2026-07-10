import json
import tempfile
import unittest
from pathlib import Path

from interaction_models.train import build_parser, load_prompt_completion_rows


class TrainDatasetTest(unittest.TestCase):
    def test_accepts_resume_checkpoint(self):
        args = build_parser().parse_args(
            ["--resume-from-checkpoint", "runs/example/checkpoint-600"]
        )

        self.assertEqual(
            args.resume_from_checkpoint, Path("runs/example/checkpoint-600")
        )

    def test_accepts_trainable_initial_adapter(self):
        args = build_parser().parse_args(["--initial-adapter", "adapters/example"])

        self.assertEqual(args.initial_adapter, Path("adapters/example"))

    def test_derives_prompt_completion_from_legacy_rows(self):
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

    def test_loads_explicit_prompt_completion_without_structured_columns(self):
        row = {
            "prompt": "Evolved interaction prompt\n",
            "completion": "<act>wait</act>\n<done/>",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            rows = load_prompt_completion_rows(path, max_samples=None)

        self.assertEqual(rows, [row])


if __name__ == "__main__":
    unittest.main()
