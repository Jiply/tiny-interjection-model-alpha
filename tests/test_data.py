import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from interaction_models.benchmarks import DOUBLE_TEXT_BENCH
from interaction_models.data import (
    build_dataset,
    events_from_record,
    examples_from_events,
    rows_from_adaptive_file,
)
from interaction_models.io import read_jsonl
from interaction_models.schema import Event
from interaction_models.synthetic import synthetic_examples, synthetic_question_examples


class DataTest(unittest.TestCase):
    def test_events_from_record_preserves_consecutive_same_speaker_messages(self):
        record = {
            "messages": [
                {"role": "user", "text": "can you help write this", "dt_ms": 0},
                {"role": "user", "text": "make it warmer actually", "dt_ms": 1300},
                {"role": "assistant", "text": "Sure.", "dt_ms": 900},
            ]
        }

        events = events_from_record(record)

        self.assertEqual(
            [event.role for event in events], ["user", "user", "assistant"]
        )
        self.assertEqual(events[1].text, "make it warmer actually")

    def test_examples_from_events_maps_wait_respond_interject_continue(self):
        wait_events = (
            Event(role="user", text="wait I mean the part where", dt_ms=0),
            Event(role="user", text="she says yes", dt_ms=900),
            Event(role="assistant", text="I will use that line.", dt_ms=700),
        )
        wait_examples = examples_from_events(
            wait_events, source="test", case_prefix="wait"
        )
        self.assertTrue(
            any(example.target.action == "wait" for example in wait_examples)
        )
        self.assertTrue(
            any(example.target.action == "respond" for example in wait_examples)
        )

        interject_events = (
            Event(role="user", text="draft it", dt_ms=0),
            Event(role="assistant", text="Here is a formal version.", dt_ms=700),
            Event(role="user", text="actually make it casual", dt_ms=600),
            Event(role="assistant", text="Here is a casual version.", dt_ms=700),
        )
        interject_examples = examples_from_events(
            interject_events, source="test", case_prefix="interject"
        )
        self.assertTrue(
            any(example.target.action == "interject" for example in interject_examples)
        )

        continue_events = (
            Event(role="user", text="give me steps", dt_ms=0),
            Event(role="assistant", text="First, gather examples.", dt_ms=700),
            Event(role="assistant", text="Second, train the adapter.", dt_ms=700),
        )
        continue_examples = examples_from_events(
            continue_events, source="test", case_prefix="continue"
        )
        self.assertTrue(
            any(example.target.action == "continue" for example in continue_examples)
        )

    def test_synthetic_examples_cover_all_actions(self):
        examples = synthetic_examples(80, seed=3)

        self.assertEqual(len(examples), 80)
        self.assertEqual(
            {example.target.action for example in examples},
            {"wait", "respond", "interject", "continue"},
        )
        self.assertEqual(len({example.case_id for example in examples}), 80)
        self.assertTrue(
            any(
                str(example.case_id).startswith("synthetic-continue-assistant")
                for example in examples
            )
        )
        self.assertTrue(
            any(
                str(example.case_id).startswith("synthetic-continue-backchannel")
                for example in examples
            )
        )

    def test_synthetic_question_boost_only_contains_grounded_responses(self):
        examples = synthetic_question_examples(80, seed=5)

        self.assertEqual(len(examples), 80)
        self.assertTrue(all(example.target.action == "respond" for example in examples))
        self.assertTrue(
            all(example.source == "synthetic-question-boost" for example in examples)
        )
        benchmark_answers = [
            example.target.messages[0]
            for example in examples
            if "benchmark" in example.events[-1].text
        ]
        self.assertTrue(benchmark_answers)
        self.assertTrue(
            all(
                "double-text" in answer.lower() and "latest" in answer.lower()
                for answer in benchmark_answers
            )
        )

    def test_synthetic_completions_preserve_semantic_content(self):
        examples = synthetic_examples(400, seed=11)
        single_responses = [
            example
            for example in examples
            if str(example.case_id).startswith("synthetic-respond-single")
        ]
        continuations = [
            example for example in examples if example.target.action == "continue"
        ]

        benchmark_answers = [
            example.target.messages[0]
            for example in single_responses
            if "benchmark" in example.events[-1].text
        ]
        self.assertTrue(benchmark_answers)
        self.assertTrue(
            all(
                "double-text" in answer.lower() and "latest" in answer.lower()
                for answer in benchmark_answers
            )
        )
        self.assertTrue(
            all(
                example.events[1].text.startswith(
                    ("First", "Option one", "Stage one", "Subject:", "The first")
                )
                and example.target.messages[0].startswith("Second")
                for example in continuations
            )
        )
        self.assertTrue(
            any(
                example.target.action == "respond"
                and len(example.events) == 2
                and ", and make it" not in example.events[-1].text
                and "actually make it" not in example.events[-1].text
                for example in examples
            )
        )
        self.assertTrue(
            any(
                example.target.action == "interject"
                and "for " not in example.target.messages[0]
                for example in examples
            )
        )
        self.assertTrue(
            any(
                example.target.action == "interject"
                and "two-line" in example.target.messages[0]
                for example in examples
            )
        )

    def test_adaptive_rows_only_use_enhancements_that_preserve_action(self):
        rows = [
            {
                "case_id": "safe",
                "prompt": "prompt one\n",
                "completion": "<act>respond</act>\n<msg>Original.</msg>\n<done/>",
                "enhanced_completion": "<act>respond</act>\n<msg>Richer answer.</msg>\n<done/>",
            },
            {
                "case_id": "changed-label",
                "prompt": "prompt two\n",
                "completion": "<act>respond</act>\n<msg>Trusted answer.</msg>\n<done/>",
                "enhanced_completion": "<act>wait</act>\n<done/>",
            },
            {
                "case_id": "wait-shape",
                "prompt": "prompt three\n",
                "completion": "<act>wait</act>\n<done/>",
                "enhanced_completion": "<act>wait</act>\n<msg></msg>\n<done/>",
            },
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "adaptive.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            selected = rows_from_adaptive_file(path)

        self.assertIn("Richer answer", selected[0]["completion"])
        self.assertIn("Trusted answer", selected[1]["completion"])
        self.assertEqual(selected[2]["completion"], rows[2]["completion"])

    def test_build_dataset_keeps_doubletextbench_as_holdout(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            counts = build_dataset(
                output_dir=output_dir,
                include_public=False,
                public_limit=0,
                synthetic_count=80,
                val_ratio=0.2,
                seed=7,
            )

            train_rows = list(read_jsonl(output_dir / "train.jsonl"))
            val_rows = list(read_jsonl(output_dir / "val.jsonl"))

        self.assertEqual(
            counts,
            {
                "train": 64,
                "val": 16,
                "doubletextbench": len(DOUBLE_TEXT_BENCH),
            },
        )
        self.assertTrue(
            all(row["source"] == "synthetic" for row in train_rows + val_rows)
        )
        self.assertEqual(
            {row["target"]["action"] for row in train_rows + val_rows},
            {"wait", "respond", "interject", "continue"},
        )
        self.assertTrue(
            all(
                row["text"] == f"{row['prompt']}{row['completion']}"
                for row in train_rows + val_rows
            )
        )
        self.assertTrue(
            all(row["prompt"].endswith("</conversation>\n") for row in train_rows)
        )
        self.assertTrue(
            all(row["completion"].endswith("<done/>") for row in train_rows)
        )


if __name__ == "__main__":
    unittest.main()
