import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from interaction_models.benchmarks import DOUBLE_TEXT_BENCH
from interaction_models.data import (
    build_dataset,
    events_from_record,
    examples_from_events,
)
from interaction_models.io import read_jsonl
from interaction_models.schema import Event
from interaction_models.synthetic import synthetic_examples


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


if __name__ == "__main__":
    unittest.main()
