import importlib.util
import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
DO_DIR = ROOT / "digital-ocean"
sys.path.insert(0, str(DO_DIR))
SPEC = importlib.util.spec_from_file_location(
    "quality_candidates", DO_DIR / "quality_candidates.py"
)
quality_candidates = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(quality_candidates)
GEN_SPEC = importlib.util.spec_from_file_location(
    "generate_candidates", DO_DIR / "generate_candidates.py"
)
generate_candidates = importlib.util.module_from_spec(GEN_SPEC)
assert GEN_SPEC.loader is not None
GEN_SPEC.loader.exec_module(generate_candidates)
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    "package_hf_dataset", DO_DIR / "package_hf_dataset.py"
)
package_hf_dataset = importlib.util.module_from_spec(PACKAGE_SPEC)
assert PACKAGE_SPEC.loader is not None
PACKAGE_SPEC.loader.exec_module(package_hf_dataset)
from candidate_utils import quality_errors


def row(action, suffix):
    if action == "wait":
        return {
            "events": [
                {
                    "role": "user",
                    "text": f"for the rollout note where {suffix}",
                    "dt_ms": 0,
                }
            ],
            "target": {"action": "wait", "messages": []},
        }
    if action == "respond":
        return {
            "events": [
                {
                    "role": "user",
                    "text": f"write the partner update for {suffix}",
                    "dt_ms": 0,
                }
            ],
            "target": {
                "action": "respond",
                "messages": [f"I will draft the partner update for {suffix}."],
            },
        }
    if action == "interject":
        return {
            "events": [
                {
                    "role": "user",
                    "text": f"draft the customer note for {suffix}",
                    "dt_ms": 0,
                },
                {
                    "role": "assistant",
                    "text": "Here is a formal version to start.",
                    "dt_ms": 700,
                },
                {"role": "user", "text": f"make it warmer for {suffix}", "dt_ms": 500},
            ],
            "target": {
                "action": "interject",
                "messages": [f"Got it, I will make the note warmer for {suffix}."],
            },
        }
    return {
        "events": [
            {
                "role": "user",
                "text": f"outline migration steps for {suffix}",
                "dt_ms": 0,
            },
            {
                "role": "assistant",
                "text": "First, confirm the affected services.",
                "dt_ms": 700,
            },
            {"role": "user", "text": "okay", "dt_ms": 400},
        ],
        "target": {
            "action": "continue",
            "messages": [f"Next, schedule the migration window for {suffix}."],
        },
    }


class DigitalOceanQualityTest(unittest.TestCase):
    def write_jsonl(self, path, rows):
        path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
            encoding="utf-8",
        )

    def test_quality_report_accepts_balanced_specific_rows(self):
        rows = [
            row(action, f"account {index}")
            for index in range(3)
            for action in quality_candidates.ACTIONS
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "valid.jsonl"
            self.write_jsonl(path, rows)

            report = quality_candidates.quality_report(path)

        self.assertEqual(report["valid"], 12)
        self.assertEqual(report["invalid"], 0)
        self.assertEqual(report["placeholder_rows"], 0)
        self.assertLess(report["duplicate_rate"], 0.01)
        self.assertEqual(
            report["action_counts"],
            {"wait": 3, "respond": 3, "interject": 3, "continue": 3},
        )

    def test_quality_report_flags_duplicates_and_placeholder_text(self):
        rows = [row("respond", "placeholder"), row("respond", "placeholder")]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            self.write_jsonl(path, rows)

            report = quality_candidates.quality_report(path)

        self.assertEqual(report["valid"], 2)
        self.assertEqual(report["placeholder_rows"], 2)
        self.assertGreater(report["duplicate_rate"], 0)

    def test_quality_errors_reject_short_placeholder_and_repeated_continue(self):
        bad = {
            "events": [
                {"role": "user", "text": "...", "dt_ms": 0},
                {
                    "role": "assistant",
                    "text": "Could you please provide more details?",
                    "dt_ms": 800,
                },
            ],
            "target": {
                "action": "continue",
                "messages": ["Could you please provide more details?"],
            },
        }

        errors = quality_errors(bad)

        self.assertIn("event 0 text is too short", errors)
        self.assertIn("target message 0 repeats prior assistant text", errors)

    def test_quality_errors_allow_continue_backchannel(self):
        good = row("continue", "database cutover")

        self.assertEqual(quality_errors(good), [])

    def test_placeholder_detection_uses_words(self):
        good = {
            "events": [
                {"role": "user", "text": "summarize the embargo plan", "dt_ms": 0}
            ],
            "target": {
                "action": "respond",
                "messages": ["I will summarize the embargo plan."],
            },
        }

        self.assertEqual(quality_errors(good), [])

    def test_placeholder_detection_allows_search_bar(self):
        good = {
            "events": [
                {
                    "role": "user",
                    "text": "Please revise the navigation layout",
                    "dt_ms": 0,
                }
            ],
            "target": {
                "action": "respond",
                "messages": ["I will move the search bar to the top header."],
            },
        }

        self.assertEqual(quality_errors(good), [])

    def test_next_action_fills_least_represented_action(self):
        counts = Counter({"continue": 4, "interject": 1, "respond": 1})

        self.assertEqual(generate_candidates.next_action(counts), "wait")

    def test_next_case_index_continues_after_accepted_and_rejected_attempts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = root / "accepted.jsonl"
            rejected = root / "rejected.jsonl"
            self.write_jsonl(accepted, [{"case_id": "do-llm-wait-17-000104"}])
            self.write_jsonl(rejected, [{"case_id": "do-llm-wait-17-000109"}])

            next_index = generate_candidates.next_case_index(accepted, rejected)

        self.assertEqual(next_index, 110)

    def test_wait_prompt_forbids_assistant_and_requires_empty_target(self):
        prompt = generate_candidates.build_prompt("wait", "technical migration")

        self.assertIn("Do not include an assistant event.", prompt)
        self.assertIn("target.messages must be []", prompt)
        self.assertIn('target.action must be exactly "wait"', prompt)
        self.assertIn('Do not use placeholders like "..."', prompt)

    def test_balanced_timing_prompt_and_validation(self):
        timing_range = ("quick", 251, 1000)
        prompt = generate_candidates.build_prompt("respond", "scheduling", timing_range)
        example = row("respond", "planning")
        example["events"].insert(
            0, {"role": "user", "text": "I need help planning", "dt_ms": 0}
        )
        example["events"][-1]["dt_ms"] = 700

        self.assertIn("between 251 and 1000 inclusive", prompt)
        self.assertEqual(generate_candidates.timing_error(example, timing_range), "")

    def test_failure_focused_prompt_and_grounding_validation(self):
        prompt = generate_candidates.build_prompt(
            "respond", "scheduling", failure_focused=True
        )
        example = row("respond", "Thursday afternoon")
        example["events"] = [
            {"role": "user", "text": "write the follow up", "dt_ms": 0},
            {
                "role": "user",
                "text": "mention Thursday afternoon availability",
                "dt_ms": 900,
            },
        ]
        raw = {"grounding_phrase": "Thursday afternoon"}

        self.assertIn('Add "grounding_phrase" at the top level.', prompt)
        self.assertEqual(
            generate_candidates.failure_focus_error(raw, example, "respond"), ""
        )

    def test_failure_focused_wait_requires_unfinished_text(self):
        good = row("wait", "the customer note about")
        bad = row("wait", "the customer note is ready")

        self.assertEqual(generate_candidates.failure_focus_error({}, good, "wait"), "")
        self.assertIn(
            "unmistakably unfinished",
            generate_candidates.failure_focus_error({}, bad, "wait"),
        )

    def test_package_hf_dataset_uses_stable_stratified_splits(self):
        rows = [
            row(action, f"account {index}")
            for index in range(5)
            for action in quality_candidates.ACTIONS
        ]
        for index, item in enumerate(rows):
            item["case_id"] = f"case-{index:03d}"
            item["source"] = "do-serverless-qwen3.5-397b-a17b"
            item["timing_bucket"] = "quick"

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "valid.jsonl"
            output = root / "hub"
            self.write_jsonl(source, rows)
            first = package_hf_dataset.package_dataset(
                source, output, "qwen3.5-397b-a17b", 0.2
            )
            first_train = (output / "data" / "train.jsonl").read_text(encoding="utf-8")
            second = package_hf_dataset.package_dataset(
                source, output, "qwen3.5-397b-a17b", 0.2
            )
            second_train = (output / "data" / "train.jsonl").read_text(encoding="utf-8")
            packaged_rows = [json.loads(line) for line in second_train.splitlines()]
            verified = package_hf_dataset.verify_package(output)
            (output / "data" / "train.jsonl").write_text(
                second_train + "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                package_hf_dataset.verify_package(output)

        self.assertEqual(first["train"], 16)
        self.assertEqual(first["validation"], 4)
        self.assertEqual(first["timing_counts"], {"quick": 20})
        self.assertEqual(first_train, second_train)
        self.assertEqual(first["files"], second["files"])
        self.assertEqual(verified["files"], second["files"])
        self.assertEqual(len({item["case_id"] for item in packaged_rows}), 16)
        self.assertTrue(
            all(item["case_id"].startswith("qwen35-") for item in packaged_rows)
        )
        self.assertTrue(
            all(
                item["text"] == f"{item['prompt']}{item['completion']}"
                for item in packaged_rows
            )
        )
        self.assertTrue(
            all(item["prompt"].endswith("</conversation>\n") for item in packaged_rows)
        )
        self.assertTrue(
            all(item["completion"].endswith("<done/>") for item in packaged_rows)
        )

    def test_extract_chat_content_handles_openai_shape(self):
        payload = {
            "choices": [{"message": {"content": '{"target":{"action":"wait"}}'}}]
        }

        self.assertEqual(
            generate_candidates.extract_chat_content(payload),
            '{"target":{"action":"wait"}}',
        )

    def test_call_do_serverless_uses_json_request(self):
        with TemporaryDirectory() as tmp:
            fake_doctl = Path(tmp) / "fake_doctl.py"
            fake_doctl.write_text(
                """#!/usr/bin/env python3
import json
import sys

request_path = sys.argv[sys.argv.index("--request") + 1]
with open(request_path, encoding="utf-8") as handle:
    request = json.load(handle)
assert request["model"] == "teacher-model"
assert request["max_tokens"] == 123
assert request["reasoning_effort"] == "none"
assert request["seed"] == 77
assert request["response_format"] == {"type": "json_object"}
print(json.dumps({"choices": [{"message": {"content": "{\\"events\\":[]}"}}]}))
""",
                encoding="utf-8",
            )
            fake_doctl.chmod(0o755)

            content = generate_candidates.call_do_serverless(
                doctl_bin=str(fake_doctl),
                model="teacher-model",
                prompt="make json",
                temperature=0.4,
                timeout=10,
                num_predict=123,
                seed=77,
            )

        self.assertEqual(content, '{"events":[]}')


if __name__ == "__main__":
    unittest.main()
