import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from interaction_models.inference import InteractionEngine, call_llama_cli
from interaction_models.schema import Event


class LlamaInferenceTest(unittest.TestCase):
    @patch("subprocess.run")
    def test_call_llama_cli_extracts_decision(self, run):
        run.return_value = SimpleNamespace(
            stdout="loading\n<act>wait</act>\n<done/>\nexiting\n"
        )

        raw = call_llama_cli(
            executable=Path("llama-cli"),
            model_path=Path("base.gguf"),
            adapter_path=Path("adapter.gguf"),
            prompt="events",
            max_new_tokens=99,
        )

        command = run.call_args.args[0]
        self.assertIn("--lora", command)
        self.assertIn("adapter.gguf", command)
        self.assertIn("4096", command)
        self.assertEqual(raw, "<act>wait</act>\n<done/>")

    @patch("interaction_models.inference.call_llama_cli")
    def test_engine_uses_llama_cli_without_loading_transformers(self, llama_cli):
        llama_cli.return_value = (
            "<act>interject</act>\n<msg>Switching now.</msg>\n<done/>"
        )
        engine = InteractionEngine(
            llama_cli_path=Path("llama-cli"),
            llama_model_path=Path("base.gguf"),
            llama_adapter_path=Path("adapter.gguf"),
        )

        decision = engine.decide(
            (
                Event(role="assistant", text="Starting the first version.", dt_ms=0),
                Event(role="user", text="Wait, make it shorter.", dt_ms=400),
            )
        )

        self.assertEqual(decision.mode, "llama.cpp")
        self.assertEqual(decision.target.action, "interject")
        self.assertEqual(decision.target.messages, ("Switching now.",))


if __name__ == "__main__":
    unittest.main()
