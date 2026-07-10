import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ArtifactRetentionTest(unittest.TestCase):
    def test_clean_targets_only_remove_caches(self):
        root_makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        teacher_makefile = (ROOT / "digital-ocean" / "Makefile").read_text(
            encoding="utf-8"
        )
        root_clean = root_makefile.split("clean:\n", maxsplit=1)[1]
        teacher_clean = teacher_makefile.split("clean:\n", maxsplit=1)[1]

        for retained in ("adapters", "data/processed", "data/raw", "runs"):
            self.assertNotIn(retained, root_clean)
        self.assertNotIn("$(DATA_DIR)", teacher_clean)
        self.assertNotIn('rm -rf "$(PACKAGE_DIR)"', teacher_clean)

    def test_training_archives_gguf_before_reporting_failed_release_gate(self):
        runner = (ROOT / "nebius" / "train_qwen_lora.sh").read_text(encoding="utf-8")
        archive = runner.index("experiments/tim-q4_k_m.gguf")
        failed_gate = runner.index('if [[ "$VERIFICATION_FAILED" -ne 0 ]]')

        self.assertLess(archive, failed_gate)
        self.assertIn('echo "QUALITY_GATE_STATUS=failed"', runner)
        self.assertIn('echo "QUALITY_GATE_STATUS=passed"', runner)
        self.assertNotIn(
            'echo "Experiment artifacts were archived, but the quality gate failed; nothing was promoted as a release."',
            runner,
        )


if __name__ == "__main__":
    unittest.main()
