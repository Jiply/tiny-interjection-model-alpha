import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


class NebiusJobTest(unittest.TestCase):
    def test_image_includes_python_headers_for_triton(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("python3-dev", dockerfile)

    def test_project_includes_sentencepiece_for_gguf_conversion(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('"sentencepiece>=0.2.0"', pyproject)

    def test_entrypoint_rejects_missing_configuration(self):
        completed = subprocess.run(
            ["bash", "nebius/job_entrypoint.sh"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("DATASET_REPO is required", completed.stderr)

    def test_submit_uses_serverless_job_and_secret_reference(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            arguments_path = tmp_path / "arguments.txt"
            fake_nebius = tmp_path / "nebius"
            fake_nebius.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$ARGUMENTS_PATH"\n',
                encoding="utf-8",
            )
            fake_nebius.chmod(0o755)
            environment = os.environ | {
                "ARGUMENTS_PATH": str(arguments_path),
                "PATH": f"{tmp_path}:{os.environ['PATH']}",
            }

            subprocess.run(
                [
                    "bash",
                    "nebius/submit_job.sh",
                    "registry.example/tim:v1",
                    "example/dataset",
                    "abc123",
                    "example/model",
                    "hf-secret",
                    "subnet-id",
                ],
                cwd=ROOT,
                env=environment,
                check=True,
            )
            arguments = arguments_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(arguments[:3], ["ai", "job", "create"])
        self.assertIn("HF_TOKEN=hf-secret", arguments)
        self.assertIn("gpu-l40s-a", arguments)
        self.assertIn("1gpu-8vcpu-32gb", arguments)
        self.assertNotIn("--registry-password", arguments)

    def test_submit_allows_public_data_without_hugging_face_secret(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            arguments_path = tmp_path / "arguments.txt"
            fake_nebius = tmp_path / "nebius"
            fake_nebius.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$ARGUMENTS_PATH"\n',
                encoding="utf-8",
            )
            fake_nebius.chmod(0o755)
            environment = os.environ | {
                "ARGUMENTS_PATH": str(arguments_path),
                "PATH": f"{tmp_path}:{os.environ['PATH']}",
            }

            subprocess.run(
                [
                    "bash",
                    "nebius/submit_job.sh",
                    "registry.example/tim:v1",
                    "example/dataset",
                    "abc123",
                    "example/model",
                    "-",
                    "subnet-id",
                ],
                cwd=ROOT,
                env=environment,
                check=True,
            )
            arguments = arguments_path.read_text(encoding="utf-8").splitlines()

        self.assertNotIn("--env-secret", arguments)


if __name__ == "__main__":
    unittest.main()
