import unittest

from interaction_models.verify import VerificationError, verify_report


class VerifyTest(unittest.TestCase):
    def test_verify_report_accepts_passing_metrics(self):
        report = {
            "doubletextbench": {
                "adapter": {
                    "total": 40,
                    "schema_valid_rate": 1.0,
                    "action_accuracy": 0.975,
                    "expected_contains_accuracy": 0.975,
                    "premature_response_rate": 0.0,
                    "results": [],
                }
            }
        }

        summary = verify_report(
            report,
            suite="doubletextbench",
            model_key="adapter",
            min_total=40,
            min_schema_valid_rate=1.0,
            min_action_accuracy=0.95,
            min_expected_contains_accuracy=0.95,
            max_premature_response_rate=0.0,
        )

        self.assertEqual(summary["total"], 40)

    def test_verify_report_rejects_failed_case_even_if_metrics_pass(self):
        report = {
            "doubletextbench": {
                "adapter": {
                    "total": 40,
                    "schema_valid_rate": 1.0,
                    "action_accuracy": 1.0,
                    "expected_contains_accuracy": 1.0,
                    "premature_response_rate": 0.0,
                    "results": [
                        {
                            "case_id": "bad",
                            "schema_valid": True,
                            "expected_action": "wait",
                            "predicted_action": "respond",
                            "contains_ok": None,
                        }
                    ],
                }
            }
        }

        with self.assertRaises(VerificationError):
            verify_report(
                report,
                suite="doubletextbench",
                model_key="adapter",
                min_total=40,
                min_schema_valid_rate=1.0,
                min_action_accuracy=0.95,
                min_expected_contains_accuracy=0.95,
                max_premature_response_rate=0.0,
            )


if __name__ == "__main__":
    unittest.main()
