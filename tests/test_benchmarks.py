import unittest

from interaction_models.benchmarks import DOUBLE_TEXT_BENCH, map_semantic_turn_action


class BenchmarkTest(unittest.TestCase):
    def test_doubletextbench_has_minimum_coverage(self):
        self.assertGreaterEqual(len(DOUBLE_TEXT_BENCH), 40)
        self.assertEqual(
            {case.target.action for case in DOUBLE_TEXT_BENCH},
            {"wait", "respond", "interject", "continue"},
        )
        self.assertEqual(
            len({case.case_id for case in DOUBLE_TEXT_BENCH}),
            len(DOUBLE_TEXT_BENCH),
        )

    def test_semantic_turn_action_mapping(self):
        self.assertEqual(map_semantic_turn_action("start_speaking"), "respond")
        self.assertEqual(map_semantic_turn_action("<|continue_listening|>"), "wait")
        self.assertEqual(map_semantic_turn_action("start_listening"), "interject")
        self.assertEqual(map_semantic_turn_action("continue_speaking"), "continue")

    def test_semantic_turn_action_mapping_rejects_unknown(self):
        with self.assertRaises(ValueError):
            map_semantic_turn_action("sleep")


if __name__ == "__main__":
    unittest.main()
