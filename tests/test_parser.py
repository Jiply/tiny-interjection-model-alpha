import unittest

from interaction_models.parser import TargetParseError, parse_target, target_to_text
from interaction_models.schema import Target


class ParserTest(unittest.TestCase):
    def test_parse_target_round_trip(self):
        target = Target(action="respond", messages=("Got it.", "Here is the version."))

        self.assertEqual(parse_target(target_to_text(target)), target)

    def test_parse_target_rejects_missing_done(self):
        with self.assertRaises(TargetParseError):
            parse_target("<act>respond</act><msg>Hello</msg>")

    def test_parse_target_rejects_wait_with_message(self):
        with self.assertRaises(TargetParseError):
            parse_target("<act>wait</act><msg>hello</msg><done/>")

    def test_parse_target_rejects_unknown_action(self):
        with self.assertRaises(TargetParseError):
            parse_target("<act>pause</act><done/>")


if __name__ == "__main__":
    unittest.main()
