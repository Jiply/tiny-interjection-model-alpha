import unittest

from interaction_models.format import build_prompt, serialize_events
from interaction_models.schema import Event


class FormatTest(unittest.TestCase):
    def test_serialize_events_is_stable(self):
        events = (
            Event(role="user", text="can you help write this", dt_ms=0),
            Event(role="user", text="make it warmer actually", dt_ms=1300),
        )

        self.assertEqual(
            serialize_events(events),
            "\n".join(
                [
                    "<conversation>",
                    '<event role="user" dt_ms="0">can you help write this</event>',
                    '<event role="user" dt_ms="1300">make it warmer actually</event>',
                    "</conversation>",
                ]
            ),
        )

    def test_build_prompt_ends_after_conversation(self):
        prompt = build_prompt((Event(role="user", text="hello", dt_ms=0),))

        self.assertTrue(prompt.endswith("</conversation>"))
        self.assertNotIn("<decision>", prompt)
        self.assertIn("<act>wait|respond|interject|continue</act>", prompt)


if __name__ == "__main__":
    unittest.main()
