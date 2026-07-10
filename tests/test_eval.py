import unittest

from interaction_models.eval import contains_expected_term


class EvalTest(unittest.TestCase):
    def test_expected_term_accepts_inflection_and_hyphenation(self):
        self.assertTrue(
            contains_expected_term(
                "Switching to customer-facing language.", "customers"
            )
        )
        self.assertTrue(
            contains_expected_term("Switching to two clear lines.", "two-line")
        )

    def test_expected_term_requires_complete_words(self):
        self.assertFalse(contains_expected_term("A doubleton request.", "double"))
        self.assertFalse(
            contains_expected_term("It should measure double-text handling.", "latest")
        )


if __name__ == "__main__":
    unittest.main()
