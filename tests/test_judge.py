import unittest
from unittest.mock import patch
from pydantic import ValidationError
from hiver.judge import Judgment, demo_examples, ReplyJudge

class JudgeTests(unittest.TestCase):
    def test_score_ranges_and_types(self):
        row = dict(relevance=2, grounding=2, usefulness=2, safety=2,
                   critical_error=False, explanation='Supported.')
        Judgment(**row)
        for replacement in [3, -1, True, '2']:
            with self.assertRaises(ValidationError):
                Judgment(**dict(row, relevance=replacement))

    def test_synthetic_inputs_have_no_answer_labels(self):
        examples = demo_examples()
        self.assertEqual(len(examples), 2)
        for example in examples:
            self.assertNotIn('expected_score', example)
            self.assertNotIn('expected_intent', example)
            self.assertNotIn('model', example)

    def test_constructor_does_not_call_api(self):
        with patch('hiver.judge.Groq') as client:
            judge = ReplyJudge()
            judge.close()
            client.assert_not_called()

if __name__ == '__main__':
    unittest.main()
