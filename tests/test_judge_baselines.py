import unittest
from hiver.judge_baselines import selected_ids

class SamplingTests(unittest.TestCase):
    def test_sampling_is_reproducible_and_order_independent(self):
        ids = [str(i) for i in range(50)]
        self.assertEqual(selected_ids(ids), selected_ids(list(reversed(ids))))
        self.assertEqual(len(set(selected_ids(ids))), 10)

    def test_rejects_duplicate_or_small_pool(self):
        for ids in [['1']*50, ['1','2']]:
            with self.assertRaises(ValueError):
                selected_ids(ids)

if __name__ == '__main__':
    unittest.main()
