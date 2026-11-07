import unittest

import pandas as pd

from mini_genrec.analysis import paired_comparison


class PairedComparisonTests(unittest.TestCase):
    def test_resampling_preserves_user_pairing_even_when_rows_are_reordered(self):
        left = pd.DataFrame({"user_id": [3, 1, 2], "rr": [.3, .1, .2], "hit10": [1, 1, 1]})
        right = pd.DataFrame({"user_id": [1, 2, 3], "rr": [0, .1, .2], "hit10": [0, 0, 0]})
        result = paired_comparison(left, right, "constant paired difference", 1000)
        self.assertAlmostEqual(result["rr_difference"], .1)
        self.assertAlmostEqual(result["rr_ci_low"], .1)
        self.assertAlmostEqual(result["rr_ci_high"], .1)
        self.assertEqual(result["hit10_difference"], 1)
        self.assertEqual(result["users_better_rr"], 3)

    def test_mismatched_or_duplicate_users_are_rejected(self):
        left = pd.DataFrame({"user_id": [1, 2], "rr": [0, 0], "hit10": [0, 0]})
        with self.assertRaises(ValueError):
            paired_comparison(left, left.iloc[:1], "missing user")
        with self.assertRaises(ValueError):
            paired_comparison(pd.concat([left, left]), left, "duplicates")


if __name__ == "__main__":
    unittest.main()
