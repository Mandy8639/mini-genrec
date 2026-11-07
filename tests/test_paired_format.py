import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mini_genrec.paired_format import order, report


class PairedFormatTests(unittest.TestCase):
    def test_order_pairs_each_user_and_balances_first_arm(self):
        users = list(range(34))
        plan = order(users)
        self.assertEqual(len(set(plan)), 68)
        self.assertEqual(sum(arm == "title" for _, arm in plan[::2]), 17)
        for i, uid in enumerate(users):
            self.assertEqual(set(plan[2*i:2*i+2]), {(uid, "title"), (uid, "id")})

    def test_failures_stay_in_coverage_and_mrr_but_not_missing_token_pairs(self):
        rows = []
        for uid in (1, 2):
            for arm in ("title", "id"):
                failed = uid == 2 and arm == "id"
                rows.append(dict(user_id=uid, arm=arm, candidate_coverage=0 if failed else 1,
                                 rr=0 if failed else 1, hit10=0 if failed else 1,
                                 error="failed" if failed else "", error_category="http" if failed else "",
                                 api_output_tokens=None if failed else 100 if arm == "title" else 50,
                                 response_provider=None if failed else "Dots", api_cost_usd=0))
        with tempfile.TemporaryDirectory() as folder:
            report(Path(folder), rows)
            result = pd.read_csv(Path(folder) / "paired_comparison.csv").iloc[0]
        self.assertEqual(result.coverage_difference, -.5)
        self.assertEqual(result.rr_difference, -.5)
        self.assertEqual(result.output_token_pairs, 1)
        self.assertEqual(result.recommendation, "Keep titles")
