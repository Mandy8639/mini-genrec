import tempfile
import unittest
from pathlib import Path

from mini_genrec.shortlist import merge_ranking, report


class ShortlistTests(unittest.TestCase):
    def test_merge_preserves_omissions_and_tail(self):
        baseline = list(range(1, 51))
        self.assertEqual(merge_ranking(baseline, []), baseline)
        final = merge_ranking(baseline, [20, 4])
        self.assertEqual(final[:5], [20, 4, 1, 2, 3])
        self.assertEqual(final[20:], baseline[20:])
        self.assertEqual(set(final), set(baseline))
        for invalid in ([21], [20, 20], [51]):
            with self.assertRaises(ValueError):
                merge_ranking(baseline, invalid)

    def test_report_counts_helped_hurt_and_fallback(self):
        baseline = {1: dict(user_id=1, rr=.05, hit10=0),
                    2: dict(user_id=2, rr=.2, hit10=1),
                    3: dict(user_id=3, rr=.01, hit10=0)}
        rows = [dict(user_id=1, rr=.1, hit10=1, ranking=[1], candidate_coverage=.05,
                     target_in_shortlist=True, outside_candidates=0, error="", api_cost_usd=0),
                dict(user_id=2, rr=.05, hit10=0, ranking=list(range(20)), candidate_coverage=1,
                     target_in_shortlist=True, outside_candidates=0, error="", api_cost_usd=0),
                dict(user_id=3, rr=.01, hit10=0, ranking=[], candidate_coverage=0,
                     target_in_shortlist=False, outside_candidates=0, error="timeout", error_category="timeout")]
        with tempfile.TemporaryDirectory() as folder:
            report(Path(folder), rows, baseline)
            text = (Path(folder) / "report.md").read_text()
        self.assertIn("Helped 1, hurt 1, unchanged 1.", text)
        self.assertIn("Full baseline fallbacks: 1; partial model rankings: 1", text)
        self.assertIn("Decision: Keep the hybrid baseline.", text)
