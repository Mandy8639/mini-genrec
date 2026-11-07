"""Behavioral checks for evaluation integrity; no real API requests."""
import json
import argparse
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from openai import OpenAI
import pandas as pd

from mini_genrec.baseline import Baseline
from mini_genrec.context import Verbalizer
from mini_genrec.data import Case, Dataset, prepare
from mini_genrec.evaluate import evaluate_llm, find_elbow, metrics, run
from mini_genrec.llm import LLM, parse_response, validate


def movies():
    return pd.DataFrame({"title": ["Alpha (2000)", "Beta (2001)", "Gamma (2002)"],
                         "genres": ["Drama", "Comedy", "Drama"]}, index=[1, 2, 3])


class CoreTests(unittest.TestCase):
    def test_candidate_gate_drops_real_non_candidates_and_unknowns(self):
        result = validate(["Alpha (2000)", "Imaginary", "Gamma (2002)",
                           " alpha (2000) ", "Beta (2001)"], movies(), [1, 2])
        self.assertEqual(result.ranking, [1, 2])
        self.assertEqual((result.suggestions, result.outside_candidates,
                          result.outside_catalog, result.duplicates), (5, 2, 1, 1))

    def test_unconstrained_probe_measures_catalog_not_candidate_membership(self):
        result = validate(["Gamma (2002)", "Made up"], movies(), [1, 2], True)
        self.assertEqual(result.ranking, [3])
        self.assertEqual(result.outside_catalog / result.suggestions, 0.5)

    def test_ambiguous_titles_rejected(self):
        catalog = movies()
        catalog.loc[2, "title"] = " ALPHA (2000) "
        self.assertEqual(validate(["Alpha (2000)"], catalog, [1, 2]).ranking, [])

    def test_mrr_hit_and_empty_rankings(self):
        self.assertAlmostEqual(metrics([2, 1, 3], 3, {1})["rr"], 1 / 3)
        self.assertEqual(metrics([], 3, {1}),
                         dict(rr=0, hit10=0, popular_top10=0, top10_count=0))
        self.assertEqual(metrics(list(range(12)), 10, set())["hit10"], 0)

    def test_refusal_truncation_and_bad_structure_are_not_rankings(self):
        for response in [dict(refusal=True, finish_reason="stop", content='{}'),
                         dict(refusal=False, finish_reason="length", content='{}'),
                         dict(refusal=False, finish_reason="stop", content='{"ranking":[1]}')]:
            with self.assertRaises(ValueError):
                parse_response(response)

    def test_context_budget_compression_and_omission(self):
        history = pd.DataFrame({"movieId": [1, 1, 2, 3], "rating": [5., 4., 3., 1.],
                                "timestamp": [10, 20, 30, 40]})
        v = Verbalizer(movies(), "gpt-4o-mini", 4096)
        prompt = v.build(history, [1, 2], 50, 50)
        self.assertEqual(prompt.repeat_events_compressed, 1)
        self.assertEqual(prompt.low_signal_dropped, 1)
        self.assertEqual(prompt.history_events, 3)
        self.assertIn("2 ratings compressed", prompt.messages[1]["content"])
        self.assertLessEqual(prompt.input_tokens, 4096)
        # Keep candidates intact even when no history fits.
        empty = history.iloc[:0]
        minimum = v.build(empty, [1, 2], 50, 5).input_tokens
        v.budget = minimum
        small = v.build(history, [1, 2], 50, 50)
        self.assertLessEqual(small.input_tokens, minimum)
        self.assertIn("Beta (2001)", small.messages[1]["content"])
        v.budget = 1
        with self.assertRaises(ValueError):
            v.build(history, [1, 2], 50, 50)

    def test_baseline_no_positive_overlap_uses_popularity(self):
        train = pd.DataFrame({"userId": [1, 2, 3, 1], "movieId": [1, 1, 1, 2],
                              "rating": [3., 3., 3., 2.]})
        baseline = Baseline(train, [1, 2, 3])
        empty = train.iloc[:0]
        self.assertEqual(baseline.rank(empty, [3, 2, 1]), [1, 2, 3])

    def test_elbow_requires_measured_complete_conditions(self):
        frame = pd.DataFrame({"ranker": ["mini-genrec"] * 4,
                              "history_length": [5, 10, 20, 50],
                              "mrr": [.2, .3, .305, .301], "errors": [0] * 4})
        self.assertEqual(find_elbow(frame)["history_length"], 10)
        frame.loc[0, "errors"] = 1
        self.assertIsNone(find_elbow(frame)["history_length"])

    def test_real_sdk_request_and_cache_with_mock_transport(self):
        requests = []

        def serve(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "mock", "object": "chat.completion", "created": 0,
                "model": "gpt-4o-mini-test", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content":
                                '{"ranking":["Alpha (2000)","Beta (2001)"]}'},
                    "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})

        client = OpenAI(api_key="unit-test-placeholder", http_client=httpx.Client(
            transport=httpx.MockTransport(serve)))
        prompt = Verbalizer(movies(), "gpt-4o-mini").build(
            pd.DataFrame(columns=["movieId", "rating", "timestamp"]), [1, 2], 50, 5)
        with tempfile.TemporaryDirectory() as folder, patch("mini_genrec.llm.OpenAI", return_value=client):
            llm = LLM("gpt-4o-mini", Path(folder))
            first, second = llm.call(prompt), llm.call(prompt)
            self.assertEqual(first, second)
            self.assertEqual(len(requests), 1)
            self.assertEqual(first["prompt_tokens"], 100)
            self.assertEqual(parse_response(first), ["Alpha (2000)", "Beta (2001)"])
            self.assertEqual(requests[0]["response_format"]["type"], "json_schema")
        client.close()

    def test_failed_response_gets_zero_without_baseline_backfill(self):
        class Broken:
            def call(self, prompt):
                return dict(refusal=False, finish_reason="length", content="{", model="mock",
                            prompt_tokens=1, completion_tokens=1, request_hash="test")

        dataset = Dataset(movies(), pd.DataFrame(), [], {1}, {})
        case = Case(1, 2, 10, pd.DataFrame(), [1, 2])
        row = evaluate_llm(Broken(), None, {}, dataset, case, "mini-genrec")
        self.assertEqual(row["ranking"], [])
        self.assertEqual(row["rr"], 0)
        self.assertEqual(row["error"], "ValueError")


@unittest.skipUnless(Path("data/ml-latest-small/ratings.csv").exists(), "download MovieLens first")
class MovieLensIntegrityTests(unittest.TestCase):
    def test_temporal_split_candidate_sets_and_reproducibility(self):
        a = prepare(Path("data"), users=20)
        b = prepare(Path("data"), users=20)
        raw = pd.read_csv("data/ml-latest-small/ratings.csv")
        self.assertTrue((a.train.timestamp < a.metadata["cutoff_timestamp"]).all())
        self.assertTrue((a.train.groupby("userId").size() >= 20).all())
        self.assertEqual([c.candidates for c in a.cases], [c.candidates for c in b.cases])
        for case in a.cases:
            self.assertEqual(case.timestamp, int(raw[raw.userId == case.user_id].timestamp.max()))
            self.assertGreaterEqual(case.timestamp, a.metadata["cutoff_timestamp"])
            self.assertTrue((case.history.timestamp < case.timestamp).all())
            self.assertNotIn(case.target, set(case.history.movieId))
            self.assertEqual(len(case.candidates), 50)
            self.assertEqual(len(set(case.candidates)), 50)
            self.assertIn(case.target, case.candidates)
            self.assertFalse((set(case.candidates) - {case.target}) & set(case.history.movieId))
            self.assertTrue(set(case.candidates).issubset(a.movies.index))

    def test_complete_api_harness_with_mocked_provider(self):
        """Exercise all 4 windows, the free probe, metrics, CSVs and PNG generation."""
        requests = []

        def serve(request):
            payload = json.loads(request.content)
            requests.append(payload)
            user = payload["messages"][1]["content"]
            titles = ["Uncatalogued test-only title (2099)"]
            if "Candidates:\n" in user:
                candidates = json.loads(user.split("Candidates:\n", 1)[1])
                titles.extend(item["title"] for item in candidates)
            return httpx.Response(200, json={
                "id": "mock", "object": "chat.completion", "created": 0,
                "model": "gpt-4o-mini-test", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": json.dumps({"ranking": titles})},
                    "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})

        client = OpenAI(api_key="unit-test-placeholder", http_client=httpx.Client(
            transport=httpx.MockTransport(serve)))
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"OPENAI_API_KEY": "unit-test-placeholder"}), \
                patch("mini_genrec.llm.OpenAI", return_value=client), redirect_stdout(io.StringIO()):
            args = argparse.Namespace(users=2, candidates=50, catalog_size=1000, seed=42,
                model="gpt-4o-mini", token_budget=4096, data_dir="data",
                cache_dir=folder + "/cache", output=folder + "/results", baseline_only=False,
                update_readme=False)
            run(args)
            out = Path(args.output)
            summary = pd.read_csv(out / "metrics.csv")
            constrained = summary[summary.ranker == "mini-genrec"]
            self.assertEqual(len(requests), 10)
            self.assertEqual(len(constrained), 4)
            self.assertTrue((constrained.users == 2).all())
            self.assertTrue((constrained.errors == 0).all())
            self.assertTrue(np.allclose(constrained.outside_candidates_rate, 1 / 51))
            self.assertTrue((constrained.top10_slot_coverage == 1).all())
            free = summary[summary.ranker == "unconstrained"].iloc[0]
            self.assertEqual(free.outside_catalog_rate, 1)
            self.assertTrue(pd.isna(free.mrr))
            self.assertTrue((out / "elbow.png").exists())
            self.assertTrue(pd.read_csv(out / "elbow.csv").genrec_mrr.notna().all())
            # The known-bad suggestion must never survive the Python catalog gate.
            rows = json.loads((out / "per_user.json").read_text())
            for row in rows:
                self.assertTrue(all(isinstance(mid, int) for mid in row["ranking"]))
        client.close()


if __name__ == "__main__":
    unittest.main()
