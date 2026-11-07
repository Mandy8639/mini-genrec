"""Frozen-runner integration tests. HTTP transport is mocked; artifacts are temporary."""
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI, APITimeoutError
import pandas as pd

from mini_genrec.baseline import Baseline, random_rank
from mini_genrec.context import Verbalizer
from mini_genrec.data import Case, Dataset
from mini_genrec.evaluate import evaluate_llm
from mini_genrec.experiment import ROOT, build_plan, load_snapshot, run
from mini_genrec.llm import RequestBudget


def args_for(folder, **overrides):
    values = dict(manifest=str(ROOT / "experiments/mini-genrec-v1"), data_dir=str(ROOT / "data"),
                  cache_dir=str(Path(folder)/"cache"), output=str(Path(folder)/"results"),
                  cohort="development", limit_users=1, models=["nvidia/nemotron-3.5-super-vl:free"],
                  request_budget=40, execute=False, baseline_only=False, dry_run=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def response(payload):
    content = payload["messages"][1]["content"]
    titles = ["Unknown test movie (2099)"]
    if "Candidates:\n" in content:
        candidates = json.loads(content.split("Candidates:\n", 1)[1])
        # Valid JSON with a missing candidate, an invalid suggestion and a duplicate.
        titles.extend(item["title"] for item in candidates[:-1])
        titles.append(candidates[0]["title"])
    return {"id": "mock", "object": "chat.completion", "created": 0, "model": payload["model"],
            "provider": "test-provider", "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps({"ranking": titles})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130, "cost": 0}}


class ContextAndDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.movies = pd.DataFrame({"title": ["Alpha (2000)", "Beta (2001)", "Gamma (2002)"],
                                    "genres": ["Drama"]*3}, index=[1, 2, 3])
        self.history = pd.DataFrame({"movieId": [1, 2, 3], "rating": [5., 3., 1.], "timestamp": [10, 20, 30]})

    def test_plain_history_keeps_neutral_ratings_and_is_chronological(self):
        v = Verbalizer(self.movies, "gpt-4o-mini")
        plain = v.build(self.history, [1, 2], 40, 20, strategy="plain_recent")
        body = plain.messages[1]["content"].split("History:\n", 1)[1].split("\n\n", 1)[0]
        self.assertLess(body.index("Alpha"), body.index("Beta"))
        self.assertLess(body.index("Beta"), body.index("Gamma"))
        self.assertIn("Beta (2001) 3/5", body)
        self.assertEqual((plain.history_events, plain.low_signal_dropped), (3, 0))
        engineered = v.build(self.history, [1, 2], 40, 20)
        self.assertEqual((engineered.history_events, engineered.low_signal_dropped), (2, 1))
        one = v.build(self.history.tail(1), [1, 2], 40, 20, strategy="plain_recent")
        v.budget = one.input_tokens
        packed = v.build(self.history, [1, 2], 40, 20, strategy="plain_recent")
        self.assertLessEqual(packed.input_tokens, v.budget)
        self.assertEqual(packed.history_events, 1)
        self.assertIn("You rated Gamma", packed.messages[1]["content"])

    def test_popularity_and_random_controls_are_candidate_order_invariant(self):
        train = pd.DataFrame({"userId": [1, 2, 3, 1], "movieId": [1, 1, 1, 2], "rating": [4.]*4})
        baseline = Baseline(train, [1, 2, 3])
        self.assertEqual(baseline.rank_popularity([3, 1, 2]), [1, 2, 3])
        self.assertEqual(random_rank([1, 2, 3], 42), random_rank([3, 2, 1], 42))
        self.assertEqual(set(random_rank([1, 2, 3], 42)), {1, 2, 3})

    def test_failures_are_classified_and_empty_or_short_rankings_are_not_parse_errors(self):
        dataset = Dataset(self.movies, pd.DataFrame(), [], {1}, {})
        case = Case(1, 2, 40, self.history, [1, 2])
        common = dict(error="", suggestions=0, outside_catalog=0, outside_candidates=0, duplicates=0)
        base = dict(prompt_tokens=1, completion_tokens=1, model="test", request_hash="test",
                    refusal=False, finish_reason="stop", content='{"ranking":[]}')

        class Fake:
            def __init__(self, result): self.result = result
            def call(self, prompt): return self.result

        for change, category in [({"refusal": True}, "refusal"), ({"finish_reason": "length"}, "truncation"),
                                  ({"content": "{"}, "json"), ({"content": '{"ranking":[2]}'}, "response_schema")]:
            row = evaluate_llm(Fake(dict(base, **change)), None, common, dataset, case, "mini-genrec")
            self.assertEqual(row["error_category"], category)
            self.assertEqual(row["candidate_coverage"], 0)
            self.assertEqual(row["rr"], 0)
        row = evaluate_llm(Fake(base), None, common, dataset, case, "mini-genrec")
        self.assertEqual(row["error"], "")
        self.assertEqual(row["candidate_coverage"], 0)

        class Timeout:
            def call(self, prompt):
                raise APITimeoutError(request=httpx.Request("POST", "https://mock.invalid"))
        row = evaluate_llm(Timeout(), None, common, dataset, case, "mini-genrec")
        self.assertEqual(row["error_category"], "timeout")


@unittest.skipUnless((ROOT / "data/ml-latest-small/ratings.csv").exists(), "download MovieLens first")
class FrozenRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = load_snapshot(ROOT / "experiments/mini-genrec-v1", ROOT / "data")

    def test_full_plan_is_exactly_330_unique_conditions_without_pilot_contamination(self):
        config, cohorts, _, cases, _, _ = self.snapshot
        selected, elbow, models, tasks = build_plan(config, cohorts, cases)
        self.assertEqual((len(selected), len(elbow), len(models), len(tasks)), (34, 21, 2, 330))
        self.assertFalse(set(selected) & {62, 418, 461})
        self.assertEqual(len({task.key for task in tasks}), 330)
        for model in models:
            for uid in elbow:
                self.assertEqual(sum(task.user_id == uid and task.model == model and task.strategy == "engineered_v1"
                                     and task.history_length == 20 and task.ranker == "mini-genrec" for task in tasks), 1)
        prefix = build_plan(config, cohorts, cases, limit=3)[0]
        self.assertEqual(prefix, selected[:3])

    def test_dry_run_and_baseline_only_never_construct_an_api_client(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), \
                patch("mini_genrec.experiment.LLM", side_effect=AssertionError("No inference allowed")):
            args = args_for(folder)
            plan = run(args)
            self.assertFalse(Path(args.output).exists())
            self.assertEqual(plan["logical_llm_conditions"], 6)
            args.baseline_only = True
            run(args)
            out = Path(args.output)
            rows = json.loads((out / "per_user.json").read_text())
            self.assertEqual({r["ranker"] for r in rows}, {"baseline", "popularity", "random"})
            self.assertTrue(all(r["candidate_coverage"] == 1 for r in rows))
            self.assertEqual(json.loads((out / "state.json").read_text())["rows"], [])
            self.assertEqual(json.loads((out / "status.json").read_text())["mode"], "baseline-only")
            self.assertEqual(len(rows), 12)
            self.assertTrue((out / "elbow.png").exists())
            self.assertTrue(pd.read_csv(out/"elbow.csv").genrec_mrr.isna().all())
            existing = Path(folder)/"existing"
            existing.mkdir()
            (existing/"metrics.csv").write_text("preserve existing report")
            args.output = str(existing)
            with self.assertRaisesRegex(ValueError, "not empty"):
                run(args)
            self.assertEqual((existing/"metrics.csv").read_text(), "preserve existing report")

    def test_tampered_snapshot_is_rejected_before_any_client(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)/"snapshot"
            shutil.copytree(ROOT / "experiments/mini-genrec-v1", destination)
            (destination/"cohorts.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                load_snapshot(destination, ROOT / "data")

    def test_mocked_execution_resumes_each_condition_and_reports_short_rankings(self):
        requests = []

        def serve(request):
            payload = json.loads(request.content)
            requests.append(payload)
            return httpx.Response(200, json=response(payload))

        def factory(**kwargs):
            return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(serve)))

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), \
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "unit-test-only"}), \
                patch("mini_genrec.llm.OpenAI", side_effect=factory):
            args = args_for(folder, execute=True)
            with patch("mini_genrec.experiment.RequestBudget", return_value=RequestBudget(2, interval=0)):
                with self.assertRaises(SystemExit): run(args)
            out = Path(args.output)
            self.assertEqual(json.loads((out/"status.json").read_text())["error_category"], "request_budget")
            self.assertEqual(len(json.loads((out/"state.json").read_text())["rows"]), 2)
            self.assertFalse((out/"metrics.csv").exists())
            budget = RequestBudget(20, interval=0)
            with patch("mini_genrec.experiment.RequestBudget", return_value=budget): run(args)
            self.assertEqual(len(requests), 6)
            self.assertEqual(budget.used, 4)
            state = json.loads((out/"state.json").read_text())
            self.assertEqual(len(state["rows"]), 6)
            self.assertEqual(len({r["condition_id"] for r in state["rows"]}), 6)
            summary = pd.read_csv(out/"metrics.csv")
            constrained = summary[summary.ranker == "mini-genrec"]
            self.assertTrue((constrained.mean_candidate_coverage == .98).all())
            self.assertTrue((constrained.incomplete_rankings == 1).all())
            self.assertTrue((constrained.errors == 0).all())
            self.assertEqual(set(summary.view), {"main", "context", "unconstrained", "elbow"})
            self.assertEqual(json.loads((out/"status.json").read_text())["status"], "complete")
            args.limit_users = 2
            with self.assertRaisesRegex(ValueError, "different experiment"):
                run(args)

    def test_http_failures_retain_status_and_category_without_sdk_error_body(self):
        for code, category, run_status in [(400, "configuration", "failed"), (401, "authentication", "failed"),
                                            (429, "rate_limit", "incomplete")]:
            def serve(request):
                return httpx.Response(code, json={"error": {"message": "private-provider-diagnostic", "type": "invalid_request_error"}})

            def factory(**kwargs):
                return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(serve)))

            with self.subTest(http_status=code), tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), \
                    patch.dict(os.environ, {"OPENROUTER_API_KEY": "unit-test-only"}), \
                    patch("mini_genrec.llm.OpenAI", side_effect=factory), \
                    patch("mini_genrec.experiment.RequestBudget", return_value=RequestBudget(1, interval=0)):
                args = args_for(folder, execute=True)
                with self.assertRaises(SystemExit): run(args)
                status_text = (Path(args.output)/"status.json").read_text()
                status = json.loads(status_text)
                self.assertEqual((status["status"], status["error_category"], status["http_status"]), (run_status, category, code))
                self.assertNotIn("private-provider-diagnostic", status_text)
                self.assertFalse((Path(args.output)/"metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
