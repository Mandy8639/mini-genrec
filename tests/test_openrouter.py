"""OpenRouter integration checks; every HTTP response is mocked, never billable."""
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI
import pandas as pd

from mini_genrec.context import Prompt
from mini_genrec.evaluate import model_folder, run_matrix
from mini_genrec.llm import LLM, RequestBudget, RunPaused


def prompt(content="test"):
    return Prompt([{"role": "user", "content": content}], 10, 0, 0, 0, 0, 0)


def completion(payload, titles):
    return {"id": "mock", "object": "chat.completion", "created": 0,
            "model": payload["model"], "provider": "mock-provider",
            "choices": [{"index": 0, "message": {"role": "assistant", "content":
                         json.dumps({"ranking": titles})}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost": 0.0}}


class OpenRouterTests(unittest.TestCase):
    def test_paid_models_and_dynamic_routers_are_rejected_before_client_creation(self):
        with tempfile.TemporaryDirectory() as folder, patch("mini_genrec.llm.OpenAI") as factory:
            for model in ["vendor/model", "openrouter/free", "openrouter/auto", "openrouter/anything:free"]:
                with self.assertRaises(ValueError):
                    LLM(model, Path(folder), "openrouter")
            factory.assert_not_called()

    def test_transport_credentials_schema_cache_isolation_and_attempt_budget(self):
        requests, clients, configs = [], [], []

        def serve(request):
            payload = json.loads(request.content)
            requests.append((request, payload))
            return httpx.Response(200, json=completion(payload, ["Alpha (2000)"]))

        def factory(**kwargs):
            configs.append(kwargs)
            client = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(serve)))
            clients.append(client)
            return client

        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                "OPENROUTER_API_KEY": "router-unit-test", "OPENAI_API_KEY": "openai-unit-test"}), \
                patch("mini_genrec.llm.OpenAI", side_effect=factory):
            budget = RequestBudget(1, interval=0)
            llm = LLM("test/model:free", Path(folder), "openrouter", budget)
            first = llm.call(prompt())
            self.assertEqual(first, llm.call(prompt()))
            self.assertEqual(budget.used, 1)
            self.assertEqual(len(requests), 1)
            request, payload = requests[0]
            self.assertEqual(str(request.url), "https://openrouter.ai/api/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer router-unit-test")
            self.assertEqual(payload["provider"], {"require_parameters": True})
            self.assertEqual(payload["max_tokens"], 2400)
            self.assertNotIn("max_completion_tokens", payload)
            self.assertEqual(configs[0]["max_retries"], 0)
            self.assertEqual(first["provider"], "mock-provider")
            self.assertEqual(first["cost_usd"], 0.0)
            with self.assertRaises(RunPaused):
                llm.call(prompt("changed prompt"))
            self.assertEqual(len(requests), 1)
            # Identical model/prompt on a different backend must not reuse router outputs.
            other = LLM("test/model:free", Path(folder), "openai")
            second = other.call(prompt())
            self.assertNotEqual(first["request_hash"], second["request_hash"])
            self.assertEqual(len(requests), 2)
        for client in clients:
            client.close()

    def test_quota_error_pauses_instead_of_becoming_ranking_failure_or_retry_storm(self):
        seen = []

        def serve(request):
            seen.append(request)
            return httpx.Response(429, json={"error": {"message": "Daily quota", "code": 429}})

        client = OpenAI(api_key="unit-test", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(serve)))
        with tempfile.TemporaryDirectory() as folder, patch("mini_genrec.llm.OpenAI", return_value=client):
            llm = LLM("test/model:free", Path(folder), "openrouter", RequestBudget(10, interval=0))
            with self.assertRaises(RunPaused):
                llm.call(prompt())
            self.assertEqual(len(seen), 1)
            self.assertEqual(list(Path(folder).glob("*.json")), [])
        client.close()

    def test_reasoning_and_output_budget_settings_are_sent_and_separate_cache_entries(self):
        payloads = []

        def serve(request):
            payload = json.loads(request.content)
            payloads.append(payload)
            return httpx.Response(200, json=completion(payload, ["Alpha (2000)"]))

        client = OpenAI(api_key="unit-test", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(serve)))
        with tempfile.TemporaryDirectory() as folder, patch("mini_genrec.llm.OpenAI", return_value=client):
            hashes = []
            for reasoning, maximum in [("default", 2400), ("off", 2400), ("off", 8192)]:
                llm = LLM("test/model:free", Path(folder), "openrouter", RequestBudget(3, interval=0),
                          reasoning=reasoning, max_output_tokens=maximum)
                hashes.append(llm.call(prompt())["request_hash"])
            self.assertEqual(len(set(hashes)), 3)
            self.assertNotIn("reasoning", payloads[0])
            self.assertEqual(payloads[1]["reasoning"], {"enabled": False})
            self.assertEqual(payloads[2]["max_tokens"], 8192)
        client.close()


@unittest.skipUnless(Path("data/ml-latest-small/ratings.csv").exists(), "download MovieLens first")
class MatrixTests(unittest.TestCase):
    def test_two_model_matrix_resumes_after_shared_budget_and_preserves_candidates(self):
        requests, clients = [], []

        def serve(request):
            payload = json.loads(request.content)
            requests.append(payload)
            text = payload["messages"][1]["content"]
            titles = ["Unknown test-only movie (2099)"]
            if "Candidates:\n" in text:
                titles.extend(movie["title"] for movie in json.loads(text.split("Candidates:\n", 1)[1]))
            return httpx.Response(200, json=completion(payload, titles))

        def factory(**kwargs):
            client = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(serve)))
            clients.append(client)
            return client

        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"OPENROUTER_API_KEY": "unit-test"}), \
                patch("mini_genrec.llm.OpenAI", side_effect=factory), redirect_stdout(io.StringIO()):
            args = argparse.Namespace(users=2, candidates=50, catalog_size=1000, seed=42,
                backend="openrouter", model=None, models=["test/a:free", "test/b:free"],
                request_budget=40, model_unused=None, token_budget=4096, data_dir="data",
                cache_dir=folder + "/cache", output=folder + "/results", baseline_only=False,
                update_readme=False)
            root = Path(args.output)
            with patch("mini_genrec.evaluate.RequestBudget", return_value=RequestBudget(6, interval=0)):
                with self.assertRaises(SystemExit):
                    run_matrix(args)
            self.assertEqual(len(requests), 6)
            self.assertFalse((root / "comparison.csv").exists())
            status = json.loads((root / model_folder(args.models[0]) / "status.json").read_text())
            self.assertEqual(status["status"], "incomplete")
            budget = RequestBudget(40, interval=0)
            with patch("mini_genrec.evaluate.RequestBudget", return_value=budget):
                run_matrix(args)
            self.assertEqual(len(requests), 20)  # Cached first six were not sent again.
            self.assertEqual(budget.used, 14)
            comparison = pd.read_csv(root / "comparison.csv")
            self.assertEqual(set(comparison.model), set(args.models))
            self.assertTrue((comparison.errors == 0).all())
            self.assertTrue((root / "comparison.md").exists())
            cases = [json.loads((root / model_folder(model) / "cases.json").read_text()) for model in args.models]
            self.assertEqual(cases[0], cases[1])
            for model in args.models:
                status = json.loads((root / model_folder(model) / "status.json").read_text())
                self.assertEqual(status["status"], "complete")
        for client in clients:
            client.close()


if __name__ == "__main__":
    unittest.main()
