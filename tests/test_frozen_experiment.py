"""Check frozen inputs and separation from the observed pilot; no inference calls."""
import hashlib
import json
from pathlib import Path
import unittest

from mini_genrec.context import Verbalizer, SYSTEM, response_format
from mini_genrec.data import prepare

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "experiments" / "mini-genrec-v1"


def read(name):
    return json.loads((FROZEN / name).read_text())


def messages_hash(messages):
    data = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


class FrozenArtifactTests(unittest.TestCase):
    def test_checksums_cover_every_frozen_file(self):
        manifest = read("checksums.json")
        expected = {p.name for p in FROZEN.iterdir() if p.is_file() and p.name != "checksums.json"}
        self.assertEqual(set(manifest["files"]), expected)
        for name, digest in manifest["files"].items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((FROZEN / name).read_bytes()).hexdigest(), digest)

    def test_cohorts_partition_cases_and_elbow_requires_fifty_events(self):
        cohorts = read("cohorts.json")
        cases = {c["user_id"]: c for c in read("cases.json")["cases"]}
        dev = set(cohorts["development"]["user_ids"])
        evaluation = set(cohorts["evaluation"]["user_ids"])
        elbow = set(cohorts["elbow"]["user_ids"])
        self.assertEqual(dev, {62, 418, 461})
        self.assertFalse(dev & evaluation)
        self.assertEqual(dev | evaluation, set(cases))
        self.assertEqual((len(dev), len(evaluation), len(elbow)), (3, 34, 21))
        self.assertEqual(elbow, {uid for uid in evaluation if cases[uid]["history_event_count"] >= 50})
        for cohort in ("development", "evaluation", "elbow"):
            ids = cohorts[cohort]["user_ids"]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(ids), cohorts[cohort]["count"])
        for cohort in ("evaluation", "elbow"):
            self.assertEqual(cohorts["selection_order"][cohort], cohorts[cohort]["user_ids"])

    def test_protocol_is_free_only_disabled_for_inference_and_budget_is_consistent(self):
        config = read("config.json")
        self.assertFalse(config["execution"]["inference_enabled"])
        self.assertEqual(config["execution"]["runner_integration"], "pending")
        self.assertEqual(config["llm"]["backend"], "openrouter")
        self.assertTrue(config["llm"]["free_only"])
        self.assertEqual(len(config["llm"]["models"]), 2)
        for model in config["llm"]["models"]:
            self.assertTrue(model.endswith(":free"))
            self.assertFalse(model.startswith("openrouter/"))
        self.assertEqual(config["llm"]["response_format"], response_format())
        self.assertEqual(config["context"]["system_message"], SYSTEM)
        cohorts = read("cohorts.json")
        total = 0
        for stage in config["planned_runs"]:
            requests = len(cohorts[stage["cohort"]]["user_ids"]) * len(config["llm"]["models"]) * len(stage["history_lengths"])
            self.assertEqual(stage["logical_requests"], requests)
            total += requests
        self.assertEqual(total, 330)
        self.assertEqual(total, config["planned_core_logical_requests"])

    def test_every_case_has_an_unseen_target_and_unique_catalog_candidates(self):
        catalog = read("catalog.json")
        ids = {m["movieId"] for m in catalog["movies"]}
        self.assertEqual(len(ids), 1000)
        self.assertTrue(set(catalog["popular_movie_ids"]).issubset(ids))
        self.assertEqual(len(catalog["popular_movie_ids"]), 100)
        cases = read("cases.json")["cases"]
        self.assertEqual(len({c["user_id"] for c in cases}), len(cases))
        for case in cases:
            with self.subTest(user=case["user_id"]):
                choices = case["candidate_movie_ids"]
                history = case["history"]
                seen = {row["movieId"] for row in history}
                self.assertEqual(len(history), case["history_event_count"])
                self.assertGreaterEqual(len(history), 20)
                self.assertEqual(len(choices), 50)
                self.assertEqual(len(set(choices)), 50)
                self.assertTrue(set(choices).issubset(ids))
                self.assertTrue(seen.issubset(ids))
                self.assertIn(case["target_movie_id"], choices)
                self.assertFalse(set(choices) & seen)
                self.assertTrue(all(row["timestamp"] < case["target_timestamp"] for row in history))

    def test_development_cases_exactly_preserve_the_observed_pilot(self):
        config = read("config.json")
        frozen = {c["user_id"]: c for c in read("cases.json")["cases"]}
        for source in config["split"]["pilot_case_sources"]:
            pilot = json.loads((ROOT / source).read_text())
            self.assertEqual(sorted(c["user_id"] for c in pilot), [62, 418, 461])
            for case in pilot:
                saved = frozen[case["user_id"]]
                self.assertEqual(saved["target_movie_id"], case["target"])
                self.assertEqual(saved["target_timestamp"], case["timestamp"])
                self.assertEqual(saved["candidate_movie_ids"], case["candidates"])
                self.assertEqual([r["movieId"] for r in saved["history"]], case["history_movie_ids"])

    def test_prompt_examples_belong_to_development_and_match_recorded_fingerprints(self):
        records = read("prompt_fingerprints.json")["fingerprints"]
        indexed = {(r["user_id"], r["variant"], r["history_length"]): r for r in records}
        case_ids = {c["user_id"] for c in read("cases.json")["cases"]}
        expected = {(uid, "constrained", length) for uid in case_ids for length in [5, 10, 20, 50]}
        expected |= {(uid, "unconstrained", 20) for uid in case_ids}
        self.assertEqual(len(records), 185)
        self.assertEqual(set(indexed), expected)
        for example in read("prompt_examples.json")["examples"]:
            self.assertIn(example["user_id"], {62, 418, 461})
            key = (example["user_id"], example["variant"], example["history_length"])
            self.assertEqual(messages_hash(example["messages"]), indexed[key]["messages_sha256"])


@unittest.skipUnless(all((ROOT / "data" / "ml-latest-small" / name).exists()
                         for name in ("ratings.csv", "movies.csv")), "download MovieLens first")
class FrozenReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read("config.json")
        cls.dataset = prepare(ROOT / "data", users=100000, candidates=50, catalog_size=1000, seed=42)

    def test_current_dataset_reproduces_frozen_catalog_cases_and_split(self):
        config = self.config
        dataset = self.dataset
        for field in ("ratings_sha256", "movies_sha256", "cutoff_timestamp", "train_ratings", "train_users"):
            self.assertEqual(config["dataset"][field], dataset.metadata[field])
        self.assertTrue((dataset.train.timestamp < config["dataset"]["cutoff_timestamp"]).all())
        catalog = read("catalog.json")
        self.assertEqual(catalog["movies"], [{"movieId": int(mid), "title": str(row.title), "genres": str(row.genres)}
                                           for mid, row in dataset.movies.iterrows()])
        self.assertEqual(catalog["popular_movie_ids"], sorted(map(int, dataset.popular)))
        frozen = {c["user_id"]: c for c in read("cases.json")["cases"]}
        self.assertEqual(set(frozen), {c.user_id for c in dataset.cases})
        for case in dataset.cases:
            saved = frozen[case.user_id]
            self.assertEqual(saved["target_movie_id"], case.target)
            self.assertEqual(saved["target_timestamp"], case.timestamp)
            self.assertEqual(saved["candidate_movie_ids"], case.candidates)
            self.assertEqual(saved["history"], [{"movieId": int(r.movieId), "rating": float(r.rating), "timestamp": int(r.timestamp)}
                                                for r in case.history.itertuples()])

    def test_both_model_contexts_reproduce_every_frozen_prompt(self):
        cases = {c.user_id: c for c in self.dataset.cases}
        records = read("prompt_fingerprints.json")
        self.assertEqual(records["applies_to_models"], self.config["llm"]["models"])
        for model in self.config["llm"]["models"]:
            verbalizer = Verbalizer(self.dataset.movies, model, self.config["context"]["estimated_input_token_budget"])
            self.assertEqual(verbalizer.encoding.name, self.config["context"]["token_estimator"])
            for record in records["fingerprints"]:
                case = cases[record["user_id"]]
                prompt = verbalizer.build(case.history, case.candidates, case.timestamp,
                                          record["history_length"], record["variant"] == "unconstrained")
                with self.subTest(model=model, user=case.user_id, variant=record["variant"], length=record["history_length"]):
                    self.assertEqual(messages_hash(prompt.messages), record["messages_sha256"])
                    self.assertEqual(prompt.input_tokens, record["approx_input_tokens"])
                    self.assertEqual(prompt.history_events, record["retained_history_events"])
                    self.assertLessEqual(prompt.input_tokens, self.config["context"]["estimated_input_token_budget"])


if __name__ == "__main__":
    unittest.main()
