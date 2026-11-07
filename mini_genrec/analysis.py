"""Audit and compare completed frozen results, without API calls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import metrics
from .experiment import ROOT
from .llm import parse_response, validate


def archive_responses(folder: Path, cache_dir: Path):
    """Copy only response fields, never request headers or credentials, into the run."""
    status = json.loads((folder / "status.json").read_text())
    if status.get("status") != "complete" or status.get("mode") != "api":
        raise ValueError("Only completed API experiments can archive responses.")
    state = json.loads((folder / "state.json").read_text())
    keys = {r["request_hash"] for r in state["rows"] if r.get("request_hash")}
    allowed = {"content", "refusal", "finish_reason", "model", "provider", "prompt_tokens", "completion_tokens",
               "reasoning_tokens", "cost_usd", "request_hash", "response_id", "latency_seconds", "backend"}
    responses, missing = {}, []
    for key in sorted(keys):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid response cache hash.")
        path = cache_dir / (key + ".json")
        if not path.exists():
            missing.append(key)
            continue
        response = json.loads(path.read_text())
        responses[key] = {name: value for name, value in response.items() if name in allowed}
    archive = dict(schema_version=1, responses=responses, missing_hashes=missing)
    (folder / "responses.json").write_text(json.dumps(archive, ensure_ascii=False, indent=2) + "\n")
    return archive


def paired_comparison(left, right, label, samples=10000, seed=42):
    """Percentile bootstrap over paired users; failed responses retain their zero scores."""
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples.")
    if left.user_id.duplicated().any() or right.user_id.duplicated().any():
        raise ValueError("Each side must have one observation per user.")
    if set(left.user_id) != set(right.user_id) or left.empty:
        raise ValueError("Paired comparisons require identical nonempty user sets.")
    a, b = left.set_index("user_id").sort_index(), right.set_index("user_id").sort_index()
    indices = np.random.default_rng(seed).integers(0, len(a), size=(samples, len(a)))
    result = dict(comparison=label, users=len(a), bootstrap_samples=samples, seed=seed)
    for metric in ("rr", "hit10"):
        differences = a[metric].to_numpy(dtype=float) - b[metric].to_numpy(dtype=float)
        means = differences[indices].mean(axis=1)
        result[metric + "_difference"] = float(differences.mean())
        result[metric + "_ci_low"], result[metric + "_ci_high"] = map(float, np.quantile(means, [.025, .975]))
    result["users_better_rr"] = int((a.rr > b.rr).sum())
    result["users_tied_rr"] = int((a.rr == b.rr).sum())
    result["users_worse_rr"] = int((a.rr < b.rr).sum())
    return result


def analyze(folder: Path, samples=10000, seed=42, update_readme=False):
    status = json.loads((folder / "status.json").read_text())
    if status.get("status") != "complete" or status.get("mode") != "api":
        raise ValueError("Analysis requires a completed API experiment, not partial or baseline-only results.")
    run = json.loads((folder / "run.json").read_text())
    records = json.loads((folder / "per_user.json").read_text())
    state = json.loads((folder / "state.json").read_text())
    snapshot = ROOT / "experiments" / run["experiment_id"]
    cases = {c["user_id"]: c for c in json.loads((snapshot / "cases.json").read_text())["cases"]}
    catalog = json.loads((snapshot / "catalog.json").read_text())
    popular = set(catalog["popular_movie_ids"])
    catalog_ids = {m["movieId"] for m in catalog["movies"]}
    movies = pd.DataFrame(catalog["movies"]).set_index("movieId")
    archive = json.loads((folder / "responses.json").read_text())["responses"] if (folder / "responses.json").exists() else {}
    raw_audited = 0
    llm = [row for row in records if row["model"]]
    if len(llm) != run["logical_llm_conditions"] or len(state["rows"]) != len(llm):
        raise ValueError("Model condition count differs from the completed plan.")
    keys = [r["condition_id"] for r in llm]
    if len(keys) != len(set(keys)) or set(keys) != {r["condition_id"] for r in state["rows"]}:
        raise ValueError("Duplicate, foreign or missing condition IDs.")
    for row in records:
        case = cases[row["user_id"]]
        ranking = row["ranking"]
        allowed = catalog_ids if row["ranker"] == "unconstrained" else set(case["candidate_movie_ids"])
        if len(ranking) != len(set(ranking)) or not set(ranking).issubset(allowed):
            raise ValueError("Persisted ranking violates the catalog/candidate gate.")
        expected = metrics(ranking, case["target_movie_id"], popular)
        if any(not np.isclose(expected[key], row[key]) for key in expected):
            raise ValueError("Persisted metrics do not match the actual ranking.")
        if row.get("request_hash") in archive and not row.get("error"):
            response = archive[row["request_hash"]]
            validated = validate(parse_response(response), movies, case["candidate_movie_ids"],
                                 row["ranker"] == "unconstrained")
            for key in ("ranking", "suggestions", "outside_candidates", "outside_catalog", "duplicates"):
                if getattr(validated, key) != row[key]:
                    raise ValueError("Archived raw response does not reproduce persisted validation results.")
            raw_audited += 1
    frame = pd.DataFrame(records)
    main = frame[(frame.history_length == 20) & (frame.ranker != "unconstrained")]
    baseline = main[main.ranker == "baseline"]
    comparisons = []
    engineered = {}
    for model in run["models"]:
        engineered[model] = main[(main.model == model) & (main.strategy == "engineered_v1")]
        plain = main[(main.model == model) & (main.strategy == "plain_recent")]
        for data in (engineered[model], plain):
            if set(data.user_id) != set(run["selected_users"]):
                raise ValueError("Main conditions do not cover the frozen evaluation cohort.")
        comparisons.extend([
            paired_comparison(engineered[model], baseline, f"{model}: engineered minus hybrid baseline", samples, seed),
            paired_comparison(engineered[model], plain, f"{model}: engineered minus plain", samples, seed)])
    if len(run["models"]) == 2:
        first, second = run["models"]
        comparisons.append(paired_comparison(engineered[first], engineered[second],
                                              f"{first} minus {second}: engineered", samples, seed))
    table = pd.DataFrame(comparisons)
    table.to_csv(folder / "paired_comparisons.csv", index=False)
    known_costs = [row["api_cost_usd"] for row in llm if row.get("api_cost_usd") is not None]
    summary = pd.read_csv(folder / "metrics.csv").fillna({"model": ""})
    main_summary = summary[summary.view == "main"]
    lines = ["# Frozen evaluation audit", "", f"Audited {len(records)} predictions, including {len(llm)} LLM conditions. "
             "Recomputed rank metrics and verified catalog/candidate membership from the saved rankings. "
             f"Replayed title validation from archived raw responses for {raw_audited} successful conditions.", "",
             "| Ranker (20-event window) | Users | MRR | Hit@10 | Popular share | Candidate coverage |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in main_summary.itertuples():
        label = row.model or row.ranker
        lines.append(f"| {label} | {row.users} | {row.mrr:.4f} | {row.hit10:.1%} "
                     f"| {row.popular_top10_fraction:.1%} | {row.mean_candidate_coverage:.1%} |")
    lines.extend(["", "## Context comparison", "",
                  "| Model | Strategy | MRR | Hit@10 | Estimated input tokens | Native input tokens | Retained events |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for model in run["models"]:
        for strategy in ("engineered_v1", "plain_recent"):
            row = summary[(summary.view == "context") & (summary.model == model) & (summary.strategy == strategy)].iloc[0]
            lines.append(f"| {model} | {strategy} | {row.mrr:.4f} | {row.hit10:.1%} "
                         f"| {row.mean_approx_input_tokens:.0f} | {row.mean_api_input_tokens:.0f} | {row.mean_retained_history_events:.1f} |")
    lines.extend(["", "## Paired comparisons", "",
             f"Differences use the same {len(run['selected_users'])} users. Intervals are 95% percentile bootstrap intervals "
             f"with {samples:,} paired-user resamples and seed {seed}. Request/parse failures remain zero-score observations. "
             "These exploratory intervals are not adjusted for multiple comparisons; they do not establish production superiority.", "",
             "| Comparison (left minus right) | MRR difference | 95% interval | Hit@10 difference | 95% interval |",
             "|---|---:|---|---:|---|"])
    for row in table.itertuples():
        lines.append(f"| {row.comparison} | {row.rr_difference:+.4f} | [{row.rr_ci_low:+.4f}, {row.rr_ci_high:+.4f}] "
                     f"| {row.hit10_difference:+.1%} | [{row.hit10_ci_low:+.1%}, {row.hit10_ci_high:+.1%}] |")
    lines.extend(["", "## What broke first", ""])
    for model in run["models"]:
        rows = [r for r in llm if r["model"] == model]
        constrained = [r for r in rows if r["ranker"] != "unconstrained"]
        free = [r for r in rows if r["ranker"] == "unconstrained"]
        count = sum(r["suggestions"] for r in constrained)
        outside = sum(r["outside_candidates"] for r in constrained)
        free_count = sum(r["suggestions"] for r in free)
        free_outside = sum(r["outside_catalog"] for r in free)
        short = sum(r["candidate_coverage"] < 1 for r in constrained)
        failures = sum(bool(r["error"]) for r in rows)
        candidate_rate = f"{outside/count:.1%}" if count else "unavailable"
        catalog_rate = f"{free_outside/free_count:.1%}" if free_count else "unavailable"
        lines.append(f"- `{model}`: {failures}/{len(rows)} request/parse failures; "
                     f"{outside}/{count} constrained suggestions outside candidates ({candidate_rate}); "
                     f"{short}/{len(constrained)} incomplete validated rankings. Unconstrained catalog mismatches: "
                     f"{free_outside}/{free_count} ({catalog_rate}).")
    lines.extend(["", "Catalog mismatches include title/year matching failures and do not prove a movie is fictional. "
                  "Invalid suggestions were dropped; no candidate backfill was used.", "",
                  f"Provider-reported cost: ${sum(known_costs):.4f} across {len(known_costs)}/{len(llm)} known cost fields. "
                  "Unknown costs have not been replaced with zero.", "",
                  f"The history sweep uses the same {run['elbow_users']} users with 50+ past catalog ratings at all four lengths. "
                  "The 20-event predictions are reused from the main experiment; no duplicate inference is counted.", "",
                  "## History-length interpretation", ""])
    elbows = json.loads((folder / "elbow.json").read_text())
    for model in run["models"]:
        point = elbows[model]
        if point["history_length"] is None:
            lines.append(f"- `{model}`: no elbow declared. {point['reason']}")
        elif point["history_length"] == 50:
            lines.append(f"- `{model}`: the rule selects 50 events, the largest tested history. "
                         "A plateau beyond that boundary has not been demonstrated.")
        else:
            lines.append(f"- `{model}`: the descriptive 0.01-MRR rule selects {point['history_length']} events; "
                         "this is not an independently validated optimum.")
    lines.extend(["",
                  "![Matched-cohort history sweep](elbow.png)", "",
                  "[All conditions and diagnostics](report.md) · [Metrics CSV](metrics.csv) · "
                  "[Paired comparison CSV](paired_comparisons.csv) · [History sweep CSV](elbow.csv)", ""])
    (folder / "analysis.md").write_text("\n".join(lines))
    if update_readme:
        path = ROOT / "README.md"
        original = path.read_text()
        relative = os.path.relpath(folder.resolve(), ROOT)
        text = "\n".join(lines).replace("# Frozen evaluation audit", "## Completed frozen evaluation", 1)
        for filename in ("elbow.png", "report.md", "metrics.csv", "paired_comparisons.csv", "elbow.csv"):
            text = text.replace(f"]({filename})", f"]({relative}/{filename})")
        start, end = "<!-- FROZEN_RESULTS_START -->", "<!-- FROZEN_RESULTS_END -->"
        if start in original:
            before, rest = original.split(start, 1)
            _, after = rest.split(end, 1)
            original = before + start + "\n" + text + end + after
        else:
            original += "\n" + start + "\n" + text + end + "\n"
        original = original.replace("The larger frozen-cohort LLM experiment has not yet been run.",
                                    "The completed frozen-cohort LLM results are recorded at the end of this README.")
        path.write_text(original)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--archive-responses", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    args = parser.parse_args()
    if args.archive_responses:
        archive_responses(args.results, args.cache_dir)
    print(analyze(args.results, args.samples, args.seed, args.update_readme).to_string(index=False))


if __name__ == "__main__":
    main()
