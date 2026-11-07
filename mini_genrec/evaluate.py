"""Run with: python -m mini_genrec.evaluate [--baseline-only]."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from openai import (APIError, APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError, NotFoundError,
                    PermissionDeniedError, UnprocessableEntityError)

from .baseline import Baseline
from .context import Verbalizer
from .data import prepare
from .llm import LLM, RequestBudget, RunPaused, parse_response, validate
from .report import write_report

LENGTHS = (5, 10, 20, 50)


class ConfigurationFailure(SystemExit):
    """A sanitized fatal API failure, retaining only status and classification."""

    def __init__(self, category, http_status):
        super().__init__("API authentication, permission, model or request configuration failed. "
                         "Check credentials/model. Completed users and cached responses are preserved.")
        self.category = category
        self.http_status = http_status


def metrics(ranking: list[int], target: int, popular: set[int]) -> dict:
    rank = ranking.index(target) + 1 if target in ranking else None
    return {"rr": 1 / rank if rank else 0.0,
            "hit10": int(rank is not None and rank <= 10),
            "popular_top10": sum(mid in popular for mid in ranking[:10]),
            "top10_count": len(ranking[:10])}


def summarize(rows: pd.DataFrame, expected_users: int) -> pd.DataFrame:
    records = []
    for (ranker, length), group in rows.groupby(["ranker", "history_length"], sort=False):
        unconstrained = ranker == "unconstrained"
        total = int(group.suggestions.sum())
        top_count = int(group.top10_count.sum())
        records.append({
            "ranker": ranker, "history_length": int(length), "users": len(group),
            "successful_evaluations": int((group.error == "").sum()),
            "mrr": None if unconstrained else float(group.rr.mean()),
            "hit10": None if unconstrained else float(group.hit10.mean()),
            "popular_top10_fraction": float(group.popular_top10.sum() / top_count)
                if top_count else None,
            "top10_slot_coverage": float(top_count / (10 * expected_users)),
            "suggestions": total,
            "outside_catalog_rate": float(group.outside_catalog.sum() / total) if total else None,
            "outside_candidates_rate": float(group.outside_candidates.sum() / total)
                if total and not unconstrained else None,
            "duplicate_suggestions": int(group.duplicates.sum()),
            "mean_approx_input_tokens": float(group.approx_input_tokens.mean()),
            "mean_api_input_tokens": float(group.api_input_tokens.mean())
                if group.api_input_tokens.notna().any() else None,
            "mean_latency_seconds": float(group.latency_seconds.mean())
                if "latency_seconds" in group and group.latency_seconds.notna().any() else None,
            "api_cost_usd": float(group.api_cost_usd.sum())
                if "api_cost_usd" in group and group.api_cost_usd.notna().all() else None,
            "mean_retained_history_events": float(group.history_events.mean()),
            "low_signal_events_dropped": int(group.low_signal_dropped.sum()),
            "budget_events_dropped": int(group.budget_dropped.sum()),
            "repeat_events_compressed": int(group.repeat_events_compressed.sum()),
            "errors": int((group.error != "").sum()),
            "truncated_responses": int((group.response_finish_reason == "length").sum())
                if "response_finish_reason" in group else 0,
            "mean_candidate_coverage": float(group.candidate_coverage.mean())
                if "candidate_coverage" in group and group.candidate_coverage.notna().any() else None,
            "incomplete_rankings": int((group.candidate_coverage < 1).sum())
                if "candidate_coverage" in group else 0,
            "failure_categories": json.dumps(group.loc[group.error != "", "error_category"].value_counts().to_dict(), sort_keys=True)
                if "error_category" in group else "{}",
        })
    return pd.DataFrame(records)


def find_elbow(summary: pd.DataFrame) -> dict:
    rows = summary[summary.ranker == "mini-genrec"].sort_values("history_length")
    if len(rows) != len(LENGTHS) or rows.errors.sum() or not rows.mrr.notna().all():
        return {"history_length": None, "reason": "Requires all four measured LLM conditions without errors."}
    # Descriptive, not statistically inferred: smallest length within 0.01 MRR of best.
    best = float(rows.mrr.max())
    choice = rows[rows.mrr >= best - 0.01].iloc[0]
    return {"history_length": int(choice.history_length), "mrr": float(choice.mrr),
            "best_mrr": best, "tolerance": 0.01,
            "reason": "Smallest history within 0.01 absolute MRR of the best observed point; exploratory."}


def run(args, request_budget=None) -> pd.DataFrame:
    load_dotenv()
    backend = getattr(args, "backend", "openai")
    key_name = "OPENROUTER_API_KEY" if backend == "openrouter" else "OPENAI_API_KEY"
    if not args.baseline_only and not os.getenv(key_name):
        raise SystemExit(f"{key_name} is missing. Set it in .env or use --baseline-only. No LLM results fabricated.")
    if backend == "openrouter" and not args.baseline_only:
        if not args.model.endswith(":free") or args.model.startswith("openrouter/"):
            raise SystemExit("Choose an explicit named OpenRouter model ending in :free; paid models/routers are disabled.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = prepare(Path(args.data_dir), args.users, args.candidates, args.catalog_size, args.seed)
    (output / "status.json").write_text(json.dumps({"status": "running", "backend": backend,
                                                   "model": args.model}))
    print(json.dumps(dataset.metadata, indent=2), flush=True)
    baseline = Baseline(dataset.train, dataset.movies.index)
    verbalizer = Verbalizer(dataset.movies, args.model, args.token_budget)
    llm = None if args.baseline_only else LLM(args.model, Path(args.cache_dir), backend, request_budget,
        reasoning=getattr(args, "reasoning", "default"),
        max_output_tokens=getattr(args, "max_output_tokens", 2400))
    manifest = [dict(user_id=c.user_id, target=c.target, timestamp=c.timestamp,
                     candidates=c.candidates,
                     history_movie_ids=list(map(int, c.history.movieId))) for c in dataset.cases]
    (output / "cases.json").write_text(json.dumps(manifest, indent=2))
    records, prompts = [], []
    for i, case in enumerate(dataset.cases, start=1):
        for length in LENGTHS:
            prompt = verbalizer.build(case.history, case.candidates, case.timestamp, length)
            prompt_stats = asdict(prompt)
            prompt_stats.pop("messages")
            prompt_stats["approx_input_tokens"] = prompt_stats.pop("input_tokens")
            common = dict(user_id=case.user_id, history_length=length, **prompt_stats,
                          api_input_tokens=None, api_output_tokens=None, suggestions=0,
                          outside_catalog=0, outside_candidates=0, duplicates=0, error="")
            baseline_ranking = baseline.rank(case.history.tail(length), case.candidates)
            base_common = dict(common)
            base_common.update(approx_input_tokens=None, history_events=len(case.history.tail(length)),
                               low_signal_dropped=0, budget_dropped=0, repeat_events_compressed=0)
            records.append(dict(base_common, ranker="baseline", ranking=baseline_ranking,
                                **metrics(baseline_ranking, case.target, dataset.popular)))
            # This local preview is reproducible and contains public MovieLens data only.
            prompts.append(dict(user_id=case.user_id, history_length=length,
                                messages=prompt.messages, **prompt_stats))
            if llm:
                records.append(evaluate_llm(llm, prompt, common, dataset, case, "mini-genrec"))
        if llm:
            prompt = verbalizer.build(case.history, case.candidates, case.timestamp, 20, True)
            stats = asdict(prompt)
            stats.pop("messages")
            stats["approx_input_tokens"] = stats.pop("input_tokens")
            common = common.copy()
            common.update(history_length=20, **stats)
            records.append(evaluate_llm(llm, prompt, common, dataset, case, "unconstrained"))
        print(f"Evaluated {i}/{len(dataset.cases)} users", flush=True)
        # Preserve each completed user's work; API cache also survives interrupted runs.
        (output / "per_user.json").write_text(json.dumps(records, indent=2))
    frame = pd.DataFrame(records)
    summary = summarize(frame, len(dataset.cases))
    # Baseline does not consume tokens; prompt-size measurements belong to the LLM.
    summary.loc[summary.ranker == "baseline", ["mean_approx_input_tokens", "mean_api_input_tokens"]] = np.nan
    summary.to_csv(output / "metrics.csv", index=False)
    elbow_rows = []
    for length in LENGTHS:
        measurements = pd.DataFrame([p for p in prompts if p["history_length"] == length])
        measured_llm = summary[(summary.ranker == "mini-genrec") & (summary.history_length == length)]
        base = summary[(summary.ranker == "baseline") & (summary.history_length == length)].iloc[0]
        elbow_rows.append(dict(history_length=length,
                               approx_input_tokens=float(measurements.approx_input_tokens.mean()),
                               retained_events=float(measurements.history_events.mean()),
                               baseline_mrr=float(base.mrr),
                               genrec_mrr=float(measured_llm.iloc[0].mrr) if len(measured_llm) else None))
    pd.DataFrame(elbow_rows).to_csv(output / "elbow.csv", index=False)
    (output / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2))
    metadata = dict(dataset.metadata, model=args.model, backend=backend, token_budget=args.token_budget,
                    mode="baseline-only" if args.baseline_only else "api",
                    llm_skip_reason=(f"{key_name} unavailable" if not os.getenv(key_name)
                                     else "--baseline-only selected") if args.baseline_only else None,
                    run_at=datetime.now(timezone.utc).isoformat(),
                    planned_llm_requests=len(dataset.cases) * 5,
                    config=vars(args), elbow=find_elbow(summary))
    (output / "run.json").write_text(json.dumps(metadata, indent=2))
    write_report(output, summary, metadata, args.update_readme)
    (output / "status.json").write_text(json.dumps({"status": "complete", "backend": backend,
                                                   "model": args.model}))
    print(f"Saved measured results to {output.resolve()}", flush=True)
    return summary


def evaluate_llm(llm, prompt, common, dataset, case, ranker) -> dict:
    row = dict(common, ranker=ranker, ranking=[], error_category="", http_status=None,
               candidate_count=len(case.candidates), candidate_coverage=None)
    try:
        response = llm.call(prompt)
        row.update(api_input_tokens=response["prompt_tokens"],
                   api_output_tokens=response["completion_tokens"],
                   response_model=response["model"], request_hash=response["request_hash"])
        row.update(response_finish_reason=response.get("finish_reason"),
                   response_content_characters=len(response.get("content") or ""))
        if response.get("refusal"):
            row["error_category"] = "refusal"
        elif response.get("finish_reason") != "stop":
            row["error_category"] = "truncation" if response.get("finish_reason") == "length" else "finish_reason"
        row.update(latency_seconds=response.get("latency_seconds"),
                   response_provider=response.get("provider"), response_backend=response.get("backend"),
                   api_reasoning_tokens=response.get("reasoning_tokens"),
                   api_cost_usd=response.get("cost_usd"), response_id=response.get("response_id"))
        validated = validate(parse_response(response), dataset.movies, case.candidates,
                             unconstrained=ranker == "unconstrained")
        row.update(asdict(validated))
        if ranker != "unconstrained":
            row["candidate_coverage"] = len(validated.ranking) / len(case.candidates)
    except (AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError,
            UnprocessableEntityError) as error:
        # Stop on invalid credentials/model/schema rather than issuing repeated billable calls.
        category = ("authentication" if isinstance(error, AuthenticationError) else
                    "permission" if isinstance(error, PermissionDeniedError) else
                    "model_or_endpoint" if isinstance(error, NotFoundError) else "configuration")
        raise ConfigurationFailure(category, getattr(error, "status_code", None)) from None
    except (APIError, ValueError, KeyError, TypeError) as error:
        # Do not log exception text: SDK errors can contain credentials or request headers.
        row["error"] = type(error).__name__
        row["http_status"] = getattr(error, "status_code", None)
        if not row["error_category"]:
            row["error_category"] = (
                "timeout" if isinstance(error, APITimeoutError) else
                "connection" if isinstance(error, APIConnectionError) else
                "http" if isinstance(error, APIError) else
                "json" if isinstance(error, json.JSONDecodeError) else "response_schema")
        if ranker != "unconstrained":
            row["candidate_coverage"] = 0.0
    row.update(metrics(row["ranking"], case.target, dataset.popular))
    return row


def model_folder(model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", model).strip(".")[:100] or "model"
    return slug + "-" + hashlib.sha256(model.encode()).hexdigest()[:8]


def run_matrix(args):
    """Same deterministic users/candidates for every model; quota is shared, not per model."""
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    models = args.models or [args.model]
    if args.backend == "openrouter" and not args.baseline_only:
        if any(not model.endswith(":free") or model.startswith("openrouter/") for model in models):
            raise SystemExit("Every OpenRouter model must be an explicit named :free variant. No requests made.")
    budget = RequestBudget(args.request_budget) if args.backend == "openrouter" else None
    multiple = bool(args.models)
    if multiple:
        for name in ("comparison.csv", "comparison.md"):
            (root / name).unlink(missing_ok=True)
    summaries = []
    for model in models:
        config = argparse.Namespace(**vars(args))
        config.model = model
        if multiple:
            config.output = str(root / model_folder(model))
            config.update_readme = False
        try:
            summary = run(config, budget)
        except RunPaused as error:
            folder = Path(config.output)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "status.json").write_text(json.dumps({"status": "incomplete", "reason": str(error),
                "model": model, "backend": args.backend}))
            raise SystemExit(f"{error} Run incomplete; no aggregate comparison published. "
                             "Cached requests are preserved. Rerun the same command to resume.") from None
        summaries.append(summary.assign(model=model, backend=args.backend))
    if multiple:
        comparison = pd.concat(summaries, ignore_index=True)
        comparison.to_csv(root / "comparison.csv", index=False)
        from .report import number
        lines = ["# Model comparison", "", "Same users, candidates and seed for every model. "
                 "Request/parse failures count as misses; inspect errors and coverage before comparing quality.",
                 "", "| Model | Ranker | History | Users | MRR | Hit@10 | Popular share | Slot coverage | Errors |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in comparison[comparison.history_length == 20].itertuples():
            mrr = "n/a" if row.ranker == "unconstrained" else number(row.mrr)
            hit10 = "n/a" if row.ranker == "unconstrained" else number(row.hit10)
            lines.append(f"| {row.model} | {row.ranker} | 20 | {row.users} | {mrr} "
                         f"| {hit10} | {number(row.popular_top10_fraction, True)} "
                         f"| {number(row.top10_slot_coverage, True)} | {row.errors} |")
        (root / "comparison.md").write_text("\n".join(lines) + "\n")
        if args.update_readme:
            print("Matrix results saved to comparison.md; README's original run was preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--catalog-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["openai", "openrouter"], default="openai")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model")
    model_group.add_argument("--models", nargs="+", help="Compare explicit model IDs on identical cases")
    parser.add_argument("--request-budget", type=int, default=40,
                        help="Maximum uncached OpenRouter attempts across this invocation (1-50)")
    parser.add_argument("--reasoning", choices=["default", "off"], default="default",
                        help="OpenRouter reasoning control; off requires a model supporting disabling reasoning")
    parser.add_argument("--max-output-tokens", type=int, default=2400)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output", default="results")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    if args.model is None:
        if args.backend == "openrouter" and not args.models and not args.baseline_only:
            parser.error("OpenRouter requires --model or --models with named :free variants.")
        args.model = "gpt-4o-mini"
    if args.users < 1 or args.candidates < 10 or args.catalog_size <= args.candidates or args.token_budget < 1:
        parser.error("Need users >= 1, candidates >= 10, catalog-size > candidates, token-budget > 0.")
    if not 1 <= args.request_budget <= 50:
        parser.error("--request-budget must be between 1 and 50.")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive.")
    if args.backend != "openrouter" and args.reasoning != "default":
        parser.error("--reasoning is only supported with --backend openrouter.")
    run_matrix(args)


if __name__ == "__main__":
    main()
