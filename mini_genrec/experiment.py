"""Run the frozen experiment. Defaults to a local dry run; inference needs --execute."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

from .baseline import Baseline, random_rank
from .context import Verbalizer, response_format
from .data import Case, Dataset, prepare
from .evaluate import evaluate_llm, find_elbow, metrics, summarize
from .llm import LLM, RequestBudget, RunPaused

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (5, 10, 20, 50)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class Condition:
    model: str
    user_id: int
    ranker: str
    strategy: str
    history_length: int

    @property
    def key(self):
        return digest(asdict(self))


def load_snapshot(folder: Path, data_dir: Path):
    """Validate the historical snapshot and restore its exact cases, not a new sample."""
    checksums = json.loads((folder / "checksums.json").read_text())
    for name, expected in checksums["files"].items():
        if Path(name).name != name:
            raise ValueError("Snapshot checksum entries must be filenames.")
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    config = json.loads((folder / "config.json").read_text())
    if config["llm"]["response_format"] != response_format():
        raise ValueError("Response schema drifted from the frozen contract.")
    cohorts = json.loads((folder / "cohorts.json").read_text())
    saved = json.loads((folder / "cases.json").read_text())["cases"]
    catalog = json.loads((folder / "catalog.json").read_text())
    for name in ("ratings.csv", "movies.csv"):
        path = data_dir / "ml-latest-small" / name
        if not path.exists():
            raise ValueError("Download MovieLens with the legacy --baseline-only command first.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != config["dataset"][name.split(".")[0] + "_sha256"]:
            raise ValueError(f"Dataset fingerprint mismatch: {name}")
    # Reconstruct training from the recorded protocol; test inputs come from the manifest.
    regenerated = prepare(data_dir, users=100000, candidates=config["candidates"]["count"],
                          catalog_size=config["dataset"]["catalog_size"], seed=config["candidates"]["seed"])
    for field in ("cutoff_timestamp", "train_ratings", "train_users"):
        if regenerated.metadata[field] != config["dataset"][field]:
            raise ValueError(f"Training split drift: {field}")
    movies = pd.DataFrame(catalog["movies"]).set_index("movieId")
    expected_movies = regenerated.movies.reset_index().to_dict("records")
    if expected_movies != catalog["movies"] or set(catalog["popular_movie_ids"]) != regenerated.popular:
        raise ValueError("Frozen catalog or popularity group differs from training data.")
    regenerated_cases = {c.user_id: c for c in regenerated.cases}
    cases = {}
    for entry in saved:
        case = Case(entry["user_id"], entry["target_movie_id"], entry["target_timestamp"],
                    pd.DataFrame(entry["history"]), entry["candidate_movie_ids"])
        original = regenerated_cases.get(case.user_id)
        if original is None or case.user_id in cases:
            raise ValueError("Invalid or duplicated frozen user.")
        if (case.target != original.target or case.timestamp != original.timestamp
                or case.candidates != original.candidates
                or case.history.to_dict("records") != original.history[["movieId", "rating", "timestamp"]].to_dict("records")):
            raise ValueError(f"Case drift for user {case.user_id}")
        cases[case.user_id] = case
    dev = set(cohorts["development"]["user_ids"])
    evaluation = set(cohorts["evaluation"]["user_ids"])
    elbow = set(cohorts["elbow"]["user_ids"])
    if dev & evaluation or dev | evaluation != set(cases):
        raise ValueError("Development/evaluation cohorts must partition the frozen cases.")
    if elbow != {uid for uid in evaluation if len(cases[uid].history) >= 50}:
        raise ValueError("Elbow cohort must be the evaluation subset with 50+ history events.")
    for name in ("development", "evaluation", "elbow"):
        ids = cohorts[name]["user_ids"]
        if len(ids) != len(set(ids)) or cohorts[name]["count"] != len(ids):
            raise ValueError(f"Invalid cohort count: {name}")
    fingerprints = json.loads((folder / "prompt_fingerprints.json").read_text())["fingerprints"]
    dataset = Dataset(movies, regenerated.train, list(cases.values()),
                      set(catalog["popular_movie_ids"]), regenerated.metadata)
    return config, cohorts, dataset, cases, fingerprints, digest(checksums)


def build_plan(config, cohorts, cases, cohort="evaluation", limit=None, models=None):
    selected = list(cohorts[cohort]["user_ids"])
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit-users must be positive.")
        selected = selected[:limit]
    models = list(models or config["llm"]["models"])
    if len(set(models)) != len(models) or not set(models).issubset(config["llm"]["models"]):
        raise ValueError("Models must be unique members of the frozen model list.")
    if any(not model.endswith(":free") or model.startswith("openrouter/") for model in models):
        raise ValueError("Only named frozen :free variants are allowed.")
    elbow = [uid for uid in selected if len(cases[uid].history) >= 50]
    tasks = []
    for model in models:
        for uid in selected:
            tasks.extend([
                Condition(model, uid, "mini-genrec", "engineered_v1", 20),
                Condition(model, uid, "mini-genrec", "plain_recent", 20),
                Condition(model, uid, "unconstrained", "engineered_v1", 20)])
        for uid in elbow:
            tasks.extend(Condition(model, uid, "mini-genrec", "engineered_v1", n) for n in (5, 10, 50))
    return selected, elbow, models, tasks


def common_row(case, length, prompt=None):
    stats = (dict(history_events=len(case.history.tail(length)), window_events=len(case.history.tail(length)),
                  low_signal_dropped=0, budget_dropped=0, repeat_events_compressed=0, approx_input_tokens=None)
             if prompt is None else asdict(prompt))
    stats.pop("messages", None)
    if "input_tokens" in stats:
        stats["approx_input_tokens"] = stats.pop("input_tokens")
    return dict(user_id=case.user_id, history_length=length, **stats, api_input_tokens=None,
                api_output_tokens=None, suggestions=0, outside_catalog=0, outside_candidates=0,
                duplicates=0, error="", error_category="", http_status=None)


def baseline_rows(dataset, cases, selected, elbow, seed, parameters=None):
    parameters = parameters or {}
    options = {k: parameters[k] for k in ("cf_weight", "popularity_weight", "positive_rating_threshold", "co_rating_shrinkage")
               if k in parameters}
    baseline = Baseline(dataset.train, dataset.movies.index, **options)
    rows = []
    for uid in selected:
        case = cases[uid]
        for length in LENGTHS if uid in elbow else (20,):
            rankings = {"baseline": baseline.rank(case.history.tail(length), case.candidates),
                        "popularity": baseline.rank_popularity(case.candidates),
                        "random": random_rank(case.candidates, uid, seed)}
            for ranker, ranking in rankings.items():
                rows.append(dict(common_row(case, length), model="", strategy="raw_recent", ranker=ranker,
                                 ranking=ranking, candidate_count=len(case.candidates), candidate_coverage=1.0,
                                 **metrics(ranking, case.target, dataset.popular)))
    return rows


def build_views(records, selected, elbow):
    """Reuse 20-event predictions while keeping main and elbow denominators separate."""
    frame = pd.DataFrame(records)
    views = []
    selectors = {
        "main": (frame.history_length == 20) & (frame.ranker != "unconstrained") & (frame.strategy != "plain_recent"),
        "context": (frame.history_length == 20) & (frame.ranker == "mini-genrec"),
        "unconstrained": frame.ranker == "unconstrained",
        "elbow": frame.user_id.isin(elbow) & (frame.ranker != "unconstrained") & (frame.strategy != "plain_recent"),
    }
    for view, mask in selectors.items():
        expected = len(elbow) if view == "elbow" else len(selected)
        for (model, strategy), group in frame[mask].groupby(["model", "strategy"], dropna=False):
            summary = summarize(group, expected)
            # Incomplete runs never have a publishable aggregate. Zeroes are real misses only.
            if (summary.users != expected).any():
                raise ValueError("Refusing to summarize an incomplete condition.")
            views.append(summary.assign(view=view, model=model, strategy=strategy))
    return pd.DataFrame([record for view in views for record in view.to_dict("records")])


def write_outputs(output, records, selected, elbow, mode, models, tasks, prompts):
    from .report import number
    summary = build_views(records, selected, elbow)
    summary.to_csv(output / "metrics.csv", index=False)
    lines = ["# Frozen experiment results", "", f"Mode: **{mode}**. Users: {len(selected)}; elbow subset: {len(elbow)}.",
             "", "| View | Model | Ranker | Strategy | History | Users | MRR | Hit@10 | Candidate coverage | Errors |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        mrr = "n/a" if row.ranker == "unconstrained" else number(row.mrr)
        hit = "n/a" if row.ranker == "unconstrained" else number(row.hit10)
        lines.append(f"| {row.view} | {row.model or 'local'} | {row.ranker} | {row.strategy} | {row.history_length} "
                     f"| {row.users} | {mrr} | {hit} | {number(row.mean_candidate_coverage, True)} | {row.errors} |")
    if mode == "baseline-only":
        lines.extend(["", "LLM conditions were not run. No LLM scores or elbow are inferred."])
    else:
        for model in models:
            group = summary[(summary.view == "elbow") & (summary.model == model)]
            if len(group):
                lines.extend(["", f"{model}: {find_elbow(group)['reason']}"])
    lines.extend(["", "## Response diagnostics", "",
                  "| Model | Strategy | History | Invalid candidates | Catalog mismatches | Incomplete rankings | Failure categories |",
                  "|---|---|---:|---:|---:|---:|---|"])
    for row in summary[(summary.view == "context") | (summary.view == "unconstrained") | (summary.view == "elbow")].itertuples():
        if row.model:
            lines.append(f"| {row.model} | {row.strategy} ({row.ranker}) | {row.history_length} "
                         f"| {number(row.outside_candidates_rate, True)} | {number(row.outside_catalog_rate, True)} "
                         f"| {row.incomplete_rankings} | `{row.failure_categories}` |")
    from .report import write_frozen_elbow
    write_frozen_elbow(output, summary, tasks, prompts, elbow, models, mode)
    if elbow:
        lines.extend(["", "![History sweep](elbow.png)", "", "[History sweep CSV](elbow.csv)"])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def run(args):
    config, cohorts, dataset, cases, fingerprints, manifest_hash = load_snapshot(Path(args.manifest), Path(args.data_dir))
    selected, elbow, models, tasks = build_plan(config, cohorts, cases, args.cohort, args.limit_users, args.models)
    verbalizers = {model: Verbalizer(dataset.movies, model, config["context"]["estimated_input_token_budget"])
                   for model in models}
    frozen = {(r["user_id"], r["variant"], r["history_length"]): r for r in fingerprints}
    prompts = {}
    for task in tasks:
        case = cases[task.user_id]
        prompt = verbalizers[task.model].build(case.history, case.candidates, case.timestamp, task.history_length,
                                              task.ranker == "unconstrained", strategy=task.strategy)
        if task.strategy == "engineered_v1":
            variant = "unconstrained" if task.ranker == "unconstrained" else "constrained"
            expected = frozen[(case.user_id, variant, task.history_length)]
            if digest(prompt.messages) != expected["messages_sha256"] or prompt.input_tokens != expected["approx_input_tokens"]:
                raise ValueError("Engineered prompt drifted from the frozen snapshot.")
        prompts[task.key] = prompt
    source_files = ["experiment.py", "context.py", "baseline.py", "evaluate.py", "llm.py", "data.py"]
    identity = dict(manifest_sha256=manifest_hash, cohort=args.cohort, selected_users=selected, elbow_users=elbow,
                    models=models, mode="baseline-only" if args.baseline_only else "api",
                    source_sha256={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                   for name in source_files})
    run_id = digest(identity)[:16]
    plan = dict(experiment_id=config["experiment_id"], run_id=run_id, cohort=args.cohort, selected_users=selected,
                users=len(selected), elbow_users=len(elbow), models=models, logical_llm_conditions=len(tasks),
                stages={"main": len(selected)*len(models), "plain_history": len(selected)*len(models),
                        "unconstrained": len(selected)*len(models), "elbow_additional": len(elbow)*len(models)*3},
                baseline_predictions=3*(len(selected)+len(elbow)*3), mode=identity["mode"])
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute and not args.baseline_only:
        print("Dry run only: no output files or model calls. Use --baseline-only for local controls; --execute for inference.")
        return plan
    if args.execute:
        load_dotenv()
        if not os.getenv(config["llm"]["credential_env_var"]):
            raise ValueError("OPENROUTER_API_KEY is missing. No inference started.")
    output = Path(args.output) if args.output else ROOT / "results" / "frozen" / config["experiment_id"] / run_id
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["identity"] != identity:
            raise ValueError("Output directory belongs to a different experiment/mode; choose another directory.")
    else:
        if any(p.name != ".DS_Store" for p in output.iterdir()):
            raise ValueError("Output directory is not empty and has no compatible checkpoint; choose a new directory.")
        state = dict(identity=identity, rows=[], invocations=[])
    completed = {row["condition_id"]: row for row in state["rows"]}
    planned_keys = {task.key for task in tasks}
    if len(completed) != len(state["rows"]) or not set(completed).issubset(planned_keys):
        raise ValueError("Checkpoint contains duplicate or foreign conditions.")
    for name in ("metrics.csv", "report.md", "elbow.csv", "elbow.png", "elbow.json"):
        (output / name).unlink(missing_ok=True)
    save(output / "run.json", dict(plan, identity=identity, frozen_config=config))
    status = dict(status="running", run_id=run_id, mode=identity["mode"])
    save(output / "status.json", status)
    budget = RequestBudget(args.request_budget if args.request_budget is not None else config["llm"]["request_budget_per_invocation"],
                           config["llm"]["minimum_request_interval_seconds"])
    invocation = dict(started_at=datetime.now(timezone.utc).isoformat(), uncached_attempts=0, status="running")
    state["invocations"].append(invocation)
    save(state_path, state)
    clients = {}
    try:
        controls = baseline_rows(dataset, cases, selected, elbow, config["candidates"]["seed"],
                                 config["baselines"]["existing_hybrid"])
        if args.execute:
            for task in tasks:
                if task.key in completed:
                    continue
                if task.model not in clients:
                    clients[task.model] = LLM(task.model, Path(args.cache_dir), "openrouter", budget,
                                             reasoning=config["llm"]["reasoning"],
                                             max_output_tokens=config["llm"]["max_output_tokens"])
                case = cases[task.user_id]
                before = budget.used
                row = evaluate_llm(clients[task.model], prompts[task.key],
                                   common_row(case, task.history_length, prompts[task.key]), dataset, case, task.ranker)
                row.update(model=task.model, strategy=task.strategy, condition_id=task.key,
                           cache_hit=(budget.used == before) if row.get("request_hash") else None)
                state["rows"].append(row)
                invocation["uncached_attempts"] = budget.used
                save(state_path, state)
                print(f"Completed {len(state['rows'])}/{len(tasks)} LLM conditions", flush=True)
        records = controls + state["rows"]
        save(output / "per_user.json", records)
        write_outputs(output, records, selected, elbow, identity["mode"], models, tasks, prompts)
        status.update(status="complete", completed_llm_conditions=len(state["rows"]))
        invocation["status"] = "complete"
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        # Never include SDK exception bodies, credentials or request headers in artifacts.
        paused = isinstance(error, (RunPaused, KeyboardInterrupt))
        category = getattr(error, "category", "configuration" if isinstance(error, SystemExit) else "runner")
        if isinstance(error, KeyboardInterrupt):
            category = "interrupted"
        status.update(status="incomplete" if paused else "failed", error_category=category,
                      http_status=getattr(error, "http_status", None), completed_llm_conditions=len(state["rows"]))
        for name in ("metrics.csv", "report.md", "elbow.csv", "elbow.png", "elbow.json"):
            (output / name).unlink(missing_ok=True)
        invocation["status"] = status["status"]
        raise SystemExit(f"Run {status['status']} ({category}); checkpoints preserved. "
                         "Rerun the same command to resume. No partial aggregate published.") from None
    finally:
        invocation["uncached_attempts"] = budget.used
        save(state_path, state)
        save(output / "status.json", status)
        for client in clients.values():
            client.client.close()
    print(f"Saved {identity['mode']} results to {output.resolve()}", flush=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "experiments" / "mini-genrec-v1"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--cache-dir", default=str(ROOT / "cache"))
    parser.add_argument("--cohort", choices=["evaluation", "development"], default="evaluation")
    parser.add_argument("--limit-users", type=int)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--request-budget", type=int)
    parser.add_argument("--output")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default: validate and print the plan, without writing results")
    mode.add_argument("--baseline-only", action="store_true")
    mode.add_argument("--execute", action="store_true", help="Explicitly enable free-model API requests")
    args = parser.parse_args()
    if args.request_budget is not None and not 1 <= args.request_budget <= 50:
        parser.error("--request-budget must be between 1 and 50.")
    try:
        run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
