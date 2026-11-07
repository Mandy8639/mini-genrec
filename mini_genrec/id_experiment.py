"""Compare integer-ID output with the saved title arm on the same frozen cases."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import os

from dotenv import load_dotenv
import httpx
import pandas as pd

from .analysis import archive_responses, paired_comparison
from .context import Verbalizer, response_format
from .evaluate import evaluate_llm
from .experiment import ROOT, common_row, digest, load_snapshot, save
from .llm import LLM, RequestBudget, RunPaused, Validated, parse_response

TITLE_RUN = ROOT / "results/frozen-evaluation"
OUTPUT = ROOT / "results/id-comparison"
TITLE_INSTRUCTION = ('Use each exact title once, including its year; return '
                     '{"ranking": [titles]}. Never suggest any other movie.')
ID_INSTRUCTION = ('Use each candidate movieId exactly once; return '
                  '{"ranking": [movieIds]} with integer IDs. Never suggest any other movie.')


def id_schema():
    schema = response_format()
    schema["json_schema"]["schema"]["properties"]["ranking"]["items"]["type"] = "integer"
    return schema


def id_prompt(prompt, verbalizer):
    text, candidates = prompt.messages[1]["content"].rsplit("\nCandidates:\n", 1)
    if not text.endswith(TITLE_INSTRUCTION):
        raise ValueError("Frozen ranking instruction changed.")
    messages = [prompt.messages[0], {"role": "user", "content":
                text[:-len(TITLE_INSTRUCTION)] + ID_INSTRUCTION + "\nCandidates:\n" + candidates}]
    tokens = verbalizer.tokens(messages)
    tokens += (len(verbalizer.encoding.encode(json.dumps(id_schema()))) -
               len(verbalizer.encoding.encode(json.dumps(response_format()))))
    if tokens > verbalizer.budget:
        raise ValueError("ID instruction exceeds the frozen input budget.")
    return replace(prompt, messages=messages, input_tokens=tokens)


def decode_ids(response, dataset, case):
    ids = parse_response(response, int)
    candidates, seen, ranking = set(case.candidates), set(), []
    outside = unknown = duplicates = 0
    for mid in ids:
        unknown += mid not in dataset.movies.index
        if mid not in candidates:
            outside += 1
        elif mid in seen:
            duplicates += 1
        else:
            seen.add(mid)
            ranking.append(mid)
    return Validated(ranking, len(ids), outside, unknown, duplicates)


class IdLLM(LLM):
    def call(self, prompt):
        return super().call(prompt, schema=id_schema())


def preflight(models, pending):
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is missing.")
    with httpx.Client(base_url="https://openrouter.ai/api/v1/", timeout=20) as client:
        response = client.get("models")
        response.raise_for_status()
        catalog = {item["id"]: item for item in response.json()["data"]}
        for model in models:
            item = catalog.get(model, {})
            if (not model.endswith(":free") or model.startswith("openrouter/")
                    or any(float(item.get("pricing", {}).get(field, "nan")) != 0
                           for field in ("prompt", "completion"))
                    or "structured_outputs" not in item.get("supported_parameters", [])
                    or (item.get("reasoning") or {}).get("mandatory")):
                raise ValueError(f"Free structured-output preflight failed: {model}")
        response = client.get("key", headers={"Authorization": f"Bearer {key}"})
        response.raise_for_status()
        quota = response.json()["data"].get("free_model_daily_requests") or {}
        if quota.get("remaining", 0) < pending:
            raise ValueError("Not enough remaining free requests for the next batch.")
        print(f"{len(models)} frozen model(s) confirmed free; {quota['remaining']} free requests remaining.", flush=True)


def report(output, records, title_rows, models):
    id_rows = pd.DataFrame(records)
    title = pd.DataFrame(title_rows)
    summary, paired = [], []
    lines = ["# Movie ID versus title output", "",
             "Same 34 users, candidate order, candidate titles/genres, engineered 20-event history, models, and generation settings. "
             "Only the output instruction and schema item type changed. The title arm is historical, so provider/model drift and sampling may affect differences.",
             "", "| Model | Output | MRR | Hit@10 | Coverage | Failures |",
             "|---|---|---:|---:|---:|---:|"]
    for model in models:
        old, new = title[title.model == model], id_rows[id_rows.model == model]
        paired.append(paired_comparison(new, old, model + ": ID minus title"))
        for name, arm in (("Title", old), ("ID", new)):
            s = dict(model=model, output=name, users=len(arm), mrr=float(arm.rr.mean()),
                     hit10=float(arm.hit10.mean()), coverage=float(arm.candidate_coverage.mean()),
                     failures=int(arm.error.astype(bool).sum()), incomplete=int((arm.candidate_coverage < 1).sum()),
                     outside_candidates=int(arm.outside_candidates.sum()), duplicates=int(arm.duplicates.sum()))
            summary.append(s)
            lines.append(f"| {model} | {name} | {s['mrr']:.4f} | {s['hit10']:.1%} | {s['coverage']:.1%} | {s['failures']} |")
    lines.extend(["", "## Matched differences (ID minus title)", "",
                  "Output token averages include only pairs with reported token counts on both sides. "
                  "Ranking and coverage use all users, including failed requests.", "",
                  "| Model | Coverage difference | MRR difference (95% paired interval) | Hit@10 difference | Title → ID output tokens | Token pairs | Same provider* |",
                  "|---|---:|---|---:|---:|---:|---:|"])
    for model, row in zip(models, paired):
        a = id_rows[id_rows.model == model].set_index("user_id")
        b = title[title.model == model].set_index("user_id").reindex(a.index)
        token_pairs = a.api_output_tokens.notna() & b.api_output_tokens.notna()
        tokens = f"{b.loc[token_pairs, 'api_output_tokens'].mean():.1f} → {a.loc[token_pairs, 'api_output_tokens'].mean():.1f}" if token_pairs.any() else "Unavailable"
        known = a.response_provider.notna() & b.response_provider.notna() if "response_provider" in a and "response_provider" in b else pd.Series(False, index=a.index)
        same = int((a.loc[known, "response_provider"] == b.loc[known, "response_provider"]).sum()) if known.any() else 0
        row.update(coverage_difference=float((a.candidate_coverage-b.candidate_coverage).mean()),
                   token_pairs=int(token_pairs.sum()),
                   title_mean_output_tokens=float(b.loc[token_pairs, "api_output_tokens"].mean()) if token_pairs.any() else None,
                   id_mean_output_tokens=float(a.loc[token_pairs, "api_output_tokens"].mean()) if token_pairs.any() else None,
                   known_provider_pairs=int(known.sum()), same_provider_pairs=same)
        lines.append(f"| {model} | {row['coverage_difference']*100:+.1f} pp | {row['rr_difference']:+.4f} "
                     f"[{row['rr_ci_low']:+.4f}, {row['rr_ci_high']:+.4f}] | {row['hit10_difference']*100:+.1f} pp | {tokens} | {int(token_pairs.sum())} | {same}/{int(known.sum())} |")
    cost = [r["api_cost_usd"] for r in records if r.get("api_cost_usd") is not None]
    lines += ["", "*Among pairs with known provider labels. Same provider does not rule out model drift.", "",
              "Intervals use 10,000 paired-user bootstrap resamples with seed 42; exploratory, with no multiple-comparison adjustment.", "",
              f"Provider-reported cost: ${sum(cost):.4f} across {len(cost)}/{len(records)} conditions with cost metadata.", ""]
    pd.DataFrame(summary).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired_comparisons.csv", index=False)
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(output, execute=False, budget_size=40):
    config, cohorts, dataset, cases, fingerprints, snapshot_hash = load_snapshot(
        ROOT / "experiments/mini-genrec-v1", ROOT / "data")
    title_bytes = (TITLE_RUN / "per_user.json").read_bytes()
    title_run = json.loads((TITLE_RUN / "run.json").read_text())
    title_status = json.loads((TITLE_RUN / "status.json").read_text())
    users, models = cohorts["evaluation"]["user_ids"], config["llm"]["models"]
    if (title_status["status"] != "complete" or title_run["selected_users"] != users
            or title_run["models"] != models or title_run["identity"]["manifest_sha256"] != snapshot_hash):
        raise ValueError("Saved title arm does not match the frozen plan.")
    title_rows = [r for r in json.loads(title_bytes) if r["ranker"] == "mini-genrec"
                  and r["history_length"] == 20 and r["strategy"] == "engineered_v1"]
    keys = {(model, uid) for model in models for uid in users}
    if len(title_rows) != len(keys) or {(r["model"], r["user_id"]) for r in title_rows} != keys:
        raise ValueError("Missing or duplicate title comparison rows.")
    expected = {r["user_id"]: r for r in fingerprints if r["variant"] == "constrained" and r["history_length"] == 20}
    prompts = {}
    for model in models:
        v = Verbalizer(dataset.movies, model, config["context"]["estimated_input_token_budget"])
        for uid in users:
            case = cases[uid]
            title_prompt = v.build(case.history, case.candidates, case.timestamp, 20)
            if digest(title_prompt.messages) != expected[uid]["messages_sha256"]:
                raise ValueError("Title prompt differs from the frozen reference.")
            prompts[model, uid] = id_prompt(title_prompt, v)
    identity = dict(snapshot=snapshot_hash, title_sha256=hashlib.sha256(title_bytes).hexdigest(),
                    models=models, users=users, conditions=len(keys), schema=id_schema(),
                    prompt_sha256=digest([p.messages for p in prompts.values()]),
                    source_sha256={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                   for name in ("id_experiment.py", "llm.py", "evaluate.py", "context.py")})
    print(f"{len(keys)} ID conditions; no title reruns. Plan {digest(identity)[:16]}.", flush=True)
    if not execute:
        return
    state_path = output / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["identity"] != identity:
            raise ValueError("Existing checkpoint belongs to a different plan.")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Choose an empty output directory.")
        state = dict(identity=identity, rows=[], invocations=[])
    completed = {(r["model"], r["user_id"]) for r in state["rows"]}
    if len(completed) != len(state["rows"]) or not completed.issubset(keys):
        raise ValueError("Invalid checkpoint conditions.")
    if keys - completed:
        preflight(models, min(budget_size, len(keys - completed)))
    output.mkdir(parents=True, exist_ok=True)
    budget = RequestBudget(budget_size, config["llm"]["minimum_request_interval_seconds"])
    status = dict(status="running", mode="api")
    invocation = dict(uncached_attempts=0)
    state["invocations"].append(invocation)
    save(output / "status.json", status)
    save(state_path, state)
    client = None
    try:
        for model in models:
            for uid in users:
                if (model, uid) in completed:
                    continue
                if client is None or client.model != model:
                    if client:
                        client.client.close()
                    client = IdLLM(model, ROOT / "cache", "openrouter", budget,
                                   reasoning=config["llm"]["reasoning"],
                                   max_output_tokens=config["llm"]["max_output_tokens"])
                case, prompt = cases[uid], prompts[model, uid]
                row = evaluate_llm(client, prompt, common_row(case, 20, prompt),
                                   dataset, case, "mini-genrec", decode=decode_ids)
                state["rows"].append(dict(row, model=model))
                invocation["uncached_attempts"] = budget.used
                save(state_path, state)
                print(f"{len(state['rows'])}/{len(keys)} recorded", flush=True)
        report(output, state["rows"], title_rows, models)
        status["status"] = "complete"
    except (RunPaused, KeyboardInterrupt) as error:
        status.update(status="incomplete", reason=getattr(error, "category", "interrupted"))
        raise SystemExit(f"Paused: {status['reason']}. Resume with the same command.") from None
    except (SystemExit, Exception) as error:
        status.update(status="failed", reason=getattr(error, "category", type(error).__name__))
        raise SystemExit(f"Stopped: {status['reason']}. Checkpoint preserved.") from None
    finally:
        invocation["uncached_attempts"] = budget.used
        save(state_path, state)
        save(output / "status.json", dict(status, completed=len(state["rows"])))
        if client:
            client.client.close()
    archive_responses(output, ROOT / "cache")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--request-budget", type=int, choices=range(1, 51), default=40)
    args = parser.parse_args()
    run(args.output, args.execute, args.request_budget)


if __name__ == "__main__":
    main()
