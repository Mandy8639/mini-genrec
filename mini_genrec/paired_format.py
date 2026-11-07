"""Fresh, interleaved title-vs-ID experiment for the frozen Dots cohort."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from .analysis import archive_responses, paired_comparison
from .context import Verbalizer, response_format
from .evaluate import evaluate_llm
from .experiment import ROOT, common_row, digest, load_snapshot, save
from .id_experiment import IdLLM, decode_ids, id_prompt, id_schema, preflight
from .llm import LLM, RequestBudget, RunPaused

MODEL = "dots-studio/dots-3-note-preview:free"
OUTPUT = ROOT / "results/paired-format"
RULE = ("Prefer IDs for Dots efficiency if the 95% paired coverage interval is entirely above zero, "
        "mean reported output tokens decrease on matched pairs, and neither ranking interval is entirely below zero. "
        "Otherwise keep titles. This is not a test of ranking noninferiority.")


def order(users):
    return [(uid, arm) for i, uid in enumerate(users)
            for arm in (("title", "id") if i % 2 == 0 else ("id", "title"))]


def report(output, rows):
    frame = pd.DataFrame(rows)
    ids, titles = frame[frame.arm == "id"], frame[frame.arm == "title"]
    comparison = paired_comparison(ids, titles, "ID minus title")
    coverage = paired_comparison(ids.assign(rr=ids.candidate_coverage),
                                 titles.assign(rr=titles.candidate_coverage), "Coverage: ID minus title")
    a = ids.set_index("user_id").sort_index()
    b = titles.set_index("user_id").reindex(a.index)
    available = a.api_output_tokens.notna() & b.api_output_tokens.notna()
    before, after = (float(b.loc[available, "api_output_tokens"].mean()),
                     float(a.loc[available, "api_output_tokens"].mean())) if available.any() else (None, None)
    providers = [(x, y) for x, y in zip(a.get("response_provider", [None]*len(a)),
                                        b.get("response_provider", [None]*len(b)))
                 if pd.notna(x) and pd.notna(y)]
    same_provider = sum(x == y for x, y in providers)
    comparison.update(coverage_difference=coverage["rr_difference"], coverage_ci_low=coverage["rr_ci_low"],
                      coverage_ci_high=coverage["rr_ci_high"], output_token_pairs=int(available.sum()),
                      title_mean_output_tokens=before, id_mean_output_tokens=after,
                      same_provider_pairs=same_provider, known_provider_pairs=len(providers))
    prefer_ids = (coverage["rr_ci_low"] > 0 and after is not None and after < before
                  and comparison["rr_ci_high"] >= 0 and comparison["hit10_ci_high"] >= 0)
    comparison["recommendation"] = "IDs for Dots efficiency" if prefer_ids else "Keep titles"
    pd.DataFrame([comparison]).to_csv(output / "paired_comparison.csv", index=False)
    lines = ["# Interleaved Dots format comparison", "",
             "34 frozen users, 50 identical ordered candidates and engineered 20-event histories. "
             "Title and ID requests were sent back to back, alternating which format went first. "
             "Both arms used a fresh shared cache namespace, with distinct keys for each request schema.", "",
             "| Output | Candidate coverage | MRR | Hit@10 | Failed requests |",
             "|---|---:|---:|---:|---:|"]
    for arm, group in (("Title", titles), ("ID", ids)):
        lines.append(f"| {arm} | {group.candidate_coverage.mean():.1%} | {group.rr.mean():.4f} | "
                     f"{group.hit10.mean():.1%} | {group.error.astype(bool).sum()}/{len(group)} |")
    lines += ["", "ID minus title; 95% percentile intervals over 10,000 paired-user bootstrap resamples (seed 42):", "",
              f"- Coverage: {coverage['rr_difference']*100:+.1f} pp "
              f"[{coverage['rr_ci_low']*100:+.1f}, {coverage['rr_ci_high']*100:+.1f}].",
              f"- MRR: {comparison['rr_difference']:+.4f} [{comparison['rr_ci_low']:+.4f}, {comparison['rr_ci_high']:+.4f}].",
              f"- Hit@10: {comparison['hit10_difference']*100:+.1f} pp "
              f"[{comparison['hit10_ci_low']*100:+.1f}, {comparison['hit10_ci_high']*100:+.1f}].",
              f"- Output tokens: {before:.1f} → {after:.1f}, across {int(available.sum())} pairs with usage on both sides." if available.any()
              else "- Output tokens: no paired usage available.",
              f"- Same provider: {same_provider}/{len(providers)} pairs with known provider labels.",
              "", f"**Decision: {comparison['recommendation']}.** {RULE}",
              "", "All users are retained in ranking and coverage metrics; failed requests score zero. "
              "The fixed cohort is small, and intervals are exploratory. Provider labels cannot verify backend version stability."]
    known = [r["api_cost_usd"] for r in rows if r.get("api_cost_usd") is not None]
    errors = Counter(r["error_category"] for r in rows if r["error"])
    lines += ["", f"Failure categories: {dict(errors)}. Reported cost: ${sum(known):.4f} over {len(known)}/{len(rows)} conditions.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(output, execute=False, limit=40):
    config, cohorts, dataset, cases, fingerprints, snapshot_hash = load_snapshot(
        ROOT / "experiments/mini-genrec-v1", ROOT / "data")
    users = cohorts["evaluation"]["user_ids"]
    if MODEL not in config["llm"]["models"] or len(users) != 34:
        raise ValueError("Unexpected frozen model or cohort.")
    expected = {r["user_id"]: r for r in fingerprints if r["variant"] == "constrained" and r["history_length"] == 20}
    v = Verbalizer(dataset.movies, MODEL, config["context"]["estimated_input_token_budget"])
    prompts = {}
    for uid in users:
        case = cases[uid]
        title = v.build(case.history, case.candidates, case.timestamp, 20)
        if digest(title.messages) != expected[uid]["messages_sha256"]:
            raise ValueError("Title prompt changed from the frozen reference.")
        prompts[uid, "title"], prompts[uid, "id"] = title, id_prompt(title, v)
    plan = order(users)
    protocol = dict(model=MODEL, snapshot=snapshot_hash, order=[list(k) for k in plan],
                    prompt_sha256=digest([prompts[k].messages for k in plan]),
                    schemas={"title": response_format(), "id": id_schema()},
                    primary="candidate_coverage", secondary=["api_output_tokens", "rr", "hit10"],
                    bootstrap_samples=10000, bootstrap_seed=42, decision_rule=RULE,
                    source_sha256={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                   for name in ("paired_format.py", "id_experiment.py", "llm.py", "evaluate.py")})
    print(f"Plan {digest(protocol)[:16]}: {len(plan)} fresh conditions, alternating title-first and ID-first.", flush=True)
    if not execute:
        return
    state_path = output / "state.json"
    cache = ROOT / "cache" / ("paired-format-" + digest(protocol)[:16])
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["identity"] != protocol:
            raise ValueError("Checkpoint differs from this protocol.")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty.")
        if cache.exists() and any(cache.iterdir()):
            raise ValueError("Fresh run requires an unused cache namespace.")
        state = dict(identity=protocol, rows=[], invocations=[])
    completed = [(r["user_id"], r["arm"]) for r in state["rows"]]
    if completed != plan[:len(completed)]:
        raise ValueError("Checkpoint does not follow the interleaved plan.")
    pending = plan[len(completed):]
    if pending:
        preflight([MODEL], len(pending))
    output.mkdir(parents=True, exist_ok=True)
    save(output / "protocol.json", protocol)
    save(state_path, state)
    status = dict(status="running", mode="api")
    save(output / "status.json", status)
    budget = RequestBudget(limit, config["llm"]["minimum_request_interval_seconds"])
    invocation = {"uncached_attempts": 0}
    state["invocations"].append(invocation)
    clients = {}
    try:
        for uid, arm in pending:
            if arm not in clients:
                factory = IdLLM if arm == "id" else LLM
                clients[arm] = factory(MODEL, cache, "openrouter", budget,
                                       reasoning=config["llm"]["reasoning"],
                                       max_output_tokens=config["llm"]["max_output_tokens"])
            prompt, case = prompts[uid, arm], cases[uid]
            started = datetime.now(timezone.utc).isoformat()
            row = evaluate_llm(clients[arm], prompt, common_row(case, 20, prompt), dataset, case,
                               "mini-genrec", decode=decode_ids if arm == "id" else None)
            state["rows"].append(dict(row, model=MODEL, arm=arm, started_at=started))
            invocation["uncached_attempts"] = budget.used
            save(state_path, state)
            print(f"{len(state['rows'])}/{len(plan)} recorded", flush=True)
        report(output, state["rows"])
        status["status"] = "complete"
    except (RunPaused, KeyboardInterrupt) as error:
        status.update(status="incomplete", reason=getattr(error, "category", "interrupted"))
        raise SystemExit(f"Paused: {status['reason']}. Checkpoint saved.") from None
    except (SystemExit, Exception) as error:
        status.update(status="failed", reason=getattr(error, "category", type(error).__name__))
        raise SystemExit(f"Stopped: {status['reason']}. Checkpoint saved.") from None
    finally:
        invocation["uncached_attempts"] = budget.used
        save(state_path, state)
        save(output / "status.json", dict(status, completed=len(state["rows"])))
        for client in clients.values():
            client.client.close()
    archive_responses(output, cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--request-budget", type=int, choices=range(2, 51, 2), default=40)
    args = parser.parse_args()
    run(args.output, args.execute, args.request_budget)


if __name__ == "__main__":
    main()
