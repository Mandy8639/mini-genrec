"""Test whether Dots ID reranking improves the saved hybrid baseline's top 10."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd

from .analysis import archive_responses, paired_comparison
from .context import Verbalizer
from .evaluate import evaluate_llm, metrics
from .experiment import ROOT, common_row, digest, load_snapshot, save
from .id_experiment import IdLLM, TITLE_RUN, decode_ids, id_prompt, id_schema, preflight
from .llm import RequestBudget, RunPaused
from .paired_format import MODEL

OUTPUT = ROOT / "results/shortlist"
K = 20
RULE = "Prefer reranking only if the lower bound of the 95% paired Hit@10 interval is above zero; otherwise keep the hybrid baseline."


def merge_ranking(baseline, proposed):
    top, tail = baseline[:K], baseline[K:]
    if len(set(proposed)) != len(proposed) or not set(proposed).issubset(top):
        raise ValueError("Validated proposal must contain unique shortlist IDs only.")
    accepted = set(proposed)
    return list(proposed) + [mid for mid in top if mid not in accepted] + tail


def report(output, rows, baseline):
    frame = pd.DataFrame(rows).sort_values("user_id")
    base = pd.DataFrame(baseline.values()).sort_values("user_id")
    comparison = paired_comparison(frame, base, "Dots rerank minus hybrid baseline")
    recommend = comparison["hit10_ci_low"] > 0
    comparison["recommendation"] = "Prefer reranking" if recommend else "Keep the hybrid baseline"
    before = base.set_index("user_id").hit10
    after = frame.set_index("user_id").hit10
    helped, hurt = int((after > before).sum()), int((after < before).sum())
    failures = Counter(row["error_category"] for row in rows if row["error"])
    coverage = sum(row["target_in_shortlist"] for row in rows)
    fallback = sum(not row["ranking"] for row in rows)
    partial = sum(0 < len(row["ranking"]) < K for row in rows)
    comparison.update(shortlist_targets=coverage, hit10_helped=helped, hit10_hurt=hurt,
                      full_fallbacks=fallback, partial_rankings=partial)
    pd.DataFrame([comparison]).to_csv(output / "paired_comparison.csv", index=False)
    cost = [row["api_cost_usd"] for row in rows if row.get("api_cost_usd") is not None]
    (output / "report.md").write_text("\n".join([
        "# Dots reranking of the hybrid top 20", "",
        f"Targets in the unmodified baseline shortlist: **{coverage}/{len(rows)} ({coverage/len(rows):.1%})**. "
        "No held-out targets were inserted. Every user is scored on the full 50 candidates.", "",
        "| Method | MRR | Hit@10 |",
        "|---|---:|---:|",
        f"| Hybrid baseline | {base.rr.mean():.4f} | {base.hit10.mean():.1%} |",
        f"| Dots rerank + baseline fallback | {frame.rr.mean():.4f} | {frame.hit10.mean():.1%} |",
        "", f"- **Hit@10 difference:** {comparison['hit10_difference']*100:+.1f} pp "
        f"(95% paired interval [{comparison['hit10_ci_low']*100:+.1f}, {comparison['hit10_ci_high']*100:+.1f}]). "
        f"Helped {helped}, hurt {hurt}, unchanged {len(rows)-helped-hurt}.",
        f"- MRR difference: {comparison['rr_difference']:+.4f} "
        f"(95% paired interval [{comparison['rr_ci_low']:+.4f}, {comparison['rr_ci_high']:+.4f}]).",
        f"- Full baseline fallbacks: {fallback}; partial model rankings: {partial}; request/parse failures: {dict(failures)}.",
        f"- Mean valid shortlist coverage before fallback: {frame.candidate_coverage.mean():.1%}; "
        f"outside-shortlist suggestions: {int(frame.outside_candidates.sum())}.",
        f"- Provider-reported cost: ${sum(cost):.4f} across {len(cost)}/{len(rows)} requests with cost metadata.",
        "", f"**Decision: {comparison['recommendation']}.** {RULE}", "",
        "Returned valid IDs are ordered first within the shortlist, followed by omitted shortlist items in their baseline order. "
        "The baseline tail (positions 21–50) is untouched. Failed or empty model rankings leave the entire baseline order unchanged.",
        "", "Intervals use 10,000 paired-user bootstrap resamples, seed 42. This small frozen cohort is exploratory, "
        "and the baseline’s shortlist depends on the original sampled candidate sets.", ""
    ]), encoding="utf-8")


def run(output, execute=False):
    config, cohorts, dataset, cases, fingerprints, snapshot_hash = load_snapshot(
        ROOT / "experiments/mini-genrec-v1", ROOT / "data")
    users = cohorts["evaluation"]["user_ids"]
    source = (TITLE_RUN / "per_user.json").read_bytes()
    run_info = json.loads((TITLE_RUN / "run.json").read_text())
    status_info = json.loads((TITLE_RUN / "status.json").read_text())
    if (len(users) != 34 or MODEL not in config["llm"]["models"] or status_info["status"] != "complete"
            or run_info["identity"]["manifest_sha256"] != snapshot_hash or run_info["selected_users"] != users):
        raise ValueError("Saved baseline does not match the frozen cohort.")
    baseline_rows = [r for r in json.loads(source) if r["ranker"] == "baseline" and r["history_length"] == 20]
    baseline = {r["user_id"]: r for r in baseline_rows}
    if set(baseline) != set(users) or len(baseline_rows) != len(users):
        raise ValueError("Missing or duplicate baseline case.")
    expected = {r["user_id"]: r for r in fingerprints if r["variant"] == "constrained" and r["history_length"] == 20}
    v = Verbalizer(dataset.movies, MODEL, config["context"]["estimated_input_token_budget"])
    prompts, shortlisted = {}, {}
    for uid in users:
        case, base = cases[uid], baseline[uid]
        full = base["ranking"]
        if len(full) != len(case.candidates) or set(full) != set(case.candidates):
            raise ValueError("Saved baseline is not a permutation of the frozen candidates.")
        if any(base[k] != value for k, value in metrics(full, case.target, dataset.popular).items()):
            raise ValueError("Saved baseline scores do not match the held-out target.")
        reference = v.build(case.history, case.candidates, case.timestamp, 20)
        if digest(reference.messages) != expected[uid]["messages_sha256"]:
            raise ValueError("Frozen history prompt changed.")
        short = replace(case, candidates=full[:K])
        title = v.build(short.history, short.candidates, short.timestamp, 20)
        if title.messages[1]["content"].rsplit("\nCandidates:\n", 1)[0] != reference.messages[1]["content"].rsplit("\nCandidates:\n", 1)[0]:
            raise ValueError("Shortlisting changed the engineered history or task text.")
        shortlisted[uid], prompts[uid] = short, id_prompt(title, v)
    protocol = dict(model=MODEL, users=users, snapshot=snapshot_hash,
                    baseline_sha256=hashlib.sha256(source).hexdigest(),
                    shortlists={str(uid): shortlisted[uid].candidates for uid in users},
                    prompts_sha256=digest([prompts[uid].messages for uid in users]),
                    schema=id_schema(), primary="hit10", bootstrap_samples=10000, bootstrap_seed=42,
                    decision_rule=RULE, fallback="Append omitted shortlist IDs in baseline order; leave tail unchanged.",
                    source_sha256={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                   for name in ("shortlist.py", "id_experiment.py", "llm.py", "evaluate.py")})
    n_targets = sum(shortlisted[uid].target in shortlisted[uid].candidates for uid in users)
    print(f"Plan {digest(protocol)[:16]}: {len(users)} free requests; shortlist contains {n_targets}/{len(users)} targets.", flush=True)
    if not execute:
        return
    state_path = output / "state.json"
    cache = ROOT / "cache" / ("shortlist-" + digest(protocol)[:16])
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["identity"] != protocol:
            raise ValueError("Checkpoint differs from the frozen protocol.")
    else:
        if (output.exists() and any(output.iterdir())) or (cache.exists() and any(cache.iterdir())):
            raise ValueError("Fresh run requires empty output and cache namespaces.")
        state = dict(identity=protocol, rows=[], invocations=[])
    if [r["user_id"] for r in state["rows"]] != users[:len(state["rows"])]:
        raise ValueError("Invalid checkpoint order.")
    pending = users[len(state["rows"]):]
    if pending:
        preflight([MODEL], len(pending))
    output.mkdir(parents=True, exist_ok=True)
    save(output / "protocol.json", protocol)
    save(state_path, state)
    status = dict(status="running", mode="api")
    save(output / "status.json", status)
    budget = RequestBudget(34, config["llm"]["minimum_request_interval_seconds"])
    invocation = {"uncached_attempts": 0}
    state["invocations"].append(invocation)
    client = None
    try:
        if pending:
            client = IdLLM(MODEL, cache, "openrouter", budget, reasoning=config["llm"]["reasoning"],
                           max_output_tokens=config["llm"]["max_output_tokens"])
        for uid in pending:
            case, prompt = shortlisted[uid], prompts[uid]
            row = evaluate_llm(client, prompt, common_row(case, 20, prompt), dataset, case,
                               "mini-genrec", decode=decode_ids)
            final = merge_ranking(baseline[uid]["ranking"], row["ranking"])
            state["rows"].append(dict(row, model=MODEL, final_ranking=final,
                                      target_in_shortlist=case.target in case.candidates)
                                 | metrics(final, case.target, dataset.popular))
            invocation["uncached_attempts"] = budget.used
            save(state_path, state)
            print(f"{len(state['rows'])}/{len(users)} recorded", flush=True)
        report(output, state["rows"], baseline)
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
        if client:
            client.client.close()
    archive_responses(output, cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output, args.execute)
