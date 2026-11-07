"""Generate tables, an elbow chart, and README results from measured rows only."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

# Keep matplotlib's writable cache within the system temporary directory.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mini-genrec-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def number(value, percent=False):
    if pd.isna(value):
        return "not run / unavailable"
    return f"{value:.1%}" if percent else f"{value:.4f}"


def write_report(output: Path, summary: pd.DataFrame, metadata: dict, update_readme: bool):
    elbow = pd.read_csv(output / "elbow.csv")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.plot(elbow.approx_input_tokens, elbow.baseline_mrr, "o--", color="#666666",
            label="Baseline (matched history window)")
    available = elbow.genrec_mrr.notna()
    if available.any():
        ax.plot(elbow.loc[available, "approx_input_tokens"], elbow.loc[available, "genrec_mrr"],
                "o-", color="#b20710", label="mini-GenRec (measured API outputs)")
    else:
        ax.text(0.5, 0.93, f"LLM not run: {metadata['llm_skip_reason']}\nNo LLM curve or elbow is inferred.",
                transform=ax.transAxes, ha="center", va="top", color="#b20710", fontsize=10)
    for row in elbow.itertuples():
        y = row.genrec_mrr if pd.notna(row.genrec_mrr) else row.baseline_mrr
        ax.annotate(f"{row.history_length} events", (row.approx_input_tokens, y),
                    xytext=(-5, 10) if row.history_length == 50 else (5, 10),
                    ha="right" if row.history_length == 50 else "left",
                    textcoords="offset points", fontsize=9)
    ax.set(xlabel="Mean approximate LLM input tokens (history + candidates + schema)",
           ylabel="Mean reciprocal rank (MRR)",
           title=f"mini-GenRec: history budget sweep · {metadata['evaluated_users']} users\n{metadata['model']}",
           ylim=(0, min(1.0, max(0.3, float(elbow.baseline_mrr.max()) * 1.6,
                               float(elbow.genrec_mrr.max()) * 1.6 if available.any() else 0))))
    ax.legend(loc="upper right" if available.any() else "lower right", fontsize=8,
              frameon=True, framealpha=0.95)
    fig.savefig(output / "elbow.png", dpi=160)
    plt.close(fig)

    lines = [f"Run mode: **{metadata['mode']}**. Backend: `{metadata.get('backend', 'openai')}`. "
             f"Model configured: `{metadata['model']}`. "
             f"Seed {metadata['seed']}; {metadata['evaluated_users']} users; "
             f"{metadata['candidates']} candidates; {metadata['catalog_size']} catalog movies.",
             "", "| Ranker (20-event window) | Users | MRR | Hit@10 | Popular share of top-10 |",
             "|---|---:|---:|---:|---:|"]
    main = summary[(summary.history_length == 20) & (summary.ranker != "unconstrained")]
    for row in main.itertuples():
        lines.append(f"| {row.ranker} | {row.users} | {number(row.mrr)} | {number(row.hit10)} "
                     f"| {number(row.popular_top10_fraction, True)} |")
    if metadata["mode"] == "baseline-only":
        lines.append("| mini-GenRec | 0 | not run | not run | not run |")
    lines.extend(["", "| History window | Mean retained high-signal events | Approx. input tokens | Baseline MRR | mini-GenRec MRR |",
                  "|---:|---:|---:|---:|---:|"])
    for row in elbow.itertuples():
        lines.append(f"| {row.history_length} | {row.retained_events:.1f} | {row.approx_input_tokens:.0f} "
                     f"| {number(row.baseline_mrr)} | {number(row.genrec_mrr)} |")
    lines.extend(["", "![MRR versus approximate input tokens](results/elbow.png)", "",
                  f"Elbow: {metadata['elbow']['reason']}", "", "### What broke first", ""])
    b20 = main[main.ranker == "baseline"].iloc[0]
    lines.append(f"- Data coverage: {metadata['eligible_users']} users qualified after the temporal/catalog/history "
                 f"filters; this run sampled {metadata['evaluated_users']}. "
                 f"{metadata['test_targets_below_4_stars']} held-out ratings were below 4 stars. "
                 "MovieLens ratings are an imperfect proxy for next watches or satisfaction.")
    lines.append(f"- Baseline misses: {int(round((1-b20.hit10)*b20.users))}/{b20.users} targets "
                 f"missed the top 10 at 20 events. Its popular-item top-10 share was "
                 f"{number(b20.popular_top10_fraction, True)}.")
    prompt_frame = pd.read_json(output / "prompts.json")
    p50 = prompt_frame[prompt_frame.history_length == 50]
    lines.append(f"- Context at 50 events: {int(p50.low_signal_dropped.sum())} neutral ratings omitted; "
                 f"{int(p50.budget_dropped.sum())} high-signal events dropped for the token budget; "
                 f"{int(p50.repeat_events_compressed.sum())} repeat events compressed. "
                 "Counts are across evaluated users. MovieLens normally has one rating per user/movie.")
    if metadata["mode"] == "baseline-only":
        lines.append(f"- LLM not run: {metadata['llm_skip_reason']}. **0 LLM calls were made**; "
                     f"the full run needs {metadata['planned_llm_requests']} logical requests before retries "
                     "(four history windows plus one unconstrained probe per user). "
                     "LLM MRR, Hit@10, hallucination, popularity bias, and the elbow remain unmeasured.")
    else:
        for row in summary[summary.ranker != "baseline"].itertuples():
            lines.append(f"- {row.ranker}, {row.history_length} events: "
                         f"{row.suggestions} raw suggestions; out-of-catalog rate "
                         f"{number(row.outside_catalog_rate, True)}; outside-candidate rate "
                         f"{number(row.outside_candidates_rate, True)}; {row.duplicate_suggestions} duplicates; "
                         f"{row.errors}/{row.users} failed responses; "
                         f"{getattr(row, 'truncated_responses', 0)} truncated responses; top-10 slot coverage "
                         f"{number(row.top10_slot_coverage, True)}.")
    lines.extend(["", "Measured artifacts: [metrics.csv](results/metrics.csv), "
                  "[elbow.csv](results/elbow.csv), [run.json](results/run.json), "
                  "[per_user.json](results/per_user.json), [cases.json](results/cases.json), "
                  "[prompts.json](results/prompts.json).", ""])
    text = "\n".join(lines)
    # README-relative paths differ from paths inside the results directory.
    (output / "report.md").write_text(text.replace(
        "](results/", "]("), encoding="utf-8")
    if update_readme:
        path = Path("README.md")
        original = path.read_text()
        start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
        before, rest = original.split(start, 1)
        _, after = rest.split(end, 1)
        # Resolve an alternate output directory correctly in README links.
        relative = os.path.relpath(output.resolve(), path.parent.resolve())
        text = text.replace("](results/", f"]({relative}/")
        path.write_text(before + start + "\n" + text + end + after)
