# mini-GenRec

A weekend-sized, **unfine-tuned** experiment inspired by [GenRec: Towards LLM-Native Recommendation at Netflix](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3), Ying Li, Arjun Rao and Shradha Sehgal, Netflix Tech Blog, July 30, 2026. The source post was fetched and read for this implementation.

Netflix trains a foundation model and a catalog scoring head together, then scores without decoding recommendations. This repo uses an off-the-shelf OpenAI model to **generate a candidate ranking**, with a code-enforced catalog gate. It reproduces the context and measurement ideas, not Netflix's architecture, training, serving costs, or claimed performance.

## Method in ten lines

1. Download the public [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/latest/) archive; retain its upstream README/license locally.
2. Freeze training strictly before the dataset's 80th-percentile timestamp; choose the 1,000 most-rated movies using only that period.
3. Fit the baseline on users with 20+ catalog ratings in that training period; use no held-out or future ratings for similarities or popularity.
4. Test users need 20+ strictly earlier catalog ratings; hold out each user's latest **overall** rating, only if it is in the catalog and at/after the cutoff.
5. Fix 50 candidates per user: the target plus 49 uniformly sampled, previously unrated catalog distractors; shuffle once with seed 42.
6. Baseline: 80% normalized item-CF affinity plus 20% log popularity; cosine similarities use positive (4+ star) interactions and overlap shrinkage.
7. Verbalize the last 5, 10, 20 or 50 catalog ratings with titles, stars, genres and relative age. Ratings are proxies, not verified watches.
8. Omit neutral (2.5–3.5 star) ratings, compress repeated movie ratings, prioritize strong preferences, and pack events within a token budget.
9. Ask `gpt-4o-mini` (configurable) to order exact candidate titles; drop unknown, ambiguous, noncandidate and duplicate titles without backfilling.
10. Measure MRR, Hit@10, popularity share and hallucinations; run one free-catalog probe per user and plot quality against input tokens.

## Run

Python **3.10+** recommended (verified with 3.12). No fine-tuning, server, database or GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY in .env, or export it in your shell. Never commit it.
python -m mini_genrec.evaluate --users 20 --update-readme
python -m unittest discover -s tests -v
```

For a free local run without credentials:

```bash
python -m mini_genrec.evaluate --baseline-only --users 20 --update-readme
```

## Compare free models through OpenRouter

OpenRouter is supported using the existing OpenAI-compatible SDK transport, with the fixed endpoint `https://openrouter.ai/api/v1`. Set **`OPENROUTER_API_KEY`** in `.env`; an OpenAI key is not required for this backend. No credential values are included in requests' cache keys, prompts, reports, or logs.

List the current public model catalog without an API key:

```bash
python -m mini_genrec.openrouter
```

Pick explicit named models ending in **`:free`** that advertise `structured_outputs`. The listing distinguishes JSON-schema support from generic `response_format` support. Support depends on the actual serving endpoint, so first run one user. Availability and model IDs change; a model appearing in the catalog is not a completed compatibility or recommendation-quality test. [Public model catalog](https://openrouter.ai/api/v1/models), [structured-output documentation](https://openrouter.ai/docs/guides/features/structured-outputs).

For example, these two IDs advertised structured outputs when the catalog was checked on November 6, 2026 and subsequently passed live one-user smoke tests:

```bash
# One model, one user: 5 logical requests, including all four history windows.
python -m mini_genrec.evaluate --backend openrouter \
  --model nvidia/nemotron-3.5-super-vl:free --reasoning off \
  --users 1 --output results/openrouter-smoke-nemotron

# Two models, three identical test users: 30 logical requests total.
python -m mini_genrec.evaluate --backend openrouter \
  --models nvidia/nemotron-3.5-super-vl:free dots-studio/dots-3-note-preview:free \
  --reasoning off --users 3 --output results/openrouter-comparison
```

The matrix produces a directory per model containing the normal artifacts, plus **`comparison.csv`** and **`comparison.md`** at its root. Every model uses the same user IDs, targets and candidate sets. Each cache key includes the backend and complete request; resolved model, provider (when returned), usage and original request latency are recorded. Matrix mode preserves the README's original measured run even with `--update-readme`. Separate invocations can compare different samples: compare only matching seeds, cohort sizes, catalog/candidate settings, token budgets and dataset hashes.

This backend intentionally accepts only **named `:free` variants**. Paid models, `openrouter/auto`, and `openrouter/free` are rejected; the latter would mix models within an experiment. Requests specify `provider.require_parameters=true` so incompatible endpoints cannot silently ignore the JSON-schema setting. We do not fall back to other model IDs or relax output validation. No compatible endpoint means a configuration error to investigate, not permission to use a paid variant. Provider implementations still vary; Python validates every response.

The current no-purchase free tier allows **20 requests/minute and 50 requests/day across the account**; failed attempts also count. Other activity on the account consumes the same allowance. This is a rate-limited service, not unlimited free inference. See [OpenRouter's free-model limits](https://openrouter.ai/collections/free-models) and [account limits API](https://openrouter.ai/docs/api/reference/limits).

For OpenRouter, the runner spaces uncached requests at least **3.1 seconds apart**, disables SDK retries, and shares a default **40-attempt budget across all models in one invocation**. Adjust with `--request-budget 1..50`. This guard is not an account-wide quota tracker. Cache hits consume no local attempt budget. On quota errors or budget exhaustion, the run stops with an `incomplete` status; rerun the identical command later to reuse cached responses. No partial aggregate model comparison is published. The default 20-user experiment needs 100 logical requests per model, so plan to resume across days; failed requests may require additional quota. Treat only directories with `status.json` set to `complete` as completed runs, since older report files can remain during an interrupted rerun.

`--reasoning off` sends OpenRouter's `reasoning.enabled=false`; use it only with models that support disabling reasoning. Models requiring reasoning can consume the entire output allowance before emitting a ranking. `--max-output-tokens` defaults to 2400 and includes reasoning plus visible output. Both settings are part of the cache identity. API-reported reasoning tokens, response finish reason, visible response length and cost (where returned) are recorded. See [OpenRouter reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

**Live OpenRouter smoke tests have now run.** The initial Liquid model failed four of five requests at the output limit; Nvidia and Dots completed all five requests each. The live results below are separate from mocked automated tests, whose outputs never enter the checked-in measured results. The `.env` key remains local and ignored by Git.

The default API experiment makes **100 logical requests per model**: 20 users × (four constrained windows + one unconstrained 20-event probe). On the OpenAI backend SDK retries may add requests. Start with `--users 3 --output results/smoke` for 15 logical requests. Defaults: `--backend openai`, `--model gpt-4o-mini`, `--token-budget 4096`, `--candidates 50`, `--catalog-size 1000`, `--seed 42`; all are configurable. Model overrides must support Chat Completions, structured JSON output and temperature 0. The tokenizer falls back to `cl100k_base` for unknown model names; this is a cross-model estimate, not their native tokenization. API-reported token counts are recorded separately. First run needs network access for data and tokenizer assets.

`--token-budget` covers tokenizer-counted messages, schema and a 64-token framing allowance; it is an **estimated input budget**, not a guaranteed provider limit. API-reported usage is stored separately. Candidate metadata is never truncated: an impossible budget raises an error. `prompts.json` records exactly what would be sent; the target label is never identified to the LLM. History lengths are **caps**: users with fewer than 50 past catalog ratings contribute their available history to that condition. Per-user `window_events` and retained-event counts expose this. Baseline uses the same raw history window; LLM context compaction can discard or reorder events within that window. The baseline's fixed blend is a simple stand-in, not a production-tuned model.

`.env` is loaded locally and ignored by Git. `data/` and `cache/` are also ignored. Cache keys hash the complete request (model, prompts, schema and generation settings); rerunning an interrupted API experiment reuses completed responses. `per_user.json` is checkpointed after each completed user. Authentication/configuration errors stop the run; other request/parse failures yield an empty ranking, zero MRR/Hit@10, and a recorded error. SDK error bodies and secrets are never written to artifacts. No fallback baseline scores are reported as LLM predictions. Temperature 0 does not guarantee deterministic provider responses; retain cached outputs and the recorded resolved model for comparison.

## Evaluation definitions and limitations

- **MRR** averages the reciprocal target rank; absent targets get 0. **Hit@10** averages target membership in the first 10 unique valid results. All requested users remain in these denominators, including failures.
- **Catalog** means the selected MovieLens subset, not all movies ever made. Hallucination here is a membership error, not proof a title is fictional. Free-catalog recommendations get no candidate list: `outside_catalog_rate` is unmatched raw titles / all returned title strings. Exact matching allows Unicode/case/whitespace normalization, but requires the year and MovieLens's title wording; aliases can inflate the measured rate.
- The constrained run also reports **outside-candidate rate** (including real catalog movies outside the supplied 50). It uses structural [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs), with **no title enum** in the schema. Candidate membership is validated independently in Python. The unconstrained probe keeps the same JSON structure but removes the candidate list and catalog instruction.
- **Popularity bias** is the micro-averaged fraction of valid top-10 results in the most-rated 10% of catalog movies, using training counts. Report slot coverage alongside it so short/empty rankings cannot conceal failures. Unmatched suggestions stay in hallucination denominators; duplicates are counted and removed. There is no candidate-filling fallback.
- Histories may contain observations after the common training cutoff, but strictly before their own target. This is a frozen baseline with an updated user profile. All ties at the target timestamp are excluded from history; the largest movie ID breaks ties when choosing the latest target. Users whose actual latest movie is outside the catalog are excluded, never assigned an older substitute.
- Uniform negatives and a small sample make this easier than full-catalog retrieval. Rated movies may be disliked; next-rating prediction is not satisfaction prediction. MovieLens and the LLM's possible prior knowledge of it do not represent Netflix traffic. No significance or superiority claim is justified by this demo.
- The exploratory **elbow** is the smallest tested history within **0.01 absolute MRR** of the best observed LLM result. All four conditions must finish without request/parse errors before an elbow is declared. It is not an independently validated optimum. The main table always uses the prespecified 20-event window.
- Raw dataset SHA-256 hashes, shared cutoff, candidates, user IDs, model, run arguments, API token usage, prompts and per-user outputs are saved for audit. `requirements.txt` uses bounded compatible versions; `requirements-lock.txt` records the environment used for the checked-in run.

<!-- OPENROUTER_LIVE_START -->
## Live OpenRouter comparison

Measured on 3 identical users, 50 candidates each, seed 42, and the same 1,000-movie catalog. Both models used `--reasoning off`, a 2,400-token output cap and a 4,096 estimated input-token cap. This is a smoke-sized comparison; it does not establish statistical superiority. One of the three held-out ratings was below 4 stars.

| Method (20-event window) | MRR | Hit@10 | Popular share of top-10 |
|---|---:|---:|---:|
| Popularity + item-CF | 0.0670 | 0.0% | 13.3% |
| Nvidia Nemotron | 0.1175 | 33.3% | 6.7% |
| Dots | 0.0358 | 0.0% | 10.0% |

### What broke first in live testing

- Initial Liquid smoke test: 4/5 responses hit the 2,400-token output limit without a usable ranking. The model requires reasoning, so disabling reasoning is not a valid fix for it. The failed conditions remain recorded in [the Liquid report](results/openrouter-smoke/report.md).
- Xing smoke test: its endpoint returned HTTP 429 before completing a user. The account allowance check still showed 994 requests remaining, so this was not exhaustion of that daily allowance. This incomplete run is excluded from the quality comparison.
- Dots: 100/644 raw constrained suggestions were outside the candidate sets (15.5%) and were dropped. 10/12 validated candidate rankings contained fewer than 50 movies. Unconstrained catalog-mismatch rate: 7/30 (23.3%). Exact title matching can count aliases as catalog mismatches.
- Nvidia Nemotron: 131/727 raw constrained suggestions were outside the candidate sets (18.0%) and were dropped. 3/12 validated candidate rankings contained fewer than 50 movies. Unconstrained catalog-mismatch rate: 5/30 (16.7%). Exact title matching can count aliases as catalog mismatches.
- Reliability: 0/30 evaluated model conditions failed to return parseable responses; cached responses are included in these counts. All requests used named `:free` variants. Cost metadata was available for 20/30 conditions and totaled $0.00; missing cost fields remain unknown rather than being filled with zero.

### History sweep

Each model was evaluated at history caps of 5, 10, 20 and 50 on the same users. The plots use estimated input tokens; native API input counts are saved separately. The reported elbow rule is exploratory, and the 50-event condition can contain fewer events for users with shorter histories.

Dots: the descriptive 0.01-MRR rule selects a history cap of 5 (best observed MRR 0.0358).

![Dots MRR versus approximate input tokens](results/openrouter-comparison/dots-studio_dots-3-note-preview_free-952e7a96/elbow.png)

[Dots full report](results/openrouter-comparison/dots-studio_dots-3-note-preview_free-952e7a96/report.md) · [Elbow CSV](results/openrouter-comparison/dots-studio_dots-3-note-preview_free-952e7a96/elbow.csv)

Nvidia Nemotron: the descriptive 0.01-MRR rule selects a history cap of 50 (best observed MRR 0.3590).

![Nvidia Nemotron MRR versus approximate input tokens](results/openrouter-comparison/nvidia_nemotron-3.5-super-vl_free-f96f1282/elbow.png)

[Nvidia Nemotron full report](results/openrouter-comparison/nvidia_nemotron-3.5-super-vl_free-f96f1282/report.md) · [Elbow CSV](results/openrouter-comparison/nvidia_nemotron-3.5-super-vl_free-f96f1282/elbow.csv)

[Full comparison table](results/openrouter-comparison/comparison.md) · [Comparison CSV](results/openrouter-comparison/comparison.csv)
<!-- OPENROUTER_LIVE_END -->

## Original baseline run

This 20-user baseline-only run predates the live OpenRouter evaluation. Its missing LLM measurements refer to this original run, not the later experiments in the OpenRouter result directories.

<!-- RESULTS_START -->
Run mode: **baseline-only**. Backend: `openai`. Model configured: `gpt-4o-mini`. Seed 42; 20 users; 50 candidates; 1000 catalog movies.

| Ranker (20-event window) | Users | MRR | Hit@10 | Popular share of top-10 |
|---|---:|---:|---:|---:|
| baseline | 20 | 0.1901 | 0.4500 | 21.0% |
| mini-GenRec | 0 | not run | not run | not run |

| History window | Mean retained high-signal events | Approx. input tokens | Baseline MRR | mini-GenRec MRR |
|---:|---:|---:|---:|---:|
| 5 | 3.8 | 2047 | 0.3122 | not run / unavailable |
| 10 | 7.2 | 2147 | 0.2320 | not run / unavailable |
| 20 | 14.1 | 2351 | 0.1901 | not run / unavailable |
| 50 | 26.3 | 2708 | 0.1617 | not run / unavailable |

![MRR versus approximate input tokens](results/elbow.png)

Elbow: Requires all four measured LLM conditions without errors.

### What broke first

- Data coverage: 37 users qualified after the temporal/catalog/history filters; this run sampled 20. 13 held-out ratings were below 4 stars. MovieLens ratings are an imperfect proxy for next watches or satisfaction.
- Baseline misses: 11/20 targets missed the top 10 at 20 events. Its popular-item top-10 share was 21.0%.
- Context at 50 events: 256 neutral ratings omitted; 0 high-signal events dropped for the token budget; 0 repeat events compressed. Counts are across evaluated users. MovieLens normally has one rating per user/movie.
- LLM not run: OPENAI_API_KEY unavailable. **0 LLM calls were made**; the full run needs 100 logical requests before retries (four history windows plus one unconstrained probe per user). LLM MRR, Hit@10, hallucination, popularity bias, and the elbow remain unmeasured.

Measured artifacts: [metrics.csv](results/metrics.csv), [elbow.csv](results/elbow.csv), [run.json](results/run.json), [per_user.json](results/per_user.json), [cases.json](results/cases.json), [prompts.json](results/prompts.json).
<!-- RESULTS_END -->
