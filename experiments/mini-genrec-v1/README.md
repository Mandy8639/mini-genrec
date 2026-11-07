# Frozen experiment: mini-genrec-v1

This directory freezes the inputs and planned protocol after the three-user OpenRouter pilot. It contains no model responses or new evaluation results. Freezing these inputs made **zero model calls**.

| Cohort | Users | Role |
|---|---:|---|
| Development | 3 | Previously evaluated users 62, 418 and 461; prompt debugging and compatibility checks |
| Evaluation | 34 | Remaining eligible users, previously unused in the Nvidia/Dots LLM evaluations |
| Elbow | 21 | Evaluation subset with at least 50 strictly earlier catalog ratings |

The 37 cases preserve the existing 80th-percentile temporal cutoff, 1,000-movie catalog and seed-42 candidate sampling. No split, candidate ordering, baseline weight, prompt or ranker behavior was changed. The evaluation and elbow selection orders are explicit lists: smaller batches must take prefixes of these lists rather than drawing a new random sample.

## Contents

- `config.json`: models, generation settings, budget, baseline parameters, prompt contract, evaluation rules, planned conditions and the original source commit/hash references.
- `cohorts.json`: fixed membership and selection order. The elbow cohort is intentionally a subset of evaluation, not a third disjoint partition.
- `catalog.json`: all 1,000 catalog movies with titles and genres, plus the training-period popular group.
- `cases.json`: targets, target timestamps, ordered 50-item candidate sets and complete eligible rating histories for all 37 users.
- `prompt_fingerprints.json`: hashes and context counts for all 185 existing engineered-history conditions (four constrained history caps and one unconstrained probe per user). The two selected models currently use the same `cl100k_base` estimate, so the fingerprints apply to both.
- `prompt_examples.json`: exact constrained and unconstrained 20-event messages for development user 62.
- `MOVIELENS_README.txt`: the unmodified upstream dataset documentation, citation and usage conditions. These conditions apply to MovieLens-derived artifacts here and in the existing result directories; they are separate from any future code license.
- `checksums.json`: SHA-256 digests of the other files in this directory.

The source commit identifies the prototype before these freeze files were added. Source-file hashes record provenance; rendered prompt fingerprints are the check against accidental changes to the existing prompt behavior. Dataset fingerprints must match before using these cases with a newly downloaded dataset.

## Execution boundary

**This is an input/protocol snapshot, not an inference command.** `execution.inference_enabled` is false and runner integration is pending. The current `mini_genrec.evaluate` CLI does not load this directory. In particular, passing `--users 34` to that legacy CLI does not select these 34 evaluation users.

The plain-history condition and random/popularity-only controls are planned in the configuration but are not implemented by this freeze. Their addition and frozen-manifest runner integration are the next work items. No final evaluation should start until those changes honor this manifest.

The planned two-model core has 330 logical requests: 68 engineered 20-event rankings, 68 plain-history rankings, 68 unconstrained probes and 126 additional elbow conditions. The elbow reuses the main run's 20-event engineered results. This is a request plan, not a report that those requests have run. Failed attempts may consume additional quota; cache hits do not require a new provider request.

Do not quietly edit frozen files after collecting results. An intentional protocol or cohort change should get a new named snapshot, with its differences recorded. The initial Git checkpoint protects the old prototype; these files protect the evaluation inputs.

## Verify locally

From the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_frozen_experiment.py' -v
```

The checks verify file digests, free-only model configuration, request arithmetic, cohort boundaries, candidate constraints, compatibility with the recorded pilot and reproduction of the current prompt fingerprints. Dataset/prompt reproduction checks need the downloaded dataset and cached tokenizer assets; artifact-only checks work without the dataset. No API key or model inference is used.
