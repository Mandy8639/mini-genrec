# Dots reranking of the hybrid top 20

Targets in the unmodified baseline shortlist: **24/34 (70.6%)**. No held-out targets were inserted. Every user is scored on the full 50 candidates.

| Method | MRR | Hit@10 |
|---|---:|---:|
| Hybrid baseline | 0.2220 | 50.0% |
| Dots rerank + baseline fallback | 0.2486 | 44.1% |

- **Hit@10 difference:** -5.9 pp (95% paired interval [-20.6, +11.8]). Helped 3, hurt 5, unchanged 26.
- MRR difference: +0.0266 (95% paired interval [-0.0840, +0.1399]).
- Full baseline fallbacks: 0; partial model rankings: 0; request/parse failures: {}.
- Mean valid shortlist coverage before fallback: 100.0%; outside-shortlist suggestions: 0.
- Provider-reported cost: $0.0000 across 34/34 requests with cost metadata.

**Decision: Keep the hybrid baseline.** Prefer reranking only if the lower bound of the 95% paired Hit@10 interval is above zero; otherwise keep the hybrid baseline.

Returned valid IDs are ordered first within the shortlist, followed by omitted shortlist items in their baseline order. The baseline tail (positions 21–50) is untouched. Failed or empty model rankings leave the entire baseline order unchanged.

Intervals use 10,000 paired-user bootstrap resamples, seed 42. This small frozen cohort is exploratory, and the baseline’s shortlist depends on the original sampled candidate sets.
