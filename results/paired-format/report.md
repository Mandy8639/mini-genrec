# Interleaved Dots format comparison

34 frozen users, 50 identical ordered candidates and engineered 20-event histories. Title and ID requests were sent back to back, alternating which format went first. Both arms used a fresh shared cache namespace, with distinct keys for each request schema.

| Output | Candidate coverage | MRR | Hit@10 | Failed requests |
|---|---:|---:|---:|---:|
| Title | 92.0% | 0.1837 | 29.4% | 1/34 |
| ID | 97.0% | 0.2264 | 41.2% | 0/34 |

ID minus title; 95% percentile intervals over 10,000 paired-user bootstrap resamples (seed 42):

- Coverage: +5.0 pp [+0.2, +12.2].
- MRR: +0.0427 [-0.0592, +0.1492].
- Hit@10: +11.8 pp [-2.9, +29.4].
- Output tokens: 590.8 → 288.8, across 33 pairs with usage on both sides.
- Same provider: 33/33 pairs with known provider labels.

**Decision: IDs for Dots efficiency.** Prefer IDs for Dots efficiency if the 95% paired coverage interval is entirely above zero, mean reported output tokens decrease on matched pairs, and neither ranking interval is entirely below zero. Otherwise keep titles. This is not a test of ranking noninferiority.

All users are retained in ranking and coverage metrics; failed requests score zero. The fixed cohort is small, and intervals are exploratory. Provider labels cannot verify backend version stability.

Failure categories: {'provider_rejection': 1}. Reported cost: $0.0000 over 67/68 conditions.
