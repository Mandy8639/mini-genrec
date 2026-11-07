# Movie ID versus title output

Same 34 users, candidate order, candidate titles/genres, engineered 20-event history, models, and generation settings. Only the output instruction and schema item type changed. The title arm is historical, so provider/model drift and sampling may affect differences.

| Model | Output | MRR | Hit@10 | Coverage | Failures |
|---|---|---:|---:|---:|---:|
| nvidia/nemotron-3.5-super-vl:free | Title | 0.2464 | 38.2% | 97.6% | 0 |
| nvidia/nemotron-3.5-super-vl:free | ID | 0.2404 | 38.2% | 95.0% | 0 |
| dots-studio/dots-3-note-preview:free | Title | 0.2358 | 35.3% | 90.8% | 1 |
| dots-studio/dots-3-note-preview:free | ID | 0.2901 | 38.2% | 97.4% | 0 |

## Matched differences (ID minus title)

Output token averages include only pairs with reported token counts on both sides. Ranking and coverage use all users, including failed requests.

| Model | Coverage difference | MRR difference (95% paired interval) | Hit@10 difference | Title → ID output tokens | Token pairs | Same provider* |
|---|---:|---|---:|---:|---:|---:|
| nvidia/nemotron-3.5-super-vl:free | -2.6 pp | -0.0059 [-0.0674, +0.0466] | +0.0 pp | 665.1 → 285.1 | 34 | 34/34 |
| dots-studio/dots-3-note-preview:free | +6.6 pp | +0.0543 [-0.0565, +0.1688] | +2.9 pp | 613.0 → 293.8 | 33 | 33/33 |

*Among pairs with known provider labels. Same provider does not rule out model drift.

Intervals use 10,000 paired-user bootstrap resamples with seed 42; exploratory, with no multiple-comparison adjustment.

Provider-reported cost: $0.0000 across 68/68 conditions with cost metadata.
