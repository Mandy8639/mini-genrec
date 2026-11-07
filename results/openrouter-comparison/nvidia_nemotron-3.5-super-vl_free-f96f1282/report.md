Run mode: **api**. Backend: `openrouter`. Model configured: `nvidia/nemotron-3.5-super-vl:free`. Seed 42; 3 users; 50 candidates; 1000 catalog movies.

| Ranker (20-event window) | Users | MRR | Hit@10 | Popular share of top-10 |
|---|---:|---:|---:|---:|
| baseline | 3 | 0.0670 | 0.0000 | 13.3% |
| mini-genrec | 3 | 0.1175 | 0.3333 | 6.7% |

| History window | Mean retained high-signal events | Approx. input tokens | Baseline MRR | mini-GenRec MRR |
|---:|---:|---:|---:|---:|
| 5 | 5.0 | 2140 | 0.4333 | 0.1564 |
| 10 | 8.3 | 2247 | 0.1414 | 0.1317 |
| 20 | 14.7 | 2436 | 0.0670 | 0.1175 |
| 50 | 30.7 | 2910 | 0.1361 | 0.3590 |

![MRR versus approximate input tokens](elbow.png)

Elbow: Smallest history within 0.01 absolute MRR of the best observed point; exploratory.

### What broke first

- Data coverage: 37 users qualified after the temporal/catalog/history filters; this run sampled 3. 1 held-out ratings were below 4 stars. MovieLens ratings are an imperfect proxy for next watches or satisfaction.
- Baseline misses: 3/3 targets missed the top 10 at 20 events. Its popular-item top-10 share was 13.3%.
- Context at 50 events: 32 neutral ratings omitted; 0 high-signal events dropped for the token budget; 0 repeat events compressed. Counts are across evaluated users. MovieLens normally has one rating per user/movie.
- mini-genrec, 5 events: 165 raw suggestions; out-of-catalog rate 0.0%; outside-candidate rate 9.1%; 0 duplicates; 0/3 failed responses; 0 truncated responses; top-10 slot coverage 100.0%.
- mini-genrec, 10 events: 174 raw suggestions; out-of-catalog rate 0.6%; outside-candidate rate 14.4%; 0 duplicates; 0/3 failed responses; 0 truncated responses; top-10 slot coverage 100.0%.
- mini-genrec, 20 events: 178 raw suggestions; out-of-catalog rate 0.6%; outside-candidate rate 16.3%; 0 duplicates; 0/3 failed responses; 0 truncated responses; top-10 slot coverage 100.0%.
- mini-genrec, 50 events: 210 raw suggestions; out-of-catalog rate 0.0%; outside-candidate rate 29.5%; 0 duplicates; 0/3 failed responses; 0 truncated responses; top-10 slot coverage 100.0%.
- unconstrained, 20 events: 30 raw suggestions; out-of-catalog rate 16.7%; outside-candidate rate not run / unavailable; 0 duplicates; 0/3 failed responses; 0 truncated responses; top-10 slot coverage 83.3%.

Measured artifacts: [metrics.csv](metrics.csv), [elbow.csv](elbow.csv), [run.json](run.json), [per_user.json](per_user.json), [cases.json](cases.json), [prompts.json](prompts.json).
