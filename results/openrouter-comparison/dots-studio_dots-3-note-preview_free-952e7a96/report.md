Run mode: **api**. Backend: `openrouter`. Model configured: `dots-studio/dots-3-note-preview:free`. Seed 42; 3 users; 50 candidates; 1000 catalog movies.

| Ranker (20-event window) | Users | MRR | Hit@10 | Popular share of top-10 |
|---|---:|---:|---:|---:|
| baseline | 3 | 0.0670 | 0.0000 | 13.3% |
| mini-genrec | 3 | 0.0358 | 0.0000 | 10.0% |

| History window | Mean retained high-signal events | Approx. input tokens | Baseline MRR | mini-GenRec MRR |
|---:|---:|---:|---:|---:|
| 5 | 5.0 | 2140 | 0.4333 | 0.0340 |
| 10 | 8.3 | 2247 | 0.1414 | 0.0288 |
| 20 | 14.7 | 2436 | 0.0670 | 0.0358 |
| 50 | 30.7 | 2910 | 0.1361 | 0.0326 |

![MRR versus approximate input tokens](elbow.png)

Elbow: Smallest history within 0.01 absolute MRR of the best observed point; exploratory.

### What broke first

- Data coverage: 37 users qualified after the temporal/catalog/history filters; this run sampled 3. 1 held-out ratings were below 4 stars. MovieLens ratings are an imperfect proxy for next watches or satisfaction.
- Baseline misses: 3/3 targets missed the top 10 at 20 events. Its popular-item top-10 share was 13.3%.
- Context at 50 events: 32 neutral ratings omitted; 0 high-signal events dropped for the token budget; 0 repeat events compressed. Counts are across evaluated users. MovieLens normally has one rating per user/movie.
- mini-genrec, 5 events: 146 raw suggestions; out-of-catalog rate 4.8%; outside-candidate rate 4.8%; 0 duplicates; 0/3 failed responses; 0 truncated responses; top-10 slot coverage 100.0%.
- mini-genrec, 10 events: 164 raw suggestions; out-of-catalog rate 9.8%; outside-candidate rate 18.3%; 0 duplicates; 0/3 failed responses; 0 truncated responses; top-10 slot coverage 100.0%.
- mini-genrec, 20 events: 169 raw suggestions; out-of-catalog rate 7.7%; outside-candidate rate 19.5%; 0 duplicates; 0/3 failed responses; 0 truncated responses; top-10 slot coverage 100.0%.
- mini-genrec, 50 events: 165 raw suggestions; out-of-catalog rate 9.1%; outside-candidate rate 18.2%; 0 duplicates; 0/3 failed responses; 0 truncated responses; top-10 slot coverage 100.0%.
- unconstrained, 20 events: 30 raw suggestions; out-of-catalog rate 23.3%; outside-candidate rate not run / unavailable; 0 duplicates; 0/3 failed responses; 0 truncated responses; top-10 slot coverage 76.7%.

Measured artifacts: [metrics.csv](metrics.csv), [elbow.csv](elbow.csv), [run.json](run.json), [per_user.json](per_user.json), [cases.json](cases.json), [prompts.json](prompts.json).
