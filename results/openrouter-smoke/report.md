Run mode: **api**. Backend: `openrouter`. Model configured: `liquid/lfm-2.5-2.6b:free`. Seed 42; 1 users; 50 candidates; 1000 catalog movies.

| Ranker (20-event window) | Users | MRR | Hit@10 | Popular share of top-10 |
|---|---:|---:|---:|---:|
| baseline | 1 | 0.0833 | 0.0000 | 10.0% |
| mini-genrec | 1 | 0.0000 | 0.0000 | not run / unavailable |

| History window | Mean retained high-signal events | Approx. input tokens | Baseline MRR | mini-GenRec MRR |
|---:|---:|---:|---:|---:|
| 5 | 5.0 | 2150 | 0.1000 | 0.0000 |
| 10 | 6.0 | 2177 | 0.0909 | 0.0000 |
| 20 | 9.0 | 2272 | 0.0833 | 0.0000 |
| 50 | 28.0 | 2833 | 0.1250 | 0.0000 |

![MRR versus approximate input tokens](elbow.png)

Elbow: Requires all four measured LLM conditions without errors.

### What broke first

- Data coverage: 37 users qualified after the temporal/catalog/history filters; this run sampled 1. 0 held-out ratings were below 4 stars. MovieLens ratings are an imperfect proxy for next watches or satisfaction.
- Baseline misses: 1/1 targets missed the top 10 at 20 events. Its popular-item top-10 share was 10.0%.
- Context at 50 events: 22 neutral ratings omitted; 0 high-signal events dropped for the token budget; 0 repeat events compressed. Counts are across evaluated users. MovieLens normally has one rating per user/movie.
- mini-genrec, 5 events: 0 raw suggestions; out-of-catalog rate not run / unavailable; outside-candidate rate not run / unavailable; 0 duplicates; 1/1 failed responses; 1 truncated responses; top-10 slot coverage 0.0%.
- mini-genrec, 10 events: 0 raw suggestions; out-of-catalog rate not run / unavailable; outside-candidate rate not run / unavailable; 0 duplicates; 1/1 failed responses; 1 truncated responses; top-10 slot coverage 0.0%.
- mini-genrec, 20 events: 0 raw suggestions; out-of-catalog rate not run / unavailable; outside-candidate rate not run / unavailable; 0 duplicates; 1/1 failed responses; 1 truncated responses; top-10 slot coverage 0.0%.
- mini-genrec, 50 events: 0 raw suggestions; out-of-catalog rate not run / unavailable; outside-candidate rate not run / unavailable; 0 duplicates; 1/1 failed responses; 1 truncated responses; top-10 slot coverage 0.0%.
- unconstrained, 20 events: 10 raw suggestions; out-of-catalog rate 30.0%; outside-candidate rate not run / unavailable; 0 duplicates; 0/1 failed responses; 0 truncated responses; top-10 slot coverage 70.0%.

Measured artifacts: [metrics.csv](metrics.csv), [elbow.csv](elbow.csv), [run.json](run.json), [per_user.json](per_user.json), [cases.json](cases.json), [prompts.json](prompts.json).
