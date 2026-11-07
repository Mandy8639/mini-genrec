# Model comparison

Same users, candidates and seed for every model. Request/parse failures count as misses; inspect errors and coverage before comparing quality.

| Model | Ranker | History | Users | MRR | Hit@10 | Popular share | Slot coverage | Errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| nvidia/nemotron-3.5-super-vl:free | baseline | 20 | 3 | 0.0670 | 0.0000 | 13.3% | 100.0% | 0 |
| nvidia/nemotron-3.5-super-vl:free | mini-genrec | 20 | 3 | 0.1175 | 0.3333 | 6.7% | 100.0% | 0 |
| nvidia/nemotron-3.5-super-vl:free | unconstrained | 20 | 3 | n/a | n/a | 28.0% | 83.3% | 0 |
| dots-studio/dots-3-note-preview:free | baseline | 20 | 3 | 0.0670 | 0.0000 | 13.3% | 100.0% | 0 |
| dots-studio/dots-3-note-preview:free | mini-genrec | 20 | 3 | 0.0358 | 0.0000 | 10.0% | 100.0% | 0 |
| dots-studio/dots-3-note-preview:free | unconstrained | 20 | 3 | n/a | n/a | 34.8% | 76.7% | 0 |
