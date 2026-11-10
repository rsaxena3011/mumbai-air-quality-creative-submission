# Mumbai Air Quality Challenge — Most Creative Submission

**Google Mumbai Data Challenge: Air Quality Mapping and Forecasting (Zindi)**

By Ratnesh Saxena — Zindi username: **saxenawave** (solo entry)

*An analysis and storytelling submission: what Mumbai's air-quality data revealed about itself when we forecast it the way a real forecaster has to — using only the data that had actually been published at the moment each decision was made.*

---

## The idea in one paragraph

Availability-aware PM2.5 forecasting for Mumbai: every model input is rebuilt at each decision time from only the rows that had actually been published by then (`available_at <= decision`), so the system never trains on data a real forecaster could not have seen. The entry pairs a LightGBM level model with persistence and climatology anchors under horizon-band weights, validated on forward-chaining folds plus two private-like windows. Along the way the data itself told stories: stations report with multi-day delays (a 17-row morning on 9 November arrived only in the next drop), the 7–8 November ground truth landed a day late, and an extreme pollution episode on 9 November defeated every anchor equally — findings documented in a full data audit and experiment log.

## 1. The dataset has a memory problem — and that became the method

The challenge data covers four Mumbai stations — Borivali East, Malad West, Byculla and Shivaji Nagar — with 47,546 observed station-hours spanning 19 February 2025 to 31 October 2026, and a test set of 2,592 hourly predictions (4 stations × 648 hours) to produce.

Our data audit (3 November 2026) found the fact that shaped everything else: **every observation in the dataset was published late.** The lag between an observation's timestamp and its `available_at` publication time, across all 47,546 rows:

| Publication lag (available_at − timestamp) | Hours |
|---|---|
| Minimum | 39.3 h |
| Median | 84.0 h (3.5 days) |
| 95th percentile | 112.0 h |
| Maximum | 381.6 h (~16 days) |
| Rows published in under 39 hours | **0** |

Nothing in this dataset was knowable less than ~39 hours after the hour it describes — and the freshest month was the stalest: October 2026 rows carried a median lag of 102.1 hours. A conventional walk-forward validation, which quietly assumes an observation is usable one hour after it occurs, therefore prices a 1–48 hour forecasting horizon that the real decision never faces. At the real submission decision, the scored hours sit far beyond the freshest usable observation.

So we rebuilt the validation around one rule:

> **At a decision time T, a history row is in the information set if and only if `available_at <= T`.**

No modelled lag offset, no synthetic delay — the recorded publication times are used as-is, because the lag structure is batch-driven: retrieval batches land on discrete days, so the freshest visible edge at any cutoff is set by the last batch before it, not by any per-row lag quantile.

## 2. What honest timing does to the leaderboard illusion

We ran the same three statistical methods under three information regimes: the original folds on raw history, the original folds on cleaned history, and the availability-aware folds. Pooled RMSE over the same 615 scored rows:

| Method | Old folds, raw | Old folds, cleaned | Availability-aware | Timing effect |
|---|---|---|---|---|
| Persistence | 15.002 | 15.002 | **23.486** | +56.5% |
| Climatology | 60.898 | 23.862 | **24.127** | +1.1% |
| 50/50 blend | 32.554 | 13.800 | **14.153** | +2.6% |

Two findings matter:

- **Persistence collapses under honest timing.** Its pooled margin over climatology shrank from 8.86 RMSE to 0.64 — effectively a tie. The old folds flatter persistence precisely where the real decision punishes it.
- **On the private-like window, the ranking outright flips.** In a fold built to mirror the real decision geometry (decision 28 October, scored window 29–31 October), persistence was the *worst* method (31.229) and cleaned climatology the best by a wide margin (12.605), with the blend second (16.226).

Measured, not asserted: the effective horizons the availability-aware folds actually priced ran from **96 to 239 hours** beyond each station's freshest visible observation — against the nominal 1–48 hours the original design claimed. Under our pre-registered materiality test, the pivot was declared **MATERIAL**, and every later anchor weight in this entry was read off availability-aware folds, never the old ones.

## 3. Stories the data told while we listened

**The five-day hole.** There are zero observations for 24–28 October 2026 — a five-day silence sitting exactly where naive validation folds would live. Two of the original six folds are simply empty. Any validation design that does not reckon with this hole is grading itself on days that never happened.

**Sensor garbage wearing a lab coat.** Nineteen rows exceed 1,000 µg/m³ (maximum 241,587.6 µg/m³; one value is literally 99,999.01), and every one carries the dataset's `flag_high_1000` marker. Left in, those 19 rows single-handedly poison long-run averages: raw climatology scored a pooled RMSE of 60.898 — worse than the public benchmark of 56.66 — while the same climatology with the flagged rows removed scored 23.862. Cleaning was not hygiene; it was the difference between losing to the benchmark and beating simple methods.

**Flatlines that are outages, not clean air.** Zero readings in this dataset are sometimes recorded sensor outages, not pristine air: Shivaji Nagar has a 313-row run of zeros spanning 19 April – 3 May 2025, and Malad West a 62-row run on 27–30 May 2026. We retained them, inventoried them, and refused to pretend them away — a real forecaster's next data drop can contain the same artefact at the fresh edge.

**Ground truth that arrives late — and an episode that beat everyone.** In the final data drops, truth kept landing after the fact: the 7–8 November ground truth arrived a day late, and the extreme pollution episode of the morning of 9 November surfaced first as just 17 rows in one drop, completed only by the next. Scored against that 17-row morning, every member of the ensemble missed by roughly the same amount — LightGBM 173.16, persistence 172.81, climatology 184.78 RMSE — an extreme observed episode that no anchor could anticipate. When the final drop completed the episode (169 rows), it still dominated the fresh validation gate (fold RMSE 115.388). We report this plainly: in volatile episodes, this system degrades, and it degrades honestly — our control experiments (below) prove the miss is a property of the data, not a defect of the pipeline.

## 4. The method

- **Information discipline first.** Every feature and every training example is built from the knowable set at its decision time (`available_at <= decision`). The target column's duplicate (`mean`, byte-identical to `pm25` on all 47,546 rows) is allowlisted out of every feature path — as a feature it would be pure leakage.
- **Model.** A LightGBM level model, blended with persistence and climatology anchors under horizon-band weights — the anchors' composition read off the availability-aware folds, where the blend was the best pooled method (14.153) and climatology ruled the private-like window.
- **Validation.** Forward-chaining folds plus two private-like windows that mirror the real decision's geometry, with a pre-registered falsification rule for the availability pivot (declared material above, in writing, before the result was known).
- **Rehearsal discipline.** The pipeline was rehearsed end-to-end on the 5 November and 9 November data drops ("Update Day Mirror": score the standing model against newly unmasked ground truth before any retraining), so the 10 November final run was a procedure, not an improvisation.
- **Fidelity controls.** On the final run, the pipeline first had to reproduce two frozen control windows exactly before it was allowed to build anything. It did: the private-like control ensembles recomputed to 10.4526 and 6.4012, matching the frozen values at full float precision, every member matching. That is how we know the 9 November miss is the atmosphere's fault, not the code's.

## 5. Results, stated with their dates

Public scores on Zindi move as the organisers' data updates re-score the board, so each figure below carries the window and date it belongs to:

| What | Figure | Context |
|---|---|---|
| Banked submission (ID zwp2trdk) | **8.153** RMSE | Public score on the original 27–28 October public window, at banking |
| Same banked submission, re-scored | **109.7724809** | After Zindi's final data update re-scored the public board |
| Finals partner (ID 3qFChYyT) | **108.4434336** | Nov 9-history re-based candidate, selected at the close |
| Phase-4 final-data candidate (ID dMSSKsYA) | **108.4434336** | Built from the complete final Train.csv on 10 Nov; tied the partner exactly, so the earlier selection stood |

**Finals pair selected at the 10 November close: zwp2trdk + 3qFChYyT** — verified on the submissions page, twice, hours before the 23:59:59 IST deadline. Three submissions were made in total. The private leaderboard, which decides the main prizes, is scored on the 11–12 November window; whatever it returns, the validation story above is the part of this entry we can already stand behind.

## 6. Why this is the creative submission

The prediction task asked: *what will Mumbai's PM2.5 be?* This submission asks the question underneath it: *what could anyone honestly have known, and when?* The creative act here is a change of tense — forecasting evaluated in the tense of publication, not the tense of the clock. It turned a lag column most pipelines ignore into the organising principle of the whole entry, and it produced findings that outlive this leaderboard: report delays are batch-shaped, not smooth; a five-day hole can hollow out a validation design; 19 garbage rows can flip a method comparison; and an honest system should be able to prove, with frozen controls, that its worst miss was the weather.

---

*Entry by Ratnesh Saxena (Zindi: saxenawave) for the "Most Creative Submission" side prize, Google Mumbai Data Challenge: Air Quality Mapping and Forecasting. All figures are taken from the entry's own data audit (3 Nov 2026), availability-aware validation record (3 Nov 2026), experiment diagnoses (8 Nov 2026), and final re-base and finals records (10 Nov 2026). No figure is estimated or extrapolated.*
