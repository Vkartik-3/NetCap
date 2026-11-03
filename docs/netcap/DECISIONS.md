# NetCap decisions

Every choice that the brief or the user left open, with the reason. Newest phase last. Anything
dropped or deferred is also listed under "Not done" in `docs/netcap/RESULTS.md`.

## Phase 1 follow-ups

| # | Decision | Reason |
|---|---|---|
| 1.1 | MAWI decode and aggregation rewritten to work in chunks: the IPFIX stream is read in 64 MB blocks and decoded about 4M records at a time; totals, protocol mix and throughput are added chunk by chunk; top talkers are grouped by DuckDB (3 GB memory limit, spills to disk) from temporary Parquet files that are deleted afterwards. | Exact top-N needs every flow; a disk-spilling group-by is the simplest exact method with bounded memory. A test checks the chunked result equals the in-memory one. |
| 1.2 | The pcap was downloaded again (sha256 identical to Phase 1) and converted again, because the IPFIX stream had been deleted. | The re-measurement must cover the whole pipeline. |
| 1.3 | Phase 1's first MAWI result files are kept, marked superseded; only `phase1_mawi_202510011400_chunked*.json` numbers are used for memory and throughput. | Never hide a run; user rule: only re-measured numbers may be used. |
| 1.4 | Swapping is measured with `vm_stat` swap-outs before and after the run. They are system-wide (all processes), so 0 proves no swapping; a non-zero value is only an upper bound. | macOS has no per-process swap counter. |
| 1.5 | Disk cleanup deletes only regenerable things (extracted CSVs whose tarballs are kept, the GÉANT extract, the Phase 0 re-run scratch copy, tool source trees, the 6.7 GB benchmark tables). Each target records its rebuild command (`results/netcap/phase1_cleanup.json`). The user-wide pip cache (5.7 GB) was not touched. | Free space without losing anything that cannot be rebuilt; the pip cache is not this project's. |
| 1.6 | Repository history begins with one complete NetCap project commit. | Keep the project history concise. |
| 1.7 | Detailed forecasting instructions live in `research/README.md`; the root README is the platform overview. | Keep setup details separate from the product overview. |

## Phase 2 frozen specifications (written before any Phase 2 result)

Data and folds
- Abilene samples are the forecasting module's: window 12, targets per segment (never across a gap), horizons
  1, 6, 12 steps (5, 30, 60 min). Folds are the forecasting module's four: original (60/20/20 chronological) and
  A, B, C from `scripts/10_robustness.py` (`FOLDS`). Equality is tested against `common.py`.
- CESNET: 1-hour institution series (283 institutions, 6,717 hours, `n_bytes`), horizons 1 and 6 hours,
  the same four fold fractions. A cell counts only when its target, last value and window are known.
- Errors in Mbit/s. MAE, RMSE, sMAPE (2|e|/(|y|+|ŷ|), 0 when both are 0) over known cells.

Methods (all forecasts clipped at 0)
- persistence; same time yesterday; same time last week; mean of the same time on the last 7 days
  (missing lookups fall back to the last value).
- blend = last + w (7-day mean − last), one w per fold and horizon from 0, 0.01, …, 1 on validation.
- ETS(A,N,A): additive error, no trend, daily additive season; (alpha, gamma) per flow from
  {0.01, 0.03, 0.1, 0.3, 0.6} × {0, 0.01, 0.05, 0.15}, chosen per flow and horizon on validation MAE;
  state starts from the first week; unknown steps do not update the state.
- SARIMA-type: a seasonal ARIMA with period 288 (Abilene) is not practical in statsmodels, so the
  seasonal ARIMA is replaced by **dynamic harmonic regression**: ARMA(2,1) errors plus a constant and
  Fourier terms (3 daily and 2 weekly harmonics), fitted by maximum likelihood on the training rows of
  each fold, order fixed in advance (not tuned). Exact h-step Kalman forecasts with parameters held fixed.
  Run on the 10 busiest flows (by training mean) only, for cost; those flows get their own comparison
  table with every other method on the same cells.
- ridge (Abilene only): multi-output linear model on the last 12 values of all flows, penalty from
  10^-2 … 10^4 (× n) on validation (forecasting module Phase 8 estimator). Not run on CESNET: 283 × 12 = 3,396
  inputs against about 4,000 training samples in the original fold.
- gradient boosting (the one GBM): scikit-learn `HistGradientBoostingRegressor`, pooled over all flows,
  absolute-error loss, target log1p(y) − log1p(last); features: log1p(last), the other 11 window values,
  same time yesterday / last week / 7-day mean at the target and the 7-day mean at the origin (all
  relative to the last value), time of day and day of week of the target, flow id (categorical).
  Settings fixed in advance, not tuned: 300 iterations, learning rate 0.1, 63 leaves, 200 samples per
  leaf, L2 1.0, no early stopping, 400,000 random training rows, seed 42.

Burst attempts (flow 86, LOSA → CHIN; at least three, each reported whatever the result)
1. fallback rule on the blend: persistence for a flow and step when |log1p(7-day mean) − log1p(last)|
   exceeds k training standard deviations of that flow; k from {3, 4, 5, 6, 8} on validation, ties within
   0.05% go to the smaller k (forecasting module Phase 10 rule).
2. the same rule on the GBM; 3. the same rule on ETS.
4. ridge with inputs clipped to each flow's training maximum (the forecasting module found linear models
   extrapolate on out-of-range inputs).
5. smooth gate on the blend: weight w / (1 + (z/tau)^2), tau from {1, 2, 3, 5, 8} and w from a 0.05 grid,
   both on validation.

Uncertainty and significance
- Split-conformal 90% intervals: per flow, margin = the ceil((n+1)·0.9)-th smallest validation |error|;
  test coverage (two-sided and upper) and mean width reported for every method.
- Diebold-Mariano (Newey-West, one-day lag) and a 95% moving-block bootstrap with one-day blocks on the
  per-step MAE, against persistence and (GBM) against the best simple baseline chosen on validation.
- "Selected on validation": the method with the lowest validation MAE; its test result is the honest
  headline, whichever method it is.

## Neural retrain rules (written before the timing run)

- Scope (user decision): Abilene only, 60-minute horizon, original fold, 3 seeds (42, 43, 44); the base
  CNN-LSTM (residual, original-units MAE loss) and the seasonal CNN-LSTM with the fallback rule, built,
  trained and scored with the forecasting module's `common.py` unchanged (`prepare_abilene`, `context_dataset`,
  `build_cnn_lstm`, `make_loss("mae_raw")`, `fit_or_load`, `predict`) and the settings of
  `scripts/07_features.py` (200 epochs max, patience 15, learning rate 1e-3, batch 128). 6 models.
- Timing gate: train one model for one epoch on this machine's CPU. Projection = seconds per epoch x
  200 (the maximum; early stopping epochs are unknown in advance) x 6 models. If the projection is
  above 3 hours, the retrain is skipped and recorded as "not run: needs GPU".
- "Reproduced" is claimed only if the forecasting module's published test gain (section 21.3: base +1.1%,
  seasonal + rule +10.8% vs persistence) lies within my 3 per-seed test gains' mean +/- 2 standard
  deviations; otherwise the result is reported as "not reproduced" with both numbers.
- TensorFlow lives only in a separate environment (`.venv-tf`); forecasts are exported to Parquet and the
  API reads only those files.

## Phase 2 decisions made after seeing results (labelled wherever they appear)

| # | Decision | Reason |
|---|---|---|
| 2.1 | The frozen ridge (level in raw units) is kept and reported as it came out. | Rule: report results as they are. |
| 2.2 | A second ridge, `ridge_basedef`, was ADDED after the first Phase 2 run: the forecasting module's own definition (predict the change y − last; scaled or raw version and penalty chosen on validation). It is labelled "added after seeing results" in the results file. | My frozen ridge used a different target than the forecasting module, so it was not a like-for-like baseline; adding the base definition (chosen on validation only) makes the comparison fair without replacing anything. |
| 2.3 | The GBM's large test gain (+18% at 60 min, first run) was checked for look-ahead before being used: a test rewrites every value after a target's origin and requires identical features (`test_gbm_features_use_nothing_after_the_origin`). It passed. | A result much better than the forecasting module's best model must be checked before it is believed. |
| 2.4 | CESNET has 283 institutions, more than scikit-learn's 255-category limit, so the first CESNET run crashed at the GBM. For panels with more than 255 series the series id is replaced by a numeric feature: the series' mean log1p value over its own training targets. Abilene (144 flows) keeps the categorical id. Decided after the crash and before any CESNET result existed. | Structural limit of the library; the replacement uses training data only. |
| 2.5 | The frozen SARIMA-type model (ARMA(2,1) errors, no differencing) was worse than persistence on the 10 busiest flows in every fold. Cause found: it reverts to the training level, and two flows (LOSA→CHIN, CHIN→LOSA) shifted level between training and test (test bias −95 and +81 Mbit/s at 60 min). Fix attempted, ADDED as `arima_dhr_sel` and labelled: order chosen per flow and horizon on validation from (2,0,1), (1,1,1), (0,1,1) (the forecasting module also chose ARIMA orders on validation, including differenced ones). The frozen result is kept and reported. | A legitimate fix for a diagnosed cause, chosen on validation only. |

## Phase 4 decisions

| # | Decision | Reason |
|---|---|---|
| 4.1 | STL uses `seasonal=13` (statsmodels default 7). In a unit test on synthetic data, robust STL with the default window put single-step spikes into the seasonal component (residual 0.08 for a +3.7 log spike); with 13 or more they stay in the residual. Chosen before any evaluation run. | The detector must see spikes in the residual. |
| 4.2 | STL decompositions run in 6 parallel processes. One decomposition of a 47,520-step Abilene series took 43 s while sharing the CPU, which projected to about 3.5 hours for the two Abilene cases (over the 3-hour compute limit). Each series is decomposed independently, so the result is identical. | Stay under the compute limit without changing the method. |

## Superseded result files (kept, never overwritten)

| File | Superseded by | Reason |
|---|---|---|
| `phase1_mawi_202510011400.json`, `phase1_mawi_202510011400_rerun.json` | `phase1_mawi_202510011400_chunked_lowmem*.json` | whole trace held in memory (16.8 GB peak); memory and timing numbers not used |
| `phase1_mawi_202510011400_chunked*.json` (attempt 1) | `..._chunked_lowmem*.json` | system-wide swap-outs during the run (92,796) while an idle control had 0 |
| `phase2_forecast_abilene_quickcheck_primary_h12.json` | `phase2_forecast_abilene.json` | one fold and horizon, no ARIMA: a functional check before the full run |
| `phase2_forecast_abilene_run1_frozen_arima_only.json` | `phase2_forecast_abilene.json` | before the post-hoc `arima_dhr_sel` variant (2.5) existed; every other number is identical in both (the run is seeded and deterministic; checked) |

## Timing labels and the neural retrain (user decision, 2026-11-02)

- Phase 2 and Phase 4 ran at the same time (load averages of 16-42 were observed), so their timings are labelled
  "upper bound, measured under CPU contention" (`timing_label` added to the result files by
  `netcap.ops.label_timings`). Throughput claims use only idle benchmarks (`phase6_idle_*.json`).
  From now on every result file records the load average when it is saved (`cpu_at_save`).
- The neural retrain and its timing gate were skipped by user decision: "not run: needs GPU (no NVIDIA GPU on
  this machine)" (`phase_neural_retrain.json`, with the command to run it on a GPU). NetCap makes no claim from
  the forecasting module's neural results.

## Phase 3 decisions

| # | Decision | Reason |
|---|---|---|
| 3.1 | The upgrade rule fixed in advance ("the link that is worst in the most N-1 cases": KSCY–IPLS) did not change the worst failure at all, because that overload is on ATLA→IPLS (2.48 Gbit/s). A second rule was ADDED and labelled: upgrade the most loaded link of the worst N-1 case to 9.92 Gbit/s (the capacity of the other core links; ASSUMED). The first result is kept (`phase3_scenarios_run1_single_upgrade_rule.json`, superseded only in that it lacks the second rule). | The first rule is a real negative result; the second answers the planning question directly. |
| 3.2 | TE confidence interval: the LP is solved at every hour of the measured data (3,405 steps, normal and under the worst failure); 95% interval for the mean and median LP/OSPF ratio from a day-block bootstrap (2,000 resamples). | User condition 3: the LP with confidence intervals. |

## Phase 6 (after Docker was started, user permission)
- `.dockerignore` added so the build context excludes data, raw files and virtual environments.
- Provenance no longer needs `git` (the image has none): it records `NETCAP_COMMIT` or "unknown (git not available)".
- `POST /api/te` takes a JSON body model (it rejected `{}` with 422); test extended.
- `phase6_quality_run1_no_docker.json` is the quality run before Docker was available (superseded for the container status only).

## Neural retrain on a GPU (2026-11-03, user decision)
- Run on a rented NVIDIA A100 (Thunder Compute) instead of this machine's CPU; the timing gate was measured on that GPU
  (8.17 s per epoch, projection 2.72 h, under the 3-hour limit). Scope, settings and the "reproduced" rule unchanged.
- Fixes needed on the host, made before any metric was seen: (1) `LD_LIBRARY_PATH` set to pip's CUDA libraries, because
  TensorFlow 2.21 did not load `libcusolver.so.11`; (2) the export alignment check gained `atol=1e-6` because 16 true zeros
  come back from `common.py`'s scaler as 5.5e-7 (raw units) and failed `rtol` alone; (3) `save_result` now runs before the
  export, because the first full run computed the metrics and lost them when the export check raised. The 6 trained models
  were kept and reloaded for the re-run (no retraining).
- The run itself is labelled "upper bound, measured under CPU contention" by `cpu_at_save` (shared host); no CPU timing is
  claimed from it.

## Warning-F1 interval (2026-11-03, added after seeing results)
- Paired day-block bootstrap of warning F1 (point forecasts, 60 min, original test period): 2,000 resamples of
  one-day blocks (288 rows), seed 42, the same rows for every method; F1 recomputed from the summed counts of each
  resample; difference = method − persistence; 95% percentile interval.
- Changed after seeing the first output: a resample with no actual events and no warnings now has F1 **undefined**
  (dropped), not 0, because 0/0 was being scored as 0 (14 of 2,000 resamples). Shares "better / worse than
  persistence" are over defined resamples only (a unit test caught them counting undefined ones as ties).
