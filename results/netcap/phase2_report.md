# Phase 2 report: forecasting

Sources: `phase2_forecast_abilene.json`, `phase2_forecast_cesnet_1h.json` (specification frozen in
`docs/netcap/DECISIONS.md` before the first run; post-hoc additions labelled there, 2.1-2.5).
Timings in those files are **upper bounds, measured under CPU contention** (Phase 2 and 4 ran together);
idle throughput is in `phase6_idle_forecast_throughput.json`.

## What was built
`netcap/forecast/`: panel and folds (the forecasting module's four rolling-origin folds, tested equal to `common.py`),
persistence, same time yesterday / last week, 7-day mean, blend, ETS(A,N,A), SARIMA-type (ARMA errors + Fourier
seasonality, 10 busiest flows), ridge (frozen definition + the forecasting module's definition, labelled post hoc), one
gradient-boosting model, burst attempts, split-conformal 90% intervals, Diebold-Mariano and day-block bootstrap.
TF-free (tested in a subprocess). Test forecasts with intervals in `data_netcap/forecasts/`.

## Checks against the forecasting module (same data, same folds)
- Persistence, Abilene 5 min: val 1,028,174.22 and test 1,204,681.67 raw units, equal to `results/metrics.csv`
  within float32 precision (test `test_persistence_equals_project_logged_metrics`).
- Blend test change vs persistence at 5/30/60 min: −2.3 / −2.8 / −1.7%, the forecasting module's section 19.1 numbers exactly.
- Base-definition ridge: −4.5 / −12.9 / −13.3%, the forecasting module's numbers exactly.

## Results: Abilene (test MAE change vs persistence; method chosen on validation)
| Fold | 5 min | 30 min | 60 min |
|---|---|---|---|
| original | gbm +11.9% [+11.1, +12.7] | gbm +15.9% [+14.2, +17.3] | gbm +18.0% [+14.6, +20.3] |
| A | gbm_rule +9.4% | gbm +14.3% [+12.1, +16.2] | gbm +16.9% [+15.0, +18.6] |
| B | gbm_rule +13.8% | gbm_rule +16.5% | gbm +18.4% [+16.9, +19.6] |
| C | gbm +13.5% [+11.9, +15.0] | gbm +18.4% [+17.2, +19.7] | gbm +20.5% [+18.5, +22.6] |

Brackets: 95% day-block bootstrap; every Diebold-Mariano p < 1e-20. The GBM also beats the best simple baseline
chosen on validation (the blend) in all 12 cases, by +8.4% to +19.3% (each interval excludes 0).
Persistence test MAE (original): 3.212 / 5.300 / 6.157 Mbit/s. The GBM was checked for look-ahead before use
(test rewrites every value after the origin; features unchanged).

Other methods (original fold, 5/30/60 min): ETS +2.6 / −0.6 / +0.6%; seasonal naive and 7-day mean −53% to −232%;
frozen ridge −80 / −109 / −98%.

## Burst attempts (flow 86, LOSA→CHIN; original fold, test MAE in Mbit/s, persistence 27.0 / 62.5 / 89.9)
| Attempt | 5 min | 30 min | 60 min | all flows, 60 min vs persistence |
|---|---|---|---|---|
| blend (no fix) | 45.9 | 123.6 | 167.5 | −1.7% |
| 1. fallback rule on blend (k on validation) | 31.6 | 96.6 | 130.4 | +2.5% |
| 5. smooth gate on blend | 32.2 | 78.8 | 120.7 | +3.3% |
| 3. fallback rule on ETS | 34.4 | 124.5 | 137.7 | +0.6% |
| 4. ridge with clipped inputs | 85.7 | 182.5 | 210.0 | −87.1% |
| GBM (no fix needed) / 2. rule on GBM | 27.2 / 27.2 | 62.1 / 62.1 | 83.0 / 83.0 | +18.0% |
The rules help the blend on flow 86 but do not bring it to persistence; clipping does not rescue the linear
model. The GBM (log-ratio target, trees cannot extrapolate) is already at or below persistence on flow 86,
so the rule changes almost nothing for it (it switches 0.03-0.75% of forecasts).

## SARIMA-type model (10 busiest flows)
Frozen ARMA(2,1)+Fourier: worse than persistence in every fold and horizon (−12% to −123%); diagnosed (DECISIONS 2.5)
as reversion to the training level on two flows whose level shifted. Added variant with the order chosen on
validation: −13.2% to +5.3% (original fold +5.1 / +5.3 / +0.4%). The GBM is better than both on these flows.

## Intervals
Split-conformal 90% (per-flow margins from validation): test coverage 84.8-86.1% in the original fold for every method
(the test period is burstier than validation, as the forecasting module found), 89-90% in folds A-C (GBM, 60 min).

## CESNET (283 institutions, 1-hour bytes; method chosen on validation)
GBM or GBM + rule selected in all 8 cases: +16.6% to +20.3% at 1 hour, +44.0% to +50.2% at 6 hours over persistence
(persistence is weak at 6 hours on hourly data with a daily cycle; seasonal baselines also gain +25-38% there).
90% interval coverage 88-93%.

## Failures and open issues
- Frozen ridge and frozen SARIMA-type models failed (kept). Two post-hoc variants added, labelled.
- CESNET's 283 series exceeded scikit-learn's 255-category limit (crash; DECISIONS 2.4).
- Neural models: not run (needs GPU), by user decision; no neural claims.
