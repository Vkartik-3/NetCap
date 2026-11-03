# Phase 4 report: anomaly detection

Source: `phase4_anomaly.json` (timings are upper bounds, measured under CPU contention). Thresholds chosen on
the validation period (max F1), scored on test. INJECTED anomalies are synthetic, written into copies of real series.

## What was built
`netcap/anomaly/detect.py`: static threshold baseline (robust z vs training median/MAD), STL robust z (window 13,
retrospective), Isolation Forest on causal features; injection (spike ×3-8, level shift up ×1.8-3 / down ×0.15-0.5
for 1-6 h, dropout to 0 for 30 min-3 h; 300 attempts per period, kept apart by a day); weak labels from R2-R4.
Flagged test cells stored in Postgres `anomalies` (injected = true).

## Injected anomalies (test period)
| Case | Detector | Avg precision | F1 | Precision | Recall | Event recall: spike / shift up / shift down / dropout |
|---|---|---|---|---|---|---|
| Abilene (142 active flows, 262 events) | threshold | 0.159 | 0.220 | 0.692 | 0.131 | 0.00 / 0.00 / 0.01 / 0.72 |
| | STL | 0.165 | 0.206 | 0.539 | 0.127 | 0.00 / 0.00 / 0.00 / 0.72 |
| | Isolation Forest | 0.025 | 0.067 | 0.041 | 0.173 | 0.56 / 0.19 / 0.29 / 1.00 |
| CESNET (283 series, 286 events) | threshold | 0.186 | 0.268 | 0.537 | 0.178 | 0.00 / 0.01 / 0.03 / 0.90 |
| | STL | 0.141 | 0.202 | 0.503 | 0.126 | 0.02 / 0.04 / 0.05 / 0.94 |
| | Isolation Forest | 0.075 | 0.186 | 0.202 | 0.172 | 0.13 / 0.10 / 0.24 / 1.00 |
Base rates: 0.46% (Abilene) and 1.90% (CESNET) of test cells.

Reading: with one threshold per detector chosen for max F1, the threshold is set by dropouts (which score very high
in log space), so spikes and level shifts are almost never flagged; Isolation Forest catches more kinds but with
low precision. No detector is good at all four kinds. STL does not beat the simple threshold.

## Weak labels (raw Abilene, rules R2-R4)
65,095 R2 cells, 1,098 R3 cells, 0 R4 cells; almost all in the training period (val 0, test 1), so a validation
threshold cannot be chosen. Scored instead with a fixed budget set in advance (flag the top 1% of cells):
| Detector | Avg precision (all rows) | Recall R2 / R3 at top 1% | Precision vs weak labels |
|---|---|---|---|
| threshold | 0.142 | 0.119 / 0.384 | 0.117 |
| STL | 0.007 | 0.021 / 0.421 | 0.027 |
| Isolation Forest | 0.025 | 0.022 / 0.353 | 0.026 |
Base rate 0.95%. Day-long flow outages (R2) are absorbed into STL's trend and look normal to it; the static threshold
sees the zeros.

## Failures and attempts
- STL default window absorbed spikes (found by a unit test; window 13, DECISIONS 4.1).
- STL projected to about 3.5 h on two cases; parallelised (DECISIONS 4.2). First Phase 4 run stopped for a missing
  upsert key (bug fixed, tested).

## Open issues
- A per-kind threshold or a combined detector would likely do better; not tried (time).
