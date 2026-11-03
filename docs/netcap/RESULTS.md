# NetCap results

Every number here is copied from a file in `results/netcap/` (named in each section), produced by the command
recorded inside that file, with the code version, seed, date and data version. `results/netcap/summary.json`
collects the headline values with their source paths. Labels used throughout:
**ASSUMED** (an input, not a measurement), **INJECTED** (synthetic anomalies), **upper bound, measured under CPU
contention** (timings taken while other heavy jobs ran), **added after seeing results** (post-hoc, never replacing
the original).

## Status

| Phase | Status | Notes |
|---|---|---|
| 0 Audit | done | base scripts reproduced; no trained models published by the forecasting module |
| 1 Data layer | done | incl. all 8 follow-ups (bounded-memory MAWI, per-table rows, cleanup, commits, README, coverage, Docker plan) |
| 2 Forecasting | done | GBM included; SARIMA-type and frozen ridge failed (kept); 2 labelled post-hoc variants |
| 3 Scenarios + TE | done | gate passed 144/144; N-1 sweep; LP with bootstrap intervals |
| 4 Anomaly detection | done (weak results) | detectors catch dropouts, rarely spikes or shifts |
| Neural retrain | done (GPU, A100, 2026-11-03) | 3 of 3 reference numbers inside the 3-seed band; the seasonal match is weak (wide band) |
| 5 API + dashboard | done | FastAPI + static Plotly page; TF-free (tested) |
| 6 Engineering | done | CI runs on GitHub (10 of 10 `ci` runs succeeded); Compose and kind stacks run and checked end to end |
| 7 Results and docs | done | this file, summary.json, RESUME_FACTS.md, INTERVIEW_NOTES.md |

## Phase 1: data layer (`phase1_summary.json`, `phase1_report.md`)
- Abilene: 9,629,270 rows in 7 tables, of which **6,967,296 are measured** OD traffic (48,384 five-minute steps × 144
  flows; 5,883,120 in kept steps); 2,612,736 are derived link loads; the rest is reference data.
- CESNET-TimeSeries24 institutions: **10,446,225 rows** (10-minute), 1,880,182 (hourly), 78,667 (daily): 12,405,074
  total from 1,244,197,731 bytes of CSV.
- Every ingestion is idempotent: re-runs change 0 Postgres rows and 0 Parquet files (row checksums compared).
- Query latency (CESNET 10-minute table, 10,446,225 rows, warm cache, median; identical results across variants):
  one institution for one week: 186.4 ms full scan → 0.37 ms with a B-tree on (institution_id, ts) (about 500×);
  top-10 institutions for one day: 192.8 → 3.69 ms (52×); network hourly totals for one month: the index does not
  help (235.4 ms); monthly range partitions give 71.6 ms (3.2×). Idle re-measurement: see Phase 6.
- MAWI 2025-10-01 14:00 (real WIDE traffic): **207,302,561 IP packets reconciled exactly** across an independent pcap
  walk, softflowd's counters and the 56,523,852 decoded flow records; 0 export failures. Bounded-memory pipeline:
  1.85 GB peak footprint, 0 system-wide swap-outs, 415.8 s to process exit, 0.09 s teardown; pipeline peak RSS 1.41 GB vs 7.86 GB for the
  superseded whole-trace-in-memory version (`phase1_mawi_202510011400.json`).

## Phase 2: forecasting (`phase2_forecast_abilene.json`, `phase2_forecast_cesnet_1h.json`, `phase2_report.md`)
- Forecasting module reproduced on identical folds: persistence (= `metrics.csv`), blend −2.3 / −2.8 / −1.7% and the base
  ridge definition −4.5 / −12.9 / −13.3% at 5 / 30 / 60 min (the forecasting module's section 19.1 numbers exactly).
- **The gradient-boosting model (alone, or with the fallback rule in 3 cases) is chosen on validation in all 12 Abilene cases**; the GBM beats persistence on test by
  +8.8% to +20.5% (60 min: +18.0% [+14.6, +20.3] original fold; +16.9%, +18.4%, +20.5% in folds A, B, C; 95%
  day-block bootstrap; every Diebold-Mariano p < 1e-20). It also beats the best simple baseline chosen on validation
  in all 12 cases (+8.4% to +19.3%). Checked for look-ahead before use.
- For comparison, the forecasting module's best published model (seasonal CNN-LSTM + rule, GPU) was +8.7% to +13.1% at
  60 min over the same four folds (its section 23.1). That is the forecasting module's number, not reproduced by NetCap.
- Flow 86 (LOSA→CHIN bursts): fallback rule and smooth gate help the blend (60 min: 167.5 → 130.4 / 120.7 Mbit/s),
  not back to persistence (89.9); clipping does not rescue ridge; the GBM is at 83.0 without any fix.
- Negative: frozen ridge (−80% to −109%), frozen SARIMA-type model (−12% to −123% on the 10 busiest flows; the
  validation-chosen order, added after seeing results, gives −13% to +5%).
- 90% split-conformal intervals, original test period: 84.8-86.1% coverage for 10 of the 14 methods (GBM 85.3-85.9%);
  lower for ridge (74.8-80.4%), ridge_clip (75.7-81.2%), ridge_basedef (83.5-85.0%) and mean7 (84.5%). Folds A-C:
  GBM 89.3-90.3% at every horizon. The GPU-retrained CNN-LSTMs: 85.4-85.7% (`phase_neural_coverage.json`).
- CESNET (283 institutions, hourly): GBM chosen in all 8 cases, +16.6% to +20.3% at 1 hour and +44.0% to +50.2% at
  6 hours over persistence.

## Phase 3: scenarios and TE (`phase3_scenarios.json`, `phase3_report.md`)
- Gate: OSPF ECMP routing equals matrix A for **144/144 flows**.
- Measured demand: 40,855 steps; network total mean 3,281 Mbit/s, max 12,864 Mbit/s. Busiest link peak 75.4%,
  worst p95 9.3%.
- N-1: the IPLS–KSCY link failure overloads ATLA→IPLS (2.48 Gbit/s) to **297%**; every other link failure stays
  at or below 77%. Router failures strand 2.5% to 79% of traffic (their own flows).
- Upgrade rule fixed in advance: no effect (negative result). Rule added after seeing results (ATLA–IPLS to
  9.92 Gbit/s, ASSUMED): worst N-1 case 297% → **77%**.
- Upgrade search (added after seeing results; `upgrade_search`): every core link pair at 2x and 4x its capacity
  (ASSUMED sizes), 30 candidates, against no failure + all 27 N-1 link and router cases, ranked by worst-case reduction
  per Gbit/s added. **Only 2 of 30 candidates reduce the worst case at all**, both on ATLA–IPLS: it is the single
  binding link. Best per Gbit/s: ATLA–IPLS 2.48 → 4.96 Gbit/s (297% → 149%, 59.9 pp per Gbit/s); the greedy plan to
  get below 80% is ATLA–IPLS 2.48 → 9.92 Gbit/s (+7.44 Gbit/s per direction) → **77.3%**, the same answer as the added rule.
- Size-free: ATLA–IPLS carries up to 7.37 Gbit/s in its worst N-1 case, so it needs **at least 9.22 Gbit/s** to stay
  below 80%; above 9.54 Gbit/s it stops being the bottleneck and WASH→NYCM (77.3% when CHIN–IPLS fails) takes over.
- Peak-to-average sensitivity (`upgrade_search.peak_factor_sensitivity`; factors ASSUMED, loads scale linearly):

  | Peak factor | No-failure max | Worst N-1 now | After ATLA–IPLS → 9.92 | Cases ≥ 80% after | ATLA–IPLS for < 80% | Greedy plan |
  |---|---|---|---|---|---|---|
  | 1.0 | 75.4% | 297% | 77.3% | 0 / 28 | 9.22 Gbit/s | 2 upgrades, +7.44 Gbit/s |
  | 1.2 | 90.4% | 357% | 92.8% | 26 / 28 | 11.06 Gbit/s | 14 upgrades, +126.48 Gbit/s |
  | 1.5 | 113.0% | 446% | 116.0% | 26 / 28 | 13.83 Gbit/s | 14 upgrades, +126.48 Gbit/s |
  | 2.0 | 150.7% | 595% | 154.6% | 26 / 28 | 18.44 Gbit/s | 14 upgrades, +126.48 Gbit/s |

  The chosen upgrade holds only up to a peak factor of **1.035** (0.8 / 0.773). Beyond that the network is uniformly
  tight under failures (26 of 28 cases over 80% at 1.2), so the plan becomes doubling most core links. Plans at
  1.2-2.0 coincide because candidate sizes are coarse (2x / 4x); the greedy objective is total overload above 80% per
  Gbit/s, because a max-only greedy stalled on a tie between mirror links (IPLS→CHIN and WASH→NYCM, both 92.8% at 1.2).
- TE bound (LP / OSPF max utilisation), hourly over all data: mean 0.704 [0.696, 0.712] normally, 0.493 [0.483,
  0.504] under the worst failure (95% day-block bootstrap); 0.51 on the 20 busiest steps.
- Better forecasts, not better warnings: the GBM has 14% lower link-load error than persistence at 60 min but a
  warning F1 of 0.890 vs 0.896: difference **−0.007, 95% CI [−0.114, +0.081]** (paired day-block bootstrap, 2,000
  resamples; GBM better in 28% of resamples, worse in 57%). No method's F1 difference from persistence excludes 0,
  including the best point values (blend_gate +0.009 [−0.001, +0.039]; base CNN-LSTM +0.011 [−0.000, +0.046]).
- Why the intervals are wide: the test period has 1,091 warning events (link-steps at or above 50%) on only **3 of 29
  days**, and **95% of them on one day**. Warning F1 here is essentially one congestion episode, so the data cannot rank
  methods by warning quality. Even methods that never warn (F1 0) have an interval reaching 0, because resamples
  without that day leave events that persistence misses too.

## Phase 4: anomaly detection (`phase4_anomaly.json`, `phase4_report.md`)
- INJECTED anomalies, test period: best average precision 0.165 (Abilene, STL) and 0.186 (CESNET, threshold) against
  base rates of 0.46% and 1.90%. Dropouts are found (72-100%), spikes and level shifts mostly are not (0-56%).
- Weak labels (rules R2-R4, raw Abilene): the static threshold has the best average precision (0.142 vs base rate
  0.0095); STL absorbs day-long outages into its trend (0.007).

## Neural retrain (`phase_neural_retrain.json`)
Run on 1x NVIDIA A100-SXM4-80GB (Thunder Compute, TensorFlow 2.21.0, commit `bb79b11`), 2026-11-03. Abilene, 60 min
(12 steps), original fold, seeds 42/43/44, 6 models trained in 7.7 min (early stopping; timing gate 8.2 s per epoch).
Test MAE gain vs persistence (3-seed mean forecast; "reproduced" = reference inside the 3-seed mean ± 2 sd):

| Model | Test gain | Per seed | Band (mean ± 2 sd) | Reference (section 21.3) | Reproduced |
|---|---|---|---|---|---|
| base CNN-LSTM | +0.79% | +0.05 / +0.91 / +0.69 | [−0.34, +1.45] | +1.1% | yes |
| seasonal CNN-LSTM | +2.18% | −1.62 / +3.14 / +3.32 | [−4.00, +7.22] | +2.9% | yes (weak: wide band) |
| seasonal + fallback rule (k = 5) | **+11.22%** | +10.18 / +10.84 / +10.93 | [+9.83, +11.47] | +10.8% | **yes** |

- 90% interval coverage of the exported 3-seed mean forecasts: base 85.7%, seasonal 85.4%, rule 85.4% (GBM 85.7%,
  persistence 86.1% on the same test period; `phase_neural_coverage.json`).
- The rule is the clean reproduction. The seasonal band is so wide that it would contain almost any value; the base model
  barely beats persistence (band includes 0). The rule falls back to persistence on 0.29% of test points.
- On the same test rows the GBM is better: +18.0% vs +11.2% (Phase 2), and 0.426 vs 0.446 pp link-load MAE (Phase 3).
- Warnings (Phase 3 `forecast_driven`, 60 min, point forecasts): base 0.908, rule 0.889, **seasonal 0.000**. The seasonal
  model never predicts any link at or above 50% (its highest planned load is 0.44 while actual reaches 0.75): it smooths
  away the bursts that cause warnings. The fallback rule recovers them. Same pattern as the other smooth methods
  (seasonal_day, mean7, ridge: F1 0).
- Two fixes were needed on the GPU host (both committed): TensorFlow did not find pip's `libcusolver.so.11` (fixed with
  `LD_LIBRARY_PATH` in `infra/gpu/run_retrain.sh`), and the export alignment check had no absolute tolerance at zero
  (16 of 1,170,864 cells: 0 vs 5.5e-7 after the scaler round trip). Metrics are now saved before the export.

## Phase 5: API and dashboard
FastAPI service (`netcap_app/api.py`): health, Prometheus `/metrics`, topology with measured utilisation, link
series, forecast summary and per-flow forecasts with 90% intervals, live scenario evaluation (growth / failure /
upgrade, ASSUMED inputs labelled), the TE LP, anomalies (from Postgres), every result file, and the limitations
page. Static Plotly dashboard at `/` (tabs: overview with the warnings finding, network map, forecasts, scenarios,
anomalies, limitations). Structured JSON request logs. TF-free (tested). All endpoints return 200 on the real results.

## Phase 6: engineering quality (`phase6_quality.json`, `phase6_ci_runs.json`, `phase6_idle_*.json`)
- Tests: **55 passed, 0 failed, 0 skipped** with the raw data present, Docker up (`phase6_quality.json`, commit `ae9c4dd`;
  in CI, tests that need raw data are skipped). Earlier runs had 48 (`phase6_quality_run2_48_tests.json`, superseded).
- Coverage, same run: **46.1%** overall; by package: store 100%, db 85.7%, netcap_app 86.5%, data 80.0%, scenario 63.5%,
  netcap (root modules) 62.8%, forecast 43.1%, anomaly 42.7%, pipeline 9.1%, bench / ops / report 0% (CLIs exercised
  only by the recorded runs). Not padded.
- ruff, yamllint, kubeconform (7/7 resources valid, strict), kustomize build, `docker compose config`: all pass.
- **Docker Compose (Postgres, API, Prometheus, ingest job) and Kubernetes on kind (Postgres StatefulSet, 2 API replicas, ingest Job, NodePort) were run and checked end to end: every endpoint 200, database reachable, Prometheus scraping the API, ingest re-run changes nothing (`phase6_stack_compose.json`, `phase6_stack_kind.json`).** Docker 28.3.2, kind; image 1.59 GB. Two bugs found and fixed by running it: provenance needed `git` (absent in the image), and `POST /api/te` rejected a JSON body (422).
- CI (`.github/workflows/ci.yml`): lint, tests with a Postgres service and coverage, image build, compose config,
  kubeconform. Runs on every push to GitHub: 10 of 10 `ci` runs succeeded, latest on `1e70b0a` (`phase6_ci_runs.json`).

Benchmarks re-run after all other jobs (load average recorded; macOS's Storage scan kept the 5-minute load just
above 4 for three of them, so they are labelled "upper bound, measured under CPU contention"):

| Benchmark | Result | Label |
|---|---|---|
| MAWI trace, chunked pipeline (`phase6_idle_mawi_process.json`) | 411.3 s to exit, 0.07 s teardown, peak footprint 1.63 GB, packets reconciled exactly again | idle |
| MAWI swap-outs in that run | 69,364 system-wide (the earlier identical run had 0; not attributable, see caveat) | idle |
| CESNET → Parquet, 12,405,074 rows | 6.5 s, 1.90M rows/s | upper bound, under contention |
| CESNET → Postgres re-upsert, 12,405,074 rows | 60.4 s, 205k rows/s, 0 rows changed | idle (load 3.49 at save) |
| Query latency (Postgres, indexed / partitioned) | Q1 0.39 / 0.39 ms, Q2 236.9 / 74.1 ms, Q3 3.83 / 4.01 ms (heap 187.7 / 234.7 / 198.6 ms) | upper bound, under contention |
| Forecasting, Abilene 60 min (2,341,440 forecasts) | GBM fit 6.4 s, predict 6.8 s (342k forecasts/s); ETS select + forecast 0.78 s | upper bound, under contention |
| Forecasting, CESNET 1 h (743,121 forecasts) | GBM fit 6.9 s, predict 1.9 s (384k forecasts/s) | upper bound, under contention |

Caveat on swapping: macOS counts swap-outs for the whole system; across three runs of the same bounded-memory
pipeline (1.6-1.9 GB footprint) the count was 92,796, 0 and 69,364, while an idle `sleep` of the same length had 0.
Only the 0 is conclusive.

## Hand-off: running everything from a clean clone

```
git clone https://github.com/Vkartik-3/NetCap.git netcap && cd netcap
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
# data (not in the repository):
#   Abilene: download X01.gz..X24.gz, A, links, topo-2003-04-10.txt into raw/ (see research/README.md)
#   CESNET:  institutions.tar.gz and times.tar.gz from doi:10.5281/zenodo.13382427 into raw_cesnet/, then
mkdir -p raw_cesnet/extracted && tar -xzf raw_cesnet/institutions.tar.gz -C raw_cesnet/extracted && tar -xzf raw_cesnet/times.tar.gz -C raw_cesnet/extracted
# Postgres: either docker compose up -d db (port 5434; set NETCAP_DATABASE_URL=postgresql://netcap:netcap@localhost:5434/netcap)
#           or any local Postgres 17 with user/password/database netcap (default URL uses port 5433)
.venv/bin/python -m netcap.pipeline.ingest parquet abilene && .venv/bin/python -m netcap.pipeline.ingest postgres abilene
.venv/bin/python -m netcap.pipeline.ingest parquet cesnet  && .venv/bin/python -m netcap.pipeline.ingest postgres cesnet
.venv/bin/python -m netcap.bench.query_latency
.venv/bin/python -m netcap.forecast.run abilene && .venv/bin/python -m netcap.forecast.run cesnet
.venv/bin/python -m netcap.scenario.run
.venv/bin/python -m netcap.anomaly.run
.venv/bin/python -m netcap.ops.quality && .venv/bin/python -m netcap.report.summary
.venv/bin/uvicorn netcap_app.api:app --port 8000          # dashboard at http://localhost:8000/
# MAWI (optional; needs softflowd 1.1.1 built as in docs/netcap/NOTES.md 5 and the 5.4 GB pcap in raw_mawi/):
.venv/bin/python -m netcap.pipeline.mawi_ingest 202510011400 --delete-raw --chunk-records 1000000 --duckdb-memory 1GB
```

What you still need to do by hand:
1. (Done here; to repeat) **Docker**: `docker compose up -d --build db api prometheus`, then
   `docker compose --profile jobs run --rm ingest-abilene`, open http://localhost:8000/ and http://localhost:9090/.
2. (Done here; to repeat) **Kubernetes on kind**: `kind create cluster --name netcap --config infra/k8s/kind-config.yaml`,
   `docker build -f infra/Dockerfile -t netcap:1.0 . && kind load docker-image netcap:1.0 --name netcap`,
   `kubectl apply -k infra/k8s && kubectl apply -f infra/k8s/job-ingest.yaml`, then http://localhost:8080/.
   Change the Postgres password in `infra/k8s/postgres.yaml` before any shared use.
3. (Done 2026-11-03; to repeat) **Neural retrain on a GPU machine** (NVIDIA, Linux, Python 3.12): copy the repo and `raw/`,
   then `bash infra/gpu/run_retrain.sh all` (venv, pip, GPU check, timing gate, full run; every step logged to `logs/`).
4. **Git**: the repository uses `https://github.com/Vkartik-3/NetCap.git` as `origin`.
5. Hosted deployment: not done.
6. Re-enable normal sleep if you disabled it: `sudo pmset -a disablesleep 0`.

## Not done
- GÉANT (optional, skipped). MAWI traces 2 and 3 (user instruction).
- Per-kind anomaly thresholds or a combined detector (not tried).
