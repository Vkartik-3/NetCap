# NetCap

**Backbone capacity planning on real traffic: forecasting, failure scenarios, traffic engineering and anomaly detection, served through an API and dashboard.**

[![ci](https://github.com/Vkartik-3/NetCap/actions/workflows/ci.yml/badge.svg)](https://github.com/Vkartik-3/NetCap/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.13-blue)
![tests](https://img.shields.io/badge/tests-48%20passing-brightgreen)
![k8s](https://img.shields.io/badge/kubernetes-kind%20tested-326ce5)

**Live dashboard (static snapshot):** https://vkartik-3.github.io/NetCap/

![NetCap network view: peak link utilisation on the Abilene backbone](docs/images/dashboard_network.png)

NetCap answers the questions a capacity planner asks about a backbone: *how busy is each link, what will traffic
look like in the next hour, what breaks if a link or router fails, which upgrade fixes it, and how much better
could routing be?* It does this on measured traffic (the Abilene research backbone, 12 routers and 144
origin-destination flows at 5-minute resolution), with every number traceable to a result file that records the
command, code version, seed, date and data version.

---

## Key results

| Question | Answer | Evidence |
|---|---|---|
| Does the routing model match reality? | OSPF shortest paths with ECMP reproduce the dataset's routing matrix for **144 / 144 flows** | [`phase3_routing_gate.json`](results/netcap/phase3_routing_gate.json) |
| How well can traffic be forecast 60 minutes ahead? | A gradient-boosting model beats persistence by **+18.0%** (95% CI +14.6 to +20.3), and by +16.9 to +20.5% in three more rolling-origin folds; chosen on validation every time | [`phase2_forecast_abilene.json`](results/netcap/phase2_forecast_abilene.json) |
| What is the biggest failure risk? | Losing the IPLS–KSCY link pushes the 2.48 Gbit/s ATLA→IPLS link to **297%** of capacity | [`phase3_scenarios.json`](results/netcap/phase3_scenarios.json) |
| Which upgrade removes it? | ATLA–IPLS to 9.92 Gbit/s (assumed size): worst single failure drops to **77%** | same |
| How much could traffic engineering help? | The min-max-utilisation LP bound is **0.70×** OSPF's peak on a typical hour (95% CI 0.696 to 0.712) and **0.49×** under the worst failure | same (3,405 LPs per case) |
| Do better forecasts give better congestion warnings? | **No.** The most accurate model has 14% lower link-load error than persistence but a warning F1 of 0.890 vs 0.896 | same, `forecast_driven` |
| Does the packet pipeline lose data? | **207,302,561** packets reconciled exactly across an independent pcap walk, the flow meter and 56,523,852 flow records | [`phase1_mawi_…_chunked_lowmem.json`](results/netcap/phase1_mawi_202510011400_chunked_lowmem.json) |
| Does it run as a service? | Docker Compose and Kubernetes (kind) stacks pass an end-to-end check: every endpoint 200, Postgres reachable, Prometheus scraping | [`phase6_stack_kind.json`](results/netcap/phase6_stack_kind.json) |

Full results, the phase status table and every caveat: **[docs/netcap/RESULTS.md](docs/netcap/RESULTS.md)**.

---

## Architecture

```mermaid
flowchart LR
  subgraph Sources["Real data"]
    A[Abilene traffic matrices<br/>48,384 steps x 144 flows]
    C[CESNET-TimeSeries24<br/>12.4M rows, 283 institutions]
    M[MAWI pcap<br/>207M packets]
  end
  subgraph Ingest["Data layer (idempotent)"]
    P[(Parquet store<br/>DuckDB)]
    G[(PostgreSQL<br/>upserts, partitions)]
  end
  subgraph Analytics
    F[Forecasting<br/>baselines, ETS, ARIMA,<br/>ridge, GBM, conformal]
    S[Scenario engine<br/>OSPF/ECMP, N-1,<br/>growth, upgrades]
    T[TE LP<br/>min-max utilisation]
    D[Anomaly detection<br/>threshold, STL, iForest]
  end
  API[FastAPI + dashboard<br/>Prometheus /metrics<br/>JSON logs]
  A --> P & G
  C --> P & G
  M -->|softflowd, IPFIX/TCP,<br/>chunked decode| P & G
  P --> F --> S
  P --> S --> T
  P --> D --> G
  F & S & T & G --> API
```

---

## What's inside

### Data layer
- Idempotent ingestion into **Parquet** (content-hashed, atomic replace, manifest) and **PostgreSQL** (COPY to staging,
  `ON CONFLICT … WHERE IS DISTINCT FROM`). A re-run changes 0 rows and 0 files, verified with row checksums.
- **CESNET-TimeSeries24** institutions: 10,446,225 rows at 10 minutes (12,405,074 over three aggregations).
- **MAWI packet trace → flow records**: softflowd exporting IPFIX over TCP to a custom collector and numpy decoder, then
  chunked aggregation with DuckDB spill. Peak memory went from 16.8 GB to **1.85 GB**.
- **Query tuning** on the 10.4M-row table: a composite B-tree takes a one-institution, one-week query from **186 ms
  to 0.37 ms**. For a month-wide aggregate the index is useless and monthly range partitions win (230 → 72 ms).

### Forecasting
- Persistence, seasonal naive, 7-day mean, blend, ETS, SARIMA-type (ARMA errors + Fourier seasonality), ridge and a
  pooled **gradient-boosting** model, all on the **same four rolling-origin folds**, at 5, 30 and 60 minutes.
- Every choice (weights, penalties, thresholds, model) is made on **validation**; test is only scored.
- **Split-conformal 90% intervals**, **Diebold-Mariano** tests and **day-block bootstrap** confidence intervals.
- Five attempts at the burst problem (one flow, LOSA→CHIN, with long out-of-range bursts), all reported.
- Also run on 283 CESNET institution series: +16.6 to +20.3% at 1 hour, +44 to +50% at 6 hours over persistence.

![Forecasts tab](docs/images/dashboard_forecasts.png)

### Scenario engine and traffic engineering
- NetworkX topology, OSPF shortest paths with per-router ECMP splitting, validated against ground truth.
- Growth (with labelled **assumed** rates), N-1 sweeps over all 15 link pairs and 12 routers, capacity upgrades,
  combined scenarios, stranded-traffic accounting.
- A **min-max link-utilisation linear program** (HiGHS) as a lower bound for any routing, solved hourly across the
  whole dataset with bootstrap confidence intervals.

![Scenario tab: the IPLS-KSCY failure](docs/images/dashboard_scenarios.png)

### Anomaly detection
- Static robust-z threshold, **STL** robust residuals and **Isolation Forest**, scored against **injected** synthetic
  anomalies (spikes, level shifts, dropouts) and **weak labels** from data-quality rules (outages).
- Honest outcome: dropouts are found, spikes and level shifts mostly are not (best average precision 0.165-0.186).

### Service and operations
- **FastAPI**: topology, link series, forecasts with intervals, live scenario evaluation, TE, anomalies, results.
- **Dashboard** (Plotly, no build step): overview, network map, forecasts, scenarios, anomalies, limitations.
- **Prometheus** `/metrics`, structured JSON request logs, health checks.
- **Docker** image, **Docker Compose** (Postgres, API, Prometheus, ingest jobs), **Kubernetes** manifests (Postgres
  StatefulSet + Secret, 2 API replicas with probes and limits, NodePort, ingest Job), **GitHub Actions** CI.
- The serving path never imports TensorFlow (tested).

![Overview tab](docs/images/dashboard_overview.png)

---

## Quick start

**Local**
```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest                                   # 48 tests (those needing raw data skip without it)
.venv/bin/uvicorn netcap_app.api:app --port 8000   # dashboard: http://localhost:8000
```

**Docker Compose**
```bash
docker compose up -d --build db api prometheus
docker compose --profile jobs run --rm ingest-abilene     # loads raw/ into Parquet and Postgres
open http://localhost:8000        # dashboard
open http://localhost:9090        # Prometheus
```

**Kubernetes (kind)**
```bash
kind create cluster --name netcap --config infra/k8s/kind-config.yaml
docker build -f infra/Dockerfile -t netcap:1.0 . && kind load docker-image netcap:1.0 --name netcap
kubectl apply -k infra/k8s && kubectl apply -f infra/k8s/job-ingest.yaml
open http://localhost:8080
```

**Static dashboard for GitHub Pages**: `python -m netcap.report.static_site` writes `site/` (published on the `gh-pages` branch).

**Reproduce every result** (after downloading the datasets below):
```bash
python -m netcap.pipeline.ingest parquet abilene && python -m netcap.pipeline.ingest postgres abilene
python -m netcap.pipeline.ingest parquet cesnet  && python -m netcap.pipeline.ingest postgres cesnet
python -m netcap.bench.query_latency
python -m netcap.forecast.run abilene && python -m netcap.forecast.run cesnet
python -m netcap.scenario.run
python -m netcap.anomaly.run
python -m netcap.ops.quality && python -m netcap.report.summary
```

---

## Repository layout

| Path | Contents |
|---|---|
| `netcap/data`, `netcap/store`, `netcap/db`, `netcap/pipeline` | Loaders, Parquet writer, Postgres schema and upserts, ingestion CLIs |
| `netcap/forecast` | Panels and folds, methods, evaluation (conformal, DM, bootstrap), runner |
| `netcap/scenario` | Routing engine, scenarios, TE linear program |
| `netcap/anomaly` | Detectors, injection, evaluation |
| `netcap/bench`, `netcap/ops`, `netcap/report` | Benchmarks, measurement and quality tooling, summary builder, static-site export |
| `netcap_app` | FastAPI service and dashboard |
| `infra`, `docker-compose.yml`, `.github/workflows` | Dockerfile, Kubernetes manifests, Prometheus config, CI |
| `tests_netcap` | Test suite |
| `research` | Forecasting research code (`common.py`, phase scripts, CNN-LSTM models), its log, figures and TensorFlow requirements |
| `results/netcap` | Every result as JSON with provenance, plus a report per phase |
| `docs/netcap` | Results, decisions, limitations, working notes |

More detail: [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md).

---

## Limitations

One link-level network (Abilene, 2004 traffic on 2003 OSPF weights; times of day are inferred); 5-minute averages
hide sub-minute bursts; growth rates and upgrade sizes are assumptions; the LP is a bound, not a deployable routing;
injected anomalies are synthetic. The neural-network retrain has not been run yet (it needs an NVIDIA GPU). Full list:
[docs/netcap/LIMITATIONS.md](docs/netcap/LIMITATIONS.md). Decisions, including every change made after seeing a
result: [docs/netcap/DECISIONS.md](docs/netcap/DECISIONS.md).

---

## Data sources

| Dataset | Source | Terms |
|---|---|---|
| Abilene traffic matrices (Y. Zhang) | https://www.cs.utexas.edu/~yzhang/research/AbileneTM/ | research dataset, no license stated |
| CESNET-TimeSeries24 (institutions) | J. Koumar, K. Hynek, T. Čejka, P. Šiška, *CESNET-TimeSeries24*, Zenodo, 2024, [doi:10.5281/zenodo.13382427](https://doi.org/10.5281/zenodo.13382427) | CC-BY-4.0 |
| MAWI Working Group Traffic Archive, samplepoint-F (WIDE Project) | https://mawi.wide.ad.jp/mawi/ | only derived aggregates are stored |

Raw data is never committed; see [docs/netcap/RESULTS.md](docs/netcap/RESULTS.md) for download and setup steps.
