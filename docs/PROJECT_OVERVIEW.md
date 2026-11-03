# Project overview

NetCap is a backbone capacity-planning platform built around traffic forecasting,
scenario analysis, anomaly detection, and operational deployment. This file maps
the repository's main components and data sources.

## Forecasting components

| Path | What it is |
|---|---|
| `common.py` | Shared forecasting code: data loading, cleaning rules R1-R4 and G1-G4, scaler, windows, baselines, models, and report page |
| `scripts/` (19 files) | Phase scripts `00_setup.py` .. `10_robustness.py` and `scripts/wsl/` GPU helpers |
| `research/RESEARCH_LOG.md` | Forecasting research log with phases, decisions, and results |
| `docs/presentation/` (3 files) | Review slides and viva Q&A |
| `results/metrics.csv`, `results/03f_geant_tuning.csv`, `results/figures/` (28 files) | Logged forecasting results and figures |
| `research/README.md`, `requirements.txt`, `requirements-gpu.txt`, `NetCap_Project_Explanation_Guide.pdf` | Forecasting documentation and environment pins |
| `.gitattributes`, `.vscode/settings.json`, `raw/.gitkeep`, `processed/.gitkeep`, `models/.gitkeep` | Repository settings and placeholders |

## Platform components

| Path | What it is |
|---|---|
| `netcap/` | The NetCap package: data layer, Parquet and Postgres stores, pipelines, benchmarks, audit scripts, forecasting (`netcap/forecast`), scenario engine and TE (`netcap/scenario`), anomaly detection (`netcap/anomaly`), operations helpers (`netcap/ops`). Never imports TensorFlow. |
| `netcap_app/` | FastAPI service and the dashboard (static page). Never imports TensorFlow. |
| `netcap/neural/` | The separate neural retraining step; this is the only NetCap code that uses TensorFlow and it runs in its own environment |
| `infra/`, `docker-compose.yml`, `.github/workflows/ci.yml` | Dockerfile, Prometheus config, Kubernetes manifests for kind, CI |
| `README.md` (root) | NetCap README |
| `tests_netcap/` | NetCap tests |
| `docs/netcap/` | NetCap documentation (`NOTES.md`, and later `RESULTS.md` etc.) |
| `results/netcap/` | Every NetCap result as JSON with command, code version, seed, date and data version |
| `pyproject.toml`, `requirements.txt`, `PROJECT_OVERVIEW.md` | Packaging, pinned NetCap dependencies, this file |

Code in `netcap/data/abilene.py` implements the Abilene cleaning rules R1-R4
(`common.abilene_clean`); `tests_netcap/test_abilene.py` checks that both give identical output on the real
data. Experiment results quoted anywhere in NetCap are labelled with their section number.

## Data sources (not in the repository)

| Dataset | Source | Terms |
|---|---|---|
| Abilene traffic matrices (Y. Zhang) | https://www.cs.utexas.edu/~yzhang/research/AbileneTM/ | No license stated |
| CESNET-TimeSeries24, institutions subset | https://doi.org/10.5281/zenodo.13382427 | CC-BY-4.0. Cite: J. Koumar, K. Hynek, T. Čejka, P. Šiška, "CESNET-TimeSeries24: Time Series Dataset for Network Traffic Anomaly Detection and Forecasting", Zenodo, 2024-09-27, doi:10.5281/zenodo.13382427 (authors and title from the Zenodo record) |
| MAWI Working Group Traffic Archive, samplepoint-F (WIDE Project) | https://mawi.wide.ad.jp/mawi/ | Only derived aggregates are kept; raw traces are never stored in the repository |
| GÉANT TOTEM traffic matrices (downloaded in Phase 0, not used) | https://totem.info.ucl.ac.be/dataset.html | Cite: S. Uhlig, B. Quoitin, S. Balon, J. Lepropre, "Providing public intradomain traffic matrices to the research community", ACM SIGCOMM CCR 36(1), 2006 |
