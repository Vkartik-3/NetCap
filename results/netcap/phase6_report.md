# Phase 6 report: engineering quality
Source: `phase6_quality.json`, `phase6_idle_*.json`, `phase1_mawi_202510011400_idle.json`.
Built: Dockerfile (no TensorFlow, non-root, healthcheck), docker-compose.yml (Postgres, API, Prometheus, ingest jobs),
Kubernetes manifests for kind (namespace, Postgres StatefulSet + Secret, API Deployment with probes and limits,
NodePort Service, ingest Job, kustomization), GitHub Actions CI, JSON request logs, Prometheus metrics.
Measured (first run; current: 55 passed, coverage 46.1%, see RESULTS.md): 48 tests passed, 0 failed, 0 skipped; coverage 45.1% (store 100%, db 85.7%, data 80.0%, app 77.6%,
scenario 46.5%, forecast 43.1%, anomaly 42.7%, pipeline 9.1%, CLIs 0%); ruff, yamllint, kubeconform (7/7 valid,
strict), kustomize and compose config pass. Benchmarks re-run after the other jobs: see RESULTS.md (MAWI run idle;
three runs labelled "upper bound, measured under CPU contention" because macOS's Storage scan kept the load above 4).
Container stack (after Docker was started): Docker Compose (Postgres, API, Prometheus, ingest job) and Kubernetes on kind (Postgres StatefulSet, 2 API replicas, ingest Job, NodePort) were run and checked end to end: every endpoint 200, database reachable, Prometheus scraping the API, ingest re-run changes nothing (`phase6_stack_compose.json`, `phase6_stack_kind.json`). Bugs found by running it and fixed: provenance needed git; POST /api/te rejected a JSON body. CI runs on GitHub on every push: 10 of 10 runs succeeded (`phase6_ci_runs.json`).
